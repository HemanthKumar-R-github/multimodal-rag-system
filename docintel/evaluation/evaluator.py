"""Benchmark retrieval configurations and answer quality over a labelled QA set."""
import json
import logging
import random
import time

from ..generation import NOT_FOUND
from ..llm import parse_json
from ..text_utils import tokenize
from . import metrics as M

log = logging.getLogger(__name__)

CONFIGS = {
    "bm25": ("bm25", False),
    "dense": ("dense", False),
    "hybrid": ("hybrid", False),
    "hybrid+rerank": ("hybrid", True),
}

JUDGE_PROMPT = """You are grading a RAG system. Score strictly.
Question: {question}
Reference answer: {reference}
Retrieved context:
{context}
System answer: {answer}

Return ONLY JSON: {{"faithfulness": 1-5 (claims supported by context), "correctness": 1-5 (agrees with reference), "reason": "<one sentence>"}}"""


class Evaluator:
    def __init__(self, pipeline):
        self.p = pipeline

    def run(self, dataset: dict, configs: list[str] | None = None, k: int = 5,
            answer_config: str = "hybrid+rerank", generate: bool = True, use_llm: bool = True,
            llm_judge: bool = False) -> dict:
        configs = [c for c in (configs or list(CONFIGS)) if c in CONFIGS]
        if answer_config not in configs:
            configs.append(answer_config)
        items = dataset.get("items", [])
        per_config: dict[str, list[dict]] = {c: [] for c in configs}
        details = []

        for item in items:
            q, targets = item["question"], item.get("relevant", [])
            row = {"question": q, "reference": item.get("answer", ""), "retrieval": {}}
            retrieved_for_answer = []
            for name in configs:
                mode, rerank = CONFIGS[name]
                t0 = time.perf_counter()
                retrieved = self.p.retrieve(q, mode=mode, rerank=rerank, top_k=k)
                latency = (time.perf_counter() - t0) * 1000
                m = M.retrieval_metrics(retrieved, targets, k) if targets else {}
                m["latency_ms"] = latency
                per_config[name].append(m)
                row["retrieval"][name] = {
                    "metrics": {kk: round(v, 4) for kk, v in m.items()},
                    "top": [f"{c['filename']} p.{c.get('page')} ({c['modality']})" for c in retrieved],
                    "relevant_flags": [bool(M.matched_targets(c, targets)) for c in retrieved],
                }
                if name == answer_config:
                    retrieved_for_answer = retrieved

            if generate:
                gen = self.p.generator.generate(q, retrieved_for_answer, use_llm=use_llm)
                answer = gen["answer"]
                cited = [retrieved_for_answer[c["n"] - 1] for c in gen["citations"] if c["cited"]]
                contexts = [c["content"] for c in retrieved_for_answer]
                abstained = answer.strip() == NOT_FOUND
                am = {
                    "token_f1": M.token_f1(answer, row["reference"]) if row["reference"] else None,
                    "rouge_l": M.rouge_l(answer, row["reference"]) if row["reference"] else None,
                    "faithfulness": None if abstained else M.faithfulness(answer, contexts),
                    "citation_coverage": None if abstained else M.citation_coverage(answer),
                    "citation_precision": M.citation_precision(cited, targets) if targets else None,
                    "answer_relevance": M.answer_relevance(q, answer),
                    "abstained": float(abstained),
                }
                if llm_judge and self.p.llm.available:
                    am.update(self._judge(q, row["reference"], contexts, answer))
                row["answer"] = answer
                row["answer_method"] = gen["method"]
                row["answer_metrics"] = {kk: (round(v, 4) if isinstance(v, float) else v) for kk, v in am.items()}
            details.append(row)

        summary = {
            "n_questions": len(items),
            "k": k,
            "retrieval": {
                name: {key: M.mean(r.get(key) for r in rows) for key in (rows[0].keys() if rows else [])}
                for name, rows in per_config.items()
            },
        }
        if generate and details:
            keys = details[0]["answer_metrics"].keys()
            summary["answer"] = {key: M.mean(d["answer_metrics"].get(key) for d in details) for key in keys}
            summary["answer_config"] = answer_config
        return {"summary": summary, "details": details}

    def _judge(self, question: str, reference: str, contexts: list[str], answer: str) -> dict:
        prompt = JUDGE_PROMPT.format(question=question, reference=reference or "(none)",
                                     context="\n---\n".join(c[:700] for c in contexts), answer=answer)
        try:
            out = self.p.llm.chat_json([{"role": "user", "content": prompt}], temperature=0, max_tokens=200)
            return {
                "judge_faithfulness": (float(out["faithfulness"]) - 1) / 4,
                "judge_correctness": (float(out["correctness"]) - 1) / 4,
            }
        except Exception as exc:
            log.warning("LLM judge failed: %s", exc)
            return {}

    # Synthetic test-set generation ---------------------------------------
    def synthesize(self, n: int = 10, seed: int = 7) -> dict:
        chunks = [c for c in self.p.store.all_chunks() if len(c["content"].split()) > 15]
        random.Random(seed).shuffle(chunks)
        items = []
        for ch in chunks[: n * 2]:
            if len(items) >= n:
                break
            item = self._llm_question(ch) if self.p.llm.available else self._heuristic_question(ch)
            if item:
                item["relevant"] = [{"doc": ch["filename"], "page": ch.get("page")}]
                item["source_modality"] = ch["modality"]
                items.append(item)
        return {"name": f"synthetic-{len(items)}", "items": items}

    def _llm_question(self, chunk: dict) -> dict | None:
        prompt = (
            "Write one specific factual question that can be answered ONLY from this passage, and its short answer.\n"
            f"Passage ({chunk['modality']}):\n{chunk['content'][:1500]}\n\n"
            'Return ONLY JSON: {"question": "...", "answer": "..."}'
        )
        try:
            out = parse_json(self.p.llm.chat([{"role": "user", "content": prompt}], temperature=0.3, max_tokens=200))
            return {"question": out["question"], "answer": out["answer"]}
        except Exception:
            return self._heuristic_question(chunk)

    @staticmethod
    def _heuristic_question(chunk: dict) -> dict | None:
        sents = [s for s in M.split_sentences(chunk["content"]) if 6 <= len(s.split()) <= 40]
        if not sents:
            return None
        sent = max(sents, key=lambda s: len(set(tokenize(s))))
        terms = sorted(set(tokenize(sent, stem=False)), key=len, reverse=True)[:4]
        return {"question": f"What do the documents say about {' '.join(terms)}?", "answer": sent}


def load_dataset(path) -> dict:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        data = {"name": str(path), "items": data}
    return data
