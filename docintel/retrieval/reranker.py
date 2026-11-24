"""Second-stage rerankers that jointly score (query, passage) pairs.

- cross-encoder: transformer cross-encoder (ms-marco MiniLM by default)
- llm: listwise LLM relevance grading
- lexical: dependency-free feature reranker (term coverage, phrase/bigram hits,
  numeric matches, proximity) blended with the first-stage rank
"""
import logging
import math
import re

from ..text_utils import tokenize

log = logging.getLogger(__name__)


class LexicalReranker:
    name = "lexical-features"

    def score(self, query: str, passages: list[str], prior: list[float] | None = None) -> list[float]:
        q_toks = tokenize(query)
        q_set = set(q_toks)
        q_bigrams = set(zip(q_toks, q_toks[1:]))
        q_nums = set(re.findall(r"\d+(?:\.\d+)?", query))
        scores = []
        for i, text in enumerate(passages):
            toks = tokenize(text)
            t_set = set(toks)
            coverage = len(q_set & t_set) / max(len(q_set), 1)
            bigrams = set(zip(toks, toks[1:]))
            bigram_hit = len(q_bigrams & bigrams) / max(len(q_bigrams), 1) if q_bigrams else 0.0
            num_hit = len(q_nums & set(re.findall(r"\d+(?:\.\d+)?", text))) / max(len(q_nums), 1) if q_nums else 0.0
            proximity = self._proximity(toks, q_set)
            length_pen = 1.0 / (1.0 + math.log1p(max(len(toks) - 250, 0) / 250))
            s = (0.5 * coverage + 0.2 * bigram_hit + 0.1 * num_hit + 0.2 * proximity) * length_pen
            if prior:
                s += 0.15 * prior[i]
            scores.append(s)
        return scores

    @staticmethod
    def _proximity(toks: list[str], q_set: set[str], window: int = 25) -> float:
        if not q_set or not toks:
            return 0.0
        best = 0
        positions = [i for i, t in enumerate(toks) if t in q_set]
        for start in positions:
            seen = {toks[j] for j in positions if start <= j < start + window}
            best = max(best, len(seen))
        return best / len(q_set)


class CrossEncoderReranker:
    def __init__(self, model_name: str):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name, max_length=512)
        self.name = f"cross-encoder:{model_name}"

    def score(self, query: str, passages: list[str], prior=None) -> list[float]:
        return [float(s) for s in self.model.predict([(query, p) for p in passages], batch_size=16)]


class LLMReranker:
    name = "llm-listwise"

    def __init__(self, llm, fallback: LexicalReranker):
        self.llm, self.fallback = llm, fallback

    def score(self, query: str, passages: list[str], prior=None) -> list[float]:
        listing = "\n\n".join(f"[{i}] {p[:900]}" for i, p in enumerate(passages))
        prompt = (
            "Rate how well each passage helps answer the question on a 0-10 scale.\n"
            f"Question: {query}\n\nPassages:\n{listing}\n\n"
            'Return ONLY JSON: {"scores": [<number for passage 0>, <passage 1>, ...]}'
        )
        try:
            out = self.llm.chat_json([{"role": "user", "content": prompt}], temperature=0, max_tokens=300)
            scores = [float(s) for s in out["scores"]]
            if len(scores) == len(passages):
                return scores
        except Exception as exc:
            log.warning("LLM rerank failed, using lexical: %s", exc)
        return self.fallback.score(query, passages, prior)


class Reranker:
    def __init__(self, cfg, llm):
        backend = cfg.RERANKER_BACKEND.lower()
        lexical = LexicalReranker()
        self.impl = None
        if backend in {"auto", "cross-encoder"}:
            try:
                self.impl = CrossEncoderReranker(cfg.CROSS_ENCODER_MODEL)
            except Exception as exc:
                if backend != "auto":
                    raise
                log.info("Cross-encoder unavailable (%s)", exc.__class__.__name__)
        if self.impl is None and backend == "llm" and llm.available:
            self.impl = LLMReranker(llm, lexical)
        if self.impl is None and backend != "none":
            self.impl = lexical

    @property
    def name(self) -> str:
        return self.impl.name if self.impl else "none"

    def rerank(self, query: str, candidates: list[dict], top_n: int) -> list[dict]:
        if not self.impl or not candidates:
            return candidates[:top_n]
        n = len(candidates)
        prior = [1.0 - i / n for i in range(n)]  # normalised first-stage rank
        scores = self.impl.score(query, [c["content"] for c in candidates], prior)
        for c, s, r in zip(candidates, scores, range(1, n + 1)):
            c["first_stage_rank"] = r
            c["rerank_score"] = float(s)
        return sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)[:top_n]
