"""Grounded answer generation with numbered source citations.

With an LLM configured, answers are generated from the reranked context and must cite
[n] markers. Without one, an extractive answer is assembled from the best-matching
sentences / table rows so the system remains fully functional offline."""
import logging
import re

from .ingestion.chunker import split_sentences
from .text_utils import tokenize

log = logging.getLogger(__name__)

NOT_FOUND = "I could not find this information in the indexed documents."

SYSTEM_PROMPT = f"""You are a meticulous document analyst. Answer the user's question using ONLY the numbered sources.
Rules:
- After every factual claim add citation markers like [1] or [2][4] that point to the supporting sources.
- Sources may be prose, markdown tables, OCR of scanned pages, or descriptions of charts/figures. Quote exact numbers.
- If sources disagree, say so and cite both.
- If the sources do not contain the answer, reply exactly: "{NOT_FOUND}"
- Be concise: 1-3 short paragraphs or a bullet list."""

MODALITY_LABEL = {"text": "text", "table": "table", "image": "figure", "chart": "chart", "ocr": "scanned page"}


def source_label(chunk: dict) -> str:
    page = f", p.{chunk['page']}" if chunk.get("page") else ""
    return f"{chunk['filename']}{page}, {MODALITY_LABEL.get(chunk['modality'], chunk['modality'])}"


def build_context(chunks: list[dict], max_chars: int = 1800) -> str:
    return "\n\n".join(f"[{i}] ({source_label(c)})\n{c['content'][:max_chars]}" for i, c in enumerate(chunks, 1))


def extract_citations(answer: str, n_sources: int) -> list[int]:
    found = []
    for group in re.findall(r"\[(\d+(?:\s*[,;]\s*\d+)*)\]", answer):
        for num in re.split(r"[,;]", group):
            n = int(num.strip())
            if 1 <= n <= n_sources and n not in found:
                found.append(n)
    return found


class AnswerGenerator:
    def __init__(self, llm):
        self.llm = llm

    def generate(self, question: str, chunks: list[dict], use_llm: bool = True) -> dict:
        if not chunks:
            return {"answer": NOT_FOUND, "citations": [], "method": "none"}
        method = "extractive"
        answer = ""
        if use_llm and self.llm.available:
            try:
                answer = self.llm.chat([
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Sources:\n{build_context(chunks)}\n\nQuestion: {question}"},
                ])
                method = "llm"
            except Exception as exc:
                log.warning("LLM generation failed, falling back to extractive: %s", exc)
        if not answer:
            answer = self._extractive(question, chunks)
        cited = extract_citations(answer, len(chunks))
        return {
            "answer": answer,
            "method": method,
            "citations": [
                {
                    "n": i,
                    "chunk_id": c["id"],
                    "doc_id": c["doc_id"],
                    "filename": c["filename"],
                    "page": c.get("page"),
                    "modality": c["modality"],
                    "label": source_label(c),
                    "snippet": c["content"][:600],
                    "image": c.get("meta", {}).get("image"),
                    "cited": i in cited,
                    "scores": {k: c.get(k) for k in ("score", "rerank_score", "bm25_rank", "dense_rank")},
                }
                for i, c in enumerate(chunks, 1)
            ],
        }

    @staticmethod
    def _units(chunk: dict) -> list[str]:
        """Sentences for prose; for tables, each row rendered as 'column: value' pairs."""
        content = chunk.get("meta", {}).get("table_markdown") or chunk["content"]
        if chunk["modality"] != "table":
            return split_sentences(content.replace("\n", " ")) if chunk["modality"] != "chart" else content.splitlines()
        rows = [ln for ln in content.splitlines() if ln.startswith("|") and not set(ln) <= set("|-: ")]
        if len(rows) < 2:
            return content.splitlines()
        cells = lambda ln: [c.strip() for c in ln.strip("|").split("|")]  # noqa: E731
        header = cells(rows[0])
        where = f"Table (p.{chunk['page']})" if chunk.get("page") else "Table"
        return [f"{where}: " + ", ".join(f"{h}: {v}" for h, v in zip(header, cells(r)) if v) for r in rows[1:]]

    @classmethod
    def _extractive(cls, question: str, chunks: list[dict], max_units: int = 4) -> str:
        q = set(tokenize(question))
        if not q:
            return NOT_FOUND
        scored: list[tuple[float, int, str]] = []
        for n, c in enumerate(chunks, 1):
            for pos, unit in enumerate(cls._units(c)):
                toks = set(tokenize(unit))
                if not toks:
                    continue
                overlap = len(q & toks) / len(q)
                # prefer higher-ranked sources and earlier sentences slightly
                scored.append((overlap - 0.02 * (n - 1) - 0.002 * pos, n, unit.strip()))
        scored.sort(reverse=True)
        candidates = [(s, n, u) for s, n, u in scored if s > 0.15]
        # Pass 1: best unit per source (covers multi-part questions); pass 2: fill remaining slots.
        picked: list[tuple[int, str]] = []
        used_sources: set[int] = set()
        for s, n, u in candidates:
            if n not in used_sources and len(picked) < max_units and s >= candidates[0][0] * 0.4:
                picked.append((n, u))
                used_sources.add(n)
        for s, n, u in candidates:
            if len(picked) >= max_units:
                break
            if (n, u) not in picked and s >= candidates[0][0] * 0.6:
                picked.append((n, u))
        if not picked:
            return NOT_FOUND
        lines = [f"{u.rstrip('.')}. [{n}]" for n, u in picked]
        return "Based on the retrieved sources:\n" + "\n".join(f"- {ln}" for ln in lines)
