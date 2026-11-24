"""Retrieval and answer-quality metrics.

Relevance targets in an eval item (any combination):
  {"doc": "report.pdf", "page": 3}      chunk from that document (and page, if given)
  {"contains": "net revenue"}           chunk text contains the phrase (case-insensitive)
  {"doc": "report.pdf", "contains": ...} both conditions
"""
import math
import re
from collections import Counter

from ..ingestion.chunker import split_sentences
from ..text_utils import tokenize

CITATION_RE = re.compile(r"\[\d+(?:\s*[,;]\s*\d+)*\]")


# Relevance ---------------------------------------------------------------
def target_matches(chunk: dict, target: dict) -> bool:
    if "doc" in target and target["doc"].lower() != str(chunk.get("filename", "")).lower():
        return False
    if target.get("page") is not None and int(target["page"]) != chunk.get("page"):
        return False
    if "contains" in target and target["contains"].lower() not in chunk["content"].lower():
        return False
    return any(k in target for k in ("doc", "page", "contains"))


def matched_targets(chunk: dict, targets: list[dict]) -> set[int]:
    return {i for i, t in enumerate(targets) if target_matches(chunk, t)}


# Retrieval metrics -------------------------------------------------------
def retrieval_metrics(retrieved: list[dict], targets: list[dict], k: int) -> dict:
    top = retrieved[:k]
    rel_flags = [bool(matched_targets(c, targets)) for c in top]
    covered: set[int] = set()
    for c in top:
        covered |= matched_targets(c, targets)
    first = next((i for i, f in enumerate(rel_flags, 1) if f), None)
    dcg = sum(1 / math.log2(i + 1) for i, f in enumerate(rel_flags, 1) if f)
    ideal_hits = min(len(targets), k) or 1
    idcg = sum(1 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return {
        f"hit@{k}": float(first is not None),
        f"recall@{k}": len(covered) / len(targets) if targets else 0.0,
        f"precision@{k}": sum(rel_flags) / k,
        "mrr": 1.0 / first if first else 0.0,
        f"ndcg@{k}": min(dcg / idcg, 1.0),
    }


# Answer metrics ----------------------------------------------------------
def _strip_citations(text: str) -> str:
    return CITATION_RE.sub("", text)


def token_f1(prediction: str, reference: str) -> float:
    p, r = tokenize(_strip_citations(prediction)), tokenize(reference)
    if not p or not r:
        return 0.0
    common = sum((Counter(p) & Counter(r)).values())
    if common == 0:
        return 0.0
    precision, recall = common / len(p), common / len(r)
    return 2 * precision * recall / (precision + recall)


def rouge_l(prediction: str, reference: str) -> float:
    p, r = tokenize(_strip_citations(prediction)), tokenize(reference)
    if not p or not r:
        return 0.0
    prev = [0] * (len(r) + 1)
    for a in p:
        cur = [0]
        for j, b in enumerate(r, 1):
            cur.append(prev[j - 1] + 1 if a == b else max(prev[j], cur[j - 1]))
        prev = cur
    lcs = prev[-1]
    if lcs == 0:
        return 0.0
    precision, recall = lcs / len(p), lcs / len(r)
    return 2 * precision * recall / (precision + recall)


def answer_sentences(answer: str) -> list[str]:
    lines = [ln.strip(" -*\t") for ln in answer.splitlines()]
    sents = []
    for ln in lines:
        if ln and not ln.lower().startswith("based on the retrieved sources"):
            sents.extend(split_sentences(ln) if not ln.startswith("|") else [ln])
    return [s for s in sents if tokenize(_strip_citations(s))]


def faithfulness(answer: str, contexts: list[str], threshold: float = 0.6) -> float:
    """Share of answer sentences whose content words are mostly present in the retrieved context."""
    sents = answer_sentences(answer)
    if not sents:
        return 0.0
    ctx_vocab = set(tokenize(" ".join(contexts)))
    supported = 0
    for s in sents:
        toks = set(tokenize(_strip_citations(s)))
        if toks and len(toks & ctx_vocab) / len(toks) >= threshold:
            supported += 1
    return supported / len(sents)


def citation_coverage(answer: str) -> float:
    sents = answer_sentences(answer)
    return sum(bool(CITATION_RE.search(s)) for s in sents) / len(sents) if sents else 0.0


def citation_precision(cited_chunks: list[dict], targets: list[dict]) -> float | None:
    if not cited_chunks:
        return None
    return sum(bool(matched_targets(c, targets)) for c in cited_chunks) / len(cited_chunks)


def answer_relevance(question: str, answer: str) -> float:
    q = set(tokenize(question))
    a = set(tokenize(_strip_citations(answer)))
    return len(q & a) / len(q) if q else 0.0


def mean(values) -> float | None:
    vals = [v for v in values if v is not None]
    return round(sum(vals) / len(vals), 4) if vals else None
