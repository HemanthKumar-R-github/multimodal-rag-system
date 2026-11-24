"""Sparse lexical retrieval with Okapi BM25 over an inverted index.

Uses the Lucene IDF variant  idf = ln(1 + (N - n + 0.5) / (n + 0.5)),  which stays
positive even for terms present in most documents (the classic Robertson IDF goes to
zero/negative on small corpora and silently drops matches)."""
import math
from collections import Counter, defaultdict

from ..text_utils import tokenize


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self._ids: list[int] = []
        self._postings: dict[str, list[tuple[int, int]]] = {}
        self._doc_len: list[int] = []
        self._idf: dict[str, float] = {}
        self._avgdl = 0.0

    def build(self, chunks: list[dict]) -> None:
        postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        ids, lens = [], []
        for ch in chunks:
            toks = tokenize(f"{ch.get('filename', '')} {ch['content']}")
            if not toks:
                continue
            idx = len(ids)
            ids.append(ch["id"])
            lens.append(len(toks))
            for term, tf in Counter(toks).items():
                postings[term].append((idx, tf))
        n = len(ids)
        self._ids, self._doc_len, self._postings = ids, lens, dict(postings)
        self._avgdl = (sum(lens) / n) if n else 0.0
        self._idf = {t: math.log(1 + (n - len(p) + 0.5) / (len(p) + 0.5)) for t, p in postings.items()}

    def __len__(self) -> int:
        return len(self._ids)

    def search(self, query: str, k: int = 20) -> list[tuple[int, float]]:
        if not self._ids:
            return []
        scores: dict[int, float] = defaultdict(float)
        for term in set(tokenize(query)):
            idf = self._idf.get(term)
            if idf is None:
                continue
            for idx, tf in self._postings[term]:
                norm = self.k1 * (1 - self.b + self.b * self._doc_len[idx] / self._avgdl)
                scores[idx] += idf * tf * (self.k1 + 1) / (tf + norm)
        top = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:k]
        return [(self._ids[i], s) for i, s in top]
