"""Hybrid retrieval: BM25 + dense, fused with Reciprocal Rank Fusion.

RRF(d) = sum_r  w_r / (k + rank_r(d))   (Cormack et al., 2009)
It only uses ranks, so it needs no score calibration between retrievers.
"""
from collections import defaultdict


def reciprocal_rank_fusion(ranked_lists: dict[str, list[int]], k: int = 60,
                           weights: dict[str, float] | None = None) -> list[tuple[int, float]]:
    fused: dict[int, float] = defaultdict(float)
    for name, ids in ranked_lists.items():
        w = (weights or {}).get(name, 1.0)
        for rank, doc_id in enumerate(ids, start=1):
            fused[doc_id] += w / (k + rank)
    return sorted(fused.items(), key=lambda kv: kv[1], reverse=True)


class HybridRetriever:
    MODES = ("bm25", "dense", "hybrid")

    def __init__(self, bm25, dense, *, k_bm25: int = 25, k_dense: int = 25, rrf_k: int = 60):
        self.bm25, self.dense = bm25, dense
        self.k_bm25, self.k_dense, self.rrf_k = k_bm25, k_dense, rrf_k

    def retrieve(self, query: str, mode: str = "hybrid", top_k: int = 20,
                 weights: dict[str, float] | None = None) -> list[dict]:
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}")
        sparse = self.bm25.search(query, self.k_bm25) if mode in {"bm25", "hybrid"} else []
        dense = self.dense.search(query, self.k_dense) if mode in {"dense", "hybrid"} else []
        sparse_rank = {cid: r for r, (cid, _) in enumerate(sparse, start=1)}
        dense_rank = {cid: r for r, (cid, _) in enumerate(dense, start=1)}
        sparse_score, dense_score = dict(sparse), dict(dense)

        if mode == "bm25":
            ordered = sparse
        elif mode == "dense":
            ordered = dense
        else:
            ordered = reciprocal_rank_fusion(
                {"bm25": [c for c, _ in sparse], "dense": [c for c, _ in dense]}, k=self.rrf_k, weights=weights
            )
        return [
            {
                "chunk_id": cid,
                "score": float(score),
                "bm25_rank": sparse_rank.get(cid),
                "dense_rank": dense_rank.get(cid),
                "bm25_score": sparse_score.get(cid),
                "dense_score": dense_score.get(cid),
            }
            for cid, score in ordered[:top_k]
        ]
