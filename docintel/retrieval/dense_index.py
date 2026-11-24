"""Dense vector retrieval backed by FAISS (exact inner-product search over
L2-normalised vectors = cosine similarity), persisted to disk."""
import json
import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

try:
    import faiss

    HAS_FAISS = True
except ImportError:  # pragma: no cover
    HAS_FAISS = False


class DenseIndex:
    def __init__(self, index_dir: Path, embedder):
        self.dir = Path(index_dir)
        self.embedder = embedder
        self.dim = embedder.dim
        self._index_path = self.dir / "faiss.index"
        self._meta_path = self.dir / "faiss_meta.json"
        self._index = None
        self._np_ids: list[int] = []
        self._np_vecs = np.zeros((0, self.dim), dtype="float32")
        self._load()

    @property
    def backend(self) -> str:
        return "faiss" if HAS_FAISS else "numpy"

    def _new(self):
        return faiss.IndexIDMap2(faiss.IndexFlatIP(self.dim)) if HAS_FAISS else None

    def _load(self) -> None:
        meta = json.loads(self._meta_path.read_text()) if self._meta_path.exists() else {}
        compatible = meta.get("embedder") == self.embedder.name and meta.get("dim") == self.dim
        if HAS_FAISS and compatible and self._index_path.exists():
            self._index = faiss.read_index(str(self._index_path))
        else:
            self._index = self._new()
            if meta and not compatible:
                log.info("Embedding model changed; dense index will be rebuilt.")
        self.needs_rebuild = bool(meta) and not compatible

    def save(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        if HAS_FAISS:
            faiss.write_index(self._index, str(self._index_path))
        self._meta_path.write_text(json.dumps({"embedder": self.embedder.name, "dim": self.dim}))

    def __len__(self) -> int:
        return int(self._index.ntotal) if HAS_FAISS else len(self._np_ids)

    def add(self, ids: list[int], texts: list[str]) -> None:
        if not ids:
            return
        vecs = self.embedder.encode(texts)
        if HAS_FAISS:
            self._index.add_with_ids(vecs, np.asarray(ids, dtype="int64"))
        else:
            self._np_ids.extend(ids)
            self._np_vecs = np.vstack([self._np_vecs, vecs])

    def remove(self, ids: list[int]) -> None:
        if not ids:
            return
        if HAS_FAISS:
            self._index.remove_ids(np.asarray(ids, dtype="int64"))
        else:
            keep = [i for i, cid in enumerate(self._np_ids) if cid not in set(ids)]
            self._np_ids = [self._np_ids[i] for i in keep]
            self._np_vecs = self._np_vecs[keep]

    def rebuild(self, chunks: list[dict]) -> None:
        self._index = self._new()
        self._np_ids, self._np_vecs = [], np.zeros((0, self.dim), dtype="float32")
        for i in range(0, len(chunks), 256):
            batch = chunks[i:i + 256]
            self.add([c["id"] for c in batch], [c["content"] for c in batch])
        self.needs_rebuild = False
        self.save()

    def search(self, query: str, k: int = 20) -> list[tuple[int, float]]:
        if len(self) == 0:
            return []
        q = self.embedder.encode([query], is_query=True)
        if HAS_FAISS:
            scores, ids = self._index.search(q, min(k, len(self)))
            return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i != -1]
        sims = self._np_vecs @ q[0]
        top = np.argsort(-sims)[:k]
        return [(self._np_ids[i], float(sims[i])) for i in top]
