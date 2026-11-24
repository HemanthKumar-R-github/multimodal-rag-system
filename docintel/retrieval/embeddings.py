"""Dense embedding backends.

- sentence-transformers: local transformer bi-encoder (best quality, needs torch)
- openai: any OpenAI-compatible embeddings endpoint (e.g. AI Gateway)
- hashing: dependency-light character n-gram embedding; stateless so FAISS can be
  updated incrementally. Used automatically when nothing better is installed.
"""
import logging

import numpy as np

log = logging.getLogger(__name__)


def _normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return (x / np.clip(norms, 1e-12, None)).astype("float32")


class HashingEmbedder:
    name = "hashing-char-ngram-768"

    def __init__(self, dim: int = 768):
        from sklearn.feature_extraction.text import HashingVectorizer

        self.dim = dim
        self._char = HashingVectorizer(analyzer="char_wb", ngram_range=(3, 5), n_features=dim,
                                       alternate_sign=True, norm="l2", lowercase=True)
        self._word = HashingVectorizer(analyzer="word", ngram_range=(1, 2), n_features=dim,
                                       alternate_sign=True, norm="l2", lowercase=True, stop_words="english")

    def encode(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        dense = self._char.transform(texts).toarray() + 0.7 * self._word.transform(texts).toarray()
        return _normalize(dense)


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)
        self.name = f"st:{model_name}"
        self.dim = self.model.get_sentence_embedding_dimension()
        self._e5 = "e5" in model_name.lower()

    def encode(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        if self._e5:
            texts = [("query: " if is_query else "passage: ") + t for t in texts]
        vecs = self.model.encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vecs, dtype="float32")


class APIEmbedder:
    def __init__(self, llm, model: str):
        self.llm, self.model = llm, model
        self.name = f"api:{model}"
        self.dim = len(self.llm.embed(["dimension probe"], model)[0])

    def encode(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        out: list[list[float]] = []
        for i in range(0, len(texts), 64):
            out.extend(self.llm.embed([t[:8000] for t in texts[i:i + 64]], self.model))
        return _normalize(np.asarray(out, dtype="float32"))


def build_embedder(cfg, llm):
    backend = cfg.EMBEDDING_BACKEND.lower()
    if backend in {"auto", "sentence-transformers"}:
        try:
            return SentenceTransformerEmbedder(cfg.ST_EMBEDDING_MODEL)
        except Exception as exc:
            if backend != "auto":
                raise
            log.info("sentence-transformers unavailable (%s); trying next backend", exc.__class__.__name__)
    if backend == "openai":
        return APIEmbedder(llm, cfg.OPENAI_EMBEDDING_MODEL)
    return HashingEmbedder()
