"""End-to-end orchestration: ingest -> index -> retrieve -> rerank -> generate."""
import logging
import shutil
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from werkzeug.utils import secure_filename

from .generation import AnswerGenerator
from .ingestion import ParseContext, parse_document
from .ingestion.ocr import ocr_available
from .ingestion.vision import VisionAnalyzer
from .llm import LLMClient
from .retrieval import BM25Index, DenseIndex, HybridRetriever, Reranker, build_embedder
from .store import Store

log = logging.getLogger(__name__)


class Pipeline:
    def __init__(self, cfg):
        cfg.ensure_dirs()
        self.cfg = cfg
        self.store = Store(cfg.DB_PATH)
        self.llm = LLMClient(cfg.LLM_BASE_URL, cfg.LLM_API_KEY, cfg.LLM_MODEL, cfg.VLM_MODEL)
        self.vision = VisionAnalyzer(self.llm, cfg.ENABLE_VLM)
        self.embedder = build_embedder(cfg, self.llm)
        self.bm25 = BM25Index()
        self.dense = DenseIndex(cfg.INDEX_DIR, self.embedder)
        self.retriever = HybridRetriever(self.bm25, self.dense, k_bm25=cfg.TOP_K_BM25,
                                         k_dense=cfg.TOP_K_DENSE, rrf_k=cfg.RRF_K)
        self.reranker = Reranker(cfg, self.llm)
        self.generator = AnswerGenerator(self.llm)
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ingest")
        self._sync_indexes()
        self._resume_pending()

    # Index maintenance ----------------------------------------------------
    def _sync_indexes(self) -> None:
        with self._lock:
            chunks = self.store.all_chunks()
            self.bm25.build(chunks)
            if self.dense.needs_rebuild or len(self.dense) != len(chunks):
                log.info("Rebuilding dense index for %d chunks", len(chunks))
                self.dense.rebuild(chunks)

    def _resume_pending(self) -> None:
        for doc in self.store.list_documents():
            if doc["status"] in {"queued", "processing"}:
                self._executor.submit(self._process, doc["id"])

    # Ingestion ------------------------------------------------------------
    def allowed(self, filename: str) -> bool:
        return "." in filename and filename.rsplit(".", 1)[1].lower() in self.cfg.ALLOWED_EXTENSIONS

    def ingest_upload(self, file_storage) -> dict:
        name = secure_filename(file_storage.filename or "")
        if not name or not self.allowed(name):
            raise ValueError(f"Unsupported file type: {file_storage.filename!r}")
        dest = self.cfg.UPLOAD_DIR / f"{uuid.uuid4().hex[:10]}_{name}"
        file_storage.save(dest)
        return self.ingest_path(dest, name)

    def ingest_path(self, path: Path, display_name: str | None = None, wait: bool = False) -> dict:
        filetype = path.suffix.lower().lstrip(".")
        doc_id = self.store.add_document(display_name or path.name, str(path), filetype)
        future = self._executor.submit(self._process, doc_id)
        if wait:
            future.result()
        return self.store.get_document(doc_id)

    def _process(self, doc_id: int) -> None:
        doc = self.store.get_document(doc_id)
        if not doc:
            return
        start = time.perf_counter()
        self.store.update_document(doc_id, status="processing", error=None)
        try:
            ctx = ParseContext(asset_dir=self.cfg.ASSET_DIR, vision=self.vision,
                               chunk_size=self.cfg.CHUNK_SIZE, chunk_overlap=self.cfg.CHUNK_OVERLAP,
                               min_image_side=self.cfg.MIN_IMAGE_SIDE, ocr_dpi=self.cfg.OCR_DPI)
            result = parse_document(Path(doc["stored_path"]), doc["filetype"], doc_id, ctx)
            if not result.chunks:
                raise RuntimeError("No content could be extracted (configure a VLM or install Tesseract for scans).")
            with self._lock:
                ids = self.store.add_chunks(doc_id, result.chunks)
                self.dense.add(ids, [c["content"] for c in result.chunks])
                self.dense.save()
                stats = {**result.stats, "seconds": round(time.perf_counter() - start, 2)}
                self.store.update_document(doc_id, status="ready", pages=result.pages, stats=stats)
                self.bm25.build(self.store.all_chunks())
            log.info("Indexed document %s (%d chunks)", doc["filename"], len(ids))
        except Exception as exc:
            log.exception("Ingestion failed for %s", doc["filename"])
            self.store.update_document(doc_id, status="failed", error=str(exc)[:500])

    def delete_document(self, doc_id: int) -> bool:
        doc = self.store.get_document(doc_id)
        if not doc:
            return False
        with self._lock:
            ids = self.store.delete_document(doc_id)
            self.dense.remove(ids)
            self.dense.save()
            self.bm25.build(self.store.all_chunks())
        Path(doc["stored_path"]).unlink(missing_ok=True)
        shutil.rmtree(self.cfg.ASSET_DIR / str(doc_id), ignore_errors=True)
        return True

    # Retrieval & QA -------------------------------------------------------
    def retrieve(self, question: str, mode: str = "hybrid", rerank: bool = True,
                 top_k: int | None = None) -> list[dict]:
        top_k = top_k or self.cfg.FINAL_TOP_N
        pool = self.cfg.RERANK_CANDIDATES if rerank else top_k
        with self._lock:
            hits = self.retriever.retrieve(question, mode=mode, top_k=pool)
        by_id = self.store.get_chunks([h["chunk_id"] for h in hits])
        candidates = [{**by_id[h["chunk_id"]], **h} for h in hits if h["chunk_id"] in by_id]
        if rerank:
            return self.reranker.rerank(question, candidates, top_k)
        return candidates[:top_k]

    def answer(self, question: str, mode: str = "hybrid", rerank: bool = True,
               top_k: int | None = None, use_llm: bool = True) -> dict:
        t0 = time.perf_counter()
        chunks = self.retrieve(question, mode=mode, rerank=rerank, top_k=top_k)
        t1 = time.perf_counter()
        result = self.generator.generate(question, chunks, use_llm=use_llm)
        t2 = time.perf_counter()
        result.update({
            "question": question,
            "mode": mode,
            "reranked": rerank,
            "timings_ms": {"retrieval": round((t1 - t0) * 1000, 1), "generation": round((t2 - t1) * 1000, 1)},
        })
        return result

    def status(self) -> dict:
        docs = self.store.list_documents()
        return {
            "documents": len(docs),
            "ready": sum(d["status"] == "ready" for d in docs),
            "chunks": len(self.bm25),
            "backends": {
                "llm": self.cfg.LLM_MODEL if self.llm.available else None,
                "vlm": self.cfg.VLM_MODEL if self.vision.available else None,
                "ocr": "tesseract" if ocr_available() else None,
                "embedder": self.embedder.name,
                "vector_index": f"{self.dense.backend} (IndexFlatIP, dim={self.dense.dim})",
                "sparse_index": "BM25Okapi",
                "fusion": f"RRF (k={self.cfg.RRF_K})",
                "reranker": self.reranker.name,
            },
        }
