"""Central configuration. Every value can be overridden with an environment variable."""
import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")
load_dotenv(BASE_DIR / ".env.development.local")


def _bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


class Config:
    SECRET_KEY = os.getenv("SECRET_KEY", "change-me-in-production")
    MAX_CONTENT_LENGTH = int(os.getenv("MAX_UPLOAD_MB", "50")) * 1024 * 1024

    DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "storage"))
    UPLOAD_DIR = DATA_DIR / "uploads"
    ASSET_DIR = DATA_DIR / "assets"
    INDEX_DIR = DATA_DIR / "index"
    DB_PATH = DATA_DIR / "docintel.db"
    EVAL_DIR = BASE_DIR / "data" / "eval"

    ALLOWED_EXTENSIONS = {"pdf", "png", "jpg", "jpeg", "tif", "tiff", "bmp", "webp", "txt", "md"}

    # Chunking (word based)
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "180"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "40"))
    MIN_IMAGE_SIDE = int(os.getenv("MIN_IMAGE_SIDE", "80"))
    OCR_DPI = int(os.getenv("OCR_DPI", "200"))

    # LLM / VLM: any OpenAI-compatible endpoint (Vercel AI Gateway, OpenAI, Ollama, vLLM, ...)
    LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://ai-gateway.vercel.sh/v1")
    LLM_API_KEY = os.getenv("LLM_API_KEY") or os.getenv("AI_GATEWAY_API_KEY") or os.getenv("OPENAI_API_KEY")
    LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-4o-mini")
    VLM_MODEL = os.getenv("VLM_MODEL", "openai/gpt-4o-mini")
    ENABLE_VLM = _bool("ENABLE_VLM", True)

    # Embeddings: auto | sentence-transformers | openai | hashing
    EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "auto")
    ST_EMBEDDING_MODEL = os.getenv("ST_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "openai/text-embedding-3-small")

    # Reranking: auto | cross-encoder | llm | lexical | none
    RERANKER_BACKEND = os.getenv("RERANKER_BACKEND", "auto")
    CROSS_ENCODER_MODEL = os.getenv("CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

    # Retrieval
    TOP_K_BM25 = int(os.getenv("TOP_K_BM25", "25"))
    TOP_K_DENSE = int(os.getenv("TOP_K_DENSE", "25"))
    RRF_K = int(os.getenv("RRF_K", "60"))
    RERANK_CANDIDATES = int(os.getenv("RERANK_CANDIDATES", "20"))
    FINAL_TOP_N = int(os.getenv("FINAL_TOP_N", "5"))

    @classmethod
    def ensure_dirs(cls) -> None:
        for path in (cls.DATA_DIR, cls.UPLOAD_DIR, cls.ASSET_DIR, cls.INDEX_DIR, cls.EVAL_DIR):
            Path(path).mkdir(parents=True, exist_ok=True)
