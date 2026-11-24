"""Tesseract OCR with graceful degradation when the binary is not installed."""
import logging
import shutil
from functools import lru_cache

from PIL import Image, ImageOps

log = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def ocr_available() -> bool:
    if not shutil.which("tesseract"):
        return False
    try:
        import pytesseract  # noqa: F401

        return True
    except ImportError:
        return False


def preprocess(image: Image.Image) -> Image.Image:
    gray = ImageOps.grayscale(image)
    if min(gray.size) < 1000:
        scale = 1000 / min(gray.size)
        gray = gray.resize((int(gray.width * scale), int(gray.height * scale)), Image.LANCZOS)
    return ImageOps.autocontrast(gray)


def ocr_image(image: Image.Image) -> str:
    if not ocr_available():
        return ""
    import pytesseract

    try:
        return pytesseract.image_to_string(preprocess(image), config="--oem 3 --psm 3").strip()
    except Exception as exc:
        log.warning("OCR failed: %s", exc)
        return ""
