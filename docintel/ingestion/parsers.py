"""Multimodal parsing: PDFs (native text, tables, embedded images, vector charts,
scanned pages), standalone images and plain text, producing citable chunks."""
import logging
from dataclasses import dataclass, field
from pathlib import Path

import fitz  # PyMuPDF
import pdfplumber
from PIL import Image

from .chunker import chunk_text, clean_text, split_markdown_table, table_to_markdown
from .ocr import ocr_available, ocr_image
from .vision import VisionAnalyzer

log = logging.getLogger(__name__)


@dataclass
class ParseContext:
    asset_dir: Path
    vision: VisionAnalyzer
    chunk_size: int = 180
    chunk_overlap: int = 40
    min_image_side: int = 80
    ocr_dpi: int = 200


@dataclass
class ParseResult:
    chunks: list[dict] = field(default_factory=list)
    pages: int = 0
    stats: dict = field(default_factory=lambda: {
        "text": 0, "table": 0, "image": 0, "chart": 0, "ocr": 0, "scanned_pages": 0, "skipped_figures": 0,
    })

    def add(self, page: int | None, modality: str, content: str, **meta) -> None:
        content = content.strip()
        if not content:
            return
        self.chunks.append({"page": page, "modality": modality, "content": content, "meta": meta})
        self.stats[modality] = self.stats.get(modality, 0) + 1


def _rect_overlap(a: fitz.Rect, b: fitz.Rect) -> float:
    inter = a & b
    if inter.is_empty or a.is_empty:
        return 0.0
    return inter.get_area() / max(a.get_area(), 1e-6)


def _nearby_text(blocks: list, rect: fitz.Rect, margin: float = 70) -> str:
    """Text blocks directly above/below a figure or table (but not inside it) usually hold its caption."""
    zone = fitz.Rect(rect.x0 - 20, rect.y0 - margin, rect.x1 + 20, rect.y1 + margin)
    parts = [
        b[4].strip() for b in blocks
        if b[6] == 0 and fitz.Rect(b[:4]).intersects(zone) and _rect_overlap(fitz.Rect(b[:4]), rect) < 0.5
    ]
    return " ".join(p.replace("\n", " ") for p in parts)[:800]


def _figure_content(page_no: int | None, analysis: dict | None, context: str, ocr_text: str) -> tuple[str, str]:
    """Return (modality, content) for a figure from VLM analysis and/or OCR fallback."""
    where = f" on page {page_no}" if page_no else ""
    if analysis:
        kind = str(analysis.get("kind", "other")).lower()
        modality = "chart" if kind in {"chart", "table"} else "image"
        parts = [f"{kind.capitalize()}{where}: {analysis.get('title', '')}".strip(),
                 analysis.get("description", "")]
        if analysis.get("data"):
            parts.append("Extracted data:\n" + str(analysis["data"]))
        if analysis.get("text"):
            parts.append("Visible text: " + str(analysis["text"]))
        if context:
            parts.append("Page context: " + context)
        return modality, "\n".join(p for p in parts if p)
    if not context and not ocr_text:
        return "image", ""
    parts = [f"Figure{where}."]
    if context:
        parts.append("Caption/context: " + context)
    if ocr_text:
        parts.append("Text in figure (OCR): " + ocr_text)
    return "image", "\n".join(parts)


def _extract_tables(page: fitz.Page, plumber_page=None) -> list[tuple[fitz.Rect, list[list]]]:
    """pdfplumber keeps cell text (decimals, thousands separators) intact more reliably;
    PyMuPDF's detector is the fallback."""
    if plumber_page is not None:
        try:
            tables = [(fitz.Rect(t.bbox), t.extract()) for t in plumber_page.find_tables()]
            if tables:
                return tables
        except Exception as exc:
            log.debug("pdfplumber table extraction failed: %s", exc)
    try:
        return [(fitz.Rect(t.bbox), t.extract()) for t in page.find_tables().tables]
    except Exception as exc:
        log.debug("PyMuPDF table extraction failed: %s", exc)
        return []


def parse_pdf(path: Path, doc_id: int, ctx: ParseContext) -> ParseResult:
    result = ParseResult()
    out_dir = ctx.asset_dir / str(doc_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    seen_xrefs: set[int] = set()

    try:
        plumber_doc = pdfplumber.open(path)
    except Exception as exc:
        log.debug("pdfplumber could not open %s: %s", path, exc)
        plumber_doc = None

    with fitz.open(path) as doc:
        result.pages = doc.page_count
        for page in doc:
            pno = page.number + 1
            blocks = page.get_text("blocks")
            plumber_page = plumber_doc.pages[page.number] if plumber_doc else None
            tables = _extract_tables(page, plumber_page)
            table_rects = [r for r, _ in tables]

            text_blocks = [
                b[4] for b in blocks
                if b[6] == 0 and not any(_rect_overlap(fitz.Rect(b[:4]), tr) > 0.5 for tr in table_rects)
            ]
            page_text = clean_text("\n\n".join(text_blocks))

            # Scanned page: little/no extractable text -> VLM transcription, then OCR fallback
            if len(page_text) < 40 and (page.get_images() or not page_text):
                result.stats["scanned_pages"] += 1
                img_path = out_dir / f"page{pno}.png"
                page.get_pixmap(dpi=ctx.ocr_dpi).save(img_path)
                rel = f"{doc_id}/{img_path.name}"
                text, method = ctx.vision.transcribe_page(img_path), "vlm"
                if not text and ocr_available():
                    text, method = ocr_image(Image.open(img_path)), "tesseract"
                if text:
                    for piece in chunk_text(text, ctx.chunk_size, ctx.chunk_overlap):
                        result.add(pno, "ocr", piece, image=rel, method=method)
                else:
                    result.stats["skipped_figures"] += 1
                continue

            for piece in chunk_text(page_text, ctx.chunk_size, ctx.chunk_overlap):
                result.add(pno, "text", piece)

            for t_idx, (rect, rows) in enumerate(tables):
                md = table_to_markdown(rows)
                if not md:
                    continue
                caption = _nearby_text(blocks, rect, margin=40)
                for part in split_markdown_table(md):
                    header = f"Table {t_idx + 1} on page {pno}." + (f" Context: {caption}" if caption else "")
                    result.add(pno, "table", f"{header}\n{part}", bbox=list(rect), table_markdown=part)

            # Embedded raster figures
            for i, img in enumerate(page.get_images(full=True)):
                xref = img[0]
                if xref in seen_xrefs:
                    continue
                seen_xrefs.add(xref)
                try:
                    pix = fitz.Pixmap(doc, xref)
                    if pix.width < ctx.min_image_side or pix.height < ctx.min_image_side:
                        continue
                    if pix.n - pix.alpha >= 4:
                        pix = fitz.Pixmap(fitz.csRGB, pix)
                    img_path = out_dir / f"p{pno}_img{i}.png"
                    pix.save(img_path)
                except Exception as exc:
                    log.warning("Could not extract image xref %s: %s", xref, exc)
                    continue
                rects = page.get_image_rects(xref)
                context = _nearby_text(blocks, rects[0]) if rects else ""
                analysis = ctx.vision.describe_figure(img_path, context)
                ocr_text = "" if analysis else ocr_image(Image.open(img_path))
                modality, content = _figure_content(pno, analysis, context, ocr_text)
                if content:
                    result.add(pno, modality, content, image=f"{doc_id}/{img_path.name}",
                               bbox=list(rects[0]) if rects else None, method="vlm" if analysis else "context+ocr")
                else:
                    result.stats["skipped_figures"] += 1

            # Vector-drawn charts (no raster image): render the drawing area for the VLM
            if ctx.vision.available:
                drawings = page.get_drawings()
                if len(drawings) > 40:
                    area = fitz.Rect()
                    for d in drawings:
                        area |= d["rect"]
                    area &= page.rect
                    if area.width > 120 and area.height > 120 and not any(_rect_overlap(area, tr) > 0.6 for tr in table_rects):
                        img_path = out_dir / f"p{pno}_vector.png"
                        page.get_pixmap(dpi=150, clip=area).save(img_path)
                        context = _nearby_text(blocks, area)
                        analysis = ctx.vision.describe_figure(img_path, context)
                        if analysis:
                            modality, content = _figure_content(pno, analysis, context, "")
                            result.add(pno, modality, content, image=f"{doc_id}/{img_path.name}",
                                       bbox=list(area), method="vlm-vector")
    if plumber_doc:
        plumber_doc.close()
    return result


def parse_image(path: Path, doc_id: int, ctx: ParseContext) -> ParseResult:
    result = ParseResult(pages=1)
    out_dir = ctx.asset_dir / str(doc_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    img_path = out_dir / "image.png"
    with Image.open(path) as im:
        im.convert("RGB").save(img_path)
        ocr_text = ocr_image(im)
    rel = f"{doc_id}/{img_path.name}"

    analysis = ctx.vision.describe_figure(img_path)
    kind = str((analysis or {}).get("kind", "")).lower()
    if kind == "text" or (not analysis and len(ocr_text.split()) > 40):
        text = ctx.vision.transcribe_page(img_path) or ocr_text
        for piece in chunk_text(text, ctx.chunk_size, ctx.chunk_overlap):
            result.add(1, "ocr", piece, image=rel)
    else:
        modality, content = _figure_content(None, analysis, "", ocr_text)
        result.add(1, modality, content, image=rel, method="vlm" if analysis else "ocr")
    if not result.chunks:
        result.stats["skipped_figures"] += 1
    return result


def parse_text(path: Path, doc_id: int, ctx: ParseContext) -> ParseResult:
    result = ParseResult(pages=1)
    text = path.read_text(encoding="utf-8", errors="ignore")
    for piece in chunk_text(text, ctx.chunk_size, ctx.chunk_overlap):
        result.add(1, "text", piece)
    return result


def parse_document(path: Path, filetype: str, doc_id: int, ctx: ParseContext) -> ParseResult:
    if filetype == "pdf":
        return parse_pdf(path, doc_id, ctx)
    if filetype in {"txt", "md"}:
        return parse_text(path, doc_id, ctx)
    return parse_image(path, doc_id, ctx)
