"""Vision-language understanding for figures, charts and scanned pages."""
import logging
from pathlib import Path

from ..llm import LLMClient, parse_json

log = logging.getLogger(__name__)

FIGURE_PROMPT = """You are a document-understanding model. Analyse this figure extracted from a document.
Nearby text on the page (may contain the caption): \"\"\"{context}\"\"\"

Return ONLY JSON with keys:
  "kind": one of "chart", "table", "diagram", "photo", "text", "logo", "other"
  "title": short title or caption
  "description": 2-4 sentences explaining what the figure shows and its key takeaway
  "data": for charts/tables, a markdown table of every readable value (series, categories, numbers); else ""
  "text": any text visible in the image, verbatim
"""

PAGE_PROMPT = """Transcribe this scanned document page faithfully as markdown.
Preserve headings, paragraphs and lists. Render tables as markdown tables.
For any chart or figure, insert a line "[Figure: <description with key values>]".
Return only the transcription."""


class VisionAnalyzer:
    def __init__(self, llm: LLMClient, enabled: bool = True):
        self.llm = llm
        self.enabled = enabled and llm.available

    @property
    def available(self) -> bool:
        return self.enabled

    def describe_figure(self, image_path: Path, context: str = "") -> dict | None:
        if not self.enabled:
            return None
        try:
            raw = self.llm.vision(FIGURE_PROMPT.format(context=context[:800]), image_path)
            result = parse_json(raw)
            if isinstance(result, dict):
                return result
        except Exception as exc:
            log.warning("VLM figure analysis failed for %s: %s", image_path, exc)
        return None

    def transcribe_page(self, image_path: Path) -> str:
        if not self.enabled:
            return ""
        try:
            return self.llm.vision(PAGE_PROMPT, image_path, max_tokens=2500)
        except Exception as exc:
            log.warning("VLM page transcription failed for %s: %s", image_path, exc)
            return ""
