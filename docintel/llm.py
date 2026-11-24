"""Thin wrapper over any OpenAI-compatible chat/vision/embedding endpoint."""
import base64
import json
import logging
import re
from pathlib import Path

log = logging.getLogger(__name__)


class LLMClient:
    def __init__(self, base_url: str, api_key: str | None, model: str, vlm_model: str):
        self.model = model
        self.vlm_model = vlm_model
        self._client = None
        if api_key:
            try:
                from openai import OpenAI

                self._client = OpenAI(base_url=base_url, api_key=api_key, timeout=90, max_retries=2)
            except Exception as exc:  # pragma: no cover - import/config failure
                log.warning("LLM client unavailable: %s", exc)

    @property
    def available(self) -> bool:
        return self._client is not None

    def chat(self, messages: list[dict], *, temperature: float = 0.1, max_tokens: int = 900,
             model: str | None = None) -> str:
        if not self._client:
            raise RuntimeError("LLM is not configured (set LLM_API_KEY / AI_GATEWAY_API_KEY).")
        resp = self._client.chat.completions.create(
            model=model or self.model, messages=messages, temperature=temperature, max_tokens=max_tokens
        )
        return (resp.choices[0].message.content or "").strip()

    def chat_json(self, messages: list[dict], **kwargs) -> dict | list:
        return parse_json(self.chat(messages, **kwargs))

    def vision(self, prompt: str, image_path: str | Path, *, max_tokens: int = 900) -> str:
        data = base64.b64encode(Path(image_path).read_bytes()).decode()
        suffix = Path(image_path).suffix.lower().lstrip(".") or "png"
        mime = "jpeg" if suffix in {"jpg", "jpeg"} else suffix
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/{mime};base64,{data}"}},
            ],
        }]
        return self.chat(messages, model=self.vlm_model, max_tokens=max_tokens)

    def embed(self, texts: list[str], model: str) -> list[list[float]]:
        if not self._client:
            raise RuntimeError("LLM is not configured.")
        resp = self._client.embeddings.create(model=model, input=texts)
        return [d.embedding for d in resp.data]


def parse_json(text: str):
    """Parse JSON from a model reply that may be wrapped in a markdown fence."""
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fenced:
        text = fenced.group(1)
    start = min([i for i in (text.find("{"), text.find("[")) if i != -1], default=-1)
    if start > 0:
        text = text[start:]
    return json.loads(text)
