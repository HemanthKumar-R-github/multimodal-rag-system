"""Structure-aware chunking. Text is packed by paragraphs into overlapping word windows;
tables and figures stay atomic so they can be cited as single sources."""
import re

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")


def clean_text(text: str) -> str:
    text = text.replace("\u00ad", "").replace("\ufb01", "fi").replace("\ufb02", "fl")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)  # de-hyphenate line breaks
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]


def chunk_text(text: str, size: int = 180, overlap: int = 40) -> list[str]:
    paragraphs = [re.sub(r"\s*\n\s*", " ", p).strip() for p in re.split(r"\n\s*\n", clean_text(text))]
    units: list[str] = []
    for p in paragraphs:
        if not p:
            continue
        if len(p.split()) <= size:
            units.append(p)
        else:
            units.extend(split_sentences(p))

    chunks: list[str] = []
    current: list[str] = []
    for unit in units:
        words = unit.split()
        if len(words) > size:  # single very long sentence: hard split
            for i in range(0, len(words), size - overlap):
                chunks.append(" ".join(words[i:i + size]))
            continue
        if len(current) + len(words) > size and current:
            chunks.append(" ".join(current))
            current = current[-overlap:] if overlap else []
        current.extend(words)
    if current:
        tail = " ".join(current)
        if not chunks or tail not in chunks[-1]:
            chunks.append(tail)
    return [c for c in chunks if len(c.split()) >= 3]


def table_to_markdown(rows: list[list]) -> str:
    rows = [[("" if c is None else str(c).replace("\n", " ").strip()) for c in r] for r in rows if r]
    rows = [r for r in rows if any(r)]
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    header, body = rows[0], rows[1:]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * width]
    lines += ["| " + " | ".join(r) + " |" for r in body]
    return "\n".join(lines)


def split_markdown_table(md: str, max_rows: int = 40) -> list[str]:
    """Large tables are split by rows; every piece repeats the header so it stays self-describing."""
    lines = md.splitlines()
    if len(lines) <= max_rows + 2:
        return [md]
    head, body = lines[:2], lines[2:]
    return ["\n".join(head + body[i:i + max_rows]) for i in range(0, len(body), max_rows)]
