"""Shared tokenisation used by BM25, the lexical reranker and evaluation metrics."""
import re

STOPWORDS = frozenset("""
a an and are as at be been but by can could did do does for from had has have how i if in into is it its
of on or our so than that the their them then there these they this to was we were what when where which
who whom why will with would you your about after all also any between both each more most other over
such through under very via vs per not no
""".split())

_TOKEN = re.compile(r"[a-z0-9]+(?:[.,][0-9]+)?")


def _stem(tok: str) -> str:
    if tok[0].isdigit() or len(tok) <= 4:
        return tok
    for suffix in ("ations", "ation", "ingly", "ings", "ing", "edly", "ies", "ied", "ers", "ed", "es", "ly", "s"):
        if tok.endswith(suffix) and len(tok) - len(suffix) >= 3:
            return tok[: -len(suffix)] + ("y" if suffix in {"ies", "ied"} else "")
    return tok


def tokenize(text: str, *, stem: bool = True, drop_stopwords: bool = True) -> list[str]:
    toks = _TOKEN.findall(text.lower())
    if drop_stopwords:
        toks = [t for t in toks if t not in STOPWORDS]
    return [_stem(t) for t in toks] if stem else toks
