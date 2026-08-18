from __future__ import annotations

import re


WORD_RE = re.compile(r"\S+")


def word_count(text: str) -> int:
    return len(WORD_RE.findall(text or ""))


def split_words_with_overlap(text: str, max_words: int, overlap_words: int) -> list[str]:
    words = WORD_RE.findall(text or "")
    if not words:
        return []
    if len(words) <= max_words:
        return [" ".join(words)]

    chunks: list[str] = []
    start = 0
    stride = max(1, max_words - overlap_words)
    while start < len(words):
        window = words[start : start + max_words]
        if not window:
            break
        chunks.append(" ".join(window))
        if start + max_words >= len(words):
            break
        start += stride
    return chunks


def slugify(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9]+", "-", value.lower()).strip("-")
    return value[:80] or "doc"
