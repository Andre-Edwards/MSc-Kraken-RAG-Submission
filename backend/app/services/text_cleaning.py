from __future__ import annotations

import re
import unicodedata
from collections import Counter

from app.models import PageRecord


PAGE_LABEL_RE = re.compile(r"^\s*page\s+\d+(\s+of\s+\d+)?\s*$", re.I)
PAGE_OF_RE = re.compile(r"^\s*\d+\s+of\s+\d+\s*$", re.I)
BARE_NUM_RE = re.compile(r"^\s*\d{1,4}\s*$")
_ZW = re.compile(r"[\u200B-\u200D\uFEFF]")
_WS = re.compile(r"\s+")


def normalize_unicode(text: str) -> str:
    return unicodedata.normalize("NFKC", text)


def fix_hyphenation(text: str) -> str:
    return re.sub(r"(\w)-\n(\w)", r"\1\2", text)


def normalize_newlines_keep_lines(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = "\n".join(ln.strip() for ln in text.split("\n"))
    return text.strip()


def collapse_whitespace_keep_newlines(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    return text.strip()


def is_obvious_page_number_line(line: str) -> bool:
    s = line.strip()
    return bool(s and (PAGE_LABEL_RE.match(s) or PAGE_OF_RE.match(s)))


def is_bare_number_line(line: str) -> bool:
    s = line.strip()
    return bool(s and BARE_NUM_RE.match(s))


def remove_boundary_page_number_lines(text: str, boundary_window: int = 3) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""

    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    nonblank_idx = [i for i, ln in enumerate(lines) if ln.strip()]
    if not nonblank_idx:
        return ""

    boundary_idxs = set(nonblank_idx[:boundary_window]) | set(nonblank_idx[-boundary_window:])
    kept: list[str] = []
    for i, ln in enumerate(lines):
        s = ln.strip()
        if not s:
            kept.append(ln)
            continue
        if is_obvious_page_number_line(s):
            continue
        if i in boundary_idxs and is_bare_number_line(s):
            continue
        kept.append(ln)
    return "\n".join(kept).strip()


def basic_clean(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = normalize_unicode(text)
    text = text.replace("\ufffd", "'")
    text = fix_hyphenation(text)
    text = normalize_newlines_keep_lines(text)
    text = remove_boundary_page_number_lines(text, boundary_window=3)
    text = collapse_whitespace_keep_newlines(text)
    return text


def clean_line(line: str) -> str:
    if not isinstance(line, str):
        return ""
    line = line.replace("\u00A0", " ")
    line = _ZW.sub("", line)
    line = line.strip()
    line = _WS.sub(" ", line)
    line = re.sub(r"^(\d+)\s+\.", r"\1.", line)
    line = re.sub(r"^(\d+)\.([A-Za-z(\[])", r"\1. \2", line)
    return line


def normalize_lines(page_text: str) -> list[str]:
    raw = (page_text or "").split("\n")
    lines = [clean_line(x) for x in raw]
    return [x for x in lines if x]


def normalize_lines_keep_blanks(page_text: str) -> list[str | None]:
    raw = (page_text or "").split("\n")
    out: list[str | None] = []
    for x in raw:
        cleaned = clean_line(x)
        out.append(cleaned if cleaned else None)
    return out


def is_probably_scanned(page_text: str, min_chars: int = 80, min_alpha_ratio: float = 0.35) -> bool:
    if not isinstance(page_text, str):
        return True
    compact = re.sub(r"\s+", "", page_text)
    if len(compact) < min_chars:
        return True
    alpha_ratio = sum(ch.isalpha() for ch in compact) / max(1, len(compact))
    return alpha_ratio < min_alpha_ratio


def clean_pages(pages: list[PageRecord]) -> list[PageRecord]:
    cleaned: list[PageRecord] = []
    for page in pages:
        page.clean_text = basic_clean(page.text)
        page.likely_scanned = is_probably_scanned(page.text)
        cleaned.append(page)
    return cleaned


def compute_repeated_lines(pages: list[PageRecord], min_frac: float = 0.4, max_len: int = 120) -> set[str]:
    n_pages = len({p.page_num for p in pages})
    if n_pages <= 1:
        return set()

    line_page_count: Counter[str] = Counter()
    for page in pages:
        for line in set(normalize_lines(page.clean_text or "")):
            if len(line) > max_len:
                continue
            if re.fullmatch(r"\d+\.", line):
                continue
            if re.fullmatch(r"^[a-zA-Z]\)$", line):
                continue
            line_page_count[line] += 1

    threshold = max(2, int(n_pages * min_frac))
    repeated = {line for line, count in line_page_count.items() if count >= threshold}
    repeated |= {
        line
        for line in repeated
        if re.fullmatch(r"Page\s*\d+(\s*of\s*\d+)?", line, flags=re.I) is not None
    }
    return repeated


def remove_repeated_lines_from_text(text: str, repeated: set[str]) -> str:
    if not repeated:
        return text
    lines = normalize_lines(text)
    return "\n".join(line for line in lines if line not in repeated)
