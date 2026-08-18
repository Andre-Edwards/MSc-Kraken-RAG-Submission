from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass

from app.core.config import settings
from app.models import ChunkRecord, PageRecord
from app.services.chunking.utils import slugify, split_words_with_overlap, word_count
from app.services.text_cleaning import (
    clean_line,
    compute_repeated_lines,
    normalize_lines,
    normalize_lines_keep_blanks,
)

# numeric heading detection, appendix detection, split-heading joining,
# repeated header/footer removal, and fallback paragraph heading detection.

MAJOR_HEADING_DOT = re.compile(r"^\d+\.\s*\S")
MAJOR_HEADING_NODOT = re.compile(r"^\d{1,2}\s+[A-Z]")
SUBCLAUSE_NUM = re.compile(r"^\d+\.\d+(\.\d+)*\b")
TOC_DOTS = re.compile(r"\.{3,}")
ENDS_WITH_PAGE = re.compile(r"\s\d+$")
BULLETISH = re.compile(r"^[•\-\u2022]")
APPENDIX = re.compile(r"^(Appendix|APPENDIX)\b", re.I)
YEAR_LIKE_START = re.compile(r"^(19|20)\d{2}(\.|\s|$)")
NUM_ONLY = re.compile(r"^\d+$")
DOT_ONLY = re.compile(r"^\.$")
NUM_DOT = re.compile(r"^\d+\.$")
INLINE_HEADING_DOT = re.compile(r"(?<=[\.\)\]])\s+(\d+\.\s+[A-Z])")
INLINE_HEADING_NODOT = re.compile(r"(?<=[\.\)\]])\s+(\d{1,2}\s+[A-Z])")
INLINE_HEADING_AFTER_WORD = re.compile(r"(?<=[a-z])\s+(\d+\.\s+[A-Z])")


@dataclass
class SectionBuffer:
    doc_id: str
    section_title: str
    parts: list[tuple[int, str]]


@dataclass
class SectionBlock:
    doc_id: str
    file_name: str
    source_path: str
    section_title: str
    page_start: int
    page_end: int
    text: str


def looks_like_numeric_major_heading(line: str) -> bool:
    line = clean_line(line)
    if not line:
        return False
    if YEAR_LIKE_START.match(line):
        return False
    if TOC_DOTS.search(line):
        return False
    if (MAJOR_HEADING_DOT.match(line) or MAJOR_HEADING_NODOT.match(line)) and ENDS_WITH_PAGE.search(line):
        return False
    if len(line) < 2 or len(line) > 160:
        return False
    if SUBCLAUSE_NUM.match(line):
        return False
    return bool(MAJOR_HEADING_DOT.match(line) or MAJOR_HEADING_NODOT.match(line))


def looks_like_appendix_heading(line: str) -> bool:
    line = clean_line(line)
    if not line or len(line) < 6 or len(line) > 180:
        return False
    return bool(APPENDIX.match(line))


def maybe_join_split_numeric_heading(lines: list[str]) -> list[str]:
    out: list[str] = []
    i = 0

    def is_title_line(value: str) -> bool:
        value = clean_line(value)
        if not value or BULLETISH.match(value) or SUBCLAUSE_NUM.match(value):
            return False
        if re.fullmatch(r"\d+(\.\d+)*", value):
            return False
        return bool(re.match(r"^[A-Z(\[]", value))

    while i < len(lines):
        a = clean_line(lines[i])
        if NUM_ONLY.match(a) and i + 2 < len(lines):
            b = clean_line(lines[i + 1])
            c = clean_line(lines[i + 2])
            if DOT_ONLY.match(b) and is_title_line(c):
                out.append(f"{a}. {c}")
                i += 3
                continue
        if NUM_DOT.match(a) and i + 1 < len(lines):
            b = clean_line(lines[i + 1])
            if is_title_line(b):
                out.append(f"{a.rstrip('.')}. {b}")
                i += 2
                continue
        if NUM_ONLY.match(a) and i + 1 < len(lines):
            b = clean_line(lines[i + 1])
            if is_title_line(b):
                out.append(f"{a}. {b}")
                i += 2
                continue
        out.append(a)
        i += 1
    return [item for item in out if item]


def insert_heading_breaks(text: str) -> str:
    if not text:
        return ""
    text = INLINE_HEADING_DOT.sub(r"\n\1", text)
    text = INLINE_HEADING_NODOT.sub(r"\n\1", text)
    text = INLINE_HEADING_AFTER_WORD.sub(r"\n\1", text)
    return text


def is_fallback_heading(candidate: str, next_line: str | None) -> bool:
    candidate = clean_line(candidate)
    if not candidate:
        return False
    if re.match(r"^(19|20)\d{2}\.", candidate):
        return False
    if len(candidate) < 3 or len(candidate) > 110:
        return False
    if BULLETISH.match(candidate):
        return False
    if SUBCLAUSE_NUM.match(candidate) or looks_like_numeric_major_heading(candidate):
        return False
    if candidate.endswith("."):
        return False

    words = candidate.split()
    if len(words) >= 2:
        cap_words = sum(1 for word in words if word[:1].isupper())
        if cap_words / len(words) < 0.4:
            return False

    nxt = clean_line(next_line) if isinstance(next_line, str) else ""
    if not nxt:
        return True
    if len(nxt) >= 40 or SUBCLAUSE_NUM.match(nxt) or BULLETISH.match(nxt):
        return True
    return True


def _flush_section(
    blocks: list[SectionBlock],
    buffer: SectionBuffer,
    file_name: str,
    source_path: str,
) -> None:
    if not buffer.parts:
        return
    pages = [page for page, _ in buffer.parts]
    text = "\n".join(line for _, line in buffer.parts).strip()
    if not text:
        return
    blocks.append(
        SectionBlock(
            doc_id=buffer.doc_id,
            file_name=file_name,
            source_path=source_path,
            section_title=buffer.section_title,
            page_start=min(pages),
            page_end=max(pages),
            text=text,
        )
    )


def build_numeric_sections(doc_pages: list[PageRecord], repeated: set[str]) -> tuple[list[SectionBlock], int]:
    blocks: list[SectionBlock] = []
    doc_pages = sorted(doc_pages, key=lambda p: p.page_num)
    first = doc_pages[0]
    current = SectionBuffer(doc_id=first.doc_id, section_title="Document Overview", parts=[])
    major_count = 0
    in_appendix = False

    for page in doc_pages:
        page_text = insert_heading_breaks(page.clean_text).strip()
        lines = [line for line in normalize_lines(page_text) if line not in repeated]
        if lines:
            if re.fullmatch(r"\d{1,4}", lines[0]):
                lines = lines[1:]
            if lines and re.fullmatch(r"\d{1,4}", lines[-1]):
                lines = lines[:-1]
        lines = maybe_join_split_numeric_heading(lines)

        for line in lines:
            line = clean_line(line)
            if looks_like_appendix_heading(line):
                _flush_section(blocks, current, first.file_name, first.source_path)
                current = SectionBuffer(first.doc_id, line, [(page.page_num, line)])
                in_appendix = True
                major_count += 1
                continue
            if not in_appendix and looks_like_numeric_major_heading(line):
                _flush_section(blocks, current, first.file_name, first.source_path)
                current = SectionBuffer(first.doc_id, line, [(page.page_num, line)])
                major_count += 1
            else:
                current.parts.append((page.page_num, line))

    _flush_section(blocks, current, first.file_name, first.source_path)
    return blocks, major_count


def build_fallback_sections(doc_pages: list[PageRecord], repeated: set[str]) -> list[SectionBlock]:
    blocks: list[SectionBlock] = []
    doc_pages = sorted(doc_pages, key=lambda p: p.page_num)
    first = doc_pages[0]
    current = SectionBuffer(doc_id=first.doc_id, section_title="Document Overview", parts=[])

    for page in doc_pages:
        lines = normalize_lines_keep_blanks(page.clean_text)
        lines = [None if line in repeated else line for line in lines]

        nonblank = [i for i, line in enumerate(lines) if line is not None]
        if nonblank:
            first_i = nonblank[0]
            last_i = nonblank[-1]
            if isinstance(lines[first_i], str) and re.fullmatch(r"\d{1,4}", lines[first_i].strip()):
                lines[first_i] = None
            if isinstance(lines[last_i], str) and re.fullmatch(r"\d{1,4}", lines[last_i].strip()):
                lines[last_i] = None

        def next_nonblank(idx: int) -> str | None:
            for j in range(idx + 1, len(lines)):
                if lines[j] is not None:
                    return lines[j]
            return None

        prev_blank = True
        for i, line in enumerate(lines):
            if line is None:
                prev_blank = True
                continue
            if prev_blank and is_fallback_heading(line, next_nonblank(i)):
                heading = clean_line(line)
                _flush_section(blocks, current, first.file_name, first.source_path)
                current = SectionBuffer(first.doc_id, heading, [(page.page_num, heading)])
            else:
                current.parts.append((page.page_num, clean_line(line)))
            prev_blank = False

    _flush_section(blocks, current, first.file_name, first.source_path)
    return blocks


def build_full_document_section(doc_pages: list[PageRecord]) -> list[SectionBlock]:
    doc_pages = sorted(doc_pages, key=lambda p: p.page_num)
    first = doc_pages[0]
    parts: list[tuple[int, str]] = []
    for page in doc_pages:
        for line in normalize_lines(page.clean_text):
            parts.append((page.page_num, line))
    current = SectionBuffer(first.doc_id, "Full document", parts)
    blocks: list[SectionBlock] = []
    _flush_section(blocks, current, first.file_name, first.source_path)
    return blocks


def build_structure_aware_chunks(pages: list[PageRecord]) -> list[ChunkRecord]:
    pages_by_doc: dict[str, list[PageRecord]] = defaultdict(list)
    for page in pages:
        pages_by_doc[page.doc_id].append(page)

    chunks: list[ChunkRecord] = []
    for doc_id, doc_pages in pages_by_doc.items():
        doc_pages = sorted(doc_pages, key=lambda p: p.page_num)
        repeated = compute_repeated_lines(doc_pages)
        sections, major_count = build_numeric_sections(doc_pages, repeated)
        if major_count < 2:
            sections = build_fallback_sections(doc_pages, repeated)
        if not sections:
            sections = build_full_document_section(doc_pages)

        section_index = 0
        for section in sections:
            pieces = split_words_with_overlap(
                section.text,
                max_words=settings.section_chunk_words,
                overlap_words=settings.section_chunk_overlap,
            )
            for piece_index, piece in enumerate(pieces):
                piece_word_count = word_count(piece)
                if piece_word_count < settings.section_min_chunk_words:
                    continue
                chunk_id = f"{slugify(doc_id)}::section::{section_index:04d}-{piece_index:02d}"
                chunks.append(
                    ChunkRecord(
                        chunk_id=chunk_id,
                        doc_id=doc_id,
                        file_name=section.file_name,
                        source_path=section.source_path,
                        strategy="structure_aware",
                        text=piece,
                        page_start=section.page_start,
                        page_end=section.page_end,
                        section_title=section.section_title,
                        word_count=piece_word_count,
                        char_count=len(piece),
                    )
                )
            section_index += 1

    return chunks
