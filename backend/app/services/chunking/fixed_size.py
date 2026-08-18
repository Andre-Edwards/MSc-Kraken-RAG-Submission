from __future__ import annotations

import re
from collections import defaultdict

from app.core.config import settings
from app.models import ChunkRecord, PageRecord
from app.services.chunking.utils import slugify, word_count
from app.services.text_cleaning import compute_repeated_lines, normalize_lines


TOKEN_RE = re.compile(r"\S+")


def build_fixed_size_chunks(pages: list[PageRecord]) -> list[ChunkRecord]:
    pages_by_doc: dict[str, list[PageRecord]] = defaultdict(list)
    for page in pages:
        pages_by_doc[page.doc_id].append(page)

    chunks: list[ChunkRecord] = []
    for doc_id, doc_pages in pages_by_doc.items():
        doc_pages = sorted(doc_pages, key=lambda p: p.page_num)
        repeated = compute_repeated_lines(doc_pages)
        tokens: list[tuple[str, int]] = []

        for page in doc_pages:
            lines = [line for line in normalize_lines(page.clean_text) if line not in repeated]
            for token in TOKEN_RE.findall("\n".join(lines)):
                tokens.append((token, page.page_num))

        max_words = settings.fixed_chunk_words
        overlap = settings.fixed_chunk_overlap
        stride = max(1, max_words - overlap)
        index = 0
        start = 0

        while start < len(tokens):
            window = tokens[start : start + max_words]
            if not window:
                break
            words = [token for token, _ in window]
            pages_in_window = [page for _, page in window]
            text = " ".join(words).strip()
            if text:
                file_name = doc_pages[0].file_name
                chunk_id = f"{slugify(doc_id)}::fixed::{index:04d}"
                chunks.append(
                    ChunkRecord(
                        chunk_id=chunk_id,
                        doc_id=doc_id,
                        file_name=file_name,
                        source_path=doc_pages[0].source_path,
                        strategy="fixed_size",
                        text=text,
                        page_start=min(pages_in_window),
                        page_end=max(pages_in_window),
                        section_title="Fixed-size window",
                        word_count=word_count(text),
                        char_count=len(text),
                    )
                )
                index += 1
            if start + max_words >= len(tokens):
                break
            start += stride

    return chunks
