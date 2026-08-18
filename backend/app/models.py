from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class PageRecord:
    doc_id: str
    file_name: str
    source_path: str
    page_num: int
    text: str
    clean_text: str = ""
    likely_scanned: bool = False


@dataclass
class ChunkRecord:
    chunk_id: str
    doc_id: str
    file_name: str
    source_path: str
    strategy: str
    text: str
    page_start: int
    page_end: int
    section_title: str
    word_count: int
    char_count: int

    def metadata(self) -> dict[str, str | int | float | bool]:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "file_name": self.file_name,
            "source_path": self.source_path,
            "strategy": self.strategy,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "section_title": self.section_title,
            "word_count": self.word_count,
            "char_count": self.char_count,
        }

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


INDEX_STRATEGIES = {"fixed_size", "structure_aware"}
HYBRID_STRATEGY = "hybrid"
VALID_STRATEGIES = INDEX_STRATEGIES | {HYBRID_STRATEGY}
