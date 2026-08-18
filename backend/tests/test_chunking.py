from __future__ import annotations

import unittest

from app.core.config import settings
from app.models import PageRecord
from app.services.chunking.fixed_size import build_fixed_size_chunks
from app.services.chunking.section_aware import build_structure_aware_chunks


class ChunkingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.original = {
            "fixed_chunk_words": settings.fixed_chunk_words,
            "fixed_chunk_overlap": settings.fixed_chunk_overlap,
            "section_chunk_words": settings.section_chunk_words,
            "section_chunk_overlap": settings.section_chunk_overlap,
            "section_min_chunk_words": settings.section_min_chunk_words,
        }

    def tearDown(self) -> None:
        for name, value in self.original.items():
            setattr(settings, name, value)

    def test_fixed_size_respects_window_and_overlap(self) -> None:
        settings.fixed_chunk_words = 50
        settings.fixed_chunk_overlap = 10
        text = " ".join(f"word{i}" for i in range(120))
        pages = [
            PageRecord(
                doc_id="test-document",
                file_name="test.pdf",
                source_path="test.pdf",
                page_num=1,
                text=text,
                clean_text=text,
            )
        ]

        chunks = build_fixed_size_chunks(pages)

        self.assertEqual(len(chunks), 3)
        self.assertTrue(all(chunk.word_count <= 50 for chunk in chunks))
        first_words = chunks[0].text.split()
        second_words = chunks[1].text.split()
        self.assertEqual(first_words[-10:], second_words[:10])

    def test_structure_aware_preserves_major_section_titles(self) -> None:
        settings.section_chunk_words = 80
        settings.section_chunk_overlap = 10
        settings.section_min_chunk_words = 1
        purpose = " ".join(["purpose"] * 30)
        scope = " ".join(["scope"] * 30)
        text = f"1. PURPOSE\n{purpose}\n2. SCOPE\n{scope}"
        pages = [
            PageRecord(
                doc_id="policy",
                file_name="policy.pdf",
                source_path="policy.pdf",
                page_num=1,
                text=text,
                clean_text=text,
            )
        ]

        chunks = build_structure_aware_chunks(pages)
        titles = {chunk.section_title for chunk in chunks}

        self.assertIn("1. PURPOSE", titles)
        self.assertIn("2. SCOPE", titles)


if __name__ == "__main__":
    unittest.main()
