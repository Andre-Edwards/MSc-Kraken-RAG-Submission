from __future__ import annotations

from pathlib import Path

import fitz

from app.models import PageRecord
from app.services.text_cleaning import clean_pages, is_probably_scanned
from app.services.web_scraper import load_web_corpus


def extract_pdf_pages_pymupdf(pdf_path: Path) -> list[PageRecord]:
    doc = fitz.open(pdf_path)
    records: list[PageRecord] = []
    doc_id = pdf_path.stem
    try:
        for i in range(len(doc)):
            page = doc[i]
            text = page.get_text("text") or ""
            records.append(
                PageRecord(
                    doc_id=doc_id,
                    source_path=str(pdf_path),
                    file_name=pdf_path.name,
                    page_num=i + 1,
                    text=text,
                    likely_scanned=is_probably_scanned(text),
                )
            )
    finally:
        doc.close()
    return records


def load_pdf_corpus(pdf_dir: Path) -> list[PageRecord]:
    pdf_paths = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(f"No PDF files found in {pdf_dir}")

    pages: list[PageRecord] = []
    for path in pdf_paths:
        pages.extend(extract_pdf_pages_pymupdf(path))
    return clean_pages(pages)


def load_available_corpus(pdf_dir: Path, web_dir: Path | None = None) -> list[PageRecord]:
    pages: list[PageRecord] = []
    if pdf_dir.exists():
        for path in sorted(pdf_dir.glob("*.pdf")):
            pages.extend(extract_pdf_pages_pymupdf(path))
    if web_dir is not None and web_dir.exists():
        pages.extend(load_web_corpus(web_dir))
    if not pages:
        raise FileNotFoundError(f"No PDF or web corpus pages found in {pdf_dir} / {web_dir}")
    return clean_pages(pages)
