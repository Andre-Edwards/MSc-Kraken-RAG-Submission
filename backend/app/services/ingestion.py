from __future__ import annotations

import json
from pathlib import Path

from app.core.database import bump_index_version
from app.core.config import settings
from app.models import ChunkRecord
from app.services.chunking.fixed_size import build_fixed_size_chunks
from app.services.chunking.section_aware import build_structure_aware_chunks
from app.services.document_loader import load_available_corpus
from app.services.openai_client import OpenAIService
from app.services.vector_store import VectorStore


def _write_chunks_jsonl(strategy: str, chunks: list[ChunkRecord]) -> Path:
    settings.processed_dir.mkdir(parents=True, exist_ok=True)
    path = settings.processed_dir / f"chunks_{strategy}.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")
    return path


def _embed_in_batches(openai_service: OpenAIService, chunks: list[ChunkRecord], batch_size: int = 64) -> list[list[float]]:
    embeddings: list[list[float]] = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        embeddings.extend(openai_service.embed_texts([chunk.text for chunk in batch]))
    return embeddings


def build_chunks_for_strategy(strategy: str, pdf_dir: Path, web_dir: Path | None = None) -> list[ChunkRecord]:
    pages = load_available_corpus(pdf_dir, web_dir=web_dir)
    if strategy == "fixed_size":
        return build_fixed_size_chunks(pages)
    if strategy == "structure_aware":
        return build_structure_aware_chunks(pages)
    raise ValueError(f"Unsupported chunking strategy: {strategy}")


def ingest_documents(strategies: list[str], force: bool = False) -> dict[str, int]:
    pdf_dir = settings.resolve_pdf_dir()
    web_dir = settings.resolve_web_corpus_dir()

    vector_store = VectorStore()
    openai_service = OpenAIService()
    counts: dict[str, int] = {}

    for strategy in strategies:
        if force:
            vector_store.reset_collection(strategy)

        chunks = build_chunks_for_strategy(strategy, pdf_dir, web_dir=web_dir)
        _write_chunks_jsonl(strategy, chunks)
        embeddings = _embed_in_batches(openai_service, chunks)
        vector_store.upsert_chunks(strategy, chunks, embeddings)
        counts[strategy] = len(chunks)

    bump_index_version()
    return counts
