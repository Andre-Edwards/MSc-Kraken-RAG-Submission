from __future__ import annotations

import os
import sys

from app.core.config import settings
from app.core.database import init_db, seed_demo_user
from app.services.ingestion import ingest_documents
from app.services.vector_store import VectorStore


def _indexes_ready() -> bool:
    store = VectorStore()
    return store.count("fixed_size") > 0 and store.count("structure_aware") > 0


def _maybe_ingest() -> None:
    if not settings.auto_ingest_on_start:
        print("AUTO_INGEST_ON_START=false; skipping startup indexing.", flush=True)
        return

    if settings.force_reindex_on_start or not _indexes_ready():
        print("Indexing Kraken PDFs into persistent Chroma storage...", flush=True)
        counts = ingest_documents(
            strategies=["fixed_size", "structure_aware"],
            force=settings.force_reindex_on_start,
        )
        print(f"Indexing complete: {counts}", flush=True)
    else:
        print("Chroma indexes already exist; skipping startup indexing.", flush=True)


def main() -> None:
    settings.ensure_storage_dirs()
    init_db()
    seed_demo_user()
    _maybe_ingest()

    port = os.environ.get("PORT", "8000")
    os.execvp(
        sys.executable,
        [
            sys.executable,
            "-m",
            "uvicorn",
            "app.main:app",
            "--host",
            "0.0.0.0",
            "--port",
            port,
        ],
    )


if __name__ == "__main__":
    main()
