from __future__ import annotations

import argparse

from app.core.database import init_db, seed_demo_user
from app.services.ingestion import ingest_documents


def main() -> None:
    parser = argparse.ArgumentParser(description="Index Kraken PDFs into ChromaDB.")
    parser.add_argument("--force", action="store_true", help="Reset Chroma collections before indexing.")
    parser.add_argument(
        "--strategy",
        action="append",
        choices=["fixed_size", "structure_aware"],
        help="Index only this strategy. Can be passed more than once.",
    )
    args = parser.parse_args()

    init_db()
    seed_demo_user()
    strategies = args.strategy or ["fixed_size", "structure_aware"]
    counts = ingest_documents(strategies=strategies, force=args.force)
    for strategy, count in counts.items():
        print(f"{strategy}: {count} chunks indexed")


if __name__ == "__main__":
    main()
