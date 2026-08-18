from __future__ import annotations

import re
from typing import Any

import chromadb

from app.core.config import settings
from app.models import ChunkRecord, INDEX_STRATEGIES


COLLECTION_BY_STRATEGY = {
    "fixed_size": "kraken_fixed_size",
    "structure_aware": "kraken_structure_aware",
}

SEARCH_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "about",
    "by",
    "does",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "say",
    "says",
    "the",
    "to",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}


def _normalise_token(token: str) -> str:
    token = token.lower().strip()
    if len(token) > 4 and token.endswith("ies"):
        return f"{token[:-3]}y"
    if len(token) > 3 and token.endswith("s"):
        return token[:-1]
    return token


def _metadata_token_list(value: str | None) -> list[str]:
    if not value:
        return []
    cleaned = re.sub(r"\.pdf$", "", value, flags=re.I)
    tokens = re.findall(r"[a-zA-Z0-9]+", cleaned.lower())
    return [
        _normalise_token(token)
        for token in tokens
        if token and token not in SEARCH_STOPWORDS
    ]


def _metadata_tokens(value: str | None) -> set[str]:
    return set(_metadata_token_list(value))


def _normalised_phrase(value: str | None) -> str:
    return " ".join(_metadata_token_list(value))


def _overlap_ratio(query_tokens: set[str], metadata_tokens: set[str]) -> float:
    if not query_tokens or not metadata_tokens:
        return 0.0
    return len(query_tokens & metadata_tokens) / len(metadata_tokens)


def _metadata_boost(
    query_text: str | None,
    metadata: dict[str, Any],
    metadata_rerank_enabled: bool,
) -> float:
    if not metadata_rerank_enabled or not query_text:
        return 0.0

    query_tokens = _metadata_tokens(query_text)
    query_phrase = _normalised_phrase(query_text)
    file_name = metadata.get("file_name")
    section_title = metadata.get("section_title")
    title_tokens = _metadata_tokens(file_name)
    section_tokens = _metadata_tokens(section_title)

    title_overlap = _overlap_ratio(query_tokens, title_tokens)
    section_overlap = _overlap_ratio(query_tokens, section_tokens)
    title_phrase = _normalised_phrase(file_name)
    exact_title_match = bool(title_phrase and title_phrase in query_phrase)

    title_boost = settings.metadata_title_boost * title_overlap
    if exact_title_match:
        title_boost = settings.metadata_title_boost
    section_boost = settings.metadata_section_boost * section_overlap
    return min(settings.metadata_title_boost + settings.metadata_section_boost, title_boost + section_boost)


def _dedupe_key(hit: dict[str, Any]) -> tuple[str, str, int, int, str]:
    metadata = hit["metadata"]
    normalised_text = re.sub(r"\s+", " ", hit["text"]).strip().lower()
    return (
        str(metadata.get("file_name") or ""),
        str(metadata.get("section_title") or ""),
        int(metadata.get("page_start") or 0),
        int(metadata.get("page_end") or 0),
        normalised_text[:700],
    )


class VectorStore:
    def __init__(self) -> None:
        self.client = chromadb.PersistentClient(path=str(settings.resolve_chroma_dir()))

    def collection_name(self, strategy: str) -> str:
        if strategy not in COLLECTION_BY_STRATEGY:
            raise ValueError(f"Unsupported chunking strategy: {strategy}")
        return COLLECTION_BY_STRATEGY[strategy]

    def get_collection(self, strategy: str):
        return self.client.get_or_create_collection(
            name=self.collection_name(strategy),
            metadata={"hnsw:space": "cosine"},
        )

    def reset_collection(self, strategy: str) -> None:
        name = self.collection_name(strategy)
        try:
            self.client.delete_collection(name)
        except Exception:
            pass
        self.client.get_or_create_collection(name=name, metadata={"hnsw:space": "cosine"})

    def upsert_chunks(self, strategy: str, chunks: list[ChunkRecord], embeddings: list[list[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("Chunk and embedding counts do not match")
        collection = self.get_collection(strategy)
        if not chunks:
            return
        collection.upsert(
            ids=[chunk.chunk_id for chunk in chunks],
            documents=[chunk.text for chunk in chunks],
            metadatas=[chunk.metadata() for chunk in chunks],
            embeddings=embeddings,
        )

    def query(
        self,
        strategy: str,
        query_embedding: list[float],
        top_k: int,
        query_text: str | None = None,
        metadata_rerank_enabled: bool | None = None,
    ) -> list[dict[str, Any]]:
        collection = self.get_collection(strategy)
        collection_count = collection.count()
        if collection_count == 0:
            return []
        use_metadata_rerank = (
            settings.metadata_rerank_enabled
            if metadata_rerank_enabled is None
            else metadata_rerank_enabled
        )
        candidate_k = top_k
        if use_metadata_rerank and query_text:
            candidate_k = max(top_k, top_k * settings.metadata_rerank_candidate_multiplier)
            candidate_k = min(candidate_k, collection_count)
        result = collection.query(
            query_embeddings=[query_embedding],
            n_results=candidate_k,
            include=["documents", "metadatas", "distances"],
        )
        hits: list[dict[str, Any]] = []
        ids = result.get("ids", [[]])[0]
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        for hit_id, document, metadata, distance in zip(ids, documents, metadatas, distances):
            vector_score = max(0.0, 1.0 - float(distance))
            metadata_boost = _metadata_boost(query_text, metadata, use_metadata_rerank)
            score = min(1.0, vector_score + metadata_boost)
            hits.append(
                {
                    "chunk_id": hit_id,
                    "text": document,
                    "metadata": metadata,
                    "distance": float(distance),
                    "vector_score": vector_score,
                    "metadata_boost": metadata_boost,
                    "score": score,
                }
            )
        hits.sort(key=lambda hit: hit["score"], reverse=True)
        return hits[:top_k]

    def query_hybrid(
        self,
        query_embedding: list[float],
        top_k: int,
        query_text: str | None = None,
        metadata_rerank_enabled: bool | None = None,
    ) -> list[dict[str, Any]]:
        candidate_k = max(top_k, top_k * settings.hybrid_candidate_multiplier)
        combined: list[dict[str, Any]] = []
        for strategy in sorted(INDEX_STRATEGIES):
            hits = self.query(
                strategy,
                query_embedding=query_embedding,
                top_k=candidate_k,
                query_text=query_text,
                metadata_rerank_enabled=metadata_rerank_enabled,
            )
            for hit in hits:
                metadata = dict(hit["metadata"])
                metadata["source_strategy"] = strategy
                combined.append({**hit, "metadata": metadata, "source_strategy": strategy})

        best_by_key: dict[tuple[str, str, int, int, str], dict[str, Any]] = {}
        for hit in combined:
            key = _dedupe_key(hit)
            current = best_by_key.get(key)
            if current is None or hit["score"] > current["score"]:
                best_by_key[key] = hit

        hits = list(best_by_key.values())
        hits.sort(
            key=lambda hit: (
                hit["score"],
                hit.get("metadata_boost") or 0.0,
                hit.get("vector_score") or 0.0,
            ),
            reverse=True,
        )
        return hits[:top_k]

    def count(self, strategy: str) -> int:
        return self.get_collection(strategy).count()

    def delete_document(self, file_name: str) -> dict[str, int]:
        deleted: dict[str, int] = {}
        for strategy in sorted(INDEX_STRATEGIES):
            collection = self.get_collection(strategy)
            result = collection.get(where={"file_name": file_name})
            ids = result.get("ids", [])
            if ids:
                collection.delete(ids=ids)
            deleted[strategy] = len(ids)
        return deleted

    def delete_source_path(self, source_path: str) -> dict[str, int]:
        deleted: dict[str, int] = {}
        for strategy in sorted(INDEX_STRATEGIES):
            collection = self.get_collection(strategy)
            result = collection.get(where={"source_path": source_path})
            ids = result.get("ids", [])
            if ids:
                collection.delete(ids=ids)
            deleted[strategy] = len(ids)
        return deleted
