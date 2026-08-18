from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.core.database import init_db
from app.core.config import settings
from app.scripts.run_three_strategy_audit import (
    STRATEGIES,
    _compact_strategy_result,
    _comparison_audit,
    _label_session,
    _select_user_id,
)
from app.services.openai_client import OpenAIService
from app.services.rag import answer_question
from app.services.vector_store import VectorStore


def _slug(value: str | None) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()


def _normalise_file(value: str | None) -> str:
    text = _slug(value)
    text = re.sub(r"\bpdf\b$", "", text).strip()
    text = re.sub(r"^web\s+", "", text).strip()
    return text


def _normalise_url(value: str | None) -> str:
    parsed = urlparse(str(value or "").strip())
    if not parsed.netloc:
        return ""
    host = parsed.netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    path = re.sub(r"/+", "/", parsed.path or "/").rstrip("/")
    return f"{host}{path}".lower()


def _url_path_key(value: str | None) -> str:
    parsed = urlparse(str(value or "").strip())
    path = re.sub(r"/+", "/", parsed.path or "/").strip("/").lower()
    parts = [part for part in path.split("/") if part]
    if parts and parts[0] in {"gb", "en-gb", "uk", "en-us", "us"}:
        parts = parts[1:]
    return "/".join(parts)


def _split_source_list(value: Any) -> list[str]:
    if value in (None, ""):
        return []
    if isinstance(value, float):
        return []
    raw = str(value).strip()
    if not raw or raw.lower() == "nan" or raw.lower().startswith("none -"):
        return []
    return [part.strip() for part in re.split(r";|\n", raw) if part.strip()]


def _gold_sources(item: dict[str, Any]) -> list[dict[str, str]]:
    sources: list[dict[str, str]] = []
    raw_sources: list[str] = []
    for key in [
        "primary_expected_documents",
        "secondary_acceptable_documents",
        "primary_relevant_source",
        "source_locator",
    ]:
        raw_sources.extend(_split_source_list(item.get(key)))

    seen: set[str] = set()
    for raw in raw_sources:
        lowered = raw.lower()
        if lowered.startswith("none -"):
            continue
        is_url = lowered.startswith("http://") or lowered.startswith("https://")
        key = _normalise_url(raw) if is_url else _normalise_file(raw)
        if not key or key in seen:
            continue
        seen.add(key)
        sources.append(
            {
                "raw": raw,
                "kind": "url" if is_url else "file_or_title",
                "norm": key,
                "path_key": _url_path_key(raw) if is_url else "",
            }
        )
    return sources


def _hit_source_key(hit: dict[str, Any]) -> str:
    metadata = hit.get("metadata") or hit
    source_path = str(metadata.get("source_path") or "")
    if source_path.startswith(("http://", "https://")):
        return _normalise_url(source_path)
    return _normalise_file(metadata.get("file_name") or metadata.get("doc_id") or source_path)


def _hit_title_key(hit: dict[str, Any]) -> str:
    metadata = hit.get("metadata") or hit
    return _normalise_file(metadata.get("file_name") or metadata.get("doc_id"))


def _hit_url_path_key(hit: dict[str, Any]) -> str:
    metadata = hit.get("metadata") or hit
    return _url_path_key(metadata.get("source_path"))


def _match_gold(hit: dict[str, Any], gold_sources: list[dict[str, str]]) -> dict[str, str] | None:
    hit_source = _hit_source_key(hit)
    hit_title = _hit_title_key(hit)
    hit_path = _hit_url_path_key(hit)

    for gold in gold_sources:
        if gold["kind"] == "url":
            if gold["norm"] == hit_source:
                return gold
            if gold["path_key"] and gold["path_key"] == hit_path:
                return gold
            continue

        gold_norm = gold["norm"]
        if gold_norm and gold_norm == hit_source:
            return gold
        if gold_norm and gold_norm == hit_title:
            return gold
        if gold_norm and (gold_norm in hit_title or hit_title in gold_norm):
            return gold
    return None


def _unique_universe_sources(vector_store: VectorStore) -> set[str]:
    universe: set[str] = set()
    for strategy in ["fixed_size", "structure_aware"]:
        collection = vector_store.get_collection(strategy)
        result = collection.get(include=["metadatas"])
        for metadata in result.get("metadatas", []):
            universe.add(_hit_source_key(metadata))
    return {source for source in universe if source}


def _metrics_for_hits(
    hits: list[dict[str, Any]],
    gold_sources: list[dict[str, str]],
    top_k: int,
    universe_size: int,
) -> dict[str, Any]:
    if not gold_sources:
        return {
            "precision_at_k": None,
            "recall_at_k": None,
            "f1_at_k": None,
            "tp_chunks": None,
            "fp_chunks": None,
            "tp_sources": None,
            "fp_sources": None,
            "fn_sources": None,
            "tn_sources": None,
            "gold_source_count": 0,
            "matched_gold_sources": [],
            "primary_source_hit": None,
        }

    matched_gold_keys: set[str] = set()
    relevant_hits = 0
    predicted_sources: set[str] = set()
    predicted_non_relevant_sources: set[str] = set()
    gold_by_key = {gold["norm"]: gold for gold in gold_sources}
    primary_key = gold_sources[0]["norm"]

    for hit in hits[:top_k]:
        source_key = _hit_source_key(hit)
        if source_key:
            predicted_sources.add(source_key)
        matched = _match_gold(hit, gold_sources)
        if matched:
            relevant_hits += 1
            matched_gold_keys.add(matched["norm"])
        elif source_key:
            predicted_non_relevant_sources.add(source_key)

    precision = relevant_hits / top_k if top_k else 0.0
    recall = len(matched_gold_keys) / len(gold_sources)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    tp_sources = len(matched_gold_keys)
    fp_sources = len(predicted_non_relevant_sources)
    fn_sources = len(gold_sources) - tp_sources
    tn_sources = max(0, universe_size - tp_sources - fp_sources - fn_sources)
    return {
        "precision_at_k": precision,
        "recall_at_k": recall,
        "f1_at_k": f1,
        "tp_chunks": relevant_hits,
        "fp_chunks": max(0, top_k - relevant_hits),
        "tp_sources": tp_sources,
        "fp_sources": fp_sources,
        "fn_sources": fn_sources,
        "tn_sources": tn_sources,
        "gold_source_count": len(gold_sources),
        "matched_gold_sources": [gold_by_key[key]["raw"] for key in matched_gold_keys if key in gold_by_key],
        "primary_source_hit": primary_key in matched_gold_keys,
    }


def _compact_hits(hits: list[dict[str, Any]], gold_sources: list[dict[str, str]], top_k: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rank, hit in enumerate(hits[:top_k], start=1):
        metadata = hit.get("metadata") or hit
        matched = _match_gold(hit, gold_sources)
        rows.append(
            {
                "rank": rank,
                "chunk_id": hit.get("chunk_id"),
                "file_name": metadata.get("file_name"),
                "source_path": metadata.get("source_path"),
                "section_title": metadata.get("section_title"),
                "page_start": metadata.get("page_start"),
                "page_end": metadata.get("page_end"),
                "score": hit.get("score"),
                "vector_score": hit.get("vector_score"),
                "metadata_boost": hit.get("metadata_boost"),
                "source_strategy": hit.get("source_strategy") or metadata.get("source_strategy") or metadata.get("strategy"),
                "relevant": bool(matched),
                "matched_gold_source": matched["raw"] if matched else "",
                "text_excerpt": str(hit.get("text") or "")[:600],
            }
        )
    return rows


def _retrieve_only_rows(items: list[dict[str, Any]], top_k: int) -> dict[str, Any]:
    service = OpenAIService()
    vector_store = VectorStore()
    universe = _unique_universe_sources(vector_store)
    rows: list[dict[str, Any]] = []

    for index, item in enumerate(items, start=1):
        question = item["question"]
        print(f"[{index}/{len(items)}] {question}", flush=True)
        embedding = service.embed_texts([question])[0]
        gold_sources = _gold_sources(item)
        strategy_results: dict[str, Any] = {}
        for strategy in STRATEGIES:
            print(f"  - retrieval {strategy}", flush=True)
            if strategy == "hybrid":
                hits = vector_store.query_hybrid(
                    query_embedding=embedding,
                    top_k=top_k,
                    query_text=question,
                    metadata_rerank_enabled=True,
                )
            else:
                hits = vector_store.query(
                    strategy,
                    query_embedding=embedding,
                    top_k=top_k,
                    query_text=question,
                    metadata_rerank_enabled=True,
                )
            strategy_results[strategy] = {
                **_metrics_for_hits(hits, gold_sources, top_k, len(universe)),
                "retrieved_count": len(hits),
                "hits": _compact_hits(hits, gold_sources, top_k),
            }
        rows.append(
            {
                **item,
                "gold_sources": gold_sources,
                "strategies": strategy_results,
            }
        )
    return {"rows": rows, "universe_source_count": len(universe)}


def _full_audit_rows(items: list[dict[str, Any]], top_k: int) -> dict[str, Any]:
    init_db()
    user_id = _select_user_id()
    vector_store = VectorStore()
    universe = _unique_universe_sources(vector_store)
    run_id = f"expanded_50_audit_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    sessions: dict[str, int | None] = {strategy: None for strategy in STRATEGIES}
    rows: list[dict[str, Any]] = []

    for index, item in enumerate(items, start=1):
        question = item["question"]
        print(f"[{index}/{len(items)}] {question}", flush=True)
        gold_sources = _gold_sources(item)
        strategy_payload: dict[str, Any] = {}
        raw_results: dict[str, dict[str, Any]] = {}
        for strategy in STRATEGIES:
            print(f"  - answer/judge {strategy}", flush=True)
            result = answer_question(
                user_id=user_id,
                question=question,
                strategy=strategy,
                top_k=top_k,
                run_judge=True,
                metadata_rerank_enabled=True,
                session_id=sessions[strategy],
            )
            if sessions[strategy] is None:
                sessions[strategy] = result["session_id"]
                _label_session(result["session_id"], run_id, strategy)
            hits = result.get("evidence") or []
            compact = _compact_strategy_result(result)
            compact.update(_metrics_for_hits(hits, gold_sources, top_k, len(universe)))
            compact["hits"] = _compact_hits(hits, gold_sources, top_k)
            strategy_payload[strategy] = compact
            raw_results[strategy] = result

        print("  - comparison audit", flush=True)
        comparison = _comparison_audit(question, raw_results)
        rows.append(
            {
                **item,
                "gold_sources": gold_sources,
                "strategies": strategy_payload,
                "comparison_audit": comparison,
            }
        )

    return {"rows": rows, "run_id": run_id, "sessions": sessions, "user_id": user_id, "universe_source_count": len(universe)}


def _summaries(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    subsets = {
        "all": rows,
        "core": [row for row in rows if not row.get("exclude_from_main_eval")],
        "answerable_with_gold_source": [row for row in rows if row.get("gold_sources")],
    }
    for subset_name, subset_rows in subsets.items():
        summary[subset_name] = {}
        for strategy in STRATEGIES:
            values = [row["strategies"][strategy] for row in subset_rows if row["strategies"][strategy].get("precision_at_k") is not None]
            if not values:
                summary[subset_name][strategy] = {}
                continue
            summary[subset_name][strategy] = {
                "question_count": len(values),
                "macro_precision_at_k": sum(v["precision_at_k"] for v in values) / len(values),
                "macro_recall_at_k": sum(v["recall_at_k"] for v in values) / len(values),
                "macro_f1_at_k": sum(v["f1_at_k"] for v in values) / len(values),
                "tp_chunks": sum(v["tp_chunks"] or 0 for v in values),
                "fp_chunks": sum(v["fp_chunks"] or 0 for v in values),
                "tp_sources": sum(v["tp_sources"] or 0 for v in values),
                "fp_sources": sum(v["fp_sources"] or 0 for v in values),
                "fn_sources": sum(v["fn_sources"] or 0 for v in values),
                "tn_sources": sum(v["tn_sources"] or 0 for v in values),
                "primary_source_hit_rate": sum(1 for v in values if v.get("primary_source_hit")) / len(values),
            }
    return summary


def run(args: argparse.Namespace) -> Path:
    with args.dataset.open("r", encoding="utf-8") as f:
        items = json.load(f)
    if not isinstance(items, list):
        raise RuntimeError("Dataset JSON must be a list of question objects.")

    if args.mode == "retrieval-only":
        result = _retrieve_only_rows(items, args.top_k)
    else:
        result = _full_audit_rows(items, args.top_k)

    payload = {
        "dataset_path": str(args.dataset),
        "dataset_name": args.dataset_name or args.dataset.stem,
        "mode": args.mode,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "top_k": args.top_k,
        "strategies": STRATEGIES,
        "metadata_rerank_enabled": True,
        "min_retrieval_score": settings.min_retrieval_score,
        "metric_note": (
            "Precision@K is computed as relevant retrieved chunks divided by K. "
            "Relevance is a document/source-level proxy because the gold labels identify expected documents or URLs. "
            "Recall@K is the share of expected gold sources found in the top K."
        ),
        **result,
    }
    payload["summary"] = _summaries(payload["rows"])

    if args.output:
        output_path = args.output
    else:
        output_path = args.output_dir / (
            f"{args.output_prefix}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"JSON output: {output_path}", flush=True)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run source-level Precision@K/Recall@K/F1@K evaluation.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--dataset-name", default="")
    parser.add_argument("--mode", choices=["retrieval-only", "full-audit"], default="retrieval-only")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "runs" / "source_level",
    )
    parser.add_argument("--output-prefix", default="gold_standard_eval")
    parser.add_argument(
        "--output",
        type=Path,
        help="Write to this exact JSON path instead of a timestamped file in --output-dir.",
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
