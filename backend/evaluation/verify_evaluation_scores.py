"""Verify saved RAG evaluation summaries by recomputing metrics from row data.

This script is intended for dissertation QA. It does not run new model calls,
retrieve new chunks, or change any output files. It simply reloads the saved
JSON evaluation exports and recomputes the summary tables from the per-question
records.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


STRATEGIES = ("structure_aware", "fixed_size", "hybrid")
TOLERANCE = 1e-9


def evaluation_dir() -> Path:
    return Path(__file__).resolve().parent


def default_source_jsons() -> list[Path]:
    base = evaluation_dir() / "results"
    return [
        base / "original_source_level_at5.json",
        base / "expanded_source_level_at5.json",
    ]


def default_chunk_json() -> Path:
    return evaluation_dir() / "results" / "chunk_level_at5.json"


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} did not contain a JSON object")
    return data


def avg(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def recompute_source_summary(rows: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, float | int]]]:
    subsets = {
        "all": rows,
        "core": [row for row in rows if not row.get("exclude_from_main_eval")],
        "answerable_with_gold_source": [row for row in rows if row.get("gold_sources")],
    }
    output: dict[str, dict[str, dict[str, float | int]]] = {}
    for subset_name, subset_rows in subsets.items():
        output[subset_name] = {}
        for strategy in STRATEGIES:
            values = [
                row["strategies"][strategy]
                for row in subset_rows
                if row.get("strategies", {}).get(strategy, {}).get("precision_at_k") is not None
            ]
            if not values:
                output[subset_name][strategy] = {}
                continue
            output[subset_name][strategy] = {
                "question_count": len(values),
                "macro_precision_at_k": avg([float(item["precision_at_k"]) for item in values]),
                "macro_recall_at_k": avg([float(item["recall_at_k"]) for item in values]),
                "macro_f1_at_k": avg([float(item["f1_at_k"]) for item in values]),
                "tp_chunks": sum(int(item.get("tp_chunks") or 0) for item in values),
                "fp_chunks": sum(int(item.get("fp_chunks") or 0) for item in values),
                "tp_sources": sum(int(item.get("tp_sources") or 0) for item in values),
                "fp_sources": sum(int(item.get("fp_sources") or 0) for item in values),
                "fn_sources": sum(int(item.get("fn_sources") or 0) for item in values),
                "tn_sources": sum(int(item.get("tn_sources") or 0) for item in values),
                "primary_source_hit_rate": avg([1.0 if item.get("primary_source_hit") else 0.0 for item in values]),
            }
    return output


def recompute_chunk_summary(rows: list[dict[str, Any]], top_k: int) -> dict[str, dict[str, float | int]]:
    output: dict[str, dict[str, float | int]] = {}
    for strategy in STRATEGIES:
        values = [row["strategies"][strategy]["metrics"] for row in rows]
        output[strategy] = {
            "question_count": len(values),
            "macro_precision_at_k": avg([float(item["precision_at_k"]) for item in values]),
            "macro_passage_recall_at_k": avg([float(item["passage_recall_at_k"]) for item in values]),
            "macro_f1_at_k": avg([float(item["f1_at_k"]) for item in values]),
            "hit_rate_at_k": avg([float(item["hit_at_k"]) for item in values]),
            "mean_reciprocal_rank": avg([float(item["mrr"]) for item in values]),
            "total_relevant_retrieved_chunks": sum(int(item["relevant_retrieved_chunks"]) for item in values),
            "top_k": top_k,
        }
    return output


def compare_summaries(
    label: str,
    stored: dict[str, Any],
    recomputed: dict[str, Any],
    *,
    prefix: str = "",
) -> list[str]:
    mismatches: list[str] = []
    for key, expected in stored.items():
        current_path = f"{prefix}{key}"
        if isinstance(expected, dict):
            actual = recomputed.get(key)
            if not isinstance(actual, dict):
                mismatches.append(f"{label}: missing dict at {current_path}")
                continue
            mismatches.extend(compare_summaries(label, expected, actual, prefix=f"{current_path}."))
            continue
        actual = recomputed.get(key)
        if isinstance(expected, float):
            if actual is None or abs(float(actual) - expected) > TOLERANCE:
                mismatches.append(f"{label}: {current_path} stored={expected} recomputed={actual}")
        else:
            if actual != expected:
                mismatches.append(f"{label}: {current_path} stored={expected} recomputed={actual}")
    return mismatches


def print_source_table(name: str, summary: dict[str, Any], subset: str = "core") -> None:
    print(f"\n{name} - {subset} source-level metrics")
    print("Strategy          Questions  Precision@5  Recall@5  F1@5  PrimaryHit")
    for strategy in STRATEGIES:
        item = summary[subset][strategy]
        print(
            f"{strategy:16s} "
            f"{int(item['question_count']):9d} "
            f"{float(item['macro_precision_at_k']):11.3f} "
            f"{float(item['macro_recall_at_k']):8.3f} "
            f"{float(item['macro_f1_at_k']):5.3f} "
            f"{float(item['primary_source_hit_rate']):10.3f}"
        )


def print_chunk_table(name: str, summary: dict[str, Any]) -> None:
    print(f"\n{name} - chunk-level metrics")
    print("Strategy          Questions  Precision@5  PassageR@5  F1@5  Hit@5  MRR")
    for strategy in STRATEGIES:
        item = summary[strategy]
        print(
            f"{strategy:16s} "
            f"{int(item['question_count']):9d} "
            f"{float(item['macro_precision_at_k']):11.3f} "
            f"{float(item['macro_passage_recall_at_k']):10.3f} "
            f"{float(item['macro_f1_at_k']):5.3f} "
            f"{float(item['hit_rate_at_k']):5.3f} "
            f"{float(item['mean_reciprocal_rank']):5.3f}"
        )


def verify_source_file(path: Path) -> bool:
    data = load_json(path)
    recomputed = recompute_source_summary(data["rows"])
    mismatches = compare_summaries(path.name, data["summary"], recomputed)
    print_source_table(path.stem, recomputed)
    if mismatches:
        print(f"\nFAIL: {path}")
        for mismatch in mismatches:
            print(f"  - {mismatch}")
        return False
    print(f"PASS: stored source-level summary matches recomputed rows for {path.name}")
    return True


def verify_chunk_file(path: Path) -> bool:
    data = load_json(path)
    recomputed = recompute_chunk_summary(data["rows"], int(data["top_k"]))
    mismatches = compare_summaries(path.name, data["summary"], recomputed)
    print_chunk_table(path.stem, recomputed)
    if mismatches:
        print(f"\nFAIL: {path}")
        for mismatch in mismatches:
            print(f"  - {mismatch}")
        return False
    print(f"PASS: stored chunk-level summary matches recomputed rows for {path.name}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify saved RAG evaluation metric summaries.")
    parser.add_argument("--source-json", type=Path, action="append", help="Source-level evaluation JSON to verify.")
    parser.add_argument("--chunk-json", type=Path, help="Chunk-level evaluation JSON to verify.")
    args = parser.parse_args()

    source_paths = args.source_json or default_source_jsons()
    chunk_path = args.chunk_json or default_chunk_json()

    ok = True
    for path in source_paths:
        ok = verify_source_file(path) and ok
    ok = verify_chunk_file(chunk_path) and ok

    if not ok:
        raise SystemExit(1)
    print("\nAll evaluation summaries verified successfully.")


if __name__ == "__main__":
    main()
