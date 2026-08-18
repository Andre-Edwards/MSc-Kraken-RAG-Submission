from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


STRATEGY_LABELS = {
    "structure_aware": "Structure-aware",
    "fixed_size": "Fixed-size",
    "hybrid": "Hybrid",
}

DEFAULT_PATTERNS = {
    "old_60": "original_source_level_at5.json",
    "expanded_50": "expanded_source_level_at5.json",
}


def _evaluation_dir() -> Path:
    return Path(__file__).resolve().parent


def _retrieval_metrics_dir() -> Path:
    return _evaluation_dir() / "results"


def _latest_file(directory: Path, pattern: str) -> Path:
    matches = sorted(directory.glob(pattern), key=lambda path: path.stat().st_mtime, reverse=True)
    if not matches:
        raise FileNotFoundError(f"No files matched {pattern!r} in {directory}")
    return matches[0]


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _f1(precision: float, recall: float) -> float:
    return _safe_div(2 * precision * recall, precision + recall)


def _pct(value: Any) -> str:
    if value in (None, ""):
        return "N/A"
    return f"{float(value):.3f}"


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _matrix_record(data: dict[str, Any], dataset_label: str, subset: str, strategy: str) -> dict[str, Any] | None:
    summary = data.get("summary", {}).get(subset, {}).get(strategy)
    if not summary:
        return None

    tp = int(summary.get("tp_sources") or 0)
    fp = int(summary.get("fp_sources") or 0)
    fn = int(summary.get("fn_sources") or 0)
    tn = int(summary.get("tn_sources") or 0)
    source_precision = _safe_div(tp, tp + fp)
    source_recall = _safe_div(tp, tp + fn)
    source_f1 = _f1(source_precision, source_recall)

    return {
        "dataset": dataset_label,
        "dataset_name": data.get("dataset_name") or dataset_label,
        "dataset_path": data.get("dataset_path") or "",
        "subset": subset,
        "strategy": strategy,
        "strategy_label": STRATEGY_LABELS.get(strategy, strategy),
        "top_k": data.get("top_k"),
        "question_count": summary.get("question_count"),
        "universe_source_count": data.get("universe_source_count"),
        "tp_sources": tp,
        "fp_sources": fp,
        "fn_sources": fn,
        "tn_sources": tn,
        "tp_chunks": int(summary.get("tp_chunks") or 0),
        "fp_chunks": int(summary.get("fp_chunks") or 0),
        "macro_precision_at_k": summary.get("macro_precision_at_k"),
        "macro_recall_at_k": summary.get("macro_recall_at_k"),
        "macro_f1_at_k": summary.get("macro_f1_at_k"),
        "primary_source_hit_rate": summary.get("primary_source_hit_rate"),
        "source_micro_precision": source_precision,
        "source_micro_recall": source_recall,
        "source_micro_f1": source_f1,
    }


def _collect_records(datasets: list[tuple[str, Path]], subsets: list[str]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for label, path in datasets:
        data = _load_json(path)
        for subset in subsets:
            for strategy in data.get("strategies", []):
                record = _matrix_record(data, label, subset, strategy)
                if record:
                    record["source_json"] = str(path)
                    records.append(record)
    return records


def _matrix_text(record: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"{record['dataset']} | {record['subset']} | {record['strategy_label']}",
            f"Questions: {record['question_count']} | K={record['top_k']} | Source universe: {record['universe_source_count']}",
            "",
            "Source-level confusion matrix:",
            "                          Retrieved source   Not retrieved source",
            f"Gold source relevant      {record['tp_sources']:>16}   {record['fn_sources']:>20}",
            f"Gold source not relevant  {record['fp_sources']:>16}   {record['tn_sources']:>20}",
            "",
            (
                f"Reported macro @K: P={_pct(record['macro_precision_at_k'])} | "
                f"R={_pct(record['macro_recall_at_k'])} | F1={_pct(record['macro_f1_at_k'])}"
            ),
            (
                f"Source-matrix micro: P={_pct(record['source_micro_precision'])} | "
                f"R={_pct(record['source_micro_recall'])} | F1={_pct(record['source_micro_f1'])}"
            ),
            (
                f"Top-K chunk counts: relevant={record['tp_chunks']} | "
                f"non-relevant={record['fp_chunks']}"
            ),
            f"Primary source hit rate: {_pct(record['primary_source_hit_rate'])}",
        ]
    )


def _print_records(records: list[dict[str, Any]]) -> str:
    sections = [
        "RAG Retrieval Confusion Matrix Summary",
        "=" * 39,
        (
            "Note: matrices are source/document-level because the gold standard labels expected "
            "documents or URLs, not every relevant chunk. Report macro Precision@K, Recall@K and "
            "F1@K as the main retrieval metrics; use the confusion matrices as supporting detail."
        ),
    ]
    for record in records:
        sections.append("")
        sections.append(_matrix_text(record))
        sections.append("-" * 72)
    output = "\n".join(sections)
    print(output)
    return output


def _write_csv(records: list[dict[str, Any]], path: Path) -> None:
    fields = [
        "dataset",
        "subset",
        "strategy",
        "strategy_label",
        "top_k",
        "question_count",
        "universe_source_count",
        "tp_sources",
        "fp_sources",
        "fn_sources",
        "tn_sources",
        "tp_chunks",
        "fp_chunks",
        "macro_precision_at_k",
        "macro_recall_at_k",
        "macro_f1_at_k",
        "primary_source_hit_rate",
        "source_micro_precision",
        "source_micro_recall",
        "source_micro_f1",
        "source_json",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def _matrix_markdown(record: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"### {record['dataset']} - {record['subset']} - {record['strategy_label']}",
            "",
            (
                f"Questions: {record['question_count']} | K: {record['top_k']} | "
                f"Source universe: {record['universe_source_count']}"
            ),
            "",
            "| Gold / retrieved source | Retrieved source | Not retrieved source |",
            "|---|---:|---:|",
            f"| Gold source relevant | {record['tp_sources']} | {record['fn_sources']} |",
            f"| Gold source not relevant | {record['fp_sources']} | {record['tn_sources']} |",
            "",
            (
                f"Reported macro @K: Precision {_pct(record['macro_precision_at_k'])}, "
                f"Recall {_pct(record['macro_recall_at_k'])}, F1 {_pct(record['macro_f1_at_k'])}."
            ),
            (
                f"Source-matrix micro: Precision {_pct(record['source_micro_precision'])}, "
                f"Recall {_pct(record['source_micro_recall'])}, F1 {_pct(record['source_micro_f1'])}."
            ),
            f"Top-K chunk counts: relevant {record['tp_chunks']}, non-relevant {record['fp_chunks']}.",
            "",
        ]
    )


def _write_markdown(records: list[dict[str, Any]], path: Path) -> None:
    sections = [
        "# RAG Retrieval Confusion Matrix Summary",
        "",
        (
            "These matrices are source/document-level because the gold standard labels expected "
            "documents or URLs, not every relevant chunk. Precision@K, Recall@K and F1@K remain "
            "the main retrieval metrics; the matrices provide supporting detail for the dissertation."
        ),
        "",
    ]
    sections.extend(_matrix_markdown(record) for record in records)
    path.write_text("\n".join(sections), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Path]:
    metrics_dir = args.metrics_dir
    datasets: list[tuple[str, Path]] = []
    if args.old_json:
        datasets.append(("old_60", args.old_json))
    else:
        datasets.append(("old_60", _latest_file(metrics_dir, DEFAULT_PATTERNS["old_60"])))

    if args.expanded_json:
        datasets.append(("expanded_50", args.expanded_json))
    else:
        datasets.append(("expanded_50", _latest_file(metrics_dir, DEFAULT_PATTERNS["expanded_50"])))

    records = _collect_records(datasets, args.subsets)
    if not records:
        raise RuntimeError("No confusion matrix records could be built from the selected files.")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    text_output = _print_records(records)
    text_path = output_dir / f"confusion_matrix_summary_{timestamp}.txt"
    csv_path = output_dir / f"confusion_matrix_summary_{timestamp}.csv"
    markdown_path = output_dir / f"confusion_matrix_summary_{timestamp}.md"

    text_path.write_text(text_output, encoding="utf-8")
    _write_csv(records, csv_path)
    _write_markdown(records, markdown_path)

    print("\nSaved outputs:")
    print(f"  TXT : {text_path}")
    print(f"  CSV : {csv_path}")
    print(f"  MD  : {markdown_path}")

    return {"txt": text_path, "csv": csv_path, "markdown": markdown_path}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Display source-level confusion matrix summaries for Kraken RAG retrieval evaluations."
    )
    parser.add_argument(
        "--metrics-dir",
        type=Path,
        default=_retrieval_metrics_dir(),
        help="Directory containing evaluation JSON files.",
    )
    parser.add_argument("--old-json", type=Path, help="Old 60-question evaluation JSON. Defaults to latest.")
    parser.add_argument("--expanded-json", type=Path, help="Expanded 50-question evaluation JSON. Defaults to latest.")
    parser.add_argument(
        "--subsets",
        nargs="+",
        default=["core"],
        choices=["all", "core", "answerable_with_gold_source"],
        help="Evaluation subsets to display.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_evaluation_dir() / "runs" / "confusion_matrices",
        help="Where to save TXT/CSV/HTML outputs.",
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
