from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


STRATEGIES = ["structure_aware", "fixed_size", "hybrid"]
DEFAULT_CHUNK_GOLD_STANDARD = (
    Path(__file__).resolve().parent / "gold_standards" / "chunk_gold_standard_20.json"
)




def _evaluation_dir() -> Path:
    return Path(__file__).resolve().parent


def _load_chunk_labels(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"No chunk-level gold standard found at {path}. "
            "Run evaluation/build_gold_standards.py first."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Chunk-level gold-standard JSON must contain a top-level list.")
    required = {"id", "question", "gold_passage", "source_match", "min_groups", "groups"}
    seen: set[str] = set()
    for index, label in enumerate(payload, start=1):
        if not isinstance(label, dict):
            raise ValueError(f"Chunk label {index} is not an object.")
        missing = sorted(required - label.keys())
        if missing:
            raise ValueError(f"Chunk label {index} is missing: {', '.join(missing)}")
        question_id = str(label["id"])
        if question_id in seen:
            raise ValueError(f"Duplicate chunk-level question ID: {question_id}")
        seen.add(question_id)
    return payload


def _project_root() -> Path:
    return _evaluation_dir().parents[1]


def _latest_expanded_eval() -> Path:
    path = _evaluation_dir() / "results" / "expanded_source_level_at5.json"
    if not path.exists():
        raise FileNotFoundError(f"No expanded source-level evaluation JSON found at {path}")
    return path


def _normalise_text(value: str | None) -> str:
    text = str(value or "").lower()
    text = text.replace("â€™", "'").replace("â€œ", '"').replace("â€", '"')
    return re.sub(r"\s+", " ", text)


def _normalise_file(value: str | None) -> str:
    text = _normalise_text(value)
    text = re.sub(r"[^a-z0-9]+", " ", text)
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


def _source_key(hit: dict[str, Any]) -> str:
    source_path = str(hit.get("source_path") or "")
    if source_path.startswith(("http://", "https://")):
        return _normalise_url(source_path)
    return _normalise_file(hit.get("file_name") or hit.get("source_path"))


def _source_path_key(hit: dict[str, Any]) -> str:
    return _url_path_key(hit.get("source_path"))


def _source_matches(hit: dict[str, Any], expected_sources: list[str]) -> bool:
    hit_key = _source_key(hit)
    hit_path_key = _source_path_key(hit)
    hit_file = _normalise_file(hit.get("file_name"))
    for source in expected_sources:
        if str(source).startswith(("http://", "https://")):
            if _normalise_url(source) == hit_key:
                return True
            if _url_path_key(source) and _url_path_key(source) == hit_path_key:
                return True
        else:
            source_key = _normalise_file(source)
            if source_key == hit_key or source_key == hit_file:
                return True
            if source_key and (source_key in hit_file or hit_file in source_key):
                return True
    return False


def _phrase_present(text: str, phrase: str) -> bool:
    phrase_text = _normalise_text(phrase)
    return phrase_text in text


def _load_chunk_texts(processed_dir: Path) -> dict[str, str]:
    chunks: dict[str, str] = {}
    for path in [
        processed_dir / "chunks_structure_aware.jsonl",
        processed_dir / "chunks_fixed_size.jsonl",
    ]:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                chunks[str(row["chunk_id"])] = str(row.get("text") or "")
    return chunks


def _criterion_matches(text: str, groups: list[list[str]]) -> tuple[int, list[str], list[str]]:
    normalised = _normalise_text(text)
    matched: list[str] = []
    missing: list[str] = []
    for group in groups:
        if any(_phrase_present(normalised, phrase) for phrase in group):
            matched.append(" / ".join(group))
        else:
            missing.append(" / ".join(group))
    return len(matched), matched, missing


def _evaluate_hit(hit: dict[str, Any], label: dict[str, Any], chunk_texts: dict[str, str]) -> dict[str, Any]:
    chunk_id = str(hit.get("chunk_id") or "")
    full_text = chunk_texts.get(chunk_id) or hit.get("text_excerpt") or ""
    source_match = _source_matches(hit, label["source_match"])
    matched_count, matched_groups, missing_groups = _criterion_matches(full_text, label["groups"])
    is_relevant = source_match and matched_count >= int(label["min_groups"])
    if is_relevant:
        reason = "Source matched and required answer-bearing keyword groups were present."
    elif not source_match:
        reason = "Source did not match the labelled answer-bearing document or URL."
    else:
        reason = "Source matched, but the chunk did not contain enough answer-bearing keyword groups."
    return {
        "chunk_id": chunk_id,
        "rank": hit.get("rank"),
        "file_name": hit.get("file_name"),
        "source_path": hit.get("source_path"),
        "section_title": hit.get("section_title"),
        "pages": f"{hit.get('page_start') or ''}-{hit.get('page_end') or ''}",
        "retrieval_score": hit.get("score"),
        "vector_score": hit.get("vector_score"),
        "metadata_boost": hit.get("metadata_boost"),
        "source_strategy": hit.get("source_strategy"),
        "source_match": source_match,
        "matched_group_count": matched_count,
        "required_group_count": label["min_groups"],
        "matched_groups": matched_groups,
        "missing_groups": missing_groups,
        "chunk_relevant": is_relevant,
        "relevance_reason": reason,
        "text_excerpt": full_text[:900],
    }


def _metrics(evaluated_hits: list[dict[str, Any]], top_k: int) -> dict[str, Any]:
    relevant_hits = [hit for hit in evaluated_hits[:top_k] if hit["chunk_relevant"]]
    precision = len(relevant_hits) / top_k
    recall = 1.0 if relevant_hits else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    first_rank = relevant_hits[0]["rank"] if relevant_hits else None
    mrr = 1.0 / float(first_rank) if first_rank else 0.0
    return {
        "precision_at_k": precision,
        "passage_recall_at_k": recall,
        "f1_at_k": f1,
        "hit_at_k": 1 if relevant_hits else 0,
        "mrr": mrr,
        "relevant_retrieved_chunks": len(relevant_hits),
        "first_relevant_rank": first_rank,
    }


def _summaries(rows: list[dict[str, Any]], top_k: int) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for strategy in STRATEGIES:
        values = [row["strategies"][strategy]["metrics"] for row in rows]
        summary[strategy] = {
            "question_count": len(values),
            "macro_precision_at_k": sum(item["precision_at_k"] for item in values) / len(values),
            "macro_passage_recall_at_k": sum(item["passage_recall_at_k"] for item in values) / len(values),
            "macro_f1_at_k": sum(item["f1_at_k"] for item in values) / len(values),
            "hit_rate_at_k": sum(item["hit_at_k"] for item in values) / len(values),
            "mean_reciprocal_rank": sum(item["mrr"] for item in values) / len(values),
            "total_relevant_retrieved_chunks": sum(item["relevant_retrieved_chunks"] for item in values),
            "top_k": top_k,
        }
    return summary


def run(args: argparse.Namespace) -> Path:
    evaluation_path = args.evaluation_json or _latest_expanded_eval()
    data = json.loads(evaluation_path.read_text(encoding="utf-8"))
    rows_by_id = {row["id"]: row for row in data["rows"]}
    processed_dir = args.processed_dir or (
        _project_root() / "backend" / "storage" / "processed"
    )
    chunk_texts = _load_chunk_texts(processed_dir)
    labels = _load_chunk_labels(args.gold_standard)

    rows: list[dict[str, Any]] = []
    for label in labels:
        source_row = rows_by_id.get(label["id"])
        if not source_row:
            raise KeyError(f"Question {label['id']} was not found in {evaluation_path}")
        if _normalise_text(source_row.get("question")) != _normalise_text(label["question"]):
            raise ValueError(f"Question text mismatch for {label['id']}")
        strategy_payload: dict[str, Any] = {}
        for strategy in STRATEGIES:
            hits = source_row["strategies"][strategy]["hits"]
            evaluated_hits = [_evaluate_hit(hit, label, chunk_texts) for hit in hits[: args.top_k]]
            strategy_payload[strategy] = {
                "metrics": _metrics(evaluated_hits, args.top_k),
                "hits": evaluated_hits,
            }
        rows.append(
            {
                "id": source_row["id"],
                "question": label["question"],
                "category": label.get("category"),
                "department_use_case": label.get("department_use_case"),
                "expected_answer_summary": label.get("expected_answer_summary"),
                "gold_passage": label["gold_passage"],
                "source_match": label["source_match"],
                "minimum_keyword_groups": label["min_groups"],
                "keyword_groups": [" | ".join(group) for group in label["groups"]],
                "strategies": strategy_payload,
            }
        )

    payload = {
        "run_id": f"chunk_level_eval_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_evaluation_json": str(evaluation_path),
        "gold_standard_json": str(args.gold_standard),
        "processed_chunks_dir": str(processed_dir),
        "question_count": len(rows),
        "top_k": args.top_k,
        "strategies": STRATEGIES,
        "method_note": (
            "Chunk-level relevance is based on whether a retrieved chunk comes from the labelled source "
            "and contains manually defined answer-bearing keyword groups for the gold passage. "
            "Passage Recall@K is a hit-style recall: 1 if any top-K chunk contains the labelled answer-bearing passage, otherwise 0."
        ),
        "rows": rows,
        "summary": _summaries(rows, args.top_k),
    }

    output_path = args.output or args.output_dir / f"{payload['run_id']}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"JSON output: {output_path}")
    for strategy, summary in payload["summary"].items():
        print(
            strategy,
            f"P@{args.top_k}={summary['macro_precision_at_k']:.3f}",
            f"PassageR@{args.top_k}={summary['macro_passage_recall_at_k']:.3f}",
            f"F1@{args.top_k}={summary['macro_f1_at_k']:.3f}",
            f"Hit@{args.top_k}={summary['hit_rate_at_k']:.3f}",
            f"MRR={summary['mean_reciprocal_rank']:.3f}",
        )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a 20-question chunk-level RAG retrieval evaluation.")
    parser.add_argument("--evaluation-json", type=Path)
    parser.add_argument(
        "--gold-standard",
        type=Path,
        default=DEFAULT_CHUNK_GOLD_STANDARD,
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--processed-dir",
        type=Path,
        help="Directory containing the fixed-size and structure-aware chunk JSONL files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_evaluation_dir() / "runs" / "chunk_level",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Write to this exact JSON path instead of a timestamped file in --output-dir.",
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
