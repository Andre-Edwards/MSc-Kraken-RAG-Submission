from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.core.database import get_connection, init_db, seed_demo_user, utc_now
from app.services.openai_client import OpenAIService
from app.services.rag import answer_question


STRATEGIES = ["structure_aware", "fixed_size", "hybrid"]


def _backend_dir() -> Path:
    return Path(__file__).resolve().parents[2]


def _evaluation_dir() -> Path:
    return _backend_dir() / "evaluation"


def _load_questions(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        rows = json.load(f)
    questions: list[dict[str, str]] = []
    seen: set[str] = set()
    for row in rows:
        question = " ".join(str(row["question"]).split()).strip()
        key = question.lower()
        if question and key not in seen:
            seen.add(key)
            questions.append(
                {
                    "source": str(row.get("source") or "selected"),
                    "question": question,
                }
            )
    return questions


def _select_user_id() -> int:
    seed_demo_user()
    preferred = ["admintest@example.com", "admin@example.com", "demo@example.com"]
    placeholders = ",".join("?" for _ in preferred)
    with get_connection() as conn:
        rows = conn.execute(
            f"""
            SELECT id, lower(email) AS email
            FROM users
            WHERE lower(email) IN ({placeholders})
            """,
            preferred,
        ).fetchall()
    by_email = {row["email"]: int(row["id"]) for row in rows}
    for email in preferred:
        if email in by_email:
            return by_email[email]
    raise RuntimeError("No usable user account was found for the audit run.")


def _label_session(session_id: int, run_id: str, strategy: str) -> None:
    title = f"Three-strategy audit {run_id} - {strategy}"
    with get_connection() as conn:
        conn.execute(
            "UPDATE chat_sessions SET title = ?, updated_at = ? WHERE id = ?",
            (title, utc_now(), session_id),
        )


def _judge_value(result: dict[str, Any], key: str) -> Any:
    judge = result.get("judge") if isinstance(result.get("judge"), dict) else {}
    return judge.get(key)


def _citations_summary(result: dict[str, Any], limit: int = 5) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in (result.get("evidence") or result.get("citations") or [])[:limit]:
        rows.append(
            {
                "label": item.get("label"),
                "file_name": item.get("file_name"),
                "section_title": item.get("section_title"),
                "pages": f"{item.get('page_start')}-{item.get('page_end')}",
                "score": item.get("score"),
                "vector_score": item.get("vector_score"),
                "metadata_boost": item.get("metadata_boost"),
                "source_strategy": item.get("source_strategy"),
                "text": str(item.get("text") or "")[:700],
            }
        )
    return rows


def _compact_strategy_result(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "chat_log_id": result.get("chat_log_id"),
        "session_id": result.get("session_id"),
        "answer": result.get("answer"),
        "retrieval_score": result.get("retrieval_score"),
        "retrieved_count": result.get("retrieved_count"),
        "refused": result.get("refused"),
        "judge_verdict": _judge_value(result, "verdict"),
        "judge_overall_score": _judge_value(result, "overall_score"),
        "judge_groundedness_score": _judge_value(result, "groundedness_score"),
        "judge_citation_score": _judge_value(result, "citation_score"),
        "judge_relevance_score": _judge_value(result, "relevance_score"),
        "judge_completeness_score": _judge_value(result, "completeness_score"),
        "judge_clarity_score": _judge_value(result, "clarity_score"),
        "judge_confidence": _judge_value(result, "confidence"),
        "judge_failure_modes": ", ".join(_judge_value(result, "failure_modes") or []),
        "judge_notes": _judge_value(result, "notes"),
        "citations": _citations_summary(result),
        "latency_ms": result.get("latency_ms"),
        "models": result.get("models"),
    }


def _build_comparison_prompt(question: str, strategy_results: dict[str, dict[str, Any]]) -> str:
    payload: dict[str, Any] = {"question": question, "strategies": {}}
    for strategy, result in strategy_results.items():
        payload["strategies"][strategy] = {
            "answer": result.get("answer"),
            "retrieval_score": result.get("retrieval_score"),
            "judge": result.get("judge"),
            "citations": _citations_summary(result, limit=4),
        }
    return f"""
You are auditing three RAG answers for a dissertation comparison.

Judge the answers as a careful human evaluator would:
- prefer the answer that is most useful, faithful to the requested document/scope, complete, and well cited;
- treat small retrieval-score differences as non-decisive;
- allow "tie" when answers are effectively equally useful;
- use "none" if all answers fail or correctly refuse because the corpus cannot answer.

Return valid JSON only with exactly these keys:
- best_strategy: "structure_aware", "fixed_size", "hybrid", "tie", or "none"
- verdict: "pass", "partial", or "fail"
- notes: one concise paragraph explaining the comparison
- strategy_rationales: object with keys structure_aware, fixed_size, hybrid
- confidence: "low", "medium", or "high"

Data:
{json.dumps(payload, ensure_ascii=False, indent=2)}
""".strip()


def _comparison_audit(question: str, strategy_results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    service = OpenAIService()
    result = service.chat_with_metadata(
        messages=[
            {
                "role": "system",
                "content": "You are a calibrated RAG comparison auditor. Return valid JSON only.",
            },
            {"role": "user", "content": _build_comparison_prompt(question, strategy_results)},
        ],
        temperature=0.0,
        response_format={"type": "json_object"},
        max_tokens=1100,
    )
    try:
        audit = json.loads(result["content"])
    except json.JSONDecodeError:
        audit = {"parse_error": True, "raw": result["content"]}
    audit["comparison_model"] = result["model"]
    audit["comparison_total_tokens"] = result.get("total_tokens")
    return audit


def run(args: argparse.Namespace) -> Path:
    init_db()
    questions = _load_questions(args.questions)
    if not questions:
        raise RuntimeError("No questions were selected for the audit.")
    user_id = _select_user_id()
    run_id = f"three_strategy_audit_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

    sessions: dict[str, int | None] = {strategy: None for strategy in STRATEGIES}
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(questions, start=1):
        question = item["question"]
        print(f"[{index}/{len(questions)}] {question}")
        strategy_results: dict[str, dict[str, Any]] = {}
        for strategy in STRATEGIES:
            print(f"  - {strategy}")
            result = answer_question(
                user_id=user_id,
                question=question,
                strategy=strategy,
                top_k=args.top_k,
                run_judge=True,
                metadata_rerank_enabled=True,
                session_id=sessions[strategy],
            )
            if sessions[strategy] is None:
                sessions[strategy] = result["session_id"]
                _label_session(result["session_id"], run_id, strategy)
            strategy_results[strategy] = result

        print("  - comparison audit")
        comparison = _comparison_audit(question, strategy_results)
        rows.append(
            {
                "question_number": index,
                "question_source": item["source"],
                "question": question,
                "strategies": {
                    strategy: _compact_strategy_result(result)
                    for strategy, result in strategy_results.items()
                },
                "comparison_audit": comparison,
            }
        )

    output_dir = args.output_dir or (_evaluation_dir() / "chat_log_exports")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{run_id}.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "run_id": run_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "user_id": user_id,
                "strategies": STRATEGIES,
                "top_k": args.top_k,
                "sessions": sessions,
                "rows": rows,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"JSON output: {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run selected questions against all three retrieval strategies.")
    parser.add_argument(
        "--questions",
        type=Path,
        default=_evaluation_dir() / "chat_log_exports" / "three_strategy_selected_questions.json",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
