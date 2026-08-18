from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.core.database import get_connection, init_db, seed_demo_user, seeded_testing_accounts, utc_now
from app.services.rag import answer_question


SIMPLE_USER_QUESTIONS = [
    "What types of clients can Kraken provide exchange services to?",
    "What does a client need to do before using Kraken exchange services?",
    "Why might Kraken restrict account access or disable funding?",
    "When can Kraken suspend trading or access to trading?",
    "What does the custody statement say about safeguarding client crypto-assets?",
    "What risks should customers know about crypto-assets?",
    "How can a customer initiate a crypto-asset transfer?",
    "When can Kraken reject a crypto-asset transfer instruction?",
    "How does Kraken manage conflicts of interest?",
    "What does the privacy notice say about personal data retention?",
]


def _repo_backend_dir() -> Path:
    return Path(__file__).resolve().parents[2]


def _evaluation_dir() -> Path:
    return _repo_backend_dir() / "evaluation"


def _load_eval_questions(limit: int) -> list[dict[str, str]]:
    path = _evaluation_dir() / "rag_eval_questions.csv"
    if not path.exists():
        raise FileNotFoundError(f"Evaluation bank not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    answerable = [row for row in rows if row.get("answerable", "").strip().lower() in {"yes", "true", "1"}]
    return answerable[:limit]


def _get_testing_users(limit: int) -> list[dict[str, Any]]:
    expected_emails = [account["email"].lower() for account in seeded_testing_accounts()[:limit]]
    placeholders = ",".join("?" for _ in expected_emails)
    with get_connection() as conn:
        rows = conn.execute(
            f"""
            SELECT id, email, full_name, role
            FROM users
            WHERE lower(email) IN ({placeholders})
            ORDER BY email
            """,
            expected_emails,
        ).fetchall()
    users = [dict(row) for row in rows]
    if len(users) < limit:
        found = {user["email"].lower() for user in users}
        missing = [email for email in expected_emails if email not in found]
        raise RuntimeError(f"Missing seeded testing users: {', '.join(missing)}")
    return users[:limit]


def _judge_value(judge: dict[str, Any] | None, key: str) -> Any:
    if not isinstance(judge, dict):
        return None
    return judge.get(key)


def _feedback_from_judge(judge: dict[str, Any] | None, refused: bool) -> tuple[str, int, str]:
    if refused:
        return (
            "needs_work",
            2,
            "Synthetic simulation: answer refused. Review whether refusal was appropriate before treating this as user feedback.",
        )
    verdict = _judge_value(judge, "verdict")
    overall = _judge_value(judge, "overall_score")
    try:
        overall_int = int(overall)
    except (TypeError, ValueError):
        overall_int = 3

    if verdict == "pass" and overall_int >= 4:
        return (
            "helpful",
            5,
            "Synthetic simulation: judged as useful and sufficiently grounded. Not real user feedback.",
        )
    if verdict == "partial" or overall_int == 3:
        return (
            "needs_work",
            3,
            "Synthetic simulation: partially useful but needs review. Not real user feedback.",
        )
    return (
        "needs_work",
        2,
        "Synthetic simulation: likely weak answer based on judge output. Not real user feedback.",
    )


def _label_session(session_id: int, label: str, question: str) -> None:
    title = f"Synthetic {label}: {' '.join(question.split())}"
    if len(title) > 100:
        title = title[:97] + "..."
    with get_connection() as conn:
        conn.execute(
            "UPDATE chat_sessions SET title = ?, updated_at = ? WHERE id = ?",
            (title, utc_now(), session_id),
        )


def _insert_synthetic_feedback(
    user_id: int,
    chat_log_id: int,
    judge: dict[str, Any] | None,
    refused: bool,
) -> None:
    feedback_type, rating, comment = _feedback_from_judge(judge, refused)
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO feedback (user_id, chat_log_id, feedback_type, rating, comment, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (user_id, chat_log_id, feedback_type, rating, comment, utc_now()),
        )


def _compact_result(
    run_id: str,
    synthetic_user: dict[str, Any],
    question_source: str,
    source_id: str,
    expected_documents: str,
    result: dict[str, Any],
) -> dict[str, Any]:
    judge = result.get("judge") if isinstance(result.get("judge"), dict) else {}
    citations = result.get("citations") or []
    return {
        "run_id": run_id,
        "synthetic_user_email": synthetic_user["email"],
        "question_source": question_source,
        "source_id": source_id,
        "question": result["question"],
        "expected_documents": expected_documents,
        "chat_log_id": result["chat_log_id"],
        "session_id": result["session_id"],
        "chunking_strategy": result["chunking_strategy"],
        "retrieval_score": result.get("retrieval_score"),
        "retrieved_count": result.get("retrieved_count"),
        "refused": result.get("refused"),
        "judge_verdict": judge.get("verdict"),
        "overall_score": judge.get("overall_score"),
        "groundedness_score": judge.get("groundedness_score"),
        "citation_score": judge.get("citation_score"),
        "relevance_score": judge.get("relevance_score"),
        "completeness_score": judge.get("completeness_score"),
        "clarity_score": judge.get("clarity_score"),
        "judge_notes": judge.get("notes"),
        "top_citation_file": citations[0].get("file_name") if citations else "",
        "top_citation_section": citations[0].get("section_title") if citations else "",
        "answer_preview": result["answer"][:260],
    }


def _write_outputs(output_dir: Path, run_id: str, compact: list[dict[str, Any]], full: list[dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{run_id}_summary.csv"
    json_path = output_dir / f"{run_id}_full.json"

    if compact:
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(compact[0].keys()))
            writer.writeheader()
            writer.writerows(compact)

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "run_id": run_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "warning": "Synthetic simulation only. Do not report this as real user-testing feedback.",
                "summary_csv": str(csv_path),
                "items": full,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"Summary CSV: {csv_path}")
    print(f"Full JSON: {json_path}")


def run_simulation(args: argparse.Namespace) -> None:
    init_db()
    seed_demo_user()

    users = _get_testing_users(args.users)
    eval_questions = _load_eval_questions(args.eval_questions)
    simple_questions = SIMPLE_USER_QUESTIONS[: args.simple_questions]
    if len(simple_questions) < args.simple_questions:
        raise RuntimeError(
            f"Only found {len(simple_questions)} simple questions; requested {args.simple_questions}."
        )
    if len(eval_questions) < args.eval_questions:
        raise RuntimeError(
            f"Only found {len(eval_questions)} answerable evaluation questions; requested {args.eval_questions}."
        )

    run_id = f"synthetic_user_test_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    compact_rows: list[dict[str, Any]] = []
    full_rows: list[dict[str, Any]] = []

    cases = [
        {
            "question_source": "simple_user_question",
            "source_id": f"SIMPLE-{index + 1:02d}",
            "question": question,
            "expected_documents": "",
            "session_label": "simple",
        }
        for index, question in enumerate(simple_questions)
    ]
    cases.extend(
        {
            "question_source": "evaluation_bank",
            "source_id": row.get("id", f"EVAL-{index + 1:02d}"),
            "question": row["question"],
            "expected_documents": row.get("expected_documents", ""),
            "session_label": "eval",
        }
        for index, row in enumerate(eval_questions)
    )

    print(
        "Synthetic run design: "
        f"{len(users)} users x {len(cases)} questions each "
        f"({len(simple_questions)} simple + {len(eval_questions)} evaluation-bank) "
        f"= {len(users) * len(cases)} total queries."
    )

    for user in users:
        for case in cases:
            print(f"{user['email']} | {case['question_source']} | {case['question']}")
            result = answer_question(
                user_id=int(user["id"]),
                question=case["question"],
                strategy=args.strategy,
                top_k=args.top_k,
                run_judge=True,
                metadata_rerank_enabled=args.metadata_rerank_enabled,
            )
            _label_session(result["session_id"], case["session_label"], case["question"])
            _insert_synthetic_feedback(
                user_id=int(user["id"]),
                chat_log_id=int(result["chat_log_id"]),
                judge=result.get("judge"),
                refused=bool(result.get("refused")),
            )
            compact_rows.append(
                _compact_result(
                    run_id=run_id,
                    synthetic_user=user,
                    question_source=case["question_source"],
                    source_id=case["source_id"],
                    expected_documents=case["expected_documents"],
                    result=result,
                )
            )
            full_rows.append(
                {
                    "synthetic_user": user,
                    "case": case,
                    "result": result,
                }
            )

    _write_outputs(args.output_dir, run_id, compact_rows, full_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run synthetic user testing against the RAG pipeline.")
    parser.add_argument("--users", type=int, default=10, help="Number of seeded tester accounts to use.")
    parser.add_argument("--simple-questions", type=int, default=10, help="Number of simple user-style questions per tester.")
    parser.add_argument("--eval-questions", type=int, default=10, help="Number of evaluation-bank questions to run.")
    parser.add_argument("--strategy", default="structure_aware", choices=["fixed_size", "structure_aware"])
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--metadata-rerank-enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_evaluation_dir() / "synthetic_user_tests",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_simulation(parse_args())
