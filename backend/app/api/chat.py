from __future__ import annotations

import json
from collections import Counter, defaultdict
from secrets import compare_digest

from fastapi import APIRouter, Depends, Header, HTTPException, status

from app.api.deps import get_current_user, require_admin
from app.core.config import settings
from app.core.database import get_connection, get_retrieval_settings, utc_now
from app.models import VALID_STRATEGIES
from app.schemas import (
    ChatRequest,
    CitationClickRequest,
    DemoFeedbackRequest,
    FeedbackRequest,
    HumanAuditRequest,
    InsightResolutionRequest,
)
from app.services.openai_client import OpenAIService
from app.services.rag import answer_question
from app.services.supabase_audit import mirror_audit_event


router = APIRouter()

INSIGHT_RESOLUTION_TYPES = {"refused", "low_llm", "low_human", "missing_document"}


def _decode_json(value: str | None) -> object | None:
    if value is None:
        return None
    return json.loads(value)


def _decode_chat_row(row: dict) -> dict:
    item = dict(row)
    item["citations"] = _decode_json(item.pop("citations_json"))
    item["evidence"] = _decode_json(item.pop("evidence_json"))
    item["judge"] = _decode_json(item.pop("judge_json"))
    return item


CATEGORY_RULES: list[tuple[str, tuple[str, ...]]] = [
    (
        "Out-of-scope and forward-looking requests",
        (
            "products will kraken launch",
            "launch in 2027",
            "future product",
            "future roadmap",
            "upcoming product",
        ),
    ),
    (
        "Investment, tax and advice boundaries",
        (
            "investment advice",
            "legal advice",
            "financial advice",
            "tax advice",
            "should i invest",
            "should i buy",
            "what crypto currency should i buy",
            "what cryptocurrency should i buy",
            "cash out tomorrow",
            "how much money",
            "make from crypto",
            "make with crypto",
            "profit",
            "returns",
            "how much tax",
            "tax i will owe",
            "tax will i owe",
            "sell bitcoin",
        ),
    ),
    (
        "Service availability and jurisdictions",
        (
            "available in",
            "serve clients",
            "prohibited region",
            "restricted region",
            "country",
            "countries",
            "jurisdiction",
            "iran",
            "united states",
            "usa",
            " us",
            "united kingdom",
            " uk",
            "canada",
            "eea",
            "europe",
            "differ between",
            "differ by country",
            "between countries",
        ),
    ),
    (
        "Regulatory licensing and reporting",
        (
            "licensed",
            "regulated",
            "regulator",
            "regulatory reporting",
            "money transmitter",
            "money transmission",
            "legal disclosure",
            "reporting",
            "report my trading",
            "trading activity",
            "hmrc",
            "tax reporting",
            "compliance",
        ),
    ),
    (
        "Marketing and communications",
        (
            "marketing",
            "promotion",
            "promotional",
            "advertising",
            "advertise",
            "financial promotion",
            "communications",
        ),
    ),
    (
        "Operational availability and support",
        (
            "site unavailable",
            "site is unavailable",
            "site not reachable",
            "not reachable",
            "goes offline",
            "kraken goes offline",
            "downtime",
            "down time",
            "outage",
            "support",
            "help center",
            "help centre",
        ),
    ),
    (
        "Legal entities and contact details",
        (
            "official address",
            "registered address",
            "address",
            "office",
            "located",
            "payward",
            "legal department",
            "contact",
        ),
    ),
    (
        "Payments, cards and funding",
        (
            "credit card",
            "debit card",
            "card information",
            "card details",
            "payment card",
            "3d secure",
            "pci",
            "funding",
            "funds",
            "cash",
            "fiat",
        ),
    ),
    (
        "Account restrictions and holds",
        (
            "temporary hold",
            "hold of funds",
            "account hold",
            "restrict account",
            "account access",
            "disable funding",
            "suspend",
            "suspension",
            "restricted from using",
        ),
    ),
    (
        "Client eligibility and onboarding",
        (
            "types of clients",
            "client type",
            "prospective client",
            "eligible client",
            "eligibility",
            "onboarding",
            "due diligence",
            "verification level",
            "create an account",
            "signing up",
        ),
    ),
    (
        "Commercial exchange services and liquidity",
        (
            "exchange services",
            "exchange limits",
            "liquidity",
            "large amount",
            "commercial policy",
            "client accepts",
            "price methodology",
            "firm price",
        ),
    ),
    (
        "Citation and source access",
        (
            "citation",
            "cite",
            "cited",
            "source is not reachable",
            "document not reachable",
            "open the document",
        ),
    ),
    (
        "Transfers and withdrawals",
        ("transfer", "withdraw", "deposit", "kraken pay", "wrong address", "wrong network", "network"),
    ),
    (
        "Risk disclosures",
        (
            "micar",
            "risk disclosure",
            "financial risk",
            "risk-free",
            "white paper",
            "consensus",
            "collateral",
            "what are the risks",
            "risks with crypto",
            "crypto trading risk",
            "risk does kraken disclose",
            "risks should customers know",
            "crypto-assets",
            "human error",
        ),
    ),
    (
        "Custody and safeguarding",
        (
            "custody",
            "safeguard",
            "client asset",
            "segregat",
            "wind-down",
            "business continuity",
            "store funds",
            "store and protect",
            "keep my btc safe",
            "crypto-assets safe",
            "becomes insolvent",
            "insolvent",
        ),
    ),
    (
        "Trading and order execution",
        ("order", "execution", "trade", "trading venue", "specific instruction", "market order", "limit order"),
    ),
    (
        "Privacy and data protection",
        (
            "privacy",
            "personal data",
            "gdpr",
            "retention",
            "international transfer",
            "disclose personal",
            "share my data",
            "third parties",
            "ip address",
            "personal information",
            "data kept safe",
        ),
    ),
    (
        "Terms, complaints and e-money",
        ("terms", "complaint", "e-money", "unauthorized", "unauthorised", "terms of service"),
    ),
    (
        "Conflicts and vendors",
        ("conflict", "vendor", "procurement", "group compan", "shareholder", "management influencing"),
    ),
    (
        "Security and verification",
        (
            "security",
            "secure",
            "credential",
            "breach",
            "account",
            "verification",
            "fraud",
            "watermarked",
            "screenshot",
            "screenshotted",
            "document pictures",
        ),
    ),
    (
        "Fees, pricing and commercial policy",
        ("fee", "price", "pricing", "commercial", "client type", "provisional condition"),
    ),
    (
        "Web and product information",
        (
            "what is kraken",
            "kraken pro",
            "xstocks",
            "xstock",
            "buy crypto",
            "how to buy crypto",
            "website",
            "mobile app",
            "api",
            "staking",
            "margin",
            "futures",
            "support",
        ),
    ),
]


def _safe_decode_json(value: str | None) -> object | None:
    if not value:
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return None


def _categorise_question(question: str) -> str:
    normalized = question.lower()
    for label, terms in CATEGORY_RULES:
        if any(term in normalized for term in terms):
            return label
    return "Other"


def _judge_score(judge: object | None) -> int | None:
    if not isinstance(judge, dict):
        return None
    try:
        return int(judge.get("overall_score"))
    except (TypeError, ValueError):
        return None


def _judge_string(judge: object | None, key: str) -> str | None:
    if not isinstance(judge, dict):
        return None
    value = judge.get(key)
    return value if isinstance(value, str) else None


def _answer_preview(answer: str, limit: int = 180) -> str:
    collapsed = " ".join(answer.split())
    return collapsed if len(collapsed) <= limit else f"{collapsed[:limit].rstrip()}..."


def _human_average(row: dict) -> float:
    scores = [
        row["groundedness_score"],
        row["citation_score"],
        row["relevance_score"],
        row["completeness_score"],
        row["clarity_score"],
    ]
    return sum(scores) / len(scores)


def _insight_question_item(row: dict, judge: object | None, category: str) -> dict:
    return {
        "chat_log_id": row["id"],
        "question": row["question"],
        "answer_preview": _answer_preview(row["answer"]),
        "category": category,
        "strategy": row["chunking_strategy"],
        "created_at": row["created_at"],
        "user_email": row["email"],
        "llm_score": _judge_score(judge),
        "llm_verdict": _judge_string(judge, "verdict"),
        "llm_notes": _judge_string(judge, "notes"),
    }


def _resolution_row(resolutions: dict[tuple[str, str], dict], issue_type: str, item_key: object) -> dict | None:
    return resolutions.get((issue_type, str(item_key)))


def _apply_resolution(item: dict, resolutions: dict[tuple[str, str], dict], issue_type: str, item_key: object) -> dict:
    key = str(item_key)
    resolution = _resolution_row(resolutions, issue_type, key)
    item["issue_type"] = issue_type
    item["item_key"] = key
    item["resolved"] = resolution is not None
    item["resolved_at"] = resolution["resolved_at"] if resolution else None
    item["resolved_by_email"] = resolution["resolved_by_email"] if resolution else None
    item["resolution_note"] = resolution["note"] if resolution else None
    return item


def _is_active_issue(
    resolutions: dict[tuple[str, str], dict],
    issue_type: str,
    item_key: object,
    include_resolved: bool,
) -> bool:
    return include_resolved or _resolution_row(resolutions, issue_type, item_key) is None


def _suggest_insight_action(category: str, signals: set[str], representative_question: str | None) -> dict:
    question = (representative_question or "").lower()
    category_lower = category.lower()

    if category == "Out-of-scope and forward-looking requests":
        return {
            "action_type": "correct_refusal",
            "suggested_action": "Review refusal boundary",
        }

    if category == "Investment, tax and advice boundaries":
        return {
            "action_type": "correct_refusal",
            "suggested_action": "Confirm advice boundary",
        }

    if any(phrase in question for phrase in ["internal customer list", "account balances", "private user"]):
        return {
            "action_type": "correct_refusal",
            "suggested_action": "Correct refusal",
        }

    if {"wrong document", "citation mismatch"} & signals:
        return {
            "action_type": "retrieval_review",
            "suggested_action": "Check retrieval/citations",
        }

    if "unnecessary refusal" in signals:
        return {
            "action_type": "retrieval_review",
            "suggested_action": "Reduce false refusal",
        }

    if "missing detail" in signals:
        return {
            "action_type": "answer_review",
            "suggested_action": "Improve answer detail",
        }

    if "refused" in signals and "low llm judge score" in {signal.lower() for signal in signals}:
        return {
            "action_type": "corpus_gap",
            "suggested_action": "Check corpus coverage",
        }

    if "refused" in signals:
        return {
            "action_type": "corpus_gap",
            "suggested_action": "Check source coverage",
        }

    if "low llm judge score" in {signal.lower() for signal in signals}:
        return {
            "action_type": "judge_review",
            "suggested_action": "Review answer/judge score",
        }

    if "legal entities" in category_lower or "contact details" in category_lower:
        return {
            "action_type": "corpus_gap",
            "suggested_action": "Verify source coverage",
        }

    return {
        "action_type": "answer_review",
        "suggested_action": "Review flagged pattern",
    }


def _generate_llm_insights(insight_payload: dict) -> dict:
    if not settings.openai_api_key:
        return {
            "available": False,
            "headline": "LLM insight generation is unavailable.",
            "executive_summary": "Set OPENAI_API_KEY to generate a written management summary.",
            "key_takeaways": [],
            "suggested_actions": [],
            "watchlist": [],
            "error": "OPENAI_API_KEY is not configured.",
        }

    compact_payload = {
        "summary": insight_payload["summary"],
        "top_question_categories": insight_payload["top_question_categories"][:10],
        "unanswered_or_refused_questions": [
            {
                "question": item["question"],
                "category": item["category"],
                "llm_score": item.get("llm_score"),
                "llm_verdict": item.get("llm_verdict"),
            }
            for item in insight_payload["unanswered_or_refused_questions"][:8]
        ],
        "low_llm_judge_score_questions": [
            {
                "question": item["question"],
                "category": item["category"],
                "llm_score": item.get("llm_score"),
                "llm_verdict": item.get("llm_verdict"),
                "llm_notes": item.get("llm_notes"),
            }
            for item in insight_payload["low_llm_judge_score_questions"][:8]
        ],
        "low_human_audit_score_questions": [
            {
                "question": item["question"],
                "category": item["category"],
                "human_average_score": item.get("human_average_score"),
                "overall_verdict": item.get("overall_verdict"),
                "issue_tags": item.get("issue_tags", []),
                "comment": item.get("comment"),
            }
            for item in insight_payload["low_human_audit_score_questions"][:8]
        ],
        "improvement_queue": insight_payload["common_missing_documents"][:10],
    }

    prompt = f"""
You are analysing an admin insights dashboard for a RAG chatbot over Kraken policy and web sources.
Use only the dashboard data below. Do not invent counts, documents, users, or scores.
Treat improvement_queue items as review candidates, not automatic proof that a document is missing.
Use each item's suggested_action and action_type to distinguish corpus gaps, retrieval problems, answer-quality issues, judge calibration issues, and correct refusals.

Return valid JSON only using this schema:
{{
  "available": true,
  "headline": "one concise sentence",
  "executive_summary": "2-3 sentence management summary",
  "key_takeaways": ["3-5 concise bullets"],
  "suggested_actions": ["3-5 concrete product/corpus/retrieval actions"],
  "watchlist": ["2-4 risks or follow-up checks"]
}}

Dashboard data:
{json.dumps(compact_payload, ensure_ascii=True)}
""".strip()

    try:
        result = OpenAIService().chat_with_metadata(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You write concise product analytics summaries for an admin dashboard. "
                        "Be practical, specific, and faithful to the supplied data."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            response_format={"type": "json_object"},
            max_tokens=900,
        )
        parsed = json.loads(result["content"])
        if not isinstance(parsed, dict):
            raise ValueError("LLM response was not a JSON object")
        parsed["model"] = result.get("model")
        parsed["generated_at"] = utc_now()
        parsed["usage"] = {
            "prompt_tokens": result.get("prompt_tokens"),
            "completion_tokens": result.get("completion_tokens"),
            "total_tokens": result.get("total_tokens"),
        }
        return parsed
    except Exception as exc:
        return {
            "available": False,
            "headline": "LLM insight generation failed.",
            "executive_summary": "The dashboard metrics loaded, but the written LLM summary could not be generated.",
            "key_takeaways": [],
            "suggested_actions": [],
            "watchlist": [],
            "error": str(exc),
        }


@router.post("/query")
def query(payload: ChatRequest, user: dict = Depends(get_current_user)) -> dict:
    retrieval_settings = get_retrieval_settings()
    strategy = retrieval_settings["chunking_strategy"]
    top_k = retrieval_settings["top_k"]
    run_judge = retrieval_settings["run_judge"]
    metadata_rerank_enabled = retrieval_settings["metadata_rerank_enabled"]
    if strategy not in VALID_STRATEGIES:
        raise HTTPException(status_code=400, detail="Invalid chunking strategy")
    if payload.session_id is not None:
        with get_connection() as conn:
            session = conn.execute(
                "SELECT id FROM chat_sessions WHERE id = ? AND user_id = ?",
                (payload.session_id, user["id"]),
            ).fetchone()
        if session is None:
            raise HTTPException(status_code=404, detail="Chat session not found")
    try:
        return answer_question(
            user_id=user["id"],
            question=payload.question,
            strategy=strategy,
            top_k=top_k,
            run_judge=run_judge,
            metadata_rerank_enabled=metadata_rerank_enabled,
            session_id=payload.session_id,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/history")
def history(user: dict = Depends(get_current_user)) -> dict:
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT s.id, s.title, s.created_at, s.updated_at,
                   COUNT(l.id) AS message_count,
                   (
                       SELECT question
                       FROM chat_logs latest
                       WHERE latest.session_id = s.id
                       ORDER BY latest.id DESC
                       LIMIT 1
                   ) AS latest_question
            FROM chat_sessions s
            JOIN chat_logs l ON l.session_id = s.id
            WHERE s.user_id = ?
            GROUP BY s.id
            ORDER BY s.updated_at DESC
            LIMIT 30
            """,
            (user["id"],),
        ).fetchall()
    return {"items": [dict(row) for row in rows]}


@router.get("/history/{session_id}")
def history_session(session_id: int, user: dict = Depends(get_current_user)) -> dict:
    with get_connection() as conn:
        session = conn.execute(
            """
            SELECT id, title, created_at, updated_at
            FROM chat_sessions
            WHERE id = ? AND user_id = ?
            """,
            (session_id, user["id"]),
        ).fetchone()
        if session is None:
            raise HTTPException(status_code=404, detail="Chat session not found")
        rows = conn.execute(
            """
            SELECT id, session_id, question, answer, chunking_strategy, citations_json, evidence_json,
                   judge_json, retrieval_score, refused, top_k, run_judge, retrieved_count,
                   corpus_version, index_version, human_audit_prompted, human_audit_prompt_reason,
                   embedding_model, generation_model, judge_model,
                   embedding_latency_ms, vector_query_latency_ms,
                   generation_latency_ms, judge_latency_ms, total_latency_ms,
                   created_at
            FROM chat_logs
            WHERE session_id = ? AND user_id = ?
            ORDER BY id ASC
            """,
            (session_id, user["id"]),
        ).fetchall()
    return {
        "session": dict(session),
        "items": [_decode_chat_row(dict(row)) for row in rows],
    }


@router.delete("/history")
def clear_history(user: dict = Depends(get_current_user)) -> dict:
    with get_connection() as conn:
        conn.execute(
            """
            DELETE FROM feedback
            WHERE chat_log_id IN (
                SELECT id FROM chat_logs WHERE user_id = ?
            )
            """,
            (user["id"],),
        )
        conn.execute(
            """
            DELETE FROM human_audits
            WHERE chat_log_id IN (
                SELECT id FROM chat_logs WHERE user_id = ?
            )
            """,
            (user["id"],),
        )
        conn.execute(
            """
            DELETE FROM citation_events
            WHERE chat_log_id IN (
                SELECT id FROM chat_logs WHERE user_id = ?
            )
            """,
            (user["id"],),
        )
        cur = conn.execute("DELETE FROM chat_logs WHERE user_id = ?", (user["id"],))
        conn.execute("DELETE FROM chat_sessions WHERE user_id = ?", (user["id"],))
    return {"ok": True, "deleted": cur.rowcount}


@router.delete("/history/{chat_log_id}")
def delete_history_item(chat_log_id: int, user: dict = Depends(get_current_user)) -> dict:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT id FROM chat_sessions WHERE id = ? AND user_id = ?",
            (chat_log_id, user["id"]),
        ).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="Chat not found")
        conn.execute(
            """
            DELETE FROM feedback
            WHERE chat_log_id IN (
                SELECT id FROM chat_logs WHERE session_id = ? AND user_id = ?
            )
            """,
            (chat_log_id, user["id"]),
        )
        conn.execute(
            """
            DELETE FROM human_audits
            WHERE chat_log_id IN (
                SELECT id FROM chat_logs WHERE session_id = ? AND user_id = ?
            )
            """,
            (chat_log_id, user["id"]),
        )
        conn.execute(
            """
            DELETE FROM citation_events
            WHERE chat_log_id IN (
                SELECT id FROM chat_logs WHERE session_id = ? AND user_id = ?
            )
            """,
            (chat_log_id, user["id"]),
        )
        cur = conn.execute(
            "DELETE FROM chat_logs WHERE session_id = ? AND user_id = ?",
            (chat_log_id, user["id"]),
        )
        conn.execute("DELETE FROM chat_sessions WHERE id = ? AND user_id = ?", (chat_log_id, user["id"]))
    return {"ok": True, "deleted": cur.rowcount}


@router.post("/feedback")
def feedback(payload: FeedbackRequest, user: dict = Depends(get_current_user)) -> dict:
    created_at = utc_now()
    with get_connection() as conn:
        cur = conn.execute(
            """
            INSERT INTO feedback (user_id, chat_log_id, feedback_type, rating, comment, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                user["id"],
                payload.chat_log_id,
                payload.feedback_type,
                payload.rating,
                payload.comment,
                created_at,
            ),
        )
    feedback_id = cur.lastrowid
    mirror_audit_event(
        "feedback",
        {
            "feedback_id": feedback_id,
            "chat_log_id": payload.chat_log_id,
            "feedback_type": payload.feedback_type,
            "rating": payload.rating,
            "comment": payload.comment,
        },
        local_id=feedback_id,
        user_id=user["id"],
        chat_log_id=payload.chat_log_id,
        created_at=created_at,
    )
    return {"ok": True, "feedback_id": feedback_id}


@router.post("/demo-feedback")
def demo_feedback(payload: DemoFeedbackRequest, user: dict = Depends(get_current_user)) -> dict:
    citations_useful = payload.citations_useful.strip().lower()
    if citations_useful not in {"yes", "somewhat", "no"}:
        raise HTTPException(status_code=400, detail="Invalid citation usefulness value")
    would_use_at_work = payload.would_use_at_work.strip().lower()
    if would_use_at_work not in {"yes", "maybe", "no"}:
        raise HTTPException(status_code=400, detail="Invalid workplace use value")

    created_at = utc_now()
    with get_connection() as conn:
        cur = conn.execute(
            """
            INSERT INTO demo_feedback (
                user_id, role_department, ease_score, helpfulness_score, trustworthiness_score,
                citations_useful, incorrect_or_misleading, incorrect_notes, missing_expectation,
                improvement, would_use_at_work, final_comments, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user["id"],
                payload.role_department,
                payload.ease_score,
                payload.helpfulness_score,
                payload.trustworthiness_score,
                citations_useful,
                1 if payload.incorrect_or_misleading else 0,
                payload.incorrect_notes,
                payload.missing_expectation,
                payload.improvement,
                would_use_at_work,
                payload.final_comments,
                created_at,
            ),
        )
    demo_feedback_id = cur.lastrowid
    mirror_audit_event(
        "demo_feedback",
        {
            "demo_feedback_id": demo_feedback_id,
            "role_department": payload.role_department,
            "ease_score": payload.ease_score,
            "helpfulness_score": payload.helpfulness_score,
            "trustworthiness_score": payload.trustworthiness_score,
            "citations_useful": citations_useful,
            "incorrect_or_misleading": payload.incorrect_or_misleading,
            "incorrect_notes": payload.incorrect_notes,
            "missing_expectation": payload.missing_expectation,
            "improvement": payload.improvement,
            "would_use_at_work": would_use_at_work,
            "final_comments": payload.final_comments,
        },
        local_id=demo_feedback_id,
        user_id=user["id"],
        created_at=created_at,
    )
    return {"ok": True, "demo_feedback_id": demo_feedback_id}


@router.post("/human-audit")
def human_audit(payload: HumanAuditRequest, user: dict = Depends(get_current_user)) -> dict:
    verdict = payload.overall_verdict.strip().lower()
    if verdict not in {"pass", "partial", "fail", "unable_to_judge"}:
        raise HTTPException(status_code=400, detail="Invalid overall verdict")
    confidence = payload.reviewer_confidence.strip().lower()
    if confidence not in {"low", "medium", "high"}:
        raise HTTPException(status_code=400, detail="Invalid reviewer confidence")

    allowed_tags = {
        "wrong_document",
        "citation_mismatch",
        "missing_detail",
        "too_broad",
        "too_vague",
        "unsupported_claim",
        "unnecessary_refusal",
        "should_refuse",
        "good_answer",
        "unable_to_judge",
    }
    issue_tags = [tag for tag in payload.issue_tags if tag in allowed_tags]
    audit_payload = {
        "chat_log_id": payload.chat_log_id,
        "groundedness_score": payload.groundedness_score,
        "citation_score": payload.citation_score,
        "relevance_score": payload.relevance_score,
        "completeness_score": payload.completeness_score,
        "clarity_score": payload.clarity_score,
        "overall_verdict": verdict,
        "issue_tags": issue_tags,
        "comment": payload.comment,
        "reviewer_confidence": confidence,
    }

    with get_connection() as conn:
        chat_log = conn.execute(
            """
            SELECT id, judge_json
            FROM chat_logs
            WHERE id = ? AND user_id = ?
            """,
            (payload.chat_log_id, user["id"]),
        ).fetchone()
        if chat_log is None:
            raise HTTPException(status_code=404, detail="Chat log not found")
        judge = _decode_json(chat_log["judge_json"])
        llm_overall_score = None
        if isinstance(judge, dict):
            try:
                llm_overall_score = int(judge.get("overall_score"))
            except (TypeError, ValueError):
                llm_overall_score = None
        now = utc_now()
        conn.execute(
            """
            INSERT INTO human_audits (
                user_id, chat_log_id, groundedness_score, citation_score, relevance_score,
                completeness_score, clarity_score, overall_verdict, issue_tags_json,
                comment, reviewer_confidence, llm_judge_json_snapshot, llm_overall_score,
                created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(user_id, chat_log_id) DO UPDATE SET
                groundedness_score = excluded.groundedness_score,
                citation_score = excluded.citation_score,
                relevance_score = excluded.relevance_score,
                completeness_score = excluded.completeness_score,
                clarity_score = excluded.clarity_score,
                overall_verdict = excluded.overall_verdict,
                issue_tags_json = excluded.issue_tags_json,
                comment = excluded.comment,
                reviewer_confidence = excluded.reviewer_confidence,
                llm_judge_json_snapshot = excluded.llm_judge_json_snapshot,
                llm_overall_score = excluded.llm_overall_score,
                updated_at = excluded.updated_at
            """,
            (
                user["id"],
                payload.chat_log_id,
                payload.groundedness_score,
                payload.citation_score,
                payload.relevance_score,
                payload.completeness_score,
                payload.clarity_score,
                verdict,
                json.dumps(issue_tags),
                payload.comment,
                confidence,
                chat_log["judge_json"],
                llm_overall_score,
                now,
                now,
            ),
        )
        row = conn.execute(
            "SELECT id FROM human_audits WHERE user_id = ? AND chat_log_id = ?",
            (user["id"], payload.chat_log_id),
        ).fetchone()
    human_audit_id = row["id"]
    audit_payload["human_audit_id"] = human_audit_id
    audit_payload["llm_overall_score"] = llm_overall_score
    audit_payload["llm_judge"] = judge
    mirror_audit_event(
        "human_audit",
        audit_payload,
        local_id=human_audit_id,
        user_id=user["id"],
        chat_log_id=payload.chat_log_id,
        created_at=now,
    )
    return {"ok": True, "human_audit_id": human_audit_id}


@router.post("/citation-click")
def citation_click(payload: CitationClickRequest, user: dict = Depends(get_current_user)) -> dict:
    created_at = utc_now()
    with get_connection() as conn:
        if payload.chat_log_id is not None:
            row = conn.execute(
                "SELECT id FROM chat_logs WHERE id = ? AND user_id = ?",
                (payload.chat_log_id, user["id"]),
            ).fetchone()
            if row is None:
                raise HTTPException(status_code=404, detail="Chat log not found")
        cur = conn.execute(
            """
            INSERT INTO citation_events (
                user_id, chat_log_id, citation_label, chunk_id, file_name,
                page_start, page_end, source_strategy, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user["id"],
                payload.chat_log_id,
                payload.citation_label,
                payload.chunk_id,
                payload.file_name,
                payload.page_start,
                payload.page_end,
                payload.source_strategy,
                created_at,
            ),
        )
    citation_event_id = cur.lastrowid
    mirror_audit_event(
        "citation_click",
        {
            "citation_event_id": citation_event_id,
            "chat_log_id": payload.chat_log_id,
            "citation_label": payload.citation_label,
            "chunk_id": payload.chunk_id,
            "file_name": payload.file_name,
            "page_start": payload.page_start,
            "page_end": payload.page_end,
            "source_strategy": payload.source_strategy,
        },
        local_id=citation_event_id,
        user_id=user["id"],
        chat_log_id=payload.chat_log_id,
        created_at=created_at,
    )
    return {"ok": True, "citation_event_id": citation_event_id}


@router.get("/admin-metrics")
def admin_metrics(_: dict = Depends(require_admin)) -> dict:
    with get_connection() as conn:
        row = conn.execute(
            """
            SELECT
                COUNT(*) AS total_chats,
                COUNT(DISTINCT user_id) AS active_users,
                AVG(retrieval_score) AS avg_retrieval_score,
                SUM(CASE WHEN refused = 1 THEN 1 ELSE 0 END) AS refusals,
                SUM(CASE WHEN human_audit_prompted = 1 THEN 1 ELSE 0 END) AS human_audit_prompts
            FROM chat_logs
            """
        ).fetchone()
        audit_row = conn.execute(
            """
            SELECT
                COUNT(*) AS total_human_audits,
                AVG((groundedness_score + citation_score + relevance_score + completeness_score + clarity_score) / 5.0) AS avg_human_score,
                AVG(llm_overall_score) AS avg_llm_score_snapshot,
                SUM(CASE WHEN overall_verdict = 'pass' THEN 1 ELSE 0 END) AS human_pass,
                SUM(CASE WHEN overall_verdict = 'partial' THEN 1 ELSE 0 END) AS human_partial,
                SUM(CASE WHEN overall_verdict = 'fail' THEN 1 ELSE 0 END) AS human_fail,
                SUM(CASE WHEN overall_verdict = 'unable_to_judge' THEN 1 ELSE 0 END) AS human_unable_to_judge
            FROM human_audits
            """
        ).fetchone()
        citation_row = conn.execute(
            "SELECT COUNT(*) AS citation_clicks FROM citation_events"
        ).fetchone()
    return {
        "total_chats": row["total_chats"] or 0,
        "active_users": row["active_users"] or 0,
        "avg_retrieval_score": row["avg_retrieval_score"],
        "refusals": row["refusals"] or 0,
        "human_audit_prompts": row["human_audit_prompts"] or 0,
        "total_human_audits": audit_row["total_human_audits"] or 0,
        "avg_human_score": audit_row["avg_human_score"],
        "avg_llm_score_snapshot": audit_row["avg_llm_score_snapshot"],
        "human_verdicts": {
            "pass": audit_row["human_pass"] or 0,
            "partial": audit_row["human_partial"] or 0,
            "fail": audit_row["human_fail"] or 0,
            "unable_to_judge": audit_row["human_unable_to_judge"] or 0,
        },
        "citation_clicks": citation_row["citation_clicks"] or 0,
    }


@router.get("/admin-insights")
def admin_insights(include_resolved: bool = False, _: dict = Depends(require_admin)) -> dict:
    with get_connection() as conn:
        log_rows = [
            dict(row)
            for row in conn.execute(
                """
                SELECT l.id, l.question, l.answer, l.chunking_strategy, l.citations_json,
                       l.judge_json, l.retrieval_score, l.refused, l.created_at,
                       u.email
                FROM chat_logs l
                JOIN users u ON u.id = l.user_id
                ORDER BY l.created_at DESC, l.id DESC
                """
            ).fetchall()
        ]
        audit_rows = [
            dict(row)
            for row in conn.execute(
                """
                SELECT h.id AS audit_id, h.chat_log_id, h.groundedness_score, h.citation_score,
                       h.relevance_score, h.completeness_score, h.clarity_score,
                       h.overall_verdict, h.issue_tags_json, h.comment, h.reviewer_confidence,
                       h.llm_overall_score, h.updated_at, l.question, l.answer,
                       l.chunking_strategy, u.email
                FROM human_audits h
                JOIN chat_logs l ON l.id = h.chat_log_id
                JOIN users u ON u.id = h.user_id
                ORDER BY h.updated_at DESC, h.id DESC
                """
            ).fetchall()
        ]
        resolution_rows = [
            dict(row)
            for row in conn.execute(
                """
                SELECT r.issue_type, r.item_key, r.note, r.resolved_at, u.email AS resolved_by_email
                FROM insight_resolutions r
                JOIN users u ON u.id = r.resolved_by
                WHERE r.active = 1
                """
            ).fetchall()
        ]

    resolutions = {
        (row["issue_type"], row["item_key"]): row
        for row in resolution_rows
        if row["issue_type"] in INSIGHT_RESOLUTION_TYPES
    }
    category_stats: dict[str, dict] = defaultdict(
        lambda: {
            "category": "",
            "total": 0,
            "refusals": 0,
            "low_llm": 0,
            "low_human": 0,
            "avg_llm_score": None,
            "avg_human_score": None,
            "_llm_scores": [],
            "_human_scores": [],
        }
    )
    refused_questions = []
    low_llm_questions = []
    gap_counter: Counter[str] = Counter()
    gap_examples: dict[str, str] = {}
    gap_signals: dict[str, set[str]] = defaultdict(set)

    for row in log_rows:
        category = _categorise_question(row["question"])
        judge = _safe_decode_json(row["judge_json"])
        llm_score = _judge_score(judge)
        citations = _safe_decode_json(row["citations_json"])
        no_citations = not citations
        refused = bool(row["refused"])
        refused_active = refused and _is_active_issue(resolutions, "refused", row["id"], include_resolved)
        low_llm_active = (
            llm_score is not None
            and llm_score <= 3
            and _is_active_issue(resolutions, "low_llm", row["id"], include_resolved)
        )

        stats = category_stats[category]
        stats["category"] = category
        stats["total"] += 1
        if refused_active:
            stats["refusals"] += 1
        if llm_score is not None:
            stats["_llm_scores"].append(llm_score)
        if low_llm_active:
            stats["low_llm"] += 1

        if refused and (include_resolved or refused_active):
            refused_questions.append(
                _apply_resolution(
                    _insight_question_item(row, judge, category),
                    resolutions,
                    "refused",
                    row["id"],
                )
            )
        if llm_score is not None and llm_score <= 3 and (include_resolved or low_llm_active):
            low_llm_questions.append(
                _apply_resolution(
                    _insight_question_item(row, judge, category),
                    resolutions,
                    "low_llm",
                    row["id"],
                )
            )

        signals = []
        if refused_active:
            signals.append("refused")
        if low_llm_active:
            signals.append("low LLM judge score")
        if no_citations:
            signals.append("no citations")
        if signals:
            gap_counter[category] += 1
            gap_examples.setdefault(category, row["question"])
            for signal in signals:
                gap_signals[category].add(signal)

    low_human_questions = []
    for row in audit_rows:
        category = _categorise_question(row["question"])
        average_score = _human_average(row)
        issue_tags = _safe_decode_json(row["issue_tags_json"])
        if not isinstance(issue_tags, list):
            issue_tags = []

        stats = category_stats[category]
        stats["category"] = category
        stats["_human_scores"].append(average_score)

        needs_attention = average_score < 4 or row["overall_verdict"] in {"partial", "fail", "unable_to_judge"}
        low_human_active = (
            needs_attention
            and _is_active_issue(resolutions, "low_human", row["audit_id"], include_resolved)
        )
        if low_human_active:
            stats["low_human"] += 1
        if needs_attention and (include_resolved or low_human_active):
            low_human_questions.append(
                _apply_resolution(
                    {
                        "audit_id": row["audit_id"],
                        "chat_log_id": row["chat_log_id"],
                        "question": row["question"],
                        "answer_preview": _answer_preview(row["answer"]),
                        "category": category,
                        "strategy": row["chunking_strategy"],
                        "created_at": row["updated_at"],
                        "user_email": row["email"],
                        "human_average_score": round(average_score, 2),
                        "overall_verdict": row["overall_verdict"],
                        "reviewer_confidence": row["reviewer_confidence"],
                        "comment": row["comment"],
                        "issue_tags": issue_tags,
                        "llm_overall_score": row["llm_overall_score"],
                    },
                    resolutions,
                    "low_human",
                    row["audit_id"],
                )
            )

        if (
            low_human_active
            and any(tag in issue_tags for tag in ("wrong_document", "missing_detail", "unnecessary_refusal", "citation_mismatch"))
        ):
            gap_counter[category] += 1
            gap_examples.setdefault(category, row["question"])
            for tag in issue_tags:
                gap_signals[category].add(tag.replace("_", " "))

    top_categories = []
    for stats in category_stats.values():
        llm_scores = stats.pop("_llm_scores")
        human_scores = stats.pop("_human_scores")
        stats["avg_llm_score"] = round(sum(llm_scores) / len(llm_scores), 2) if llm_scores else None
        stats["avg_human_score"] = round(sum(human_scores) / len(human_scores), 2) if human_scores else None
        top_categories.append(stats)

    top_categories.sort(key=lambda item: (-item["total"], item["category"]))
    low_llm_questions.sort(key=lambda item: ((item["llm_score"] or 99), item["created_at"]), reverse=False)
    low_human_questions.sort(key=lambda item: (item["human_average_score"], item["created_at"]), reverse=False)

    common_missing_documents = []
    for category, count in gap_counter.most_common():
        if not _is_active_issue(resolutions, "missing_document", category, include_resolved):
            continue
        signals = sorted(gap_signals[category])
        category_total = category_stats[category]["total"]
        warning_rate = round(count / category_total, 3) if category_total else None
        common_missing_documents.append(
            _apply_resolution(
                {
                    "topic": category,
                    "count": count,
                    "category_total": category_total,
                    "warning_rate": warning_rate,
                    "representative_question": gap_examples.get(category),
                    "signals": signals,
                    **_suggest_insight_action(category, set(signals), gap_examples.get(category)),
                },
                resolutions,
                "missing_document",
                category,
            )
        )
        if len(common_missing_documents) >= 10:
            break

    resolved_counts: Counter[str] = Counter(row["issue_type"] for row in resolutions.values())

    insight_payload = {
        "generated_at": utc_now(),
        "include_resolved": include_resolved,
        "resolved_counts": {
            "refused": resolved_counts["refused"],
            "low_llm": resolved_counts["low_llm"],
            "low_human": resolved_counts["low_human"],
            "missing_document": resolved_counts["missing_document"],
        },
        "summary": {
            "total_questions_analyzed": len(log_rows),
            "total_refusals": len(refused_questions),
            "low_llm_score_questions": len(low_llm_questions),
            "low_human_score_questions": len(low_human_questions),
            "corpus_gap_topics": len(common_missing_documents),
        },
        "top_question_categories": top_categories[:10],
        "unanswered_or_refused_questions": refused_questions[:12],
        "low_llm_judge_score_questions": low_llm_questions[:12],
        "low_human_audit_score_questions": low_human_questions[:12],
        "common_missing_documents": common_missing_documents,
        "rag_evaluation_note": (
            "Keep RAG evaluation as a separate admin dashboard because Precision@K, Recall@K, F1@K, "
            "and gold-standard audit results measure retrieval quality rather than user demand."
        ),
    }
    insight_payload["llm_generated_insights"] = _generate_llm_insights(insight_payload)
    return insight_payload


@router.post("/admin-insights/resolve")
def resolve_admin_insight(payload: InsightResolutionRequest, user: dict = Depends(require_admin)) -> dict:
    issue_type = payload.issue_type.strip().lower()
    item_key = payload.item_key.strip()
    if issue_type not in INSIGHT_RESOLUTION_TYPES:
        raise HTTPException(status_code=400, detail="Invalid insight issue type")
    if not item_key:
        raise HTTPException(status_code=400, detail="Missing insight item key")

    now = utc_now()
    with get_connection() as conn:
        if payload.resolved:
            conn.execute(
                """
                INSERT INTO insight_resolutions (
                    issue_type, item_key, note, resolved_by, resolved_at, active, updated_at
                )
                VALUES (?, ?, ?, ?, ?, 1, ?)
                ON CONFLICT(issue_type, item_key) DO UPDATE SET
                    note = excluded.note,
                    resolved_by = excluded.resolved_by,
                    resolved_at = excluded.resolved_at,
                    active = 1,
                    updated_at = excluded.updated_at
                """,
                (issue_type, item_key, payload.note, user["id"], now, now),
            )
        else:
            conn.execute(
                """
                UPDATE insight_resolutions
                SET active = 0, updated_at = ?
                WHERE issue_type = ? AND item_key = ?
                """,
                (now, issue_type, item_key),
            )

    return {
        "ok": True,
        "issue_type": issue_type,
        "item_key": item_key,
        "resolved": payload.resolved,
        "updated_at": now,
    }


@router.get("/export")
def export_logs(
    _: dict = Depends(get_current_user),
    x_admin_token: str | None = Header(default=None),
) -> dict:
    if not settings.admin_export_token:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin export is disabled. Set ADMIN_EXPORT_TOKEN to enable it.",
        )
    if not x_admin_token or not compare_digest(x_admin_token, settings.admin_export_token):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid admin export token")

    with get_connection() as conn:
        users = [
            dict(row)
            for row in conn.execute(
                "SELECT id, email, full_name, role, created_at FROM users ORDER BY id"
            ).fetchall()
        ]
        chat_sessions = [dict(row) for row in conn.execute("SELECT * FROM chat_sessions ORDER BY id").fetchall()]
        chat_logs = [dict(row) for row in conn.execute("SELECT * FROM chat_logs ORDER BY id").fetchall()]
        feedback_rows = [dict(row) for row in conn.execute("SELECT * FROM feedback ORDER BY id").fetchall()]
        demo_feedback_rows = [dict(row) for row in conn.execute("SELECT * FROM demo_feedback ORDER BY id").fetchall()]
        human_audits = [dict(row) for row in conn.execute("SELECT * FROM human_audits ORDER BY id").fetchall()]
        citation_events = [dict(row) for row in conn.execute("SELECT * FROM citation_events ORDER BY id").fetchall()]

    for item in chat_logs:
        item["citations"] = _decode_json(item.pop("citations_json"))
        item["evidence"] = _decode_json(item.pop("evidence_json"))
        item["judge"] = _decode_json(item.pop("judge_json"))

    return {
        "exported_at": utc_now(),
        "users": users,
        "chat_sessions": chat_sessions,
        "chat_logs": chat_logs,
        "feedback": feedback_rows,
        "demo_feedback": demo_feedback_rows,
        "human_audits": human_audits,
        "citation_events": citation_events,
    }
