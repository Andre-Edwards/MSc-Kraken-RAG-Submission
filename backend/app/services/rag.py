from __future__ import annotations

import json
from time import perf_counter
from typing import Any

from app.core.config import settings
from app.core.database import (
    ensure_chat_session,
    get_connection,
    get_corpus_index_versions,
    human_audit_prompt_for_chat,
    touch_chat_session,
    utc_now,
)
from app.models import HYBRID_STRATEGY, VALID_STRATEGIES
from app.services.openai_client import OpenAIService
from app.services.supabase_audit import mirror_audit_event
from app.services.vector_store import VectorStore


REFUSAL = (
    "I cannot find this information in the provided Kraken policy documents. "
    "Please check the source documents or consult the relevant policy owner."
)


def _elapsed_ms(start: float) -> int:
    return round((perf_counter() - start) * 1000)


def _format_context(hits: list[dict]) -> str:
    parts: list[str] = []
    for i, hit in enumerate(hits, start=1):
        meta = hit["metadata"]
        text = hit["text"][: settings.max_context_chars_per_chunk]
        parts.append(
            "\n".join(
                [
                    f"[C{i}] {meta.get('file_name')} | {meta.get('section_title')} | pages {meta.get('page_start')}-{meta.get('page_end')}",
                    text,
                ]
            )
        )
    return "\n\n".join(parts)


def _build_answer_prompt(question: str, hits: list[dict]) -> list[dict[str, str]]:
    context = _format_context(hits)
    system = (
        "You are a compliance policy assistant for Kraken policy documents. "
        "Answer using only the supplied context. Do not use outside knowledge. "
        "Cite every factual claim with citation markers like [C1] or [C2]. "
        "If the context is insufficient, say you cannot find the information in the provided documents. "
        "Keep answers clear, practical, and cautious."
    )
    user = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _build_judge_prompt(question: str, answer: str, hits: list[dict]) -> str:
    return f"""
You are evaluating a RAG answer for a dissertation experiment.

Judge ONLY against the retrieved context below. Do not reward general plausibility, policy knowledge, or outside knowledge.
Use the full 1-5 scale. Do not collapse every imperfect answer to 1, and do not give 5 unless it is clearly excellent.
Judge like a careful human evaluator reviewing whether the answer is useful for an employee using the chatbot.

First decide whether the retrieved context contains enough information to answer the question.
Then extract up to 8 important factual claims from the answer. For each claim, check whether the cited chunk(s) directly support it.
Treat citation markers as claim-level evidence: if [C2] is used near a claim, C2 must support that claim.
If the question names a specific document, disclosure, policy, service, jurisdiction, or section, treat that named scope as important.
Answers that mostly use the wrong document or drift into adjacent policy material should lose relevance and completeness even if the text is factually plausible.
Do not over-penalise a concise answer that gives the core policy point correctly with usable citations.

Return valid JSON only with exactly these keys:
- rubric_version: "rag-judge-v3"
- answer_type: "direct_answer" or "refusal"
- retrieved_context_answerable: boolean
- groundedness_score: integer from 1 to 5
- citation_score: integer from 1 to 5
- relevance_score: integer from 1 to 5
- completeness_score: integer from 1 to 5
- clarity_score: integer from 1 to 5
- overall_score: integer from 1 to 5
- supported: boolean
- verdict: "pass", "partial", or "fail"
- confidence: "low", "medium", or "high"
- claim_checks: array of objects with keys claim, citations_used, support_status, issue
- failure_modes: array chosen from ["unsupported_claim", "wrong_citation", "missing_citation", "incomplete_answer", "missing_core_policy_detail", "scope_drift", "wrong_source_document", "overbroad_answer", "irrelevant_answer", "overconfident_answer", "unclear_answer", "should_refuse", "unnecessary_refusal"]
- unsupported_claims: array of short strings naming specific unsupported or weakly supported claims
- missing_citations: array of short strings naming claims that needed citations
- incorrect_citations: array of citation labels that do not support the nearby claim
- required_improvements: array of short actionable improvements
- notes: one specific non-boilerplate sentence explaining the judgement

Score anchors:
5 = excellent: answer directly addresses the question, all important claims are supported by the cited chunks, citations are precise, and no material information is missing.
4 = good: answer is supported and useful, with only minor omissions, mild citation imprecision, or wording issues.
3 = mixed: answer is partly useful, but misses an important policy point, has weak citation coverage, gives an over-broad answer, or includes a claim that is only partially supported.
2 = poor: answer contains major omissions, mostly uses the wrong source scope, has multiple weak/incorrect citations, or includes important unsupported claims.
1 = fail: answer is irrelevant, mostly unsupported, misleading, has no usable citation support, or should have refused but did not.

Calibration rules:
- If the answer refuses and the retrieved context is insufficient, groundedness_score, citation_score, and overall_score may be 4 or 5 because the refusal is correct.
- If the answer refuses but the retrieved context does answer the question, set failure_modes to include "unnecessary_refusal" and completeness_score must be 2 or lower.
- If the answer gives a direct answer while the retrieved context does not answer the question, set failure_modes to include "should_refuse" and verdict must be "fail".
- If any important factual claim has no citation, citation_score must be 3 or lower.
- If a citation label points to context that does not support the nearby claim, citation_score must be 2 or lower.
- If the answer uses facts not in the retrieved context, groundedness_score must be 2 or lower.
- If the answer is correct but materially incomplete, completeness_score must be 3 or lower. Minor missing detail should usually remain a 4, not a 3.
- If the question asks about a named document or policy and the answer relies mainly on another document, include "wrong_source_document" or "scope_drift"; relevance_score and completeness_score must be 3 or lower.
- If the answer includes adjacent but unnecessary policy material that could confuse the user, include "overbroad_answer"; overall_score should be 3 at best unless the core answer is still very clear.
- If the answer gives only a generic summary when the question asks for concrete policy details, completeness_score must be 3 or lower and verdict should be "partial" at best.
- Do not treat "see the policy/documentation for details" as a substitute for answering from the retrieved context; if this is the main answer, completeness_score must be 3 or lower.
- verdict should be "pass" when the answer is practically usable and no serious issue is present, even if it is not perfect.
- verdict should be "partial" when the answer is useful but materially incomplete, overbroad, or has notable citation/source-scope problems.
- verdict should be "fail" when the answer is misleading, unsupported, irrelevant, or should have refused.
- overall_score should reflect the weakest serious dimension, not a simple optimistic average.

Question:
{question}

Answer:
{answer}

Retrieved context:
{_format_context(hits)}
""".strip()


def _score(value: Any, default: int = 1) -> int:
    try:
        score = int(value)
    except (TypeError, ValueError):
        return default
    return max(1, min(5, score))


def _as_list(value: Any) -> list:
    if isinstance(value, list):
        return value
    if value in (None, ""):
        return []
    return [value]


def _normalise_judge(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {
            "rubric_version": "rag-judge-v3",
            "parse_error": True,
            "raw": raw,
            "groundedness_score": 1,
            "citation_score": 1,
            "relevance_score": 1,
            "completeness_score": 1,
            "clarity_score": 1,
            "overall_score": 1,
            "supported": False,
            "verdict": "fail",
            "confidence": "low",
            "failure_modes": ["unclear_answer"],
            "notes": "The judge response could not be parsed into the expected JSON object.",
        }

    judge = dict(raw)
    judge["rubric_version"] = "rag-judge-v3"
    score_keys = [
        "groundedness_score",
        "citation_score",
        "relevance_score",
        "completeness_score",
        "clarity_score",
    ]
    for key in score_keys:
        judge[key] = _score(judge.get(key), default=1)

    if "overall_score" in judge:
        judge["overall_score"] = _score(judge.get("overall_score"), default=min(judge[key] for key in score_keys))
    else:
        judge["overall_score"] = min(judge[key] for key in score_keys)

    if judge.get("verdict") not in {"pass", "partial", "fail"}:
        if judge["overall_score"] >= 4:
            judge["verdict"] = "pass"
        elif judge["overall_score"] >= 3:
            judge["verdict"] = "partial"
        else:
            judge["verdict"] = "fail"

    if judge.get("confidence") not in {"low", "medium", "high"}:
        judge["confidence"] = "medium"

    judge["supported"] = bool(judge.get("supported", judge["verdict"] == "pass"))
    judge["failure_modes"] = _as_list(judge.get("failure_modes"))
    judge["unsupported_claims"] = _as_list(judge.get("unsupported_claims"))
    judge["missing_citations"] = _as_list(judge.get("missing_citations"))
    judge["incorrect_citations"] = _as_list(judge.get("incorrect_citations"))
    judge["required_improvements"] = _as_list(judge.get("required_improvements"))
    judge["claim_checks"] = _as_list(judge.get("claim_checks"))
    judge.setdefault("notes", "No judge note provided.")
    calibration_text = " ".join(
        str(item)
        for item in [
            judge.get("notes", ""),
            *judge["required_improvements"],
            *judge["failure_modes"],
        ]
    ).lower()
    serious_detail_gap_markers = [
        "materially incomplete",
        "major omission",
        "missing important",
        "missing critical",
        "missing core",
        "omits the main",
        "generic summary",
        "too generic",
        "not specific enough",
        "incomplete_answer",
        "missing_core_policy_detail",
    ]
    scope_gap_markers = [
        "scope drift",
        "scope_drift",
        "wrong source",
        "wrong_source_document",
        "wrong document",
        "outside the requested document",
        "overbroad_answer",
    ]
    if any(marker in calibration_text for marker in scope_gap_markers):
        adjustments = judge.setdefault("calibration_adjustments", [])
        if judge["relevance_score"] > 3:
            judge["relevance_score"] = 3
            adjustments.append("capped_relevance_for_scope_drift")
        if judge["completeness_score"] > 3:
            judge["completeness_score"] = 3
            adjustments.append("capped_completeness_for_scope_drift")
        if judge["overall_score"] > 3:
            judge["overall_score"] = 3
            adjustments.append("capped_overall_for_scope_drift")
        if "scope_drift" not in judge["failure_modes"] and "wrong_source_document" not in judge["failure_modes"]:
            judge["failure_modes"].append("scope_drift")
        if judge["verdict"] == "pass":
            judge["verdict"] = "partial"
    if any(marker in calibration_text for marker in serious_detail_gap_markers):
        adjustments = judge.setdefault("calibration_adjustments", [])
        if judge["completeness_score"] > 3:
            judge["completeness_score"] = 3
            adjustments.append("capped_completeness_for_detail_gap")
        if judge["overall_score"] > 3:
            judge["overall_score"] = 3
            adjustments.append("capped_overall_for_detail_gap")
        if "incomplete_answer" not in judge["failure_modes"]:
            judge["failure_modes"].append("incomplete_answer")
        if judge["verdict"] == "pass":
            judge["verdict"] = "partial"
    return judge


def _citations_from_hits(hits: list[dict]) -> list[dict]:
    citations: list[dict] = []
    for i, hit in enumerate(hits, start=1):
        meta = hit["metadata"]
        citations.append(
            {
                "label": f"C{i}",
                "chunk_id": hit["chunk_id"],
                "file_name": meta.get("file_name"),
                "source_path": meta.get("source_path"),
                "document": meta.get("doc_id"),
                "section_title": meta.get("section_title"),
                "page_start": meta.get("page_start"),
                "page_end": meta.get("page_end"),
                "score": hit["score"],
                "vector_score": hit.get("vector_score"),
                "metadata_boost": hit.get("metadata_boost"),
                "source_strategy": hit.get("source_strategy") or meta.get("source_strategy") or meta.get("strategy"),
            }
        )
    return citations


def _evidence_from_hits(hits: list[dict]) -> list[dict]:
    evidence: list[dict] = []
    for i, hit in enumerate(hits, start=1):
        meta = hit["metadata"]
        evidence.append(
            {
                "label": f"C{i}",
                "chunk_id": hit["chunk_id"],
                "text": hit["text"],
                "file_name": meta.get("file_name"),
                "source_path": meta.get("source_path"),
                "section_title": meta.get("section_title"),
                "page_start": meta.get("page_start"),
                "page_end": meta.get("page_end"),
                "score": hit["score"],
                "vector_score": hit.get("vector_score"),
                "metadata_boost": hit.get("metadata_boost"),
                "source_strategy": hit.get("source_strategy") or meta.get("source_strategy") or meta.get("strategy"),
            }
        )
    return evidence


def answer_question(
    user_id: int,
    question: str,
    strategy: str,
    top_k: int,
    run_judge: bool = False,
    metadata_rerank_enabled: bool | None = None,
    session_id: int | None = None,
) -> dict:
    if strategy not in VALID_STRATEGIES:
        raise ValueError(f"Unsupported chunking strategy: {strategy}")

    total_start = perf_counter()
    openai_service = OpenAIService()
    vector_store = VectorStore()

    embedding_result = openai_service.embed_texts_with_metadata([question])
    query_embedding = embedding_result["embeddings"][0]
    vector_start = perf_counter()
    if strategy == HYBRID_STRATEGY:
        hits = vector_store.query_hybrid(
            query_embedding=query_embedding,
            top_k=top_k,
            query_text=question,
            metadata_rerank_enabled=metadata_rerank_enabled,
        )
    else:
        hits = vector_store.query(
            strategy,
            query_embedding=query_embedding,
            top_k=top_k,
            query_text=question,
            metadata_rerank_enabled=metadata_rerank_enabled,
        )
    vector_query_latency_ms = _elapsed_ms(vector_start)
    best_score = hits[0]["score"] if hits else 0.0

    refused = False
    judge = None
    generation_result = None
    judge_result = None
    if not hits or best_score < settings.min_retrieval_score:
        answer = REFUSAL
        refused = True
    else:
        generation_result = openai_service.chat_with_metadata(_build_answer_prompt(question, hits))
        answer = generation_result["content"]
        if answer.strip().lower().startswith("i cannot find"):
            refused = True
        if run_judge:
            judge_result = openai_service.judge_json_with_metadata(_build_judge_prompt(question, answer, hits))
            judge = _normalise_judge(judge_result["judge"])

    citations = _citations_from_hits(hits)
    evidence = _evidence_from_hits(hits)
    total_latency_ms = _elapsed_ms(total_start)
    generation_model = generation_result["model"] if generation_result else settings.openai_generation_model
    judge_model = None
    if run_judge:
        judge_model = (
            judge_result["model"]
            if judge_result
            else settings.openai_judge_model or settings.openai_generation_model
        )

    with get_connection() as conn:
        chat_session_id = ensure_chat_session(conn, user_id, session_id, question)
        versions = get_corpus_index_versions(conn)
        created_at = utc_now()
        cur = conn.execute(
            """
            INSERT INTO chat_logs (
                session_id, user_id, question, answer, chunking_strategy, top_k, run_judge,
                retrieved_count, citations_json, evidence_json, judge_json,
                retrieval_score, refused, embedding_model, generation_model,
                judge_model, embedding_latency_ms, vector_query_latency_ms,
                generation_latency_ms, judge_latency_ms, total_latency_ms,
                generation_prompt_tokens, generation_completion_tokens,
                generation_total_tokens, judge_prompt_tokens,
                judge_completion_tokens, judge_total_tokens, corpus_version,
                index_version, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                chat_session_id,
                user_id,
                question,
                answer,
                strategy,
                top_k,
                1 if run_judge else 0,
                len(hits),
                json.dumps(citations),
                json.dumps(evidence),
                json.dumps(judge) if judge is not None else None,
                best_score,
                1 if refused else 0,
                embedding_result["model"],
                generation_model,
                judge_model,
                embedding_result["duration_ms"],
                vector_query_latency_ms,
                generation_result["duration_ms"] if generation_result else None,
                judge_result["duration_ms"] if judge_result else None,
                total_latency_ms,
                generation_result["prompt_tokens"] if generation_result else None,
                generation_result["completion_tokens"] if generation_result else None,
                generation_result["total_tokens"] if generation_result else None,
                judge_result["prompt_tokens"] if judge_result else None,
                judge_result["completion_tokens"] if judge_result else None,
                judge_result["total_tokens"] if judge_result else None,
                versions["corpus_version"],
                versions["index_version"],
                created_at,
            ),
        )
        chat_log_id = cur.lastrowid
        touch_chat_session(conn, chat_session_id)
        human_audit_prompt = human_audit_prompt_for_chat(conn, user_id, chat_log_id, judge)
        conn.execute(
            """
            UPDATE chat_logs
            SET human_audit_prompted = ?,
                human_audit_prompt_reason = ?,
                human_audit_query_count = ?
            WHERE id = ?
            """,
            (
                1 if human_audit_prompt.get("show") else 0,
                human_audit_prompt.get("reason"),
                human_audit_prompt.get("query_count"),
                chat_log_id,
            ),
        )

    response = {
        "chat_log_id": chat_log_id,
        "session_id": chat_session_id,
        "question": question,
        "answer": answer,
        "chunking_strategy": strategy,
        "citations": citations,
        "evidence": evidence,
        "judge": judge,
        "retrieval_score": best_score,
        "refused": refused,
        "top_k": top_k,
        "run_judge": run_judge,
        "metadata_rerank_enabled": metadata_rerank_enabled
        if metadata_rerank_enabled is not None
        else settings.metadata_rerank_enabled,
        "retrieved_count": len(hits),
        "human_audit_prompt": human_audit_prompt,
        "corpus_version": versions["corpus_version"],
        "index_version": versions["index_version"],
        "latency_ms": {
            "embedding": embedding_result["duration_ms"],
            "vector_query": vector_query_latency_ms,
            "generation": generation_result["duration_ms"] if generation_result else None,
            "judge": judge_result["duration_ms"] if judge_result else None,
            "total": total_latency_ms,
        },
        "models": {
            "embedding": embedding_result["model"],
            "generation": generation_model,
            "judge": judge_model,
        },
    }
    mirror_audit_event(
        "chat_log",
        response,
        local_id=chat_log_id,
        user_id=user_id,
        session_id=chat_session_id,
        chat_log_id=chat_log_id,
        created_at=created_at,
    )
    return response
