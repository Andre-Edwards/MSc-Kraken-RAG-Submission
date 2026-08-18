from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from app.api.deps import get_current_user, require_admin
from app.core.database import get_retrieval_settings, update_retrieval_settings
from app.models import VALID_STRATEGIES
from app.schemas import RetrievalSettings


router = APIRouter()


@router.get("/retrieval", response_model=RetrievalSettings)
def retrieval_settings(_: dict = Depends(get_current_user)) -> RetrievalSettings:
    return RetrievalSettings(**get_retrieval_settings())


@router.patch("/retrieval", response_model=RetrievalSettings)
def update_settings(
    payload: RetrievalSettings,
    _: dict = Depends(require_admin),
) -> RetrievalSettings:
    if payload.chunking_strategy not in VALID_STRATEGIES:
        raise HTTPException(status_code=400, detail="Invalid chunking strategy")
    settings = update_retrieval_settings(
        chunking_strategy=payload.chunking_strategy,
        top_k=payload.top_k,
        run_judge=payload.run_judge,
        metadata_rerank_enabled=payload.metadata_rerank_enabled,
        human_audit_enabled=payload.human_audit_enabled,
        human_audit_interval=payload.human_audit_interval,
        human_audit_low_score_threshold=payload.human_audit_low_score_threshold,
        human_audit_max_per_session=payload.human_audit_max_per_session,
        human_audit_cooldown_minutes=payload.human_audit_cooldown_minutes,
    )
    return RetrievalSettings(**settings)
