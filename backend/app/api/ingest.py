from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from app.api.deps import require_admin
from app.models import INDEX_STRATEGIES
from app.schemas import IngestRequest, IngestResponse
from app.services.ingestion import ingest_documents


router = APIRouter()


@router.post("", response_model=IngestResponse)
def ingest(payload: IngestRequest, _: dict = Depends(require_admin)) -> IngestResponse:
    invalid = [strategy for strategy in payload.strategies if strategy not in INDEX_STRATEGIES]
    if invalid:
        raise HTTPException(status_code=400, detail=f"Invalid strategies: {invalid}")
    try:
        counts = ingest_documents(payload.strategies, force=payload.force)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return IngestResponse(ok=True, message="Documents indexed", counts=counts)
