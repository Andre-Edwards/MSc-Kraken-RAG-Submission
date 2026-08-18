from __future__ import annotations

import json
import logging
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from app.core.config import settings
from app.core.database import utc_now


logger = logging.getLogger(__name__)


def _enabled() -> bool:
    return bool(
        settings.supabase_audit_enabled
        and settings.supabase_url
        and settings.supabase_service_role_key
    )


def mirror_audit_event(
    event_type: str,
    payload: dict[str, Any],
    *,
    local_id: int | None = None,
    user_id: int | None = None,
    session_id: int | None = None,
    chat_log_id: int | None = None,
    created_at: str | None = None,
) -> None:
    """Mirror audit data to Supabase without affecting the live app flow."""
    if not _enabled():
        return

    base_url = str(settings.supabase_url).rstrip("/")
    url = f"{base_url}/rest/v1/rag_audit_events"
    body = json.dumps(
        {
            "event_type": event_type,
            "local_id": local_id,
            "user_id": user_id,
            "session_id": session_id,
            "chat_log_id": chat_log_id,
            "created_at": created_at or utc_now(),
            "payload": payload,
        },
        ensure_ascii=False,
    ).encode("utf-8")
    request = Request(
        url,
        data=body,
        method="POST",
        headers={
            "apikey": settings.supabase_service_role_key or "",
            "Authorization": f"Bearer {settings.supabase_service_role_key}",
            "Content-Type": "application/json",
            "Prefer": "return=minimal",
        },
    )
    try:
        with urlopen(request, timeout=6):
            return
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        logger.warning("Supabase audit mirror failed for %s: %s", event_type, exc)
