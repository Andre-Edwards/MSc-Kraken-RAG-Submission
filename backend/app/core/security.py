from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import bcrypt
import jwt

from app.core.config import settings


JWT_ALGORITHM = "HS256"


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))


def create_access_token(subject: str, extra: dict[str, Any] | None = None) -> str:
    expire = datetime.now(timezone.utc) + timedelta(minutes=settings.access_token_expire_minutes)
    payload: dict[str, Any] = {"sub": subject, "exp": expire}
    if extra:
        payload.update(extra)
    return jwt.encode(payload, settings.app_secret_key, algorithm=JWT_ALGORITHM)


def decode_access_token(token: str) -> dict[str, Any]:
    return jwt.decode(token, settings.app_secret_key, algorithms=[JWT_ALGORITHM])
