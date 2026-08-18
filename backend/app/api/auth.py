from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.deps import get_current_user
from app.core.database import get_connection, utc_now
from app.core.security import create_access_token, hash_password, verify_password
from app.schemas import LoginRequest, TokenResponse, UserCreate


router = APIRouter()


def _public_user(row) -> dict:
    return {
        "id": row["id"],
        "email": row["email"],
        "full_name": row["full_name"],
        "role": row["role"],
        "created_at": row["created_at"],
    }


@router.post("/register", response_model=TokenResponse)
def register(payload: UserCreate) -> TokenResponse:
    with get_connection() as conn:
        existing = conn.execute(
            "SELECT id FROM users WHERE lower(email) = lower(?)",
            (payload.email,),
        ).fetchone()
        if existing:
            raise HTTPException(status_code=409, detail="Email is already registered")

        cur = conn.execute(
            """
            INSERT INTO users (email, full_name, password_hash, role, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                payload.email,
                payload.full_name,
                hash_password(payload.password),
                "user",
                utc_now(),
            ),
        )
        row = conn.execute(
            "SELECT id, email, full_name, role, created_at FROM users WHERE id = ?",
            (cur.lastrowid,),
        ).fetchone()

    token = create_access_token(str(row["id"]))
    return TokenResponse(access_token=token, user=_public_user(row))


@router.post("/login", response_model=TokenResponse)
def login(payload: LoginRequest) -> TokenResponse:
    with get_connection() as conn:
        row = conn.execute(
            """
            SELECT id, email, full_name, role, password_hash, created_at
            FROM users
            WHERE lower(email) = lower(?)
            """,
            (payload.email,),
        ).fetchone()

    if row is None or not verify_password(payload.password, row["password_hash"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
        )

    token = create_access_token(str(row["id"]))
    return TokenResponse(access_token=token, user=_public_user(row))


@router.get("/me")
def me(user: dict = Depends(get_current_user)) -> dict:
    return user
