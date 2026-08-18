from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Iterator

from app.core.config import settings
from app.core.security import hash_password


def seeded_testing_accounts() -> list[dict[str, str]]:
    account_count = max(settings.testing_account_count, 20)
    return [
        {
            "email": f"{settings.testing_account_email_prefix}{index:02d}@{settings.testing_account_domain}",
            "full_name": f"Testing User {index:02d}",
            "password": f"{settings.testing_account_password_prefix}{index:02d}",
        }
        for index in range(1, account_count + 1)
    ]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def get_connection() -> Iterator[sqlite3.Connection]:
    path = settings.resolve_sqlite_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with get_connection() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT NOT NULL UNIQUE,
                full_name TEXT,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'user',
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS chat_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id)
            );

            CREATE TABLE IF NOT EXISTS chat_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER,
                user_id INTEGER NOT NULL,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                chunking_strategy TEXT NOT NULL,
                top_k INTEGER,
                run_judge INTEGER NOT NULL DEFAULT 0,
                retrieved_count INTEGER,
                citations_json TEXT NOT NULL,
                evidence_json TEXT NOT NULL,
                judge_json TEXT,
                retrieval_score REAL,
                refused INTEGER NOT NULL DEFAULT 0,
                embedding_model TEXT,
                generation_model TEXT,
                judge_model TEXT,
                embedding_latency_ms INTEGER,
                vector_query_latency_ms INTEGER,
                generation_latency_ms INTEGER,
                judge_latency_ms INTEGER,
                total_latency_ms INTEGER,
                generation_prompt_tokens INTEGER,
                generation_completion_tokens INTEGER,
                generation_total_tokens INTEGER,
                judge_prompt_tokens INTEGER,
                judge_completion_tokens INTEGER,
                judge_total_tokens INTEGER,
                created_at TEXT NOT NULL,
                FOREIGN KEY(session_id) REFERENCES chat_sessions(id),
                FOREIGN KEY(user_id) REFERENCES users(id)
            );

            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                chat_log_id INTEGER,
                feedback_type TEXT,
                rating INTEGER,
                comment TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id),
                FOREIGN KEY(chat_log_id) REFERENCES chat_logs(id)
            );

            CREATE TABLE IF NOT EXISTS demo_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                role_department TEXT,
                ease_score INTEGER NOT NULL,
                helpfulness_score INTEGER NOT NULL,
                trustworthiness_score INTEGER NOT NULL,
                citations_useful TEXT NOT NULL,
                incorrect_or_misleading INTEGER NOT NULL,
                incorrect_notes TEXT,
                missing_expectation TEXT,
                improvement TEXT,
                would_use_at_work TEXT NOT NULL,
                final_comments TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id)
            );

            CREATE TABLE IF NOT EXISTS human_audits (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                chat_log_id INTEGER NOT NULL,
                groundedness_score INTEGER NOT NULL,
                citation_score INTEGER NOT NULL,
                relevance_score INTEGER NOT NULL,
                completeness_score INTEGER NOT NULL,
                clarity_score INTEGER NOT NULL,
                overall_verdict TEXT NOT NULL,
                issue_tags_json TEXT NOT NULL,
                comment TEXT,
                reviewer_confidence TEXT NOT NULL,
                llm_judge_json_snapshot TEXT,
                llm_overall_score INTEGER,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(user_id, chat_log_id),
                FOREIGN KEY(user_id) REFERENCES users(id),
                FOREIGN KEY(chat_log_id) REFERENCES chat_logs(id)
            );

            CREATE TABLE IF NOT EXISTS citation_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                chat_log_id INTEGER,
                citation_label TEXT,
                chunk_id TEXT,
                file_name TEXT NOT NULL,
                page_start INTEGER,
                page_end INTEGER,
                source_strategy TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id),
                FOREIGN KEY(chat_log_id) REFERENCES chat_logs(id)
            );

            CREATE TABLE IF NOT EXISTS insight_resolutions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                issue_type TEXT NOT NULL,
                item_key TEXT NOT NULL,
                note TEXT,
                resolved_by INTEGER NOT NULL,
                resolved_at TEXT NOT NULL,
                active INTEGER NOT NULL DEFAULT 1,
                updated_at TEXT NOT NULL,
                UNIQUE(issue_type, item_key),
                FOREIGN KEY(resolved_by) REFERENCES users(id)
            );

            CREATE TABLE IF NOT EXISTS app_settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            """
        )
        _ensure_columns(conn)
        _backfill_chat_sessions(conn)
        _seed_app_settings(conn)


def _ensure_columns(conn: sqlite3.Connection) -> None:
    user_columns = {
        row["name"] for row in conn.execute("PRAGMA table_info(users)").fetchall()
    }
    if "role" not in user_columns:
        conn.execute("ALTER TABLE users ADD COLUMN role TEXT NOT NULL DEFAULT 'user'")

    chat_log_columns = {
        row["name"] for row in conn.execute("PRAGMA table_info(chat_logs)").fetchall()
    }
    chat_log_additions = {
        "session_id": "INTEGER",
        "top_k": "INTEGER",
        "run_judge": "INTEGER NOT NULL DEFAULT 0",
        "retrieved_count": "INTEGER",
        "embedding_model": "TEXT",
        "generation_model": "TEXT",
        "judge_model": "TEXT",
        "embedding_latency_ms": "INTEGER",
        "vector_query_latency_ms": "INTEGER",
        "generation_latency_ms": "INTEGER",
        "judge_latency_ms": "INTEGER",
        "total_latency_ms": "INTEGER",
        "generation_prompt_tokens": "INTEGER",
        "generation_completion_tokens": "INTEGER",
        "generation_total_tokens": "INTEGER",
        "judge_prompt_tokens": "INTEGER",
        "judge_completion_tokens": "INTEGER",
        "judge_total_tokens": "INTEGER",
        "corpus_version": "INTEGER",
        "index_version": "INTEGER",
        "human_audit_prompted": "INTEGER NOT NULL DEFAULT 0",
        "human_audit_prompt_reason": "TEXT",
        "human_audit_query_count": "INTEGER",
    }
    for column, definition in chat_log_additions.items():
        if column not in chat_log_columns:
            conn.execute(f"ALTER TABLE chat_logs ADD COLUMN {column} {definition}")

    feedback_columns = {
        row["name"] for row in conn.execute("PRAGMA table_info(feedback)").fetchall()
    }
    if "feedback_type" not in feedback_columns:
        conn.execute("ALTER TABLE feedback ADD COLUMN feedback_type TEXT")


def _chat_title(question: str) -> str:
    title = " ".join(question.split()).strip()
    if not title:
        return "New chat"
    return title[:77] + "..." if len(title) > 80 else title


def _backfill_chat_sessions(conn: sqlite3.Connection) -> None:
    rows = conn.execute(
        """
        SELECT id, user_id, question, created_at
        FROM chat_logs
        WHERE session_id IS NULL
        ORDER BY id
        """
    ).fetchall()
    for row in rows:
        cur = conn.execute(
            """
            INSERT INTO chat_sessions (user_id, title, created_at, updated_at)
            VALUES (?, ?, ?, ?)
            """,
            (
                row["user_id"],
                _chat_title(row["question"]),
                row["created_at"],
                row["created_at"],
            ),
        )
        conn.execute(
            "UPDATE chat_logs SET session_id = ? WHERE id = ?",
            (cur.lastrowid, row["id"]),
        )


def ensure_chat_session(
    conn: sqlite3.Connection,
    user_id: int,
    session_id: int | None,
    question: str,
) -> int:
    if session_id is not None:
        row = conn.execute(
            "SELECT id FROM chat_sessions WHERE id = ? AND user_id = ?",
            (session_id, user_id),
        ).fetchone()
        if row is None:
            raise ValueError("Chat session not found")
        return int(row["id"])

    now = utc_now()
    cur = conn.execute(
        """
        INSERT INTO chat_sessions (user_id, title, created_at, updated_at)
        VALUES (?, ?, ?, ?)
        """,
        (user_id, _chat_title(question), now, now),
    )
    return int(cur.lastrowid)


def touch_chat_session(conn: sqlite3.Connection, session_id: int) -> None:
    conn.execute(
        "UPDATE chat_sessions SET updated_at = ? WHERE id = ?",
        (utc_now(), session_id),
    )


def _seed_app_settings(conn: sqlite3.Connection) -> None:
    defaults = {
        "retrieval.chunking_strategy": "structure_aware",
        "retrieval.top_k": "5",
        "retrieval.run_judge": "false",
        "retrieval.metadata_rerank_enabled": "true" if settings.metadata_rerank_enabled else "false",
        "human_audit.enabled": "false",
        "human_audit.interval": "5",
        "human_audit.low_score_threshold": "3",
        "human_audit.max_per_session": "1",
        "human_audit.cooldown_minutes": "10",
        "corpus.version": "1",
        "index.version": "1",
    }
    now = utc_now()
    for key, value in defaults.items():
        conn.execute(
            """
            INSERT OR IGNORE INTO app_settings (key, value, updated_at)
            VALUES (?, ?, ?)
            """,
            (key, value, now),
        )


def get_retrieval_settings() -> dict:
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT key, value
            FROM app_settings
            WHERE key IN (
                'retrieval.chunking_strategy',
                'retrieval.top_k',
                'retrieval.run_judge',
                'retrieval.metadata_rerank_enabled',
                'human_audit.enabled',
                'human_audit.interval',
                'human_audit.low_score_threshold',
                'human_audit.max_per_session',
                'human_audit.cooldown_minutes'
            )
            """
        ).fetchall()
    values = {row["key"]: row["value"] for row in rows}
    return {
        "chunking_strategy": values.get("retrieval.chunking_strategy", "structure_aware"),
        "top_k": int(values.get("retrieval.top_k", "5")),
        "run_judge": values.get("retrieval.run_judge", "false").lower() == "true",
        "metadata_rerank_enabled": values.get(
            "retrieval.metadata_rerank_enabled",
            "true" if settings.metadata_rerank_enabled else "false",
        ).lower() == "true",
        "human_audit_enabled": values.get("human_audit.enabled", "false").lower() == "true",
        "human_audit_interval": int(values.get("human_audit.interval", "5")),
        "human_audit_low_score_threshold": int(values.get("human_audit.low_score_threshold", "3")),
        "human_audit_max_per_session": int(values.get("human_audit.max_per_session", "1")),
        "human_audit_cooldown_minutes": int(values.get("human_audit.cooldown_minutes", "10")),
    }


def update_retrieval_settings(
    chunking_strategy: str,
    top_k: int,
    run_judge: bool,
    metadata_rerank_enabled: bool,
    human_audit_enabled: bool,
    human_audit_interval: int,
    human_audit_low_score_threshold: int,
    human_audit_max_per_session: int,
    human_audit_cooldown_minutes: int,
) -> dict:
    updates = {
        "retrieval.chunking_strategy": chunking_strategy,
        "retrieval.top_k": str(top_k),
        "retrieval.run_judge": "true" if run_judge else "false",
        "retrieval.metadata_rerank_enabled": "true" if metadata_rerank_enabled else "false",
        "human_audit.enabled": "true" if human_audit_enabled else "false",
        "human_audit.interval": str(human_audit_interval),
        "human_audit.low_score_threshold": str(human_audit_low_score_threshold),
        "human_audit.max_per_session": str(human_audit_max_per_session),
        "human_audit.cooldown_minutes": str(human_audit_cooldown_minutes),
    }
    now = utc_now()
    with get_connection() as conn:
        for key, value in updates.items():
            conn.execute(
                """
                INSERT INTO app_settings (key, value, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value = excluded.value,
                    updated_at = excluded.updated_at
                """,
                (key, value, now),
            )
    return get_retrieval_settings()


def get_corpus_index_versions(conn: sqlite3.Connection) -> dict[str, int]:
    rows = conn.execute(
        """
        SELECT key, value
        FROM app_settings
        WHERE key IN ('corpus.version', 'index.version')
        """
    ).fetchall()
    values = {row["key"]: row["value"] for row in rows}
    return {
        "corpus_version": int(values.get("corpus.version", "1")),
        "index_version": int(values.get("index.version", "1")),
    }


def _bump_version(key: str) -> int:
    with get_connection() as conn:
        row = conn.execute("SELECT value FROM app_settings WHERE key = ?", (key,)).fetchone()
        current = int(row["value"]) if row is not None else 1
        next_value = current + 1
        conn.execute(
            """
            INSERT INTO app_settings (key, value, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET
                value = excluded.value,
                updated_at = excluded.updated_at
            """,
            (key, str(next_value), utc_now()),
        )
    return next_value


def bump_corpus_version() -> int:
    return _bump_version("corpus.version")


def bump_index_version() -> int:
    return _bump_version("index.version")


def human_audit_prompt_for_chat(
    conn: sqlite3.Connection,
    user_id: int,
    chat_log_id: int,
    judge: dict | None,
) -> dict:
    rows = conn.execute(
        """
        SELECT key, value
        FROM app_settings
        WHERE key IN (
            'human_audit.enabled',
            'human_audit.interval',
            'human_audit.low_score_threshold',
            'human_audit.max_per_session',
            'human_audit.cooldown_minutes'
        )
        """
    ).fetchall()
    values = {row["key"]: row["value"] for row in rows}
    human_audit_enabled = values.get("human_audit.enabled", "false").lower() == "true"
    interval = int(values.get("human_audit.interval", "5"))
    low_score_threshold = int(values.get("human_audit.low_score_threshold", "3"))
    max_per_session = int(values.get("human_audit.max_per_session", "1"))
    cooldown_minutes = int(values.get("human_audit.cooldown_minutes", "10"))

    if not human_audit_enabled:
        return {"show": False, "reason": "disabled"}

    existing = conn.execute(
        "SELECT id FROM human_audits WHERE user_id = ? AND chat_log_id = ?",
        (user_id, chat_log_id),
    ).fetchone()
    if existing is not None:
        return {"show": False, "reason": "already_reviewed", "reviewed": True}

    chat_log = conn.execute(
        "SELECT session_id FROM chat_logs WHERE id = ? AND user_id = ?",
        (chat_log_id, user_id),
    ).fetchone()
    session_id = chat_log["session_id"] if chat_log is not None else None

    if session_id is not None:
        query_count = conn.execute(
            """
            SELECT COUNT(*) AS count
            FROM chat_logs
            WHERE user_id = ? AND session_id = ?
            """,
            (user_id, session_id),
        ).fetchone()["count"]
    else:
        query_count = conn.execute(
            "SELECT COUNT(*) AS count FROM chat_logs WHERE user_id = ?",
            (user_id,),
        ).fetchone()["count"]

    interval_due = interval > 0 and query_count > 0 and query_count % interval == 0
    llm_overall_score = None
    if isinstance(judge, dict):
        try:
            llm_overall_score = int(judge.get("overall_score"))
        except (TypeError, ValueError):
            llm_overall_score = None
    low_score_due = llm_overall_score is not None and llm_overall_score <= low_score_threshold

    if interval_due and low_score_due:
        due_reason = "interval_and_low_score"
    elif interval_due:
        due_reason = "interval"
    elif low_score_due:
        due_reason = "low_score"
    else:
        due_reason = "not_due"

    due = interval_due or low_score_due
    reason = due_reason
    if due and session_id is not None and max_per_session > 0:
        session_prompt_count = conn.execute(
            """
            SELECT COUNT(*) AS count
            FROM chat_logs
            WHERE user_id = ? AND session_id = ? AND human_audit_prompted = 1
            """,
            (user_id, session_id),
        ).fetchone()["count"]
        if session_prompt_count >= max_per_session:
            due = False
            reason = "session_limit"

    if due and cooldown_minutes > 0:
        last_prompt = conn.execute(
            """
            SELECT created_at
            FROM chat_logs
            WHERE user_id = ? AND human_audit_prompted = 1 AND id <> ?
            ORDER BY id DESC
            LIMIT 1
            """,
            (user_id, chat_log_id),
        ).fetchone()
        if last_prompt is not None:
            last_prompt_at = datetime.fromisoformat(last_prompt["created_at"])
            if last_prompt_at.tzinfo is None:
                last_prompt_at = last_prompt_at.replace(tzinfo=timezone.utc)
            if datetime.now(timezone.utc) - last_prompt_at < timedelta(minutes=cooldown_minutes):
                due = False
                reason = "cooldown"

    return {
        "show": due,
        "reason": reason,
        "due_reason": due_reason,
        "reviewed": False,
        "query_count": query_count,
        "interval": interval,
        "low_score_threshold": low_score_threshold,
        "max_per_session": max_per_session,
        "cooldown_minutes": cooldown_minutes,
        "llm_overall_score": llm_overall_score,
    }


def _seed_user(
    conn: sqlite3.Connection,
    email: str,
    full_name: str,
    password: str,
    role: str = "user",
    reset_password: bool = False,
) -> None:
    existing = conn.execute(
        "SELECT id, role FROM users WHERE lower(email) = lower(?)",
        (email,),
    ).fetchone()
    if existing:
        if existing["role"] != role:
            conn.execute(
                "UPDATE users SET role = ? WHERE id = ?",
                (role, existing["id"]),
            )
        if reset_password:
            conn.execute(
                "UPDATE users SET full_name = ?, password_hash = ? WHERE id = ?",
                (full_name, hash_password(password), existing["id"]),
            )
        return
    conn.execute(
        """
        INSERT INTO users (email, full_name, password_hash, role, created_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            email,
            full_name,
            hash_password(password),
            role,
            utc_now(),
        ),
    )


def seed_demo_user() -> None:
    with get_connection() as conn:
        _seed_user(conn, settings.demo_email, "Demo User", settings.demo_password)
        _seed_user(conn, settings.admin_email, "Admin User", settings.admin_password, role="admin")
        if settings.seed_testing_users:
            for account in seeded_testing_accounts():
                _seed_user(
                    conn,
                    account["email"],
                    account["full_name"],
                    account["password"],
                    reset_password=True,
                )
