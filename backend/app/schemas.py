from __future__ import annotations

from pydantic import BaseModel, EmailStr, Field


class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(min_length=6)
    full_name: str | None = None


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: dict


class IngestRequest(BaseModel):
    force: bool = False
    strategies: list[str] = Field(default_factory=lambda: ["fixed_size", "structure_aware"])


class IngestResponse(BaseModel):
    ok: bool
    message: str
    counts: dict[str, int]


class WebCrawlRequest(BaseModel):
    seed_urls: list[str] = Field(min_length=1, max_length=20)
    allowed_domains: list[str] = Field(default_factory=list, max_length=20)
    max_pages: int = Field(default=25, ge=1, le=100)
    max_depth: int = Field(default=1, ge=0, le=3)


class WebCrawlResponse(BaseModel):
    ok: bool
    pages_saved: int
    visited: int
    skipped: list[dict[str, str]]
    output_path: str
    allowed_domains: list[str]
    seed_urls: list[str]
    max_pages: int
    max_depth: int
    corpus_version: int


class ChatRequest(BaseModel):
    question: str = Field(min_length=2)
    chunking_strategy: str = "structure_aware"
    top_k: int = Field(default=5, ge=1, le=10)
    run_judge: bool = False
    session_id: int | None = None


class RetrievalSettings(BaseModel):
    chunking_strategy: str = "structure_aware"
    top_k: int = Field(default=5, ge=1, le=10)
    run_judge: bool = False
    metadata_rerank_enabled: bool = True
    human_audit_enabled: bool = False
    human_audit_interval: int = Field(default=5, ge=0, le=100)
    human_audit_low_score_threshold: int = Field(default=3, ge=1, le=5)
    human_audit_max_per_session: int = Field(default=1, ge=0, le=20)
    human_audit_cooldown_minutes: int = Field(default=10, ge=0, le=1440)


class FeedbackRequest(BaseModel):
    chat_log_id: int | None = None
    feedback_type: str | None = Field(default=None, max_length=40)
    rating: int | None = Field(default=None, ge=1, le=5)
    comment: str | None = None


class DemoFeedbackRequest(BaseModel):
    role_department: str | None = Field(default=None, max_length=120)
    ease_score: int = Field(ge=1, le=5)
    helpfulness_score: int = Field(ge=1, le=5)
    trustworthiness_score: int = Field(ge=1, le=5)
    citations_useful: str = Field(max_length=20)
    incorrect_or_misleading: bool
    incorrect_notes: str | None = Field(default=None, max_length=1000)
    missing_expectation: str | None = Field(default=None, max_length=1000)
    improvement: str | None = Field(default=None, max_length=1000)
    would_use_at_work: str = Field(max_length=20)
    final_comments: str | None = Field(default=None, max_length=1000)


class HumanAuditRequest(BaseModel):
    chat_log_id: int
    groundedness_score: int = Field(ge=1, le=5)
    citation_score: int = Field(ge=1, le=5)
    relevance_score: int = Field(ge=1, le=5)
    completeness_score: int = Field(ge=1, le=5)
    clarity_score: int = Field(ge=1, le=5)
    overall_verdict: str
    issue_tags: list[str] = Field(default_factory=list)
    comment: str | None = None
    reviewer_confidence: str


class InsightResolutionRequest(BaseModel):
    issue_type: str = Field(max_length=40)
    item_key: str = Field(max_length=200)
    resolved: bool = True
    note: str | None = Field(default=None, max_length=500)


class CitationClickRequest(BaseModel):
    chat_log_id: int | None = None
    citation_label: str | None = Field(default=None, max_length=20)
    chunk_id: str | None = Field(default=None, max_length=200)
    file_name: str
    page_start: int | None = None
    page_end: int | None = None
    source_strategy: str | None = Field(default=None, max_length=40)
