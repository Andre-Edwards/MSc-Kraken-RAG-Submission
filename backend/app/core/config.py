from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


BACKEND_DIR = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BACKEND_DIR.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BACKEND_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "Kraken RAG Assistant"
    app_secret_key: str = "dev-secret-key-change-this-before-real-use-32bytes"
    access_token_expire_minutes: int = 720

    openai_api_key: str | None = None
    openai_embedding_model: str = "text-embedding-3-small"
    openai_generation_model: str = "gpt-4o-mini"
    openai_judge_model: str | None = None
    openai_final_model: str = "gpt-4.1"

    demo_email: str = "demo@example.com"
    demo_password: str = "demo1234"
    admin_email: str = "admin@example.com"
    admin_password: str = "admin1234"
    seed_testing_users: bool = False
    testing_account_count: int = 20
    testing_account_email_prefix: str = "tester"
    testing_account_domain: str = "example.com"
    testing_account_password_prefix: str = "tester"
    admin_export_token: str | None = None
    supabase_url: str | None = None
    supabase_service_role_key: str | None = None
    supabase_audit_enabled: bool = False

    kraken_pdf_dir: Path = Field(default_factory=lambda: PROJECT_ROOT / "data" / "kraken_PDFs")
    chroma_dir: Path = Field(default_factory=lambda: BACKEND_DIR / "storage" / "chroma")
    sqlite_path: Path = Field(default_factory=lambda: BACKEND_DIR / "storage" / "app.db")
    processed_dir: Path = Field(default_factory=lambda: BACKEND_DIR / "storage" / "processed")
    logs_dir: Path = Field(default_factory=lambda: BACKEND_DIR / "storage" / "logs")
    web_corpus_dir: Path = Field(default_factory=lambda: BACKEND_DIR / "storage" / "web_pages")
    web_crawl_user_agent: str = "KrakenRAGDissertationBot/1.0"
    web_crawl_default_allowed_domains: str = "kraken.com,www.kraken.com,support.kraken.com,docs.kraken.com"
    web_crawl_default_max_pages: int = 25
    web_crawl_default_max_depth: int = 1
    web_crawl_delay_seconds: float = 0.5
    web_crawl_timeout_seconds: float = 15.0

    fixed_chunk_words: int = 320
    fixed_chunk_overlap: int = 60
    section_chunk_words: int = 380
    section_chunk_overlap: int = 70
    section_min_chunk_words: int = 20

    min_retrieval_score: float = 0.18
    max_context_chunks: int = 5
    max_context_chars_per_chunk: int = 1800
    metadata_rerank_enabled: bool = True
    metadata_rerank_candidate_multiplier: int = 4
    hybrid_candidate_multiplier: int = 2
    metadata_title_boost: float = 0.22
    metadata_section_boost: float = 0.08

    cors_origins: str = (
        "http://127.0.0.1:5173,"
        "http://localhost:5173,"
        "http://127.0.0.1:3000,"
        "http://localhost:3000,"
        "http://127.0.0.1:4173,"
        "http://localhost:4173"
    )
    auto_ingest_on_start: bool = False
    force_reindex_on_start: bool = False

    def ensure_storage_dirs(self) -> None:
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        self.chroma_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.web_corpus_dir.mkdir(parents=True, exist_ok=True)

    def resolve_pdf_dir(self) -> Path:
        pdf_dir = self.kraken_pdf_dir
        if not pdf_dir.is_absolute():
            pdf_dir = (PROJECT_ROOT / pdf_dir).resolve()
        return pdf_dir

    def resolve_chroma_dir(self) -> Path:
        chroma_dir = self.chroma_dir
        if not chroma_dir.is_absolute():
            chroma_dir = (BACKEND_DIR / chroma_dir).resolve()
        return chroma_dir

    def resolve_sqlite_path(self) -> Path:
        sqlite_path = self.sqlite_path
        if not sqlite_path.is_absolute():
            sqlite_path = (BACKEND_DIR / sqlite_path).resolve()
        return sqlite_path

    def resolve_web_corpus_dir(self) -> Path:
        web_dir = self.web_corpus_dir
        if not web_dir.is_absolute():
            web_dir = (BACKEND_DIR / web_dir).resolve()
        return web_dir

    def get_cors_origins(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    def get_default_allowed_web_domains(self) -> list[str]:
        return [
            domain.strip().lower()
            for domain in self.web_crawl_default_allowed_domains.split(",")
            if domain.strip()
        ]


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_storage_dirs()
    return settings


settings = get_settings()
