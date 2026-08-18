from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import auth, chat, documents, ingest, settings as settings_api
from app.core.config import settings
from app.core.database import init_db, seed_demo_user


app = FastAPI(
    title="Kraken RAG Assistant API",
    version="0.1.0",
    description="Citation-grounded RAG backend for Kraken policy documents.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.get_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup() -> None:
    settings.ensure_storage_dirs()
    init_db()
    seed_demo_user()


@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "app": "kraken-rag-assistant"}


app.include_router(auth.router, prefix="/api/auth", tags=["auth"])
app.include_router(documents.router, prefix="/api/documents", tags=["documents"])
app.include_router(ingest.router, prefix="/api/ingest", tags=["ingest"])
app.include_router(chat.router, prefix="/api/chat", tags=["chat"])
app.include_router(settings_api.router, prefix="/api/settings", tags=["settings"])
