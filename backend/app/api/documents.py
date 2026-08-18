from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse

from app.api.deps import require_admin
from app.core.config import settings
from app.core.database import bump_corpus_version, bump_index_version, get_connection, get_corpus_index_versions
from app.schemas import WebCrawlRequest, WebCrawlResponse
from app.services.vector_store import VectorStore
from app.services.web_scraper import crawl_web_pages, delete_web_record, detect_duplicate_web_records, load_web_records


router = APIRouter()


def _resolve_pdf_file(file_name: str) -> Path:
    if Path(file_name).name != file_name:
        raise HTTPException(status_code=400, detail="Invalid file name")

    pdf_dir = settings.resolve_pdf_dir()
    candidate = pdf_dir / file_name
    if candidate.exists() and candidate.is_file():
        return candidate

    lower_name = file_name.lower()
    for path in pdf_dir.glob("*.pdf"):
        if path.name.lower() == lower_name:
            return path

    raise HTTPException(status_code=404, detail="Document not found")


def _remove_processed_chunks(file_name: str) -> dict[str, int]:
    return _remove_processed_chunks_where("file_name", file_name)


def _remove_processed_chunks_by_source_path(source_path: str) -> dict[str, int]:
    return _remove_processed_chunks_where("source_path", source_path)


def _remove_processed_chunks_where(field_name: str, value: str) -> dict[str, int]:
    removed: dict[str, int] = {}
    for path in settings.processed_dir.glob("chunks_*.jsonl"):
        kept: list[str] = []
        removed_count = 0
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                kept.append(line)
                continue
            if record.get(field_name) == value:
                removed_count += 1
            else:
                kept.append(line)
        if removed_count:
            path.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
        removed[path.stem.replace("chunks_", "")] = removed_count
    return removed


@router.get("/file/{file_name}")
def get_document_file(file_name: str) -> FileResponse:
    path = _resolve_pdf_file(file_name)
    return FileResponse(path=path, media_type="application/pdf", filename=path.name)


@router.get("")
def list_documents(_: dict = Depends(require_admin)) -> dict:
    pdf_dir = settings.resolve_pdf_dir()
    documents = []
    if pdf_dir.exists():
        for path in sorted(pdf_dir.glob("*.pdf")):
            documents.append({"file_name": path.name, "bytes": path.stat().st_size, "source_type": "pdf"})
    web_documents = [
        {
            "file_name": record.title,
            "bytes": len(record.text.encode("utf-8")),
            "source_type": "web",
            "url": record.final_url,
        }
        for record in load_web_records(settings.resolve_web_corpus_dir())
    ]
    return {
        "pdf_dir": str(pdf_dir),
        "web_corpus_dir": str(settings.resolve_web_corpus_dir()),
        "documents": documents,
        "web_documents": web_documents,
    }


@router.get("/status")
def status(_: dict = Depends(require_admin)) -> dict:
    pdf_dir = settings.resolve_pdf_dir()
    vector_store = VectorStore()
    counts = {}
    for strategy in ["fixed_size", "structure_aware"]:
        try:
            counts[strategy] = vector_store.count(strategy)
        except Exception:
            counts[strategy] = 0
    with get_connection() as conn:
        versions = get_corpus_index_versions(conn)
    web_pages_count = len(load_web_records(settings.resolve_web_corpus_dir()))
    return {
        "pdf_dir": str(pdf_dir),
        "pdf_dir_exists": pdf_dir.exists(),
        "web_corpus_dir": str(settings.resolve_web_corpus_dir()),
        "web_pages_count": web_pages_count,
        "openai_key_configured": bool(settings.openai_api_key),
        "index_counts": counts,
        **versions,
    }


@router.post("/web-crawl", response_model=WebCrawlResponse)
def crawl_web_corpus(payload: WebCrawlRequest, _: dict = Depends(require_admin)) -> WebCrawlResponse:
    try:
        result = crawl_web_pages(
            seed_urls=payload.seed_urls,
            allowed_domains=payload.allowed_domains or None,
            max_pages=payload.max_pages,
            max_depth=payload.max_depth,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    if result.get("pages_saved", 0) > 0:
        corpus_version = bump_corpus_version()
    else:
        with get_connection() as conn:
            corpus_version = get_corpus_index_versions(conn)["corpus_version"]
    result["corpus_version"] = corpus_version
    return WebCrawlResponse(**result)


@router.get("/web-crawl/duplicates")
def web_corpus_duplicates(
    similarity_threshold: float = Query(default=0.86, ge=0.5, le=1.0),
    _: dict = Depends(require_admin),
) -> dict:
    return detect_duplicate_web_records(
        settings.resolve_web_corpus_dir(),
        similarity_threshold=similarity_threshold,
    )


@router.delete("/web-crawl")
def delete_web_corpus_item(url: str = Query(..., min_length=1), _: dict = Depends(require_admin)) -> dict:
    deleted_record = delete_web_record(url, settings.resolve_web_corpus_dir())
    if deleted_record is None:
        raise HTTPException(status_code=404, detail="Web corpus item not found")

    vector_deleted = VectorStore().delete_source_path(deleted_record.final_url)
    processed_deleted = _remove_processed_chunks_by_source_path(deleted_record.final_url)
    corpus_version = bump_corpus_version()
    index_version = bump_index_version()
    return {
        "ok": True,
        "deleted": {
            "file_name": deleted_record.title,
            "bytes": len(deleted_record.text.encode("utf-8")),
            "source_type": "web",
            "url": deleted_record.final_url,
        },
        "vector_chunks_deleted": vector_deleted,
        "processed_chunks_deleted": processed_deleted,
        "corpus_version": corpus_version,
        "index_version": index_version,
        "web_corpus_dir": str(settings.resolve_web_corpus_dir()),
    }


@router.post("/upload")
async def upload_documents(
    files: list[UploadFile] = File(...),
    _: dict = Depends(require_admin),
) -> dict:
    pdf_dir = settings.resolve_pdf_dir()
    pdf_dir.mkdir(parents=True, exist_ok=True)

    uploaded = []
    for file in files:
        file_name = Path(file.filename or "").name
        if not file_name:
            raise HTTPException(status_code=400, detail="Uploaded file is missing a name")
        if file_name.lower().endswith(".pdf") is False:
            raise HTTPException(status_code=400, detail=f"{file_name} is not a PDF")

        destination = pdf_dir / file_name
        content = await file.read()
        if not content.startswith(b"%PDF"):
            raise HTTPException(status_code=400, detail=f"{file_name} does not look like a valid PDF")
        destination.write_bytes(content)
        uploaded.append({"file_name": file_name, "bytes": len(content)})

    corpus_version = bump_corpus_version()
    return {"ok": True, "uploaded": uploaded, "pdf_dir": str(pdf_dir), "corpus_version": corpus_version}


@router.delete("/{file_name}")
def delete_document(file_name: str, _: dict = Depends(require_admin)) -> dict:
    path = _resolve_pdf_file(file_name)
    deleted_file = {"file_name": path.name, "bytes": path.stat().st_size}
    path.unlink()
    vector_deleted = VectorStore().delete_document(path.name)
    processed_deleted = _remove_processed_chunks(path.name)
    corpus_version = bump_corpus_version()
    index_version = bump_index_version()
    return {
        "ok": True,
        "deleted": deleted_file,
        "vector_chunks_deleted": vector_deleted,
        "processed_chunks_deleted": processed_deleted,
        "corpus_version": corpus_version,
        "index_version": index_version,
        "pdf_dir": str(settings.resolve_pdf_dir()),
    }
