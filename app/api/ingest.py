"""Ingest documents into the knowledge base via HTTP."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, File, Query, UploadFile
from fastapi.responses import JSONResponse

from app.core.config import MAX_UPLOAD_BYTES, UPLOAD_DIR
from app.core.ingestion import SUPPORTED_SUFFIXES, ingest_source, list_indexed_documents
from app.models.schemas import (
    DocumentListResponse,
    ErrorResponse,
    IngestResponse,
    IngestedFile,
)
from app.utils.logger import log_event, log_warning

MAX_FILENAME_CHARS = 200
TEXT_SUFFIXES = {".md", ".txt", ".markdown"}


def _validate_upload(name: str, suffix: str, data: bytes, request_id: str) -> JSONResponse | None:
    if not name or name in {".", ".."}:
        return _error(400, "Filename is missing or invalid", "INVALID_FILENAME", request_id)
    if len(name) > MAX_FILENAME_CHARS:
        return _error(400, "Filename is too long", "INVALID_FILENAME", request_id)
    if suffix not in SUPPORTED_SUFFIXES:
        return _error(
            400,
            f"Unsupported file type: {name}. Allowed: {sorted(SUPPORTED_SUFFIXES)}",
            "UNSUPPORTED_TYPE",
            request_id,
        )
    if not data:
        return _error(400, "Uploaded file is empty", "EMPTY_FILE", request_id)
    if len(data) > MAX_UPLOAD_BYTES:
        return _error(
            400,
            f"{name} exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB upload limit",
            "FILE_TOO_LARGE",
            request_id,
        )
    if suffix == ".pdf" and not data.startswith(b"%PDF"):
        return _error(400, "File extension is .pdf but the content is not a PDF", "INVALID_PDF", request_id)
    if suffix in TEXT_SUFFIXES:
        try:
            data.decode("utf-8")
        except UnicodeDecodeError:
            return _error(400, "Text files must be valid UTF-8", "INVALID_TEXT", request_id)
    return None


router = APIRouter()


# Path-based ingest is disabled so the HTTP API never opens server filesystem
# paths. Bulk-index ./data with: python scripts/ingest.py --source ./data
#
# def _resolve_source(source: str) -> Path:
#     raw = Path(source)
#     path = raw.resolve() if raw.is_absolute() else (PROJECT_ROOT / raw).resolve()
#     try:
#         path.relative_to(PROJECT_ROOT)
#     except ValueError:
#         raise ValueError("source path must stay inside the project directory") from None
#     return path


def _to_response(request_id: str, summary: dict, reset: bool, ingest_ms: float) -> IngestResponse:
    return IngestResponse(
        request_id=request_id,
        source=summary["source"],
        files_seen=summary["files_seen"],
        files_ingested=summary["files_ingested"],
        files_skipped=summary["files_skipped"],
        chunks_upserted=summary["chunks_upserted"],
        reset=reset,
        ingest_ms=ingest_ms,
    )


# @router.post(
#     "/ingest",
#     response_model=IngestResponse,
#     responses={400: {"model": ErrorResponse}, 404: {"model": ErrorResponse}},
# )
# async def ingest(body: IngestRequest):
#     """Index files from a project-relative path (default `./data`)."""
#     request_id = str(uuid4())
#     try:
#         source = _resolve_source(body.source)
#     except ValueError as exc:
#         return _error(400, str(exc), "INVALID_SOURCE", request_id)
#
#     if not source.exists():
#         return _error(404, f"Source not found: {source}", "SOURCE_NOT_FOUND", request_id)
#
#     started = time.perf_counter()
#     try:
#         summary = await asyncio.to_thread(ingest_source, source, reset=body.reset)
#     except Exception as exc:
#         log_warning("ingest_api_failed", error=str(exc), request_id=request_id)
#         return _error(500, "Ingestion failed", "INGEST_FAILED", request_id)
#
#     ingest_ms = round((time.perf_counter() - started) * 1000, 2)
#     log_event("ingest_api_complete", request_id=request_id, ingest_ms=ingest_ms, **summary)
#     return _to_response(request_id, summary, body.reset, ingest_ms)


@router.post(
    "/ingest/upload",
    response_model=IngestResponse,
    responses={400: {"model": ErrorResponse}},
)
async def ingest_upload(
    file: UploadFile = File(
        ...,
        description="One PDF, Markdown, or .txt file. Repeat for more files, or use the CLI for a folder.",
    ),
    reset: bool = Query(
        False,
        description="If true, drop the existing collection before indexing this file.",
    ),
):
    """Upload a document, save it under data/uploads, and index it.

    Swagger: use the **file** picker (Choose File), then Execute.
    For the whole sample corpus: python scripts/ingest.py --source ./data
    """
    request_id = str(uuid4())
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    name = Path(file.filename or "").name
    suffix = Path(name).suffix.lower()
    data = await file.read()
    invalid = _validate_upload(name, suffix, data, request_id)
    if invalid is not None:
        return invalid

    dest = UPLOAD_DIR / name
    dest.write_bytes(data)

    started = time.perf_counter()
    try:
        summary = await asyncio.to_thread(ingest_source, dest, reset=reset)
    except Exception as exc:
        log_warning("ingest_upload_failed", error=str(exc), request_id=request_id)
        return _error(500, "Ingestion failed", "INGEST_FAILED", request_id)

    ingest_ms = round((time.perf_counter() - started) * 1000, 2)
    log_event(
        "ingest_upload_complete",
        request_id=request_id,
        files=[dest.name],
        ingest_ms=ingest_ms,
        **summary,
    )
    return _to_response(request_id, summary, reset, ingest_ms)


@router.get("/documents", response_model=DocumentListResponse)
async def list_documents():
    """List indexed source files and chunk counts."""
    payload = list_indexed_documents()
    return DocumentListResponse(
        collection=payload["collection"],
        chunk_count=payload["chunk_count"],
        files=[IngestedFile(**item) for item in payload["files"]],
    )


def _error(status: int, error: str, code: str, request_id: str) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"error": error, "code": code, "request_id": request_id},
    )
