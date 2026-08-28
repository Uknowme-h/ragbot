"""PDF / Markdown parsing, cleaning, chunking, embedding, and Chroma storage."""

from __future__ import annotations

import hashlib
import io
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import get_settings
from app.core.store import embed_texts, get_collection, reset_collection
from app.utils.logger import log_error, log_event, log_warning

SUPPORTED_SUFFIXES = {".pdf", ".md", ".txt", ".markdown"}
MANIFEST_NAME = "ingest_manifest.json"


@dataclass
class PageText:
    page_number: int
    text: str
    ocr_used: bool = False


@dataclass
class Chunk:
    text: str
    source_file: str
    page_number: int
    chunk_index: int
    file_hash: str


def _manifest_path() -> Path:
    settings = get_settings()
    return Path(settings.chroma_path) / MANIFEST_NAME


def load_manifest() -> dict:
    path = _manifest_path()
    if not path.exists():
        return {"files": {}}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"files": {}}


def save_manifest(manifest: dict) -> None:
    path = _manifest_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def file_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def clean_text(text: str) -> str:
    text = text.replace("\x00", "")
    raw_lines = text.splitlines()
    cleaned: list[str] = []
    i = 0
    while i < len(raw_lines):
        line = raw_lines[i].rstrip()
        stripped = line.strip()
        if re.fullmatch(r"\d+", stripped):
            i += 1
            continue
        if stripped.endswith("-") and i + 1 < len(raw_lines):
            nxt = raw_lines[i + 1].lstrip()
            if nxt and nxt[0].islower():
                line = stripped[:-1] + nxt
                i += 2
                cleaned.append(line)
                continue
        cleaned.append(line)
        i += 1
    text = "\n".join(cleaned)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _ocr_page(page) -> str:
    try:
        import pytesseract
        from PIL import Image
    except ImportError:
        log_warning("ocr_unavailable", reason="pytesseract_or_pillow_not_installed")
        return ""

    try:
        pix = page.get_pixmap(dpi=200)
        image = Image.open(io.BytesIO(pix.tobytes("png")))
        return pytesseract.image_to_string(image)
    except Exception as exc:
        log_warning("ocr_failed", error=str(exc))
        return ""


def extract_pdf_pages(path: Path) -> Iterator[PageText]:
    import fitz

    try:
        doc = fitz.open(path)
    except Exception as exc:
        log_error("pdf_open_failed", path=str(path), error=str(exc))
        return

    try:
        if doc.is_encrypted:
            log_warning("pdf_encrypted_skipped", path=str(path))
            return
        for index, page in enumerate(doc, start=1):
            blocks = page.get_text("blocks")
            text_blocks = [b for b in blocks if len(b) >= 5]
            text_blocks.sort(key=lambda b: (round(float(b[1]), 1), float(b[0])))
            text = "\n".join(str(b[4]) for b in text_blocks if str(b[4]).strip())
            ocr_used = False
            if not text.strip():
                ocr_text = _ocr_page(page)
                if ocr_text.strip():
                    text = ocr_text
                    ocr_used = True
                    log_event("pdf_page_ocr", path=str(path), page=index)
                else:
                    log_warning("pdf_page_empty", path=str(path), page=index)
                    continue
            yield PageText(page_number=index, text=text, ocr_used=ocr_used)
    finally:
        doc.close()


def extract_text_pages(path: Path) -> Iterator[PageText]:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = path.read_text(encoding="utf-8", errors="replace")
    if text.strip():
        yield PageText(page_number=1, text=text)


def extract_pages(path: Path) -> Iterator[PageText]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        yield from extract_pdf_pages(path)
    elif suffix in {".md", ".txt", ".markdown"}:
        yield from extract_text_pages(path)
    else:
        log_warning("unsupported_file_skipped", path=str(path))


def _splitter() -> RecursiveCharacterTextSplitter:
    settings = get_settings()
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def chunk_pages(path: Path, pages: list[PageText], file_hash: str) -> list[Chunk]:
    splitter = _splitter()
    chunks: list[Chunk] = []
    chunk_index = 0
    for page in pages:
        cleaned = clean_text(page.text)
        if not cleaned:
            continue
        for piece in splitter.split_text(cleaned):
            piece = piece.strip()
            if not piece:
                continue
            chunks.append(
                Chunk(
                    text=piece,
                    source_file=path.name,
                    page_number=page.page_number,
                    chunk_index=chunk_index,
                    file_hash=file_hash,
                )
            )
            chunk_index += 1
    return chunks


def _delete_source_chunks(source_file: str) -> None:
    collection = get_collection()
    try:
        collection.delete(where={"source_file": {"$eq": source_file}})
    except Exception as exc:
        log_warning("delete_source_failed", source_file=source_file, error=str(exc))


def _upsert_chunks(chunks: list[Chunk]) -> int:
    if not chunks:
        return 0
    collection = get_collection()
    ids = [f"{c.file_hash}:{c.chunk_index}" for c in chunks]
    documents = [c.text for c in chunks]
    metadatas = [
        {
            "source_file": c.source_file,
            "page_number": c.page_number,
            "chunk_index": c.chunk_index,
            "file_hash": c.file_hash,
        }
        for c in chunks
    ]
    embeddings = embed_texts(documents)
    collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings,
    )
    return len(chunks)


def ingest_file(path: Path, manifest: dict, *, force: bool = False) -> int:
    file_hash = file_md5(path)
    previous = manifest.get("files", {}).get(path.name)
    if previous and previous.get("md5") == file_hash and not force:
        log_event("ingest_skipped_duplicate", path=str(path), md5=file_hash)
        return 0

    pages = list(extract_pages(path))
    chunks = chunk_pages(path, pages, file_hash)
    if previous:
        _delete_source_chunks(path.name)

    stored = _upsert_chunks(chunks)
    manifest.setdefault("files", {})[path.name] = {
        "md5": file_hash,
        "chunks": stored,
        "path": str(path),
        "ingested_at": datetime.now(timezone.utc).isoformat(),
    }
    log_event("file_ingested", path=str(path), chunks=stored, md5=file_hash)
    return stored


def iter_source_files(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    files = [
        p
        for p in sorted(source.rglob("*"))
        if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
    ]
    return files


def ingest_source(source: str | Path, *, reset: bool = False) -> dict:
    source_path = Path(source).resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"Source not found: {source_path}")

    if reset:
        reset_collection()
        save_manifest({"files": {}})
        log_event("collection_reset")

    manifest = load_manifest()
    files = iter_source_files(source_path)
    total_chunks = 0
    ingested_files = 0
    skipped = 0

    for path in files:
        try:
            stored = ingest_file(path, manifest)
        except Exception as exc:
            log_error("ingest_file_failed", path=str(path), error=str(exc))
            continue
        if stored:
            ingested_files += 1
            total_chunks += stored
        else:
            skipped += 1

    save_manifest(manifest)
    summary = {
        "source": str(source_path),
        "files_seen": len(files),
        "files_ingested": ingested_files,
        "files_skipped": skipped,
        "chunks_upserted": total_chunks,
    }
    log_event("ingest_complete", **summary)
    return summary


def list_indexed_documents() -> dict:
    """Return manifest files plus the live collection chunk count."""
    settings = get_settings()
    manifest = load_manifest()
    files = []
    for name, meta in sorted(manifest.get("files", {}).items()):
        files.append(
            {
                "name": name,
                "md5": meta.get("md5"),
                "chunks": meta.get("chunks", 0),
                "ingested_at": meta.get("ingested_at"),
            }
        )
    try:
        chunk_count = get_collection().count()
    except Exception:
        chunk_count = 0
    return {
        "collection": settings.collection_name,
        "chunk_count": chunk_count,
        "files": files,
    }
