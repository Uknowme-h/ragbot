"""Process-wide embedding model and ChromaDB collection."""

from __future__ import annotations

from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from app.core.config import get_settings
from app.utils.logger import log_event

_embedder: SentenceTransformer | None = None
_chroma: chromadb.PersistentClient | None = None


def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        settings = get_settings()
        log_event("loading_embedding_model", model=settings.embedding_model)
        _embedder = SentenceTransformer(settings.embedding_model)
    return _embedder


def embed_texts(texts: list[str]) -> list[list[float]]:
    settings = get_settings()
    model = get_embedder()
    vectors = model.encode(
        texts,
        batch_size=settings.embed_batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return [v.tolist() for v in vectors]


def get_chroma() -> chromadb.PersistentClient:
    global _chroma
    if _chroma is None:
        settings = get_settings()
        Path(settings.chroma_path).mkdir(parents=True, exist_ok=True)
        _chroma = chromadb.PersistentClient(path=settings.chroma_path)
    return _chroma


def get_collection():
    settings = get_settings()
    client = get_chroma()
    return client.get_or_create_collection(
        name=settings.collection_name,
        metadata={"hnsw:space": "cosine"},
    )


def reset_collection() -> None:
    settings = get_settings()
    client = get_chroma()
    try:
        client.delete_collection(settings.collection_name)
    except Exception:
        pass
    get_collection()


def warmup() -> None:
    """Load the embedding model and open Chroma so the first request is warm."""
    get_embedder()
    get_collection()
    log_event("retrieval_stack_ready")
