"""ChromaDB retrieval with cosine-distance to similarity conversion."""

from __future__ import annotations

from dataclasses import dataclass, field

from app.core.config import get_settings
from app.core.store import embed_texts, get_collection
from app.models.schemas import Source


@dataclass
class RetrievedChunk:
    text: str
    source_file: str
    page_number: int | None
    chunk_index: int | None
    distance: float
    similarity: float


@dataclass
class RetrievalResult:
    chunks: list[RetrievedChunk] = field(default_factory=list)
    max_similarity: float = 0.0
    below_threshold: bool = True

    def sources(self) -> list[Source]:
        return [
            Source(
                source_file=c.source_file,
                page_number=c.page_number,
                chunk_index=c.chunk_index,
                similarity=round(c.similarity, 4),
            )
            for c in self.chunks
        ]


def distance_to_similarity(distance: float) -> float:
    """Chroma cosine distance is 1 - cosine_similarity, range [0, 2]."""
    return 1.0 - (float(distance) / 2.0)


def retrieve(question: str, top_k: int | None = None) -> RetrievalResult:
    settings = get_settings()
    k = top_k or settings.top_k
    collection = get_collection()
    count = collection.count()
    if count == 0:
        return RetrievalResult(chunks=[], max_similarity=0.0, below_threshold=True)

    query_embedding = embed_texts([question])[0]
    raw = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(k, count),
        include=["documents", "metadatas", "distances"],
    )

    documents = (raw.get("documents") or [[]])[0]
    metadatas = (raw.get("metadatas") or [[]])[0]
    distances = (raw.get("distances") or [[]])[0]

    chunks: list[RetrievedChunk] = []
    for doc, meta, distance in zip(documents, metadatas, distances):
        meta = meta or {}
        similarity = distance_to_similarity(distance)
        chunks.append(
            RetrievedChunk(
                text=doc or "",
                source_file=str(meta.get("source_file", "unknown")),
                page_number=meta.get("page_number"),
                chunk_index=meta.get("chunk_index"),
                distance=float(distance),
                similarity=similarity,
            )
        )

    max_sim = max((c.similarity for c in chunks), default=0.0)
    return RetrievalResult(
        chunks=chunks,
        max_similarity=max_sim,
        below_threshold=max_sim < settings.similarity_threshold,
    )
