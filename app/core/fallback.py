"""Structured fallback when retrieval confidence is too low."""

from __future__ import annotations

from app.core.config import get_settings
from app.models.schemas import QueryResponse, Source

FALLBACK_ANSWER = (
    "I don't have enough information in my knowledge base to answer this "
    "question confidently. No retrieved passage crossed the similarity threshold, "
    "so I will not guess. Try rephrasing, or ask about topics covered in the "
    "ingested documents (asyncio, FastAPI, RAG, embeddings, prompt injection, "
    "chunking, Groq, or PDF parsing)."
)


def mock_web_search(question: str) -> dict:
    """Stand-in for a production web-search tool routed on low confidence."""
    return {
        "tool": "web_search_mock",
        "status": "not_executed",
        "query": question,
        "message": (
            "Internal knowledge base had insufficient context. "
            "In production this would be routed to a web search tool."
        ),
    }


def insufficient_context_response(
    *,
    request_id: str,
    question: str,
    similarity_score: float,
    sources: list[Source] | None = None,
) -> QueryResponse:
    settings = get_settings()
    return QueryResponse(
        answer=FALLBACK_ANSWER,
        sources=sources or [],
        similarity_score=round(similarity_score, 4),
        fallback_triggered=True,
        fallback_reason=f"similarity_below_threshold ({settings.similarity_threshold})",
        request_id=request_id,
        web_search_mock=mock_web_search(question),
    )
