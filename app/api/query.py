"""POST /api/v1/query — RAG query with optional SSE streaming."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator
from uuid import uuid4

from fastapi import APIRouter
from fastapi.responses import JSONResponse, StreamingResponse

from app.core.config import get_settings
from app.core.fallback import FALLBACK_ANSWER, insufficient_context_response
from app.core.guard import check_prompt
from app.core.llm import generate, generate_stream, is_groq_unavailable, log_llm_failure
from app.core.retriever import retrieve
from app.models.schemas import ErrorResponse, QueryRequest, QueryResponse
from app.utils.logger import log_event, log_warning

router = APIRouter()


def _sse(payload: dict | str) -> str:
    if isinstance(payload, str):
        return f"data: {payload}\n\n"
    return f"data: {json.dumps(payload)}\n\n"


def _error_response(status: int, *, error: str, code: str, request_id: str) -> JSONResponse:
    headers = {"Retry-After": "5"} if status == 503 else None
    return JSONResponse(
        status_code=status,
        content={"error": error, "code": code, "request_id": request_id},
        headers=headers,
    )


@router.post(
    "/query",
    response_model=QueryResponse,
    responses={
        400: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
async def query(body: QueryRequest):
    request_id = str(uuid4())
    settings = get_settings()
    started = time.perf_counter()
    timings: dict[str, float] = {}

    t0 = time.perf_counter()
    guard = await check_prompt(body.question)
    timings["guard_ms"] = round((time.perf_counter() - t0) * 1000, 2)

    if guard.blocked:
        log_event(
            "query_blocked",
            request_id=request_id,
            question_length=len(body.question),
            guard_ms=timings["guard_ms"],
            retrieval_ms=0,
            llm_ms=0,
            total_ms=round((time.perf_counter() - started) * 1000, 2),
            similarity_score=None,
            tokens_prompt=None,
            tokens_completion=None,
            fallback_triggered=False,
            guard_triggered=True,
        )
        return _error_response(
            400,
            error="Query flagged as potentially unsafe",
            code="PROMPT_INJECTION",
            request_id=request_id,
        )

    t1 = time.perf_counter()
    retrieval = retrieve(body.question)
    timings["retrieval_ms"] = round((time.perf_counter() - t1) * 1000, 2)
    sources = retrieval.sources()

    if retrieval.below_threshold:
        response = insufficient_context_response(
            request_id=request_id,
            question=body.question,
            similarity_score=retrieval.max_similarity,
            sources=sources,
        )
        log_event(
            "query_complete",
            request_id=request_id,
            question_length=len(body.question),
            guard_ms=timings["guard_ms"],
            retrieval_ms=timings["retrieval_ms"],
            llm_ms=0,
            total_ms=round((time.perf_counter() - started) * 1000, 2),
            similarity_score=round(retrieval.max_similarity, 4),
            tokens_prompt=0,
            tokens_completion=0,
            fallback_triggered=True,
            guard_triggered=False,
        )
        if body.stream:
            return StreamingResponse(
                _fallback_sse(response),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Request-Id": request_id},
            )
        return response

    if body.stream:
        return StreamingResponse(
            _token_sse(
                request_id=request_id,
                question=body.question,
                retrieval=retrieval,
                sources=[s.model_dump() for s in sources],
                timings=timings,
                started=started,
                threshold=settings.similarity_threshold,
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "X-Request-Id": request_id,
            },
        )

    t2 = time.perf_counter()
    try:
        llm = await generate(body.question, retrieval.chunks)
    except asyncio.CancelledError:
        log_warning("query_cancelled", request_id=request_id, phase="llm")
        raise
    except Exception as exc:
        log_llm_failure(exc)
        if is_groq_unavailable(exc):
            return _error_response(
                503,
                error="Upstream LLM provider timed out or is unavailable. Retry shortly.",
                code="LLM_UNAVAILABLE",
                request_id=request_id,
            )
        return _error_response(
            502,
            error="Generation failed",
            code="LLM_ERROR",
            request_id=request_id,
        )
    timings["llm_ms"] = round((time.perf_counter() - t2) * 1000, 2)

    log_event(
        "query_complete",
        request_id=request_id,
        question_length=len(body.question),
        guard_ms=timings["guard_ms"],
        retrieval_ms=timings["retrieval_ms"],
        llm_ms=timings["llm_ms"],
        total_ms=round((time.perf_counter() - started) * 1000, 2),
        similarity_score=round(retrieval.max_similarity, 4),
        tokens_prompt=llm.prompt_tokens,
        tokens_completion=llm.completion_tokens,
        fallback_triggered=False,
        guard_triggered=False,
    )
    return QueryResponse(
        answer=llm.answer,
        sources=sources,
        similarity_score=round(retrieval.max_similarity, 4),
        fallback_triggered=False,
        request_id=request_id,
    )


async def _fallback_sse(response: QueryResponse) -> AsyncIterator[str]:
    yield _sse(
        {
            "type": "meta",
            "request_id": response.request_id,
            "similarity_score": response.similarity_score,
            "sources": [s.model_dump() for s in response.sources],
            "fallback_triggered": True,
            "fallback_reason": response.fallback_reason,
            "web_search_mock": response.web_search_mock,
        }
    )
    yield _sse({"type": "token", "token": FALLBACK_ANSWER})
    yield _sse("[DONE]")


async def _token_sse(
    *,
    request_id: str,
    question: str,
    retrieval,
    sources: list[dict],
    timings: dict[str, float],
    started: float,
    threshold: float,
) -> AsyncIterator[str]:
    yield _sse(
        {
            "type": "meta",
            "request_id": request_id,
            "similarity_score": round(retrieval.max_similarity, 4),
            "sources": sources,
            "fallback_triggered": False,
            "threshold": threshold,
        }
    )
    t2 = time.perf_counter()
    prompt_tokens = None
    completion_tokens = None
    try:
        async for kind, payload in generate_stream(question, retrieval.chunks):
            if kind == "token" and isinstance(payload, str):
                yield _sse({"type": "token", "token": payload})
            elif kind == "usage" and payload is not None:
                prompt_tokens = payload.prompt_tokens
                completion_tokens = payload.completion_tokens
    except asyncio.CancelledError:
        log_warning(
            "stream_client_disconnect",
            request_id=request_id,
            truncated=True,
            llm_ms=round((time.perf_counter() - t2) * 1000, 2),
        )
        raise
    except Exception as exc:
        log_llm_failure(exc)
        if is_groq_unavailable(exc):
            yield _sse(
                {
                    "type": "error",
                    "code": "LLM_UNAVAILABLE",
                    "error": "Upstream LLM provider timed out or is unavailable.",
                    "request_id": request_id,
                }
            )
            yield _sse("[DONE]")
            return
        yield _sse(
            {
                "type": "error",
                "code": "LLM_ERROR",
                "error": "Generation failed",
                "request_id": request_id,
            }
        )
        yield _sse("[DONE]")
        return

    timings["llm_ms"] = round((time.perf_counter() - t2) * 1000, 2)
    log_event(
        "query_complete",
        request_id=request_id,
        question_length=len(question),
        guard_ms=timings["guard_ms"],
        retrieval_ms=timings["retrieval_ms"],
        llm_ms=timings["llm_ms"],
        total_ms=round((time.perf_counter() - started) * 1000, 2),
        similarity_score=round(retrieval.max_similarity, 4),
        tokens_prompt=prompt_tokens,
        tokens_completion=completion_tokens,
        fallback_triggered=False,
        guard_triggered=False,
        stream=True,
    )
    yield _sse("[DONE]")
