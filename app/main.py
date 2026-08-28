"""FastAPI application entrypoint."""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.ingest import router as ingest_router
from app.api.query import router as query_router
from app.core.store import warmup
from app.utils.logger import log_event, log_warning


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        warmup()
    except Exception as exc:
        log_warning("warmup_failed", error=str(exc))
    log_event("api_started")
    yield
    log_event("api_stopped")


app = FastAPI(
    title="RAG Agent API",
    description=(
        "Retrieval-augmented generation over a local knowledge base, "
        "with prompt-injection screening, similarity fallback, and SSE streaming."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(query_router, prefix="/api/v1")
app.include_router(ingest_router, prefix="/api/v1")


@app.get("/health")
def health():
    return {"status": "ok"}
