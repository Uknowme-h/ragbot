"""Pydantic request and response schemas."""

from pydantic import BaseModel, Field, field_validator


class QueryRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        max_length=1024,
        description="Natural-language question (1–1024 characters after trim).",
        examples=["Why use chunk overlap when splitting documents?"],
    )
    stream: bool = Field(default=False, description="If true, stream tokens as SSE.")

    @field_validator("question")
    @classmethod
    def clean_question(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("question cannot be empty or whitespace")
        if "\x00" in cleaned:
            raise ValueError("question contains invalid characters")
        return cleaned


class Source(BaseModel):
    source_file: str
    page_number: int | None = None
    chunk_index: int | None = None
    similarity: float | None = None


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source] = Field(default_factory=list)
    similarity_score: float
    fallback_triggered: bool = False
    fallback_reason: str | None = None
    request_id: str
    web_search_mock: dict | None = None


class ErrorResponse(BaseModel):
    error: str
    code: str
    request_id: str


class IngestRequest(BaseModel):
    source: str = Field(
        default="./data",
        min_length=1,
        max_length=512,
        description="Project-relative file or directory to index.",
        examples=["./data"],
    )
    reset: bool = Field(default=False, description="Drop the existing collection first.")

    @field_validator("source")
    @classmethod
    def clean_source(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("source cannot be empty or whitespace")
        if "\x00" in cleaned:
            raise ValueError("source contains invalid characters")
        return cleaned


class IngestedFile(BaseModel):
    name: str
    md5: str | None = None
    chunks: int = 0
    ingested_at: str | None = None


class IngestResponse(BaseModel):
    request_id: str
    source: str
    files_seen: int
    files_ingested: int
    files_skipped: int
    chunks_upserted: int
    reset: bool = False
    ingest_ms: float


class DocumentListResponse(BaseModel):
    collection: str
    chunk_count: int
    files: list[IngestedFile]
