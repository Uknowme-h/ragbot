"""Application settings loaded from environment / .env file."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
UPLOAD_DIR = PROJECT_ROOT / "data" / "uploads"
MAX_UPLOAD_BYTES = 20 * 1024 * 1024


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    groq_api_key: str = ""
    groq_llm_model: str = "meta-llama/llama-4-scout-17b-16e-instruct"
    groq_guard_model: str = "meta-llama/llama-prompt-guard-2-86m"
    groq_timeout_seconds: float = 30.0

    chroma_path: str = "./chroma_store"
    collection_name: str = "knowledge_base"

    similarity_threshold: float = 0.35
    top_k: int = 5
    chunk_size: int = 800
    chunk_overlap: int = 150
    embedding_model: str = "all-MiniLM-L6-v2"
    embed_batch_size: int = 32

    guard_injection_threshold: float = 0.5


@lru_cache
def get_settings() -> Settings:
    return Settings()
