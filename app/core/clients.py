"""Shared Groq clients with a 30s timeout."""

from functools import lru_cache

from groq import AsyncGroq, Groq

from app.core.config import get_settings


@lru_cache
def get_groq() -> Groq:
    settings = get_settings()
    return Groq(api_key=settings.groq_api_key, timeout=settings.groq_timeout_seconds)


@lru_cache
def get_async_groq() -> AsyncGroq:
    settings = get_settings()
    return AsyncGroq(api_key=settings.groq_api_key, timeout=settings.groq_timeout_seconds)
