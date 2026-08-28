"""Prompt-injection pre-filter using Llama Prompt Guard 2 plus local heuristics."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from groq import APIError, APITimeoutError, RateLimitError

from app.core.clients import get_async_groq
from app.core.config import get_settings
from app.utils.logger import log_event, log_warning

# Guard model context is 512 tokens; keep a conservative character cap.
GUARD_MAX_CHARS = 1500

INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions", re.I),
    re.compile(r"disregard\s+(your\s+)?(system\s+)?prompt", re.I),
    re.compile(r"you\s+are\s+now\s+(dan|a\s+different)", re.I),
    re.compile(r"reveal\s+(your\s+)?(system\s+)?prompt", re.I),
    re.compile(r"print\s+(your\s+)?hidden\s+instructions", re.I),
    re.compile(r"jailbreak", re.I),
    re.compile(r"developer\s+mode\s+enabled", re.I),
    re.compile(r"override\s+(the\s+)?safety", re.I),
]


@dataclass
class GuardResult:
    blocked: bool
    score: float | None
    label: str
    source: str  # "guard_model" | "heuristic" | "error"
    raw: str | None = None


def heuristic_injection(question: str) -> bool:
    return any(pattern.search(question) for pattern in INJECTION_PATTERNS)


def _parse_guard_output(content: str) -> tuple[bool, float | None, str]:
    text = (content or "").strip()
    if not text:
        return False, None, "empty"

    upper = text.upper()
    if any(token in upper for token in ("INJECTION", "JAILBREAK", "MALICIOUS")):
        if "BENIGN" in upper and "INJECTION" not in upper:
            return False, 0.0, "benign"
        return True, 1.0, "injection"

    if upper in {"BENIGN", "SAFE", "0", "FALSE", "LABEL_0"}:
        return False, 0.0, "benign"
    if upper in {"1", "TRUE", "LABEL_1"}:
        return True, 1.0, "injection"

    try:
        payload = json.loads(text)
        if isinstance(payload, dict):
            label = str(payload.get("label", payload.get("class", ""))).upper()
            score = payload.get("score", payload.get("score_injection"))
            if isinstance(score, (int, float)):
                blocked = float(score) >= get_settings().guard_injection_threshold
                return blocked, float(score), label or "json"
            if "INJECTION" in label or "JAILBREAK" in label:
                return True, 1.0, label
            if "BENIGN" in label:
                return False, 0.0, label
    except json.JSONDecodeError:
        pass

    try:
        score = float(text.split()[0])
        blocked = score >= get_settings().guard_injection_threshold
        return blocked, score, "score"
    except ValueError:
        return False, None, "unparsed"


async def check_prompt(question: str) -> GuardResult:
    settings = get_settings()
    truncated = question[:GUARD_MAX_CHARS]

    if heuristic_injection(question):
        log_event("prompt_guard_blocked", source="heuristic")
        return GuardResult(
            blocked=True,
            score=1.0,
            label="heuristic_injection",
            source="heuristic",
        )

    if not settings.groq_api_key or settings.groq_api_key.startswith("your_"):
        log_warning("guard_skipped_missing_api_key")
        return GuardResult(blocked=False, score=None, label="skipped", source="error")

    try:
        client = get_async_groq()
        response = await client.chat.completions.create(
            model=settings.groq_guard_model,
            messages=[{"role": "user", "content": truncated}],
            max_tokens=32,
            temperature=0,
        )
        raw = (response.choices[0].message.content or "").strip()
        blocked, score, label = _parse_guard_output(raw)
        log_event(
            "prompt_guard_result",
            blocked=blocked,
            score=score,
            label=label,
            source="guard_model",
        )
        return GuardResult(
            blocked=blocked,
            score=score,
            label=label,
            source="guard_model",
            raw=raw,
        )
    except (APITimeoutError, RateLimitError, APIError) as exc:
        log_warning("guard_api_failed", error=str(exc), type=type(exc).__name__)
        return GuardResult(
            blocked=False,
            score=None,
            label="guard_unavailable",
            source="error",
            raw=str(exc),
        )
