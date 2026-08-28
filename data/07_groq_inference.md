# Groq Inference and Open Models

Groq runs open models on LPUs (Language Processing Units). For this
assessment the generation model is
`meta-llama/llama-4-scout-17b-16e-instruct` and the safety model is
`meta-llama/llama-prompt-guard-2-86m`. Both are available on the free
developer tier, gated by rate limits rather than per-token billing.

## Chat Completions API

Groq's HTTP API is OpenAI-compatible. The official `groq` Python SDK
exposes `chat.completions.create` with `stream=True` for token streaming.

```python
from groq import Groq

client = Groq(api_key=api_key, timeout=30.0)
stream = client.chat.completions.create(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    messages=[...],
    stream=True,
    max_tokens=1024,
)
for chunk in stream:
    delta = chunk.choices[0].delta.content
    if delta:
        print(delta, end="")
```

Set an explicit timeout (30 seconds here). On timeout the API layer returns
HTTP 503 with a `Retry-After` header instead of hanging the worker.

## Token usage

Non-streaming responses include `usage.prompt_tokens` and
`usage.completion_tokens`. Streaming responses on Groq may include a final
chunk with usage when `stream_options={"include_usage": True}` is set.
Log both numbers per request: they are the cost and latency budget for
the evaluation report.

`max_tokens=1024` caps completion length so a runaway answer cannot exhaust
the daily rate limit.

## Streaming and RAG

Retrieval must finish **before** the first generated token. The typical
SSE sequence is:

1. A `meta` event with `similarity_score` and `sources` (so the UI can
   show citations immediately).
2. Many `token` events.
3. `data: [DONE]`

Do not stream retrieved chunk text as if it were the answer. Only the
model's tokens go in `token` events.

## Rate limits and retries

The free tier is roughly tens of requests per minute. Bursting ingest-time
classification plus query-time generation can 429. Treat
`RateLimitError` like a timeout: HTTP 503, `Retry-After: 5`.

Prompt Guard calls add one extra request per query. Keep the payload tiny
(the question only) so guard latency stays well under generation latency.

## Model choice notes

Llama 4 Scout is a mixture-of-experts instruct model with a large context
window (128k on Groq). This service still sends only the top-5 chunks to
stay cheap and focused. A 128k window is not a reason to skip chunking.

Prompt Guard 2-86M is the wrong model for answering questions. It must never
be used as the generator. The 22M variant is faster and slightly less
accurate; 86M is the one wired into this API.
