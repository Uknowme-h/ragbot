# FastAPI Request Lifecycle

FastAPI is an ASGI web framework built on Starlette and Pydantic. Every HTTP
request follows a predictable lifecycle from socket accept to response.

## Path from client to handler

1. **Uvicorn** (or another ASGI server) accepts the TCP connection and parses
   the HTTP request into an ASGI scope.
2. Middleware runs in the order it was added — outermost first on the way in,
   reverse order on the way out.
3. Routing matches the method and path (for example `POST /api/v1/query`)
   and selects the path operation function.
4. **Dependency injection** resolves `Depends()` parameters, including nested
   dependencies and `yield` dependencies that act as context managers.
5. **Pydantic validation** parses the request body, query params, and headers
   into models. Invalid input short-circuits with HTTP 422 and a structured
   error list. It never reaches the handler.
6. The path function runs. If it is `async def`, FastAPI awaits it directly
   on the event loop. If it is a plain `def`, FastAPI runs it in a threadpool
   so it does not block other requests.
7. The return value is serialized (JSON by default, or a `Response` subclass
   such as `StreamingResponse`).
8. Response middleware and exception handlers wrap the result. Uncaught
   exceptions become HTTP 500 unless an exception handler is registered.

## StreamingResponse and Server-Sent Events

For token-by-token LLM output, return a `StreamingResponse` whose body is an
async generator:

```python
from fastapi.responses import StreamingResponse

async def events():
    yield "data: {\"token\": \"Hello\"}\n\n"
    yield "data: [DONE]\n\n"

return StreamingResponse(events(), media_type="text/event-stream")
```

SSE requires `text/event-stream`, one blank line after each event, and
`data:` prefixed payloads. Disable response buffering in reverse proxies
(`X-Accel-Buffering: no` for nginx) or the client will see one giant chunk.

If the client disconnects, FastAPI raises `asyncio.CancelledError` inside the
generator. Catch it to cancel downstream work (for example an LLM stream)
and then re-raise.

## Lifespan events

Use `@asynccontextmanager` with `lifespan=` on the `FastAPI` constructor to
load models, open a vector database, and close HTTP clients:

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.ready = True
    yield
    # shutdown: close clients, persist state
```

Startup work that loads a SentenceTransformer or ChromaDB client belongs
here so the first user request is not delayed by a cold model load.

## Validation limits

Pydantic `Field(min_length=..., max_length=...)` is the first guardrail on
user text. Empty bodies and oversized prompts fail at 422 before any
embedding or LLM call. Keep these limits aligned with the Prompt Guard
context window (512 tokens for Llama Prompt Guard 2).
