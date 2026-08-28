# Python Asyncio Fundamentals

Python's `asyncio` library lets a single thread run many I/O-bound tasks concurrently
using an event loop, coroutines, and awaitable objects.

## Event loop

The event loop is the scheduler. It:

1. Collects ready callbacks and coroutines.
2. Runs them until they `await` something that is not yet finished.
3. Polls I/O (sockets, files, subprocesses) and wakes the waiting coroutines.

Start the loop with `asyncio.run(main())`. Do not nest `asyncio.run()` inside an
already-running loop; use `await` instead.

## Coroutines vs tasks

A coroutine function is declared with `async def`. Calling it returns a
coroutine object; it does **not** start running until you `await` it or wrap
it in a Task.

```python
async def fetch(url: str) -> str:
    await asyncio.sleep(0.1)
    return url

async def main():
    # Sequential: total wait is the sum of sleeps
    a = await fetch("a")
    b = await fetch("b")

    # Concurrent: both start immediately
    results = await asyncio.gather(fetch("a"), fetch("b"))
```

`asyncio.create_task(coro)` schedules the coroutine on the running loop and
returns a `Task`. Always keep a reference to tasks you create, or they may be
garbage-collected before they finish.

## Awaitables

You can `await` three kinds of objects:

- Coroutines produced by `async def`
- Tasks and Futures
- Objects that implement `__await__`

Blocking the event loop (CPU-heavy work, or sync I/O such as `time.sleep` or
`requests.get`) starves every other task. Offload CPU work with
`asyncio.to_thread()` (Python 3.9+) or a `ProcessPoolExecutor`.

## Cancellation and timeouts

```python
async def with_timeout():
    try:
        return await asyncio.wait_for(fetch("slow"), timeout=2.0)
    except TimeoutError:
        return None
```

Cancelled tasks raise `asyncio.CancelledError` at the next `await`. Catch it
only if you need to run cleanup; then re-raise so the task actually stops.

## Synchronization

Use `asyncio.Lock`, `asyncio.Semaphore`, `asyncio.Event`, and `asyncio.Queue`
instead of `threading` primitives. Mixing thread locks with the event loop
easily causes deadlocks.

## Common pitfalls

- Forgetting to `await` a coroutine (the runtime emits a "coroutine was never
  awaited" warning and the work never happens).
- Using `time.sleep()` instead of `asyncio.sleep()`.
- Sharing non-thread-safe objects across `to_thread` workers without a lock.
- Creating tasks in a loop without gathering or cancelling them on shutdown.

These patterns are the foundation for async web frameworks such as FastAPI,
which run an ASGI event loop and treat every request handler as a coroutine.
