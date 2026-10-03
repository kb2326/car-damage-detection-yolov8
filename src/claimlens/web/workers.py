"""One dedicated thread per area. SQLite connections (the event store, the LLM budget and cache)
must stay on the thread that opened them, so everything that holds one runs here."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor


class Worker:
    def __init__(self, name: str, *, inline: bool = False) -> None:
        """`inline` runs each call at once on the caller's thread (tests)."""
        self._pool = None if inline else ThreadPoolExecutor(max_workers=1, thread_name_prefix=name)

    def submit[T](self, fn: Callable[[], T]) -> Future[T]:
        if self._pool is not None:
            return self._pool.submit(fn)
        future: Future[T] = Future()
        try:
            future.set_result(fn())
        except Exception as exc:
            future.set_exception(exc)
        return future

    def call[T](self, fn: Callable[[], T]) -> T:
        return self.submit(fn).result()

    def shutdown(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=False, cancel_futures=True)
