"""
Single GPU worker thread with a two-priority queue.

Priority 0 = interactive (single-frame, fast)
Priority 1 = propagation  (multi-frame, slow, cancellable)

FastAPI async handlers submit tasks and await concurrent.futures.Future results.
The bridge: loop.call_soon_threadsafe + future.set_result/set_exception.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import itertools
import queue
import threading
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass(order=True)
class _QueueItem:
    priority: int
    seq: int
    task: Callable = field(compare=False)
    future: concurrent.futures.Future = field(compare=False)


class GPUWorker:
    """
    Wraps all GPU work behind a single background thread.
    Callers submit callables and get back asyncio-compatible futures.
    """

    def __init__(self) -> None:
        self._q: queue.PriorityQueue[_QueueItem] = queue.PriorityQueue()
        self._seq = itertools.count()
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="gpu-worker", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        # Unblock queue.get() with a no-op sentinel
        sentinel = _QueueItem(
            priority=-1,
            seq=-1,
            task=lambda: None,
            future=concurrent.futures.Future(),
        )
        self._q.put(sentinel)
        if self._thread:
            self._thread.join(timeout=10)

    # ------------------------------------------------------------------
    # Submission API
    # ------------------------------------------------------------------

    async def submit(self, task: Callable, priority: int = 0) -> Any:
        """
        Schedule `task()` on the GPU thread and await its result.
        priority=0 → interactive, priority=1 → propagation.
        """
        cf_future: concurrent.futures.Future = concurrent.futures.Future()
        item = _QueueItem(
            priority=priority,
            seq=next(self._seq),
            task=task,
            future=cf_future,
        )
        self._q.put(item)
        return await asyncio.wrap_future(cf_future)

    @property
    def queue_depth(self) -> int:
        return self._q.qsize()

    # ------------------------------------------------------------------
    # Worker loop
    # ------------------------------------------------------------------

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                item = self._q.get(timeout=1.0)
            except queue.Empty:
                continue

            if self._stop_event.is_set():
                break

            try:
                result = item.task()
                # concurrent.futures.Future.set_result is thread-safe;
                # asyncio.wrap_future handles cross-thread notification internally.
                item.future.set_result(result)
            except Exception as exc:
                item.future.set_exception(exc)
            finally:
                self._q.task_done()
