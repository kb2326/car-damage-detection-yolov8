"""Runs the claims pipeline in the background, one claim at a time, on the claims worker."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from uuid import UUID

from claimlens.web.workers import Worker

log = logging.getLogger(__name__)


class ClaimRunner:
    def __init__(self, worker: Worker, process: Callable[[UUID], None]) -> None:
        self._worker = worker
        self._process = process
        self._lock = threading.Lock()
        self._running: set[UUID] = set()
        self._errors: dict[UUID, str] = {}

    def start(self, claim_id: UUID) -> bool:
        """Queue the claim; False if it is already queued or running (a double click)."""
        with self._lock:
            if claim_id in self._running:
                return False
            self._running.add(claim_id)
            self._errors.pop(claim_id, None)
        self._worker.submit(lambda: self._run(claim_id))
        return True

    def _run(self, claim_id: UUID) -> None:
        try:
            self._process(claim_id)
        except Exception as exc:  # the claim stays resumable from its log
            log.exception("processing claim %s stopped", claim_id)  # details stay server-side
            with self._lock:
                self._errors[claim_id] = f"Processing stopped unexpectedly ({type(exc).__name__})."
        finally:
            with self._lock:
                self._running.discard(claim_id)

    def running(self, claim_id: UUID) -> bool:
        with self._lock:
            return claim_id in self._running

    def error(self, claim_id: UUID) -> str | None:
        with self._lock:
            return self._errors.get(claim_id)
