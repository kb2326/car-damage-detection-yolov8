"""What the routers need, built once at start-up (the real wiring is in web/serve.py)."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from uuid import UUID

from claimlens.blobs import BlobStore
from claimlens.events.envelope import ClaimEvent
from claimlens.events.projection import ClaimState
from claimlens.events.store import SQLiteEventStore
from claimlens.intake_agent.session import IntakeSessions
from claimlens.memory.index import ClaimMemory
from claimlens.memory.records import build_record
from claimlens.web.runner import ClaimRunner
from claimlens.web.schemas import SimilarView
from claimlens.web.settings import WebSettings
from claimlens.web.workers import Worker
from claimlens.workflow import PipelineDeps, process_claim


def pipeline_processor(
    db_path: Path, deps_for: Callable[[SQLiteEventStore], PipelineDeps]
) -> Callable[[UUID], None]:
    """Process one claim with its own store, opened and closed on the calling (worker) thread."""

    def process(claim_id: UUID) -> None:
        store = SQLiteEventStore(db_path)
        try:
            process_claim(claim_id, deps_for(store))
        finally:
            store.close()

    return process


@dataclass
class WebServices:
    db_path: Path
    blobs: BlobStore
    settings: WebSettings
    runner: ClaimRunner
    intake_worker: Worker
    intake: Callable[[], IntakeSessions] | None
    upload_dir: Path
    memory: ClaimMemory | None = None
    read_only: bool = False  # the showcase: the store can never be written
    transcript: list[dict[str, str]] = field(default_factory=list)  # the showcase's recorded chat
    _sessions: list[IntakeSessions] = field(default_factory=list)

    @contextmanager
    def open_store(self) -> Iterator[SQLiteEventStore]:
        """A store for one request: open and close it inside the endpoint body, never in a
        dependency (FastAPI may run a dependency and its endpoint on different threads)."""
        store = SQLiteEventStore(self.db_path, read_only=self.read_only)
        try:
            yield store
        finally:
            store.close()

    def sessions(self) -> IntakeSessions:
        """The intake sessions, built on first use. Call only on the intake worker."""
        if self.intake is None:
            raise LookupError("the intake chat is not available")
        if not self._sessions:
            self._sessions.append(self.intake())
        return self._sessions[0]

    def similar(self, state: ClaimState, events: Sequence[ClaimEvent]) -> list[SimilarView]:
        """Similar claims in memory now (read-only). Errors show as none, never break the page."""
        if self.memory is None or not state.photos:
            return []
        try:
            mine = build_record(state, events, self.blobs.path, provisional=True)
            return [
                SimilarView(
                    claim_id=s.claim_id,
                    reason=s.reason,
                    distance=s.distance,
                    route=s.route,
                    damage=s.damage,
                )
                for s in self.memory.similar(mine)
            ]
        except Exception:
            return []
