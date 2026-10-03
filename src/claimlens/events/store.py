"""Append-only SQLite storage for claim events."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4

from claimlens.events.envelope import (
    GENESIS_HASH,
    Actor,
    ChainIntegrityError,
    ClaimEvent,
    compute_hash,
    verify_chain,
)
from claimlens.events.payloads import EventType, Payload

_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    claim_id TEXT NOT NULL,
    seq INTEGER NOT NULL,
    type TEXT NOT NULL,
    hash TEXT NOT NULL,
    data TEXT NOT NULL,
    PRIMARY KEY (claim_id, seq)
)
"""


def utc_now() -> datetime:
    return datetime.now(UTC)


class ClaimNotFoundError(LookupError):
    def __init__(self, claim_id: UUID) -> None:
        super().__init__(f"no events found for claim {claim_id}")
        self.claim_id = claim_id


class SQLiteEventStore:
    """Events are only ever inserted. Reads verify the hash chain before returning."""

    def __init__(
        self,
        path: Path | str,
        *,
        clock: Callable[[], datetime] = utc_now,
        new_id: Callable[[], UUID] = uuid4,
        read_only: bool = False,
    ) -> None:
        """`read_only` opens an existing store that can never be written (the public showcase)."""
        if read_only:
            if not Path(path).is_file():
                raise FileNotFoundError(f"no event store at {path}")
            uri = f"{Path(path).resolve().as_uri()}?mode=ro"
            self._conn = sqlite3.connect(uri, uri=True, isolation_level=None)
        else:
            if isinstance(path, Path):
                path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(path), isolation_level=None)
            self._conn.execute(_SCHEMA)
        self._clock = clock
        self._new_id = new_id

    def append(self, claim_id: UUID, payload: Payload, actor: Actor) -> ClaimEvent:
        conn = self._conn
        conn.execute("BEGIN IMMEDIATE")
        try:
            row = conn.execute(
                "SELECT seq, hash FROM events WHERE claim_id = ? ORDER BY seq DESC LIMIT 1",
                (str(claim_id),),
            ).fetchone()
            seq, prev_hash = (row[0] + 1, row[1]) if row else (1, GENESIS_HASH)
            draft = ClaimEvent(
                event_id=self._new_id(),
                claim_id=claim_id,
                seq=seq,
                type=payload.event_type.value,
                payload=payload.model_dump(mode="json"),
                actor=actor,
                occurred_at=self._clock(),
                prev_hash=prev_hash,
                hash="",
            )
            event = draft.model_copy(update={"hash": compute_hash(draft)})
            conn.execute(
                "INSERT INTO events (claim_id, seq, type, hash, data) VALUES (?, ?, ?, ?, ?)",
                (str(claim_id), seq, event.type, event.hash, event.model_dump_json()),
            )
            conn.execute("COMMIT")
        except BaseException:
            conn.execute("ROLLBACK")
            raise
        return event

    def load(self, claim_id: UUID) -> list[ClaimEvent]:
        rows = self._conn.execute(
            "SELECT seq, type, hash, data FROM events WHERE claim_id = ? ORDER BY seq",
            (str(claim_id),),
        ).fetchall()
        if not rows:
            raise ClaimNotFoundError(claim_id)
        events: list[ClaimEvent] = []
        for seq, event_type, event_hash, data in rows:
            event = ClaimEvent.model_validate_json(data)
            indexed = (event.claim_id, event.seq, event.type, event.hash)
            if indexed != (claim_id, seq, event_type, event_hash):
                raise ChainIntegrityError(claim_id, seq, "index columns do not match event data")
            events.append(event)
        verify_chain(events)
        return events

    def claim_ids(self) -> list[UUID]:
        rows = self._conn.execute("SELECT DISTINCT claim_id FROM events").fetchall()
        return [UUID(row[0]) for row in rows]

    def claims_with_photo(self, sha256: str) -> set[UUID]:
        rows = self._conn.execute(
            "SELECT DISTINCT claim_id FROM events "
            "WHERE json_extract(data, '$.type') = ? "
            "AND json_extract(data, '$.payload.sha256') = ?",
            (EventType.PHOTO_UPLOADED.value, sha256),
        ).fetchall()
        return {UUID(row[0]) for row in rows}

    def close(self) -> None:
        self._conn.close()
