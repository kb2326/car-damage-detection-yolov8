"""Writing memory: only the workflow (and human reviews) call this, never an agent."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from uuid import UUID

from claimlens.events.envelope import Actor, ActorKind
from claimlens.events.payloads import MemoryWritten, StageFailed
from claimlens.events.projection import fold
from claimlens.events.store import SQLiteEventStore
from claimlens.memory.index import ClaimMemory
from claimlens.memory.records import MemoryRecord, build_record

MEMORY_WRITER = Actor(kind=ActorKind.SYSTEM, name="workflow")


def needs_memory(store: SQLiteEventStore, claim_id: UUID) -> bool:
    """True when the latest decision or review has not been written to memory yet."""
    last_written = 0
    last_change = 0
    forgotten = False
    for event in store.load(claim_id):
        if event.type in ("RouteDecided", "HumanReviewed"):
            last_change = event.seq
        elif event.type == "MemoryWritten":
            last_written = event.seq
        elif event.type == "MemoryForgotten":
            forgotten = True
    return last_change > last_written and not forgotten


def remember(
    store: SQLiteEventStore,
    claim_id: UUID,
    memory: ClaimMemory,
    photo_path: Callable[[str], Path],
) -> None:
    """Write the claim to memory and record MemoryWritten. A failure is recorded, never raised:
    memory must never change or block a decision."""
    events = store.load(claim_id)
    try:
        record = build_record(fold(events), events, photo_path)
        memory.upsert(record)
    except Exception as exc:
        store.append(
            claim_id,
            StageFailed(stage="memory", error=f"{type(exc).__name__}: {exc}"),
            MEMORY_WRITER,
        )
        return
    store.append(
        claim_id,
        MemoryWritten(
            record_id=record.claim_id,
            fields=tuple(MemoryRecord.model_fields),
            source_seq=record.source_seq,
        ),
        MEMORY_WRITER,
    )
