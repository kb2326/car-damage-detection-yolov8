"""Claim event envelope and hash chaining."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from datetime import datetime
from enum import StrEnum
from typing import Any
from uuid import UUID

from pydantic import Field

from claimlens.domain import Frozen

GENESIS_HASH = "0" * 64


class ActorKind(StrEnum):
    SYSTEM = "system"
    AGENT = "agent"
    HUMAN = "human"


class Actor(Frozen):
    kind: ActorKind
    name: str


class ClaimEvent(Frozen):
    event_id: UUID
    claim_id: UUID
    seq: int = Field(ge=1)
    type: str
    payload: dict[str, Any]
    actor: Actor
    occurred_at: datetime
    schema_version: int = 1
    prev_hash: str
    hash: str


class ChainIntegrityError(Exception):
    """Raised when a claim's event log has been altered or is out of order."""

    def __init__(self, claim_id: UUID, seq: int, problem: str) -> None:
        super().__init__(f"claim {claim_id} event #{seq}: {problem}")
        self.claim_id = claim_id
        self.seq = seq


def compute_hash(event: ClaimEvent) -> str:
    """sha256(prev_hash + canonical JSON of every field except `hash`)."""
    body = event.model_dump(mode="json", exclude={"hash"})
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256((event.prev_hash + canonical).encode("utf-8")).hexdigest()


def verify_chain(events: Sequence[ClaimEvent]) -> None:
    prev = GENESIS_HASH
    for expected_seq, event in enumerate(events, start=1):
        if event.seq != expected_seq:
            raise ChainIntegrityError(
                event.claim_id, event.seq, f"expected sequence {expected_seq}"
            )
        if event.prev_hash != prev:
            raise ChainIntegrityError(
                event.claim_id, event.seq, "prev_hash does not match the previous event"
            )
        if compute_hash(event) != event.hash:
            raise ChainIntegrityError(
                event.claim_id, event.seq, "hash does not match event content"
            )
        prev = event.hash
