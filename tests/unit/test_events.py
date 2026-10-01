from datetime import UTC, datetime
from uuid import UUID

import pytest

from claimlens.events.envelope import (
    GENESIS_HASH,
    Actor,
    ActorKind,
    ChainIntegrityError,
    ClaimEvent,
    compute_hash,
    verify_chain,
)
from claimlens.events.payloads import (
    PAYLOAD_TYPES,
    ClaimReported,
    EventType,
    Payload,
    PhotoAccepted,
    parse_payload,
)

CLAIM = UUID(int=1)
ACTOR = Actor(kind=ActorKind.SYSTEM, name="test")


def _event(seq: int, prev_hash: str, payload: Payload) -> ClaimEvent:
    draft = ClaimEvent(
        event_id=UUID(int=100 + seq),
        claim_id=CLAIM,
        seq=seq,
        type=payload.event_type.value,
        payload=payload.model_dump(mode="json"),
        actor=ACTOR,
        occurred_at=datetime(2026, 10, 8, 12, 0, seq, tzinfo=UTC),
        prev_hash=prev_hash,
        hash="",
    )
    return draft.model_copy(update={"hash": compute_hash(draft)})


def _chain() -> list[ClaimEvent]:
    first = _event(1, GENESIS_HASH, ClaimReported(policy_id="P-1001", description="scrape"))
    second = _event(2, first.hash, PhotoAccepted(photo_id="p1", width=640, height=640))
    return [first, second]


def test_hash_is_deterministic_sha256() -> None:
    first = _chain()[0]
    assert compute_hash(first) == first.hash
    assert len(first.hash) == 64


def test_valid_chain_passes() -> None:
    verify_chain(_chain())


def test_edited_payload_breaks_chain() -> None:
    first, second = _chain()
    tampered = first.model_copy(update={"payload": {"policy_id": "P-9999", "description": "x"}})
    with pytest.raises(ChainIntegrityError, match="hash does not match"):
        verify_chain([tampered, second])


def test_rehashed_edit_is_caught_by_next_event() -> None:
    first, second = _chain()
    edited = first.model_copy(update={"payload": {"policy_id": "P-9999", "description": "x"}})
    rehashed = edited.model_copy(update={"hash": compute_hash(edited)})
    with pytest.raises(ChainIntegrityError, match="prev_hash"):
        verify_chain([rehashed, second])


def test_gap_in_sequence_is_rejected() -> None:
    first, second = _chain()
    with pytest.raises(ChainIntegrityError, match="expected sequence 2"):
        verify_chain([first, second.model_copy(update={"seq": 3})])


def test_parse_payload_round_trips() -> None:
    assert parse_payload(_chain()[0]) == ClaimReported(policy_id="P-1001", description="scrape")


def test_parse_payload_rejects_unknown_type() -> None:
    mystery = _chain()[0].model_copy(update={"type": "Mystery"})
    with pytest.raises(ValueError, match="Mystery"):
        parse_payload(mystery)


def test_every_event_type_has_a_payload_class() -> None:
    assert set(PAYLOAD_TYPES) == set(EventType)
