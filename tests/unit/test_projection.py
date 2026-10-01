from uuid import UUID

import pytest

from claimlens.domain import BoundingBox, DamageFinding, DamageType, Decision, Route
from claimlens.events.envelope import Actor, ActorKind
from claimlens.events.payloads import (
    ClaimReported,
    DamageDetected,
    PhotoAccepted,
    PhotoRejected,
    PhotoUploaded,
    RouteDecided,
    StageFailed,
)
from claimlens.events.projection import PhotoStatus, fold
from claimlens.events.store import SQLiteEventStore

CLAIM = UUID(int=7)
ACTOR = Actor(kind=ActorKind.SYSTEM, name="test")
FINDING = DamageFinding(
    photo_id="p1",
    damage_type=DamageType.DENT,
    confidence=0.9,
    bbox=BoundingBox(x1=0, y1=0, x2=64, y2=64),
    image_area_fraction=0.01,
)


def _upload(store: SQLiteEventStore, photo_id: str) -> None:
    store.append(
        CLAIM,
        PhotoUploaded(
            photo_id=photo_id,
            sha256=photo_id * 32,
            filename=f"{photo_id}.jpg",
            blob_name=f"{photo_id}.jpg",
            size_bytes=10,
        ),
        ACTOR,
    )


def test_fold_builds_claim_state(store: SQLiteEventStore) -> None:
    store.append(CLAIM, ClaimReported(policy_id="P-1001", description="scrape"), ACTOR)
    _upload(store, "p1")
    store.append(CLAIM, PhotoAccepted(photo_id="p1", width=640, height=640), ACTOR)
    detected = store.append(
        CLAIM, DamageDetected(photo_id="p1", model_version="m", findings=(FINDING,)), ACTOR
    )

    state = fold(store.load(CLAIM))

    assert state.policy_id == "P-1001"
    assert [photo.photo_id for photo in state.accepted_photos] == ["p1"]
    assert state.findings == [FINDING]
    assert state.detection_event_ids == {"p1": str(detected.event_id)}
    assert state.last_seq == 4
    assert state.decision is None


def test_fold_tracks_rejections_and_failures(store: SQLiteEventStore) -> None:
    store.append(CLAIM, ClaimReported(policy_id="P-1001", description=""), ACTOR)
    _upload(store, "p1")
    _upload(store, "p2")
    store.append(CLAIM, PhotoRejected(photo_id="p1", reason="too small"), ACTOR)
    store.append(CLAIM, StageFailed(stage="perception", photo_id="p2", error="boom"), ACTOR)

    state = fold(store.load(CLAIM))

    assert state.photos["p1"].status is PhotoStatus.REJECTED
    assert state.photos["p1"].reject_reason == "too small"
    assert state.failed("perception", "p2")
    assert not state.failed("perception", "p1")


def test_fold_records_decision(store: SQLiteEventStore) -> None:
    decision = Decision(route=Route.FAST_TRACK, rule_id="R9", reason="ok", policy_version="v0")
    store.append(CLAIM, ClaimReported(policy_id="P-1001", description=""), ACTOR)
    store.append(CLAIM, RouteDecided(decision=decision), ACTOR)
    assert fold(store.load(CLAIM)).decision == decision


def test_fold_requires_claim_reported_first(store: SQLiteEventStore) -> None:
    _upload(store, "p1")
    with pytest.raises(ValueError, match="ClaimReported"):
        fold(store.load(CLAIM))


def test_fold_rejects_an_empty_log() -> None:
    with pytest.raises(ValueError, match="empty"):
        fold([])
