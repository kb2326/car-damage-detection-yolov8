from collections.abc import Callable
from pathlib import Path
from uuid import UUID

from claimlens.events.envelope import Actor, ActorKind
from claimlens.events.payloads import HumanReviewed, MemoryForgotten, MemoryWritten, ReviewAction
from claimlens.events.projection import fold
from claimlens.events.store import SQLiteEventStore
from claimlens.intake import submit_claim
from claimlens.memory.records import build_record, phash_of
from claimlens.workflow import process_claim
from tests.fakes import FakeDetector, make_test_deps

SECRET_STORY = "my neighbour Bob Smith reversed into me"


def decided(
    tmp_path: Path, store: SQLiteEventStore, make_image: Callable[..., Path], **image: object
) -> tuple[UUID, Callable[[str], Path]]:
    deps = make_test_deps(tmp_path, store, FakeDetector())
    claim = submit_claim(
        store,
        deps.blobs,
        policy_id="P-1001",
        description=SECRET_STORY,
        photo_paths=[make_image("a.jpg", **image)],
    )
    process_claim(claim, deps)
    return claim, deps.blobs.path


def test_a_decided_claim_becomes_a_record(
    tmp_path: Path, store: SQLiteEventStore, make_image: Callable[..., Path]
) -> None:
    claim, photo_path = decided(tmp_path, store, make_image)
    events = store.load(claim)
    record = build_record(fold(events), events, photo_path)
    assert record.claim_id == str(claim)
    assert record.policy_id == "P-1001"
    assert (record.route, record.rule_id) == ("FAST_TRACK", "R9")
    assert record.damage == "dent"
    assert (record.cost_low, record.cost_high) == (150, 400)
    assert len(record.photo_phashes) == 1
    assert len(record.photo_phashes[0]) == 16
    assert "dent" in record.summary
    assert "FAST_TRACK" in record.summary
    by_type = {e.type: e.seq for e in events}
    assert by_type["RouteDecided"] in record.source_seq
    assert by_type["CostEstimated"] in record.source_seq


def test_customer_words_never_reach_memory(
    tmp_path: Path, store: SQLiteEventStore, make_image: Callable[..., Path]
) -> None:
    claim, photo_path = decided(tmp_path, store, make_image)
    events = store.load(claim)
    record = build_record(fold(events), events, photo_path)
    assert "Bob" not in record.model_dump_json()
    assert "neighbour" not in record.model_dump_json()


def test_a_review_updates_the_record(
    tmp_path: Path, store: SQLiteEventStore, make_image: Callable[..., Path]
) -> None:
    claim, photo_path = decided(tmp_path, store, make_image)
    review = HumanReviewed(reviewer="sam", action=ReviewAction.DENY, note="staged")
    store.append(claim, review, Actor(kind=ActorKind.HUMAN, name="sam"))
    events = store.load(claim)
    record = build_record(fold(events), events, photo_path)
    assert record.review_action == "deny"
    assert events[-1].seq in record.source_seq
    assert "reviewed: deny" in record.summary


def test_a_claim_without_photos_or_estimate_gives_empty_values(store: SQLiteEventStore) -> None:
    from uuid import uuid4

    from claimlens.domain import Decision, Route
    from claimlens.events.payloads import ClaimReported, RouteDecided

    claim = uuid4()
    system = Actor(kind=ActorKind.SYSTEM, name="t")
    store.append(claim, ClaimReported(policy_id="P-1001", description="x"), system)
    decision = Decision(
        route=Route.ADJUSTER_REVIEW, rule_id="R3", reason="no photos", policy_version="v"
    )
    store.append(claim, RouteDecided(decision=decision), system)
    events = store.load(claim)
    record = build_record(fold(events), events, lambda name: Path(name))
    assert record.photo_phashes == ()
    assert (record.cost_low, record.cost_high) == (0, 0)
    assert record.damage == ""


def test_phash_is_stable_for_the_same_image(
    tmp_path: Path, make_image: Callable[..., Path]
) -> None:
    path = make_image("x.png")
    assert phash_of(path) == phash_of(path)


def test_memory_events_fold_without_changing_the_claim(store: SQLiteEventStore) -> None:
    from uuid import uuid4

    from claimlens.events.payloads import ClaimReported

    claim = uuid4()
    system = Actor(kind=ActorKind.SYSTEM, name="t")
    store.append(claim, ClaimReported(policy_id="P-1001", description="x"), system)
    store.append(
        claim, MemoryWritten(record_id=str(claim), fields=("route",), source_seq=(1,)), system
    )
    store.append(claim, MemoryForgotten(reason="owner request"), system)
    assert fold(store.load(claim)).policy_id == "P-1001"
