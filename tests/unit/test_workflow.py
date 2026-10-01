import sqlite3
from collections.abc import Callable, Sequence
from pathlib import Path
from uuid import UUID

import pytest

from claimlens.domain import Route
from claimlens.events.envelope import ChainIntegrityError
from claimlens.events.projection import PhotoStatus, fold
from claimlens.events.store import SQLiteEventStore
from claimlens.intake import submit_claim
from claimlens.workflow import PipelineDeps, process_claim
from tests.fakes import FailingAgent, FakeDetector, SimulatedCrashError, make_test_deps


def _submit(deps: PipelineDeps, photos: Sequence[Path], policy_id: str = "P-1001") -> UUID:
    return submit_claim(
        deps.store, deps.blobs, policy_id=policy_id, description="scrape", photo_paths=photos
    )


def _types(store: SQLiteEventStore, claim_id: UUID) -> list[str]:
    return [event.type for event in store.load(claim_id)]


def test_clean_claim_is_fast_tracked_with_full_audit_trail(
    store: SQLiteEventStore, tmp_path: Path, make_image: Callable[..., Path]
) -> None:
    deps = make_test_deps(tmp_path, store, FakeDetector())
    claim_id = _submit(deps, [make_image("a.jpg")])

    decision = process_claim(claim_id, deps)

    assert (decision.route, decision.rule_id) == (Route.FAST_TRACK, "R9")
    assert _types(store, claim_id) == [
        "ClaimReported",
        "PhotoUploaded",
        "PhotoAccepted",
        "DamageDetected",
        "IntegrityChecked",
        "CostEstimated",
        "PolicyRetrieved",
        "AgentRecommended",
        "RouteDecided",
    ]


def test_processing_twice_changes_nothing(
    store: SQLiteEventStore, tmp_path: Path, make_image: Callable[..., Path]
) -> None:
    detector = FakeDetector()
    deps = make_test_deps(tmp_path, store, detector)
    claim_id = _submit(deps, [make_image("a.jpg")])

    first = process_claim(claim_id, deps)
    count = len(store.load(claim_id))
    second = process_claim(claim_id, deps)

    assert first == second
    assert len(store.load(claim_id)) == count
    assert detector.calls == ["p1"]


def test_resume_after_crash_does_not_repeat_work(
    store: SQLiteEventStore, tmp_path: Path, make_image: Callable[..., Path]
) -> None:
    photos = [make_image("a.jpg"), make_image("b.jpg", color=(0, 90, 0))]
    crashing = FakeDetector(crash_on="p2")
    claim_id = _submit(make_test_deps(tmp_path, store, crashing), photos)

    with pytest.raises(SimulatedCrashError):
        process_claim(claim_id, make_test_deps(tmp_path, store, crashing))

    healthy = FakeDetector()
    decision = process_claim(claim_id, make_test_deps(tmp_path, store, healthy))

    assert decision.route is Route.FAST_TRACK
    assert healthy.calls == ["p2"]
    assert _types(store, claim_id).count("DamageDetected") == 2


def test_detector_failure_is_retried_then_routed_to_adjuster(
    store: SQLiteEventStore, tmp_path: Path, make_image: Callable[..., Path]
) -> None:
    detector = FakeDetector(fail_photos=frozenset({"p1"}))
    deps = make_test_deps(tmp_path, store, detector)
    claim_id = _submit(deps, [make_image("a.jpg")])

    decision = process_claim(claim_id, deps)

    assert (decision.route, decision.rule_id) == (Route.ADJUSTER_REVIEW, "R2")
    assert detector.calls == ["p1", "p1"]


def test_duplicate_photo_in_claim_is_rejected_not_fraud(
    store: SQLiteEventStore, tmp_path: Path, make_image: Callable[..., Path]
) -> None:
    image = make_image("a.jpg")
    deps = make_test_deps(tmp_path, store, FakeDetector())
    claim_id = _submit(deps, [image, image])

    decision = process_claim(claim_id, deps)
    state = fold(store.load(claim_id))

    assert decision.route is Route.FAST_TRACK
    assert state.photos["p2"].status is PhotoStatus.REJECTED
    assert state.photos["p2"].reject_reason == "duplicate of p1"
    assert state.fraud_signals == []


def test_photo_reused_across_claims_goes_to_fraud_review(
    store: SQLiteEventStore, tmp_path: Path, make_image: Callable[..., Path]
) -> None:
    image = make_image("a.jpg")
    deps = make_test_deps(tmp_path, store, FakeDetector())
    process_claim(_submit(deps, [image]), deps)

    decision = process_claim(_submit(deps, [image], policy_id="P-1002"), deps)

    assert (decision.route, decision.rule_id) == (Route.FRAUD_REVIEW, "R1")


def test_unusable_photos_route_to_adjuster(
    store: SQLiteEventStore, tmp_path: Path, make_image: Callable[..., Path]
) -> None:
    detector = FakeDetector()
    deps = make_test_deps(tmp_path, store, detector)
    corrupt = tmp_path / "corrupt.jpg"
    corrupt.write_text("not an image", encoding="utf-8")
    claim_id = _submit(deps, [make_image("tiny.png", size=(100, 100)), corrupt])

    decision = process_claim(claim_id, deps)

    assert (decision.route, decision.rule_id) == (Route.ADJUSTER_REVIEW, "R3")
    assert detector.calls == []


def test_lapsed_policy_routes_to_adjuster(
    store: SQLiteEventStore, tmp_path: Path, make_image: Callable[..., Path]
) -> None:
    deps = make_test_deps(tmp_path, store, FakeDetector())
    decision = process_claim(_submit(deps, [make_image("a.jpg")], policy_id="P-2001"), deps)
    assert (decision.route, decision.rule_id) == (Route.ADJUSTER_REVIEW, "R4")


def test_agent_failure_routes_to_adjuster(
    store: SQLiteEventStore, tmp_path: Path, make_image: Callable[..., Path]
) -> None:
    deps = make_test_deps(tmp_path, store, FakeDetector(), agent=FailingAgent())
    decision = process_claim(_submit(deps, [make_image("a.jpg")]), deps)
    assert (decision.route, decision.rule_id) == (Route.ADJUSTER_REVIEW, "R2")


def test_tampered_log_stops_processing(
    store: SQLiteEventStore, tmp_path: Path, make_image: Callable[..., Path]
) -> None:
    deps = make_test_deps(tmp_path, store, FakeDetector())
    claim_id = _submit(deps, [make_image("a.jpg")])
    with sqlite3.connect(tmp_path / "events.db") as conn:
        conn.execute("UPDATE events SET data = replace(data, 'P-1001', 'P-1002') WHERE seq = 1")

    with pytest.raises(ChainIntegrityError):
        process_claim(claim_id, deps)
