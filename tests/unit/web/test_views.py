from pathlib import Path

from claimlens.events.projection import fold
from claimlens.events.store import SQLiteEventStore
from claimlens.intake import submit_claim
from claimlens.web.schemas import StageStatus
from claimlens.web.views import (
    agent_view,
    claim_detail,
    claim_status,
    claim_summary,
    photo_views,
    stage_timeline,
)
from tests.unit.web.helpers import decided_claim, deps, image


def _states(timeline: list[StageStatus]) -> dict[str, str]:
    return {s.key: s.state for s in timeline}


def test_a_decided_claim_has_every_pipeline_stage_done(
    tmp_path: Path, store: SQLiteEventStore
) -> None:
    claim = decided_claim(tmp_path, store)
    states = _states(stage_timeline(store.load(claim), running=False))
    for key in ("photos", "damage", "integrity", "pricing", "coverage", "agent", "decision"):
        assert states[key] == "done", key
    assert states["intake"] == "skipped"  # filed by form, not by chat
    assert states["memory"] == "skipped"  # memory is off in this pipeline
    assert states["review"] == "waiting"


def test_a_running_claim_shows_the_next_stage_running(
    tmp_path: Path, store: SQLiteEventStore
) -> None:
    d = deps(tmp_path, store)
    claim = submit_claim(
        store, d.blobs, policy_id="P-1001", description="x", photo_paths=[image(tmp_path / "a.png")]
    )
    states = _states(stage_timeline(store.load(claim), running=True))
    assert states["photos"] == "running"
    assert states["decision"] == "waiting"
    assert claim_status(fold(store.load(claim)), running=False) == "stopped"


def test_boxes_are_percentages_of_the_photo(tmp_path: Path, store: SQLiteEventStore) -> None:
    claim = decided_claim(tmp_path, store)
    events = store.load(claim)
    (view,) = photo_views(fold(events), events)
    (box,) = view.boxes
    assert (box.left, box.top, box.width, box.height) == (10.0, 10.0, 40.0, 40.0)
    assert box.label == "dent · back_bumper · 0.90"
    assert view.status == "accepted"


def test_the_agent_section_and_the_summary(tmp_path: Path, store: SQLiteEventStore) -> None:
    claim = decided_claim(tmp_path, store)
    events = store.load(claim)
    state = fold(events)
    assert state.recommendation is not None
    assert state.decision is not None
    agent = agent_view(state, events)
    assert agent is not None
    assert agent.route_suggestion == state.recommendation.route_suggestion.value
    assert agent.confidence == state.recommendation.confidence.value
    summary = claim_summary(state, events, running=False)
    assert summary.status == "decided"
    assert summary.route == state.decision.route.value
    assert summary.rule_id == state.decision.rule_id


def test_the_detail_carries_the_audit_trail(tmp_path: Path, store: SQLiteEventStore) -> None:
    claim = decided_claim(tmp_path, store)
    events = store.load(claim)
    detail = claim_detail(fold(events), events, running=False, error=None, similar=[])
    assert detail.chain_ok
    assert detail.event_count == len(events)
    assert [e.type for e in detail.events] == [e.type for e in events]
    assert detail.description == "Reversed into a bollard <b>slowly</b>"  # escaped by templates
    assert detail.cost_low is not None
