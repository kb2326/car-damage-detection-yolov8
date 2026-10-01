from uuid import UUID

from claimlens.agent.stub import StubTriageAgent
from claimlens.domain import BoundingBox, Confidence, DamageFinding, DamageType, Route
from claimlens.events.projection import ClaimState


def _state() -> ClaimState:
    return ClaimState(claim_id=UUID(int=1), policy_id="P-1001", description="")


def test_no_findings_asks_for_human_review() -> None:
    rec = StubTriageAgent().recommend(_state())
    assert rec.route_suggestion is Route.ADJUSTER_REVIEW
    assert rec.confidence is Confidence.LOW
    assert rec.open_questions


def test_findings_suggest_fast_track_with_citations() -> None:
    state = _state()
    state.findings = [
        DamageFinding(
            photo_id="p1",
            damage_type=DamageType.DENT,
            confidence=0.9,
            bbox=BoundingBox(x1=0, y1=0, x2=1, y2=1),
            image_area_fraction=0.01,
        )
    ]
    state.detection_event_ids = {"p1": "event-123"}
    rec = StubTriageAgent().recommend(state)
    assert rec.route_suggestion is Route.FAST_TRACK
    assert rec.confidence is Confidence.MEDIUM
    assert rec.citations == ("event-123",)
    assert "dent" in rec.rationale
