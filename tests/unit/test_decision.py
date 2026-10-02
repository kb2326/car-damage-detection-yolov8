from pathlib import Path
from typing import Any
from uuid import UUID

import pytest

from claimlens.decision import DecisionConfig, decide, load_decision_config
from claimlens.domain import (
    AgentRecommendation,
    BoundingBox,
    Confidence,
    CostEstimate,
    Coverage,
    DamageFinding,
    DamageType,
    FraudSignal,
    Route,
)
from claimlens.events.payloads import StageFailed
from claimlens.events.projection import ClaimState, PhotoRecord, PhotoStatus

ROOT = Path(__file__).resolve().parents[2]
CONFIG = DecisionConfig(
    version="test-policy",
    fraud_score_threshold=0.5,
    max_fast_track_cost_usd=3000,
    min_finding_confidence=0.4,
)


def _finding(confidence: float = 0.9) -> DamageFinding:
    return DamageFinding(
        photo_id="p1",
        damage_type=DamageType.DENT,
        confidence=confidence,
        bbox=BoundingBox(x1=0, y1=0, x2=64, y2=64),
        image_area_fraction=0.01,
    )


def _recommendation(**changes: Any) -> AgentRecommendation:
    base = AgentRecommendation(
        route_suggestion=Route.FAST_TRACK,
        confidence=Confidence.MEDIUM,
        rationale="ok",
        citations=("e1",),
    )
    return base.model_copy(update=changes)


def _ready_state(**changes: Any) -> ClaimState:
    state = ClaimState(claim_id=UUID(int=1), policy_id="P-1001", description="")
    state.photos["p1"] = PhotoRecord(
        photo_id="p1",
        sha256="a" * 64,
        filename="p1.jpg",
        blob_name="x.jpg",
        status=PhotoStatus.ACCEPTED,
    )
    state.findings = [_finding()]
    state.integrity_checked = True
    state.cost_estimate = CostEstimate(low=400, high=1200, basis="test")
    state.coverage = Coverage(
        policy_id="P-1001", found=True, active=True, collision=True, deductible=500
    )
    state.recommendation = _recommendation()
    for name, value in changes.items():
        setattr(state, name, value)
    return state


FRAUD = [FraudSignal(kind="photo_reuse", score=1.0, detail="Photo p1 reused.")]
FAILURE = [StageFailed(stage="perception", photo_id="p1", error="boom")]
LAPSED = Coverage(policy_id="P-1001", found=True, active=False, collision=True, deductible=500)


@pytest.mark.parametrize(
    ("changes", "route", "rule"),
    [
        ({}, Route.FAST_TRACK, "R9"),
        ({"fraud_signals": FRAUD, "failures": FAILURE}, Route.FRAUD_REVIEW, "R1"),
        ({"failures": FAILURE}, Route.ADJUSTER_REVIEW, "R2"),
        ({"photos": {}}, Route.ADJUSTER_REVIEW, "R3"),
        ({"coverage": None}, Route.ADJUSTER_REVIEW, "R4"),
        ({"coverage": LAPSED}, Route.ADJUSTER_REVIEW, "R4"),
        (
            {"cost_estimate": CostEstimate(low=2000, high=3500, basis="t")},
            Route.ADJUSTER_REVIEW,
            "R5",
        ),
        ({"findings": [_finding(confidence=0.2)]}, Route.ADJUSTER_REVIEW, "R6"),
        ({"recommendation": None}, Route.ADJUSTER_REVIEW, "R7"),
        (
            {"recommendation": _recommendation(confidence=Confidence.LOW)},
            Route.ADJUSTER_REVIEW,
            "R7",
        ),
        (
            {"recommendation": _recommendation(open_questions=("why?",))},
            Route.ADJUSTER_REVIEW,
            "R7",
        ),
        (
            {"recommendation": _recommendation(route_suggestion=Route.ADJUSTER_REVIEW)},
            Route.ADJUSTER_REVIEW,
            "R8",
        ),
    ],
)
def test_decision_rules(changes: dict[str, Any], route: Route, rule: str) -> None:
    decision = decide(_ready_state(**changes), CONFIG)
    assert (decision.route, decision.rule_id) == (route, rule)
    assert decision.policy_version == "test-policy"


def test_low_fraud_score_does_not_trigger_fraud_review() -> None:
    weak = [FraudSignal(kind="exif_mismatch", score=0.2, detail="minor")]
    assert decide(_ready_state(fraud_signals=weak), CONFIG).route is Route.FAST_TRACK


def test_repo_config_loads() -> None:
    config = load_decision_config(ROOT / "config" / "decision_policy.toml")
    assert config.version == "decision-policy-v1"
    assert config.min_finding_confidence == 0.65  # owner-approved R6 threshold (ADR 0009)
    assert config.max_fast_track_cost_usd == 3000
