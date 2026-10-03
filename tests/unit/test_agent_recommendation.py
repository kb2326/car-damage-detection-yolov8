from claimlens.agent.recommendation import (
    Recommendation,
    check_recommendation,
    to_agent_recommendation,
)
from claimlens.domain import Confidence, Route

GOOD = {
    "route_suggestion": "ADJUSTER_REVIEW",
    "confidence": "high",
    "rationale": "The description says the car was used for deliveries.",
    "evidence_ids": ["E1"],
    "policy_citations": ["STD-8.5"],
    "open_questions": [],
}
WORDINGS = {"STD-8.5": "standard", "PRM-8.5": "premium"}


def _check(**changes: object) -> list[str]:
    return check_recommendation(
        {**GOOD, **changes}, evidence_ids={"E1"}, wording="standard", wording_of=WORDINGS.get
    )


def test_a_good_recommendation_has_no_problems() -> None:
    assert _check() == []


def test_schema_problems_are_reported() -> None:
    assert any("route_suggestion" in p for p in _check(route_suggestion="DENY"))


def test_unknown_clause_is_rejected() -> None:
    assert _check(policy_citations=["STD-99.9"]) == ["clause STD-99.9 does not exist"]


def test_clause_from_another_wording_is_rejected() -> None:
    assert _check(policy_citations=["PRM-8.5"]) == [
        "clause PRM-8.5 is from the premium wording; this policy uses standard"
    ]


def test_no_citations_allowed_without_a_policy_wording() -> None:
    problems = check_recommendation(
        GOOD, evidence_ids={"E1"}, wording=None, wording_of=WORDINGS.get
    )
    assert problems == ["no policy wording is on file, so no clause may be cited"]


def test_unknown_evidence_label_is_rejected() -> None:
    assert _check(evidence_ids=["E7"]) == ["evidence E7 is not in the evidence list"]


def test_escalation_needs_a_reason() -> None:
    assert _check(rationale="  ") == ["give a reason in rationale"]


def test_conversion_maps_labels_to_event_ids() -> None:
    rec = to_agent_recommendation(Recommendation.model_validate(GOOD), {"E1": "evt-abc"})
    assert rec.route_suggestion is Route.ADJUSTER_REVIEW
    assert rec.confidence is Confidence.HIGH
    assert rec.citations == ("evt-abc",)
    assert rec.policy_citations == ("STD-8.5",)
