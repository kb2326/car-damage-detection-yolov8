from uuid import uuid4

import pytest

from claimlens.domain import AgentRecommendation, Confidence, Route
from claimlens.evals.agent_metrics import agent_quality
from claimlens.evals.golden import GoldenClaim
from claimlens.evals.triage import CaseResult
from claimlens.events.projection import ClaimState

WORDINGS = {"STD-8.5": "standard", "STD-8.1": "standard", "PRM-8.5": "premium"}


def _case(case_id: str, scenario: str, route: Route, cites: tuple[str, ...] = ()) -> GoldenClaim:
    return GoldenClaim(
        case_id=case_id,
        scenario=scenario,
        policy_id="P-1001",
        description="x",
        photos=("a.jpg",),
        expected_route=route,
        label_source="t",
        narrative=True,
        expected_citations=cites,
        must_not_fast_track_reason="" if route is Route.FAST_TRACK else "r",
    )


def _result(
    case: GoldenClaim, suggestion: Route | None, cites: tuple[str, ...] = (), error: str = ""
) -> CaseResult:
    state = ClaimState(claim_id=uuid4(), policy_id=case.policy_id, description="x")
    if suggestion is not None:
        state.recommendation = AgentRecommendation(
            route_suggestion=suggestion,
            confidence=Confidence.HIGH,
            rationale="r",
            citations=(),
            policy_citations=cites,
        )
    return CaseResult(
        case.case_id,
        case.scenario,
        case.expected_route,
        Route.ADJUSTER_REVIEW,
        state=state,
        agent_error=error,
    )


def test_quality_metrics_on_a_small_set() -> None:
    a = _case("a", "exclusion_commercial", Route.ADJUSTER_REVIEW, ("STD-8.5",))
    b = _case("b", "exclusion_racing", Route.ADJUSTER_REVIEW, ("STD-8.1",))
    c = _case("c", "benign_distractor", Route.FAST_TRACK)
    d = _case("d", "benign_distractor", Route.FAST_TRACK)
    results = [
        _result(a, Route.ADJUSTER_REVIEW, ("STD-8.5",)),  # caught, right clause
        _result(b, Route.FAST_TRACK, ("STD-8.5", "PRM-8.5")),  # missed; one wrong-wording clause
        _result(c, Route.FAST_TRACK),  # benign passed
        _result(d, None, error="AgentFailed: step limit"),  # agent failed
    ]
    q = agent_quality([a, b, c, d], results, {"P-1001": "standard"}, WORDINGS.get)
    assert q.citation_validity == pytest.approx(2 / 3)
    assert q.citation_hit_rate == pytest.approx(1 / 2)
    assert q.narrative_catch_rate == pytest.approx(1 / 2)
    assert q.benign_pass_rate == pytest.approx(1 / 2)
    assert q.agent_failure_rate == pytest.approx(1 / 4)


def test_rates_are_none_when_there_is_nothing_to_measure() -> None:
    q = agent_quality([], [], {}, WORDINGS.get)
    assert q.citation_validity is None
    assert q.narrative_catch_rate is None


def test_an_unknown_clause_on_an_unknown_policy_is_not_valid() -> None:
    case = _case("a", "exclusion_commercial", Route.ADJUSTER_REVIEW, ("STD-8.5",))
    case = case.model_copy(update={"policy_id": "P-0000"})
    result = _result(case, Route.ADJUSTER_REVIEW, ("XYZ-1.1",))
    q = agent_quality([case], [result], {}, WORDINGS.get)
    assert q.citation_validity == 0.0
