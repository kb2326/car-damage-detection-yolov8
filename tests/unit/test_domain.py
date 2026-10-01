import pytest
from pydantic import ValidationError

from claimlens.domain import (
    AgentRecommendation,
    BoundingBox,
    Confidence,
    CostEstimate,
    Coverage,
    DamageFinding,
    DamageType,
    Route,
)


def _finding(confidence: float = 0.9) -> DamageFinding:
    return DamageFinding(
        photo_id="p1",
        damage_type=DamageType.DENT,
        confidence=confidence,
        bbox=BoundingBox(x1=0, y1=0, x2=10, y2=10),
        image_area_fraction=0.01,
    )


def test_routes_never_include_a_denial() -> None:
    assert {route.value for route in Route} == {"FAST_TRACK", "ADJUSTER_REVIEW", "FRAUD_REVIEW"}


def test_finding_rejects_confidence_above_one() -> None:
    with pytest.raises(ValidationError):
        _finding(confidence=1.5)


def test_bounding_box_rejects_inverted_corners() -> None:
    with pytest.raises(ValidationError, match="x1 <= x2"):
        BoundingBox(x1=10, y1=0, x2=5, y2=10)


def test_cost_estimate_rejects_low_above_high() -> None:
    with pytest.raises(ValidationError, match="low must not exceed high"):
        CostEstimate(low=500, high=100, basis="test")


def test_coverage_confirmed_requires_found_active_and_collision() -> None:
    def cover(*, found: bool = True, active: bool = True, collision: bool = True) -> Coverage:
        return Coverage(
            policy_id="P", found=found, active=active, collision=collision, deductible=500
        )

    assert cover().confirmed
    assert not cover(found=False).confirmed
    assert not cover(active=False).confirmed
    assert not cover(collision=False).confirmed


def test_damage_type_values_are_snake_case() -> None:
    assert DamageType("glass_shatter") is DamageType.GLASS_SHATTER


def test_recommendation_defaults_to_no_open_questions() -> None:
    rec = AgentRecommendation(
        route_suggestion=Route.FAST_TRACK,
        confidence=Confidence.HIGH,
        rationale="ok",
        citations=("e1",),
    )
    assert rec.open_questions == ()
