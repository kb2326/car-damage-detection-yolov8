from pathlib import Path

import pytest

from claimlens.domain import BoundingBox, DamageFinding, DamageType, Severity
from claimlens.pricing import PartBands, estimate_cost, load_rate_card, severity_for, severity_of

ROOT = Path(__file__).resolve().parents[2]
CARD = load_rate_card(ROOT / "config" / "rate_card.toml")


def _finding(damage_type: DamageType, fraction: float, photo_id: str = "p1") -> DamageFinding:
    return DamageFinding(
        photo_id=photo_id,
        damage_type=damage_type,
        confidence=0.9,
        bbox=BoundingBox(x1=0, y1=0, x2=1, y2=1),
        image_area_fraction=fraction,
    )


def test_severity_bands() -> None:
    assert severity_for(0.01, CARD) is Severity.MINOR
    assert severity_for(0.05, CARD) is Severity.MODERATE
    assert severity_for(0.10, CARD) is Severity.MODERATE
    assert severity_for(0.50, CARD) is Severity.SEVERE


def test_no_findings_cost_nothing() -> None:
    estimate = estimate_cost([], CARD)
    assert (estimate.low, estimate.high) == (0, 0)
    assert "no damage found" in estimate.basis


def test_same_damage_type_is_priced_once_at_its_worst_severity() -> None:
    estimate = estimate_cost(
        [_finding(DamageType.DENT, 0.01), _finding(DamageType.DENT, 0.10, "p2")], CARD
    )
    assert (estimate.low, estimate.high) == (400, 1200)
    assert estimate.basis == "rate-card-2026.10-v1: dent/moderate"


def test_costs_add_up_across_damage_types() -> None:
    estimate = estimate_cost(
        [_finding(DamageType.DENT, 0.01), _finding(DamageType.SCRATCH, 0.01)], CARD
    )
    assert (estimate.low, estimate.high) == (250, 700)


def test_rate_card_must_cover_every_damage_type(tmp_path: Path) -> None:
    text = (ROOT / "config" / "rate_card.toml").read_text(encoding="utf-8")
    broken = text.split("[rates.smash]")[0]
    path = tmp_path / "rate_card.toml"
    path.write_text(broken, encoding="utf-8")
    with pytest.raises(ValueError, match="missing rates for smash"):
        load_rate_card(path)


def _part_finding(
    fraction: float, part: str | None = None, ratio: float | None = None
) -> DamageFinding:
    return DamageFinding(
        photo_id="p1",
        damage_type=DamageType.DENT,
        confidence=0.9,
        bbox=BoundingBox(x1=0, y1=0, x2=1, y2=1),
        image_area_fraction=fraction,
        part=part,
        part_area_ratio=ratio,
    )


def test_repo_card_has_part_bands() -> None:
    assert CARD.part_bands == PartBands(minor_max_ratio=0.10, moderate_max_ratio=0.30)
    assert CARD.version == "rate-card-2026.10-v1"


def test_part_ratio_decides_when_a_part_is_known() -> None:
    # A close-up fills 60% of the photo but covers only 5% of the bumper: minor.
    assert severity_of(_part_finding(0.60, "rear_bumper", 0.05), CARD) is Severity.MINOR
    assert severity_of(_part_finding(0.01, "door", 0.20), CARD) is Severity.MODERATE


def test_ratio_above_one_is_severe() -> None:
    assert severity_of(_part_finding(0.30, "mirror", 1.7), CARD) is Severity.SEVERE


def test_without_a_part_the_image_fraction_decides() -> None:
    assert severity_of(_part_finding(0.60), CARD) is Severity.SEVERE
    assert severity_of(_part_finding(0.01), CARD) is Severity.MINOR


def test_part_bands_must_be_ordered() -> None:
    with pytest.raises(ValueError, match="minor_max_ratio"):
        PartBands(minor_max_ratio=0.5, moderate_max_ratio=0.3)


def test_estimate_uses_the_part_ratio() -> None:
    close_up = estimate_cost([_part_finding(0.60, "rear_bumper", 0.05)], CARD)
    assert close_up.basis.endswith("dent/minor")
