from pathlib import Path

import pytest

from claimlens.decision import load_decision_config
from claimlens.domain import BoundingBox, Coverage, DamageFinding, DamageType, Route
from claimlens.evals.oracle import findings_from_yolo_label, oracle_route
from claimlens.pricing import load_rate_card

ROOT = Path(__file__).resolve().parents[2]
NAMES = ["crack", "dent", "glass shatter", "lamp broken", "scratch", "tire flat", "smash"]
CARD = load_rate_card(ROOT / "config" / "rate_card.toml")
CONFIG = load_decision_config(ROOT / "config" / "decision_policy.toml")
COVERED = Coverage(policy_id="P-1001", found=True, active=True, collision=True, deductible=250)


def test_polygon_label_becomes_a_pixel_box() -> None:
    (finding,) = findings_from_yolo_label(
        "1 0.1 0.1 0.3 0.1 0.3 0.2 0.1 0.2", NAMES, "p1", 640, 640
    )
    assert finding.damage_type is DamageType.DENT
    assert finding.bbox == BoundingBox(x1=64, y1=64, x2=192, y2=128)
    assert finding.image_area_fraction == pytest.approx(0.02)
    assert finding.confidence == 1.0


def test_box_label_is_supported() -> None:
    (finding,) = findings_from_yolo_label("4 0.5 0.5 0.2 0.1", NAMES, "p1", 640, 640)
    assert finding.damage_type is DamageType.SCRATCH
    assert finding.image_area_fraction == pytest.approx(0.02)


def test_malformed_line_raises() -> None:
    with pytest.raises(ValueError, match="line 1"):
        findings_from_yolo_label("1 0.1 0.2 0.3", NAMES, "p1", 640, 640)


def test_unknown_class_index_raises() -> None:
    with pytest.raises(ValueError, match="class index 9"):
        findings_from_yolo_label("9 0.5 0.5 0.2 0.1", NAMES, "p1", 640, 640)


def _finding(damage_type: DamageType, fraction: float) -> DamageFinding:
    return DamageFinding(
        photo_id="p1",
        damage_type=damage_type,
        confidence=1.0,
        bbox=BoundingBox(x1=0, y1=0, x2=1, y2=1),
        image_area_fraction=fraction,
    )


def test_oracle_fast_tracks_small_covered_damage() -> None:
    route = oracle_route([_finding(DamageType.DENT, 0.01)], COVERED, CARD, CONFIG)
    assert route is Route.FAST_TRACK


def test_oracle_escalates_expensive_damage() -> None:
    route = oracle_route([_finding(DamageType.SMASH, 0.5)], COVERED, CARD, CONFIG)
    assert route is Route.ADJUSTER_REVIEW


def test_oracle_escalates_when_nothing_is_labelled() -> None:
    assert oracle_route([], COVERED, CARD, CONFIG) is Route.ADJUSTER_REVIEW


def test_findings_from_record_uses_polygon_extents() -> None:
    from claimlens.data.records import Annotation, ImageRecord
    from claimlens.evals.oracle import findings_from_record

    record = ImageRecord(
        image_id="s:a",
        source="s",
        source_split="test",
        path="a.jpg",
        width=1000,
        height=500,
        annotations=(Annotation(label="scratch", polygon=(0.1, 0.2, 0.3, 0.2, 0.3, 0.6)),),
    )
    (finding,) = findings_from_record(record)
    assert finding.damage_type is DamageType.SCRATCH
    assert finding.bbox == BoundingBox(x1=100, y1=100, x2=300, y2=300)
    assert finding.image_area_fraction == pytest.approx(0.08)
