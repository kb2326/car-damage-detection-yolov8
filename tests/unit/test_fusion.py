from pathlib import Path

import pytest

from claimlens.data.taxonomy import load_part_groups
from claimlens.domain import BoundingBox, DamageFinding, DamageType
from claimlens.fusion import FusedDetector, calibrate_confidence, fuse
from claimlens.vision.instances import SegInstance, Segmentation

ROOT = Path(__file__).resolve().parents[2]
GROUPS = load_part_groups(ROOT / "config" / "taxonomy.toml")


def _square(x0: float, y0: float, x1: float, y1: float) -> tuple[float, ...]:
    return (x0, y0, x1, y0, x1, y1, x0, y1)


def _inst(label: str, polygon: tuple[float, ...], confidence: float = 0.9) -> SegInstance:
    return SegInstance(
        label=label, confidence=confidence, box_xyxy=(0, 0, 10, 10), polygon_xyn=polygon
    )


def test_damage_inside_a_part_gets_its_group_and_ratio() -> None:
    dent = _inst("dent", _square(0.1, 0.1, 0.2, 0.2))
    bumper = _inst("back_bumper", _square(0.0, 0.0, 0.5, 0.5))
    (fused,) = fuse([dent], [bumper], GROUPS)
    assert fused.part_group == "rear_bumper"
    assert fused.part_area_ratio == pytest.approx(0.04, abs=0.01)


def test_damage_goes_to_the_part_with_the_largest_overlap() -> None:
    dent = _inst("dent", _square(0.1, 0.1, 0.3, 0.3))
    door = _inst("front_door", _square(0.0, 0.0, 0.25, 1.0))
    fender_light = _inst("front_light", _square(0.25, 0.0, 1.0, 1.0))
    (fused,) = fuse([dent], [door, fender_light], GROUPS)
    assert fused.part_group == "door"


def test_tie_goes_to_the_smaller_part() -> None:
    dent = _inst("dent", _square(0.4, 0.4, 0.6, 0.6))
    big = _inst("hood", _square(0.0, 0.0, 1.0, 1.0))
    small = _inst("left_mirror", _square(0.3, 0.3, 0.7, 0.7))
    (fused,) = fuse([dent], [big, small], GROUPS)
    assert fused.part_group == "mirror"


def test_no_overlap_leaves_the_part_unknown() -> None:
    dent = _inst("dent", _square(0.8, 0.8, 0.9, 0.9))
    (fused,) = fuse([dent], [_inst("hood", _square(0.0, 0.0, 0.3, 0.3))], GROUPS)
    assert fused.part_group is None
    assert fused.part_area_ratio is None


def test_overlap_below_the_minimum_is_ignored() -> None:
    dent = _inst("dent", _square(0.0, 0.0, 0.5, 0.5))
    sliver = _inst("hood", _square(0.45, 0.45, 1.0, 1.0))
    (fused,) = fuse([dent], [sliver], GROUPS, min_overlap=0.10)
    assert fused.part_group is None


def test_degenerate_damage_is_not_fused() -> None:
    (fused,) = fuse([_inst("dent", (0.1, 0.1))], [_inst("hood", _square(0, 0, 1, 1))], GROUPS)
    assert fused.part_group is None


def test_calibration_is_identity_at_temperature_one_and_softens_above() -> None:
    assert calibrate_confidence(0.8, 1.0) == pytest.approx(0.8)
    assert 0.5 < calibrate_confidence(0.8, 2.0) < 0.8
    assert calibrate_confidence(0.2, 2.0) > 0.2


class _FakeSegmenter:
    def __init__(self, name: str, instances: list[SegInstance]) -> None:
        self.model_version = name
        self._instances = tuple(instances)

    def segment(self, image_path: Path) -> Segmentation:
        return Segmentation(instances=self._instances, width=200, height=100)


def test_fused_detector_returns_findings_with_parts() -> None:
    damage = _FakeSegmenter("d", [_inst("dent", _square(0.1, 0.1, 0.2, 0.2), 0.8)])
    parts = _FakeSegmenter("p", [_inst("back_bumper", _square(0.0, 0.0, 0.5, 0.5))])
    detector = FusedDetector(damage, parts, GROUPS, temperature=2.0)
    (finding,) = detector.detect(Path("x.jpg"), "p1")
    assert finding.damage_type is DamageType.DENT
    assert finding.part == "rear_bumper"
    assert finding.part_area_ratio == pytest.approx(0.04, abs=0.01)
    assert finding.confidence == pytest.approx(calibrate_confidence(0.8, 2.0))
    # Fallback size is the box (10 x 10 px on 200 x 100), as the rate card defines, not the mask.
    assert finding.image_area_fraction == pytest.approx(100 / 20000)
    assert detector.model_version == "fused:d+p+T2.00"


def test_old_findings_without_ratio_still_load() -> None:
    data = {
        "photo_id": "p1",
        "damage_type": "dent",
        "confidence": 0.9,
        "bbox": {"x1": 0, "y1": 0, "x2": 1, "y2": 1},
        "image_area_fraction": 0.1,
        "part": None,
    }
    finding = DamageFinding.model_validate(data)
    assert finding.part_area_ratio is None
    assert finding.bbox == BoundingBox(x1=0, y1=0, x2=1, y2=1)


def test_part_ratio_is_the_share_of_the_part_covered_and_never_above_one() -> None:
    # A large scratch overlaps all of a small mirror: it covers 100% of the mirror, not 900%.
    scratch = _inst("scratch", _square(0.0, 0.0, 0.6, 0.6))
    mirror = _inst("left_mirror", _square(0.1, 0.1, 0.3, 0.3))
    (fused,) = fuse([scratch], [mirror], GROUPS)
    assert fused.part_group == "mirror"
    assert fused.part_area_ratio == pytest.approx(1.0, abs=0.02)


def test_partial_overlap_counts_only_the_overlap() -> None:
    dent = _inst("dent", _square(0.4, 0.0, 0.6, 0.2))  # half on the door, half off it
    door = _inst("front_door", _square(0.0, 0.0, 0.5, 1.0))
    (fused,) = fuse([dent], [door], GROUPS)
    assert fused.part_area_ratio == pytest.approx(0.02 / 0.5, abs=0.01)


def test_a_flat_tyre_goes_to_the_wheel_not_the_bumper() -> None:
    tyre = _inst("tire_flat", _square(0.2, 0.2, 0.4, 0.4))
    bumper = _inst("front_bumper", _square(0.0, 0.0, 0.5, 0.5))  # covers all of the tyre
    wheel = _inst("wheel", _square(0.3, 0.3, 0.6, 0.6))  # covers a quarter of it
    (fused,) = fuse([tyre], [bumper, wheel], GROUPS)
    assert fused.part_group == "wheel"


def test_an_implausible_part_only_leaves_the_part_unknown() -> None:
    tyre = _inst("tire_flat", _square(0.2, 0.2, 0.4, 0.4))
    bumper = _inst("front_bumper", _square(0.0, 0.0, 0.5, 0.5))
    (fused,) = fuse([tyre], [bumper], GROUPS)
    assert fused.part_group is None
    assert fused.part_area_ratio is None


@pytest.mark.parametrize(
    ("damage", "part", "group"),
    [
        ("glass_shatter", "front_glass", "glass"),
        ("glass_shatter", "front_light", "light"),
        ("glass_shatter", "hood", None),
        ("glass_shatter", "front_bumper", None),
        ("lamp_broken", "back_light", "light"),
        ("lamp_broken", "trunk", None),
        ("dent", "hood", "hood"),
        ("crack", "front_glass", "glass"),
    ],
)
def test_damage_types_only_go_to_plausible_parts(damage: str, part: str, group: str | None) -> None:
    (fused,) = fuse(
        [_inst(damage, _square(0.2, 0.2, 0.4, 0.4))], [_inst(part, _square(0, 0, 1, 1))], GROUPS
    )
    assert fused.part_group == group


def test_plausible_parts_must_name_real_groups() -> None:
    from pydantic import ValidationError

    from claimlens.data.taxonomy import PartGroups

    with pytest.raises(ValidationError, match="unknown part group"):
        PartGroups(
            classes=GROUPS.classes,
            members=GROUPS.members,
            damage_parts={"tire_flat": ("tyre",)},
        )
