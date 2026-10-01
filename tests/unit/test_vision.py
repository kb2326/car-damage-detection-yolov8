import pytest

from claimlens.domain import DamageType
from claimlens.vision.base import findings_from_predictions, normalize_class_name

LEGACY_NAMES = {
    0: "crack",
    1: "dent",
    2: "glass shatter",
    3: "lamp broken",
    4: "scratch",
    5: "tire flat",
    6: "smash",
}


def test_normalize_maps_legacy_names() -> None:
    assert normalize_class_name("glass shatter") is DamageType.GLASS_SHATTER
    assert normalize_class_name("Lamp-Broken") is DamageType.LAMP_BROKEN


def test_normalize_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="Front-Windscreen-Damage"):
        normalize_class_name("Front-Windscreen-Damage")


def test_findings_use_the_model_class_names() -> None:
    findings = findings_from_predictions(
        photo_id="p1",
        boxes_xyxy=[[0, 0, 64, 64], [100, 100, 420, 420]],
        confidences=[0.8, 0.3],
        class_ids=[1, 4],
        class_names=LEGACY_NAMES,
        image_width=640,
        image_height=640,
    )
    assert [f.damage_type for f in findings] == [DamageType.DENT, DamageType.SCRATCH]
    assert findings[0].image_area_fraction == pytest.approx(0.01)
    assert findings[1].confidence == pytest.approx(0.3)
    assert findings[0].photo_id == "p1"


def test_boxes_are_clamped_to_the_image() -> None:
    (finding,) = findings_from_predictions(
        photo_id="p1",
        boxes_xyxy=[[-10, -10, 700, 700]],
        confidences=[0.5],
        class_ids=[6],
        class_names=LEGACY_NAMES,
        image_width=640,
        image_height=640,
    )
    assert (finding.bbox.x1, finding.bbox.x2) == (0.0, 640.0)
    assert finding.image_area_fraction == pytest.approx(1.0)


def test_unknown_class_id_raises() -> None:
    with pytest.raises(ValueError, match="class id 9"):
        findings_from_predictions(
            photo_id="p1",
            boxes_xyxy=[[0, 0, 1, 1]],
            confidences=[0.5],
            class_ids=[9],
            class_names=LEGACY_NAMES,
            image_width=640,
            image_height=640,
        )


def test_mismatched_lengths_raise() -> None:
    with pytest.raises(ValueError, match="same length"):
        findings_from_predictions(
            photo_id="p1",
            boxes_xyxy=[[0, 0, 1, 1]],
            confidences=[],
            class_ids=[1],
            class_names=LEGACY_NAMES,
            image_width=640,
            image_height=640,
        )
