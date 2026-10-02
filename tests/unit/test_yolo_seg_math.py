from pathlib import Path

import pytest

from claimlens.cli import DetectorSpec, resolve_detector
from claimlens.domain import DamageType
from claimlens.training.benchmark import benchmark_detector
from claimlens.vision.base import findings_from_predictions, polygon_area_fraction
from tests.fakes import FakeDetector


def test_unit_square_quarter_has_quarter_area() -> None:
    assert polygon_area_fraction([0.0, 0.0, 0.5, 0.0, 0.5, 0.5, 0.0, 0.5]) == pytest.approx(0.25)


def test_orientation_does_not_matter() -> None:
    clockwise = [0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.0]
    assert polygon_area_fraction(clockwise) == pytest.approx(0.25)


def test_degenerate_polygon_has_zero_area() -> None:
    assert polygon_area_fraction([]) == 0.0
    assert polygon_area_fraction([0.1, 0.1, 0.2, 0.2]) == 0.0


def test_area_fraction_is_clamped() -> None:
    assert polygon_area_fraction([-0.1, -0.1, 1.1, -0.1, 1.1, 1.1, -0.1, 1.1]) == 1.0


def test_odd_number_of_coordinates_is_rejected() -> None:
    with pytest.raises(ValueError, match="pairs"):
        polygon_area_fraction([0.1, 0.2, 0.3])


def _findings(area_fractions: list[float] | None) -> list[float]:
    findings = findings_from_predictions(
        photo_id="p1",
        boxes_xyxy=[[0, 0, 100, 100]],
        confidences=[0.9],
        class_ids=[1],
        class_names={1: "dent"},
        image_width=200,
        image_height=200,
        area_fractions=area_fractions,
    )
    assert findings[0].damage_type is DamageType.DENT
    return [f.image_area_fraction for f in findings]


def test_mask_area_replaces_box_area() -> None:
    assert _findings(None) == [pytest.approx(0.25)]
    assert _findings([0.04]) == [pytest.approx(0.04)]


def test_mismatched_area_fractions_are_rejected() -> None:
    with pytest.raises(ValueError, match="same length"):
        _findings([0.1, 0.2])


def test_no_predictions_give_no_findings() -> None:
    assert (
        findings_from_predictions(
            photo_id="p1",
            boxes_xyxy=[],
            confidences=[],
            class_ids=[],
            class_names={0: "dent"},
            image_width=10,
            image_height=10,
            area_fractions=[],
        )
        == []
    )


DAMAGE_ONLY = '[damage]\nrun = "s"\nweights = "models/damage/s/best.pt"\nmlflow_version = "2"\n'
BOTH = (
    DAMAGE_ONLY
    + "temperature = 1.4\n"
    + '[parts]\nrun = "p"\nweights = "models/parts/p/best.pt"\nmlflow_version = "1"\n'
)


def test_resolve_prefers_explicit_arguments(tmp_path: Path) -> None:
    spec = resolve_detector("legacy", tmp_path / "w.pt", tmp_path)
    assert spec == DetectorSpec(kind="legacy", weights=tmp_path / "w.pt")


def test_resolve_defaults_to_legacy_without_a_champion(tmp_path: Path) -> None:
    spec = resolve_detector(None, None, tmp_path)
    assert spec.kind == "legacy"
    assert spec.weights.name == "yolov8n-cardamage-v6.pt"


def test_resolve_uses_the_damage_champion(tmp_path: Path) -> None:
    (tmp_path / "models.toml").write_text(DAMAGE_ONLY, encoding="utf-8")
    spec = resolve_detector(None, None, tmp_path)
    assert (spec.kind, spec.weights) == ("yolo-seg", Path("models/damage/s/best.pt"))


def test_resolve_defaults_to_fused_with_both_champions(tmp_path: Path) -> None:
    (tmp_path / "models.toml").write_text(BOTH, encoding="utf-8")
    spec = resolve_detector(None, None, tmp_path)
    assert spec.kind == "fused"
    assert spec.weights == Path("models/damage/s/best.pt")
    assert spec.parts_weights == Path("models/parts/p/best.pt")
    assert spec.temperature == 1.4
    assert spec.taxonomy == tmp_path / "taxonomy.toml"


def test_fused_without_parts_is_an_error(tmp_path: Path) -> None:
    (tmp_path / "models.toml").write_text(DAMAGE_ONLY, encoding="utf-8")
    with pytest.raises(ValueError, match="--task parts"):
        resolve_detector("fused", None, tmp_path)


def test_resolve_yolo_without_champion_or_weights_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="train select"):
        resolve_detector("yolo-seg", None, tmp_path)


def test_benchmark_returns_median_milliseconds(tmp_path: Path) -> None:
    images = [tmp_path / f"{i}.jpg" for i in range(3)]
    detector = FakeDetector()
    assert benchmark_detector(detector, images) >= 0.0
    assert len(detector.calls) == 4  # one warm-up call plus one per image


def test_benchmark_needs_images() -> None:
    with pytest.raises(ValueError, match="at least one image"):
        benchmark_detector(FakeDetector(), [])


def test_weights_without_a_detector_flag_keep_the_legacy_meaning(tmp_path: Path) -> None:
    (tmp_path / "models.toml").write_text(
        '[damage]\nrun = "s"\nweights = "models/damage/s/best.pt"\nmlflow_version = "2"\n',
        encoding="utf-8",
    )
    assert resolve_detector(None, tmp_path / "old.pt", tmp_path) == DetectorSpec(
        kind="legacy", weights=tmp_path / "old.pt"
    )


def test_explicit_weights_are_not_given_the_champions_temperature(tmp_path: Path) -> None:
    (tmp_path / "models.toml").write_text(BOTH, encoding="utf-8")
    spec = resolve_detector("fused", tmp_path / "other.pt", tmp_path)
    assert spec.weights == tmp_path / "other.pt"
    assert spec.temperature is None
    champion = resolve_detector("fused", Path("models/damage/s/best.pt"), tmp_path)
    assert champion.temperature == 1.4
