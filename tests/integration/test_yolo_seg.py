import importlib.util
from pathlib import Path

import pytest

from claimlens.cli import resolve_detector
from claimlens.domain import DamageType

ROOT = Path(__file__).resolve().parents[2]
SAMPLE = ROOT / "tests" / "fixtures" / "images" / "dent_1.jpg"


def _champion() -> Path | None:
    try:
        _, weights = resolve_detector("yolo-seg", None, ROOT / "config")
    except ValueError:
        return None
    path = ROOT / weights
    return path if path.is_file() else None


pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        _champion() is None or importlib.util.find_spec("ultralytics") is None,
        reason="needs a champion in config/models.toml, its weights and `uv sync --group vision`",
    ),
]


def test_yolo_seg_returns_findings_with_mask_areas() -> None:
    from claimlens.vision.yolo_seg import YoloSegDetector

    weights = _champion()
    assert weights is not None
    detector = YoloSegDetector(weights, run=weights.parent.name)
    findings = detector.detect(SAMPLE, "p1")
    assert detector.model_version.startswith("yolo11-seg:")
    assert all(isinstance(f.damage_type, DamageType) for f in findings)
    assert all(0.0 <= f.image_area_fraction <= 1.0 for f in findings)


def test_yolo_seg_sizes_damage_by_its_box() -> None:
    """The rate card bands and the golden oracle are defined on box area (M3a gate finding)."""
    from PIL import Image

    from claimlens.vision.yolo_seg import YoloSegDetector

    weights = _champion()
    assert weights is not None
    width, height = Image.open(SAMPLE).size
    findings = YoloSegDetector(weights, run=weights.parent.name).detect(SAMPLE, "p1")
    for f in findings:
        box_area = (f.bbox.x2 - f.bbox.x1) * (f.bbox.y2 - f.bbox.y1)
        assert f.image_area_fraction == pytest.approx(box_area / (width * height), abs=1e-6)
