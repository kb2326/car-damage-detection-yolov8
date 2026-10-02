import importlib.util
from pathlib import Path

import pytest

from claimlens.cli import resolve_detector
from claimlens.domain import DamageType

ROOT = Path(__file__).resolve().parents[2]
SAMPLE = ROOT / "tests" / "fixtures" / "images" / "dent_1.jpg"


def _champion() -> Path | None:
    try:
        weights = resolve_detector("yolo-seg", None, ROOT / "config").weights
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
