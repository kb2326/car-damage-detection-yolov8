import importlib.util
from pathlib import Path

import pytest

from claimlens.domain import DamageType

ROOT = Path(__file__).resolve().parents[2]
WEIGHTS = ROOT / "models" / "legacy" / "yolov8n-cardamage-v6.pt"
SAMPLE = ROOT / "tests" / "fixtures" / "images" / "dent_1.jpg"

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not WEIGHTS.is_file() or importlib.util.find_spec("ultralytics") is None,
        reason="needs models/legacy weights and `uv sync --group vision`",
    ),
]


def test_legacy_detector_returns_known_damage_types() -> None:
    from claimlens.vision.legacy_yolo import LegacyYoloDetector

    detector = LegacyYoloDetector(WEIGHTS)
    findings = detector.detect(SAMPLE, "p1")
    assert detector.model_version == "legacy-yolov8n:yolov8n-cardamage-v6.pt"
    assert all(isinstance(f.damage_type, DamageType) for f in findings)
    assert all(f.photo_id == "p1" for f in findings)
