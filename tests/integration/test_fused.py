import importlib.util
from pathlib import Path

import pytest

from claimlens.cli import DetectorSpec, _default_detector, resolve_detector

ROOT = Path(__file__).resolve().parents[2]
SAMPLE = ROOT / "tests" / "fixtures" / "images" / "dent_1.jpg"


def _spec() -> DetectorSpec | None:
    try:
        spec = resolve_detector("fused", None, ROOT / "config")
    except ValueError:
        return None
    if spec.parts_weights is None:
        return None
    ok = (ROOT / spec.weights).is_file() and (ROOT / spec.parts_weights).is_file()
    return spec if ok else None


pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        _spec() is None or importlib.util.find_spec("ultralytics") is None,
        reason="needs damage and parts champions, their weights and `uv sync --group vision`",
    ),
]


def test_fused_detector_runs_on_a_photo(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(ROOT)
    spec = _spec()
    assert spec is not None
    detector = _default_detector(spec)
    findings = detector.detect(SAMPLE, "p1")
    assert detector.model_version.startswith("fused:")
    assert all(f.part_area_ratio is None or f.part_area_ratio >= 0 for f in findings)
