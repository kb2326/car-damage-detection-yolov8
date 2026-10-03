import json
import random
from pathlib import Path

import pytest
from PIL import Image

from claimlens.intake_agent.config import load_intake_config
from claimlens.intake_agent.photos import coach_photo
from claimlens.quality import QualityConfig, check_quality

ROOT = Path(__file__).resolve().parents[2]
CONFIG = load_intake_config(ROOT / "config" / "intake.toml")


def _noise(path: Path) -> Path:
    rng = random.Random(1)
    image = Image.new("RGB", (640, 480))
    image.putdata([(rng.randrange(256),) * 3 for _ in range(640 * 480)])
    image.save(path)
    return path


def test_a_sharp_well_lit_photo_passes(tmp_path: Path) -> None:
    assert coach_photo(_noise(tmp_path / "ok.png"), CONFIG).ok


def test_a_dark_photo_is_too_dark(tmp_path: Path) -> None:
    path = tmp_path / "dark.png"
    Image.new("RGB", (640, 480), (5, 5, 5)).save(path)
    check = coach_photo(path, CONFIG)
    assert not check.ok
    assert check.reason == "too dark"


def test_a_flat_photo_is_too_blurry(tmp_path: Path) -> None:
    path = tmp_path / "flat.png"
    Image.new("RGB", (640, 480), (128, 128, 128)).save(path)
    check = coach_photo(path, CONFIG)
    assert not check.ok
    assert check.reason == "too blurry"


def test_the_pipeline_quality_reason_comes_first(tmp_path: Path) -> None:
    path = tmp_path / "x.jpg"
    path.write_bytes(b"not an image")
    check = coach_photo(path, CONFIG)
    assert not check.ok
    assert check.reason == check_quality(path, QualityConfig()).reason


def test_every_golden_photo_the_pipeline_accepts_passes_coaching() -> None:
    golden = ROOT / "evals" / "golden" / "v2" / "claims.jsonl"
    paths = sorted(
        {
            p
            for line in golden.read_text(encoding="utf-8").splitlines()
            for p in json.loads(line)["photos"]
        }
    )
    usable = [ROOT / p for p in paths if (ROOT / p).is_file()]
    if len(usable) < 50:
        pytest.skip("golden photos are not available (data/ is not in git)")
    accepted = [p for p in usable if check_quality(p, QualityConfig()).ok]
    rejected = [p.name for p in accepted if not coach_photo(p, CONFIG).ok]
    assert rejected == []
