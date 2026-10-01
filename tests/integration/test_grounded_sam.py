import importlib.util
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SAMPLE = ROOT / "tests" / "fixtures" / "images" / "dent_1.jpg"

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        importlib.util.find_spec("transformers") is None or os.environ.get("CLAIMLENS_SLOW") != "1",
        reason="needs `uv sync --group autolabel` and CLAIMLENS_SLOW=1 (downloads ~800 MB)",
    ),
]


def test_grounded_sam_proposes_known_part_groups() -> None:
    from claimlens.autolabel.grounded_sam import GroundedSamLabeller

    labeller = GroundedSamLabeller(
        {"car door": "door", "wheel": "wheel"},
        box_threshold=0.3,
        text_threshold=0.25,
        min_score=0.3,
        max_per_group=2,
        detector="IDEA-Research/grounding-dino-tiny",
        segmenter="facebook/sam2.1-hiera-tiny",
    )
    annotations, _dropped = labeller.label(SAMPLE)
    assert all(a.label in {"door", "wheel"} for a in annotations)
    assert all(a.score is not None and len(a.polygon) >= 6 for a in annotations)
