"""Photo coaching for intake: the pipeline's quality check plus "too dark" and "too blurry".

These extra checks only decide whether to ask the customer for a retake. The pipeline's own
quality gate (claimlens.quality) is unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageFilter, ImageStat

from claimlens.intake_agent.config import IntakeConfig
from claimlens.quality import QualityConfig, check_quality


@dataclass(frozen=True)
class PhotoCheck:
    ok: bool
    reason: str = ""


def coach_photo(path: Path, config: IntakeConfig) -> PhotoCheck:
    basic = check_quality(path, QualityConfig())
    if not basic.ok:
        return PhotoCheck(ok=False, reason=basic.reason)
    with Image.open(path) as image:
        grey = image.convert("L")
    grey.thumbnail((512, 512))
    if ImageStat.Stat(grey).mean[0] < config.min_brightness:
        return PhotoCheck(ok=False, reason="too dark")
    edges = grey.filter(ImageFilter.FIND_EDGES)
    # The filter marks the image's own border as an edge; leave it out.
    inner = edges.crop((2, 2, max(edges.width - 2, 3), max(edges.height - 2, 3)))
    if ImageStat.Stat(inner).var[0] < config.min_edge_variance:
        return PhotoCheck(ok=False, reason="too blurry")
    return PhotoCheck(ok=True)
