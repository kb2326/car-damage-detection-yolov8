"""Intake quality gate: reject photos the models cannot use."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image, UnidentifiedImageError

_DEFAULT_FORMATS = frozenset({"JPEG", "PNG", "WEBP"})


@dataclass(frozen=True)
class QualityConfig:
    min_side_px: int = 320
    max_bytes: int = 15 * 1024 * 1024
    allowed_formats: frozenset[str] = _DEFAULT_FORMATS


@dataclass(frozen=True)
class QualityResult:
    ok: bool
    reason: str = ""
    width: int = 0
    height: int = 0


def check_quality(path: Path, config: QualityConfig) -> QualityResult:
    size = path.stat().st_size
    if size > config.max_bytes:
        return QualityResult(
            ok=False, reason=f"file is {size} bytes; the limit is {config.max_bytes} bytes"
        )
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            image_format = image.format or "unknown"
            width, height = image.size
    except (UnidentifiedImageError, OSError, SyntaxError):
        return QualityResult(ok=False, reason="not a readable image")
    if image_format not in config.allowed_formats:
        return QualityResult(ok=False, reason=f"unsupported format {image_format}")
    if min(width, height) < config.min_side_px:
        return QualityResult(
            ok=False,
            reason=(
                f"image too small ({width}x{height}); "
                f"the shortest side must be at least {config.min_side_px}px"
            ),
            width=width,
            height=height,
        )
    return QualityResult(ok=True, width=width, height=height)
