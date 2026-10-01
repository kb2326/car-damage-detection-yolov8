"""Helpers for data-engine tests."""

from __future__ import annotations

import random
from pathlib import Path

from PIL import Image, ImageDraw


def make_pattern_image(path: Path, seed: int, size: tuple[int, int] = (320, 240)) -> Path:
    """An image of seeded coloured rectangles: same seed, same picture."""
    rng = random.Random(seed)
    image = Image.new("RGB", size, (rng.randrange(256), rng.randrange(256), rng.randrange(256)))
    draw = ImageDraw.Draw(image)
    width, height = size
    for _ in range(6):
        x1, y1 = rng.randrange(width // 2), rng.randrange(height // 2)
        x2, y2 = x1 + rng.randrange(20, width // 2), y1 + rng.randrange(20, height // 2)
        colour = (rng.randrange(256), rng.randrange(256), rng.randrange(256))
        draw.rectangle((x1, y1, x2, y2), fill=colour)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG" if path.suffix.lower() == ".png" else "JPEG", quality=95)
    return path
