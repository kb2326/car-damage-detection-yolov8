"""Median CPU time per image for a detector, after one warm-up call."""

from __future__ import annotations

import statistics
import time
from collections.abc import Sequence
from pathlib import Path

from claimlens.vision.base import Detector


def benchmark_detector(detector: Detector, images: Sequence[Path]) -> float:
    if not images:
        raise ValueError("benchmark needs at least one image")
    detector.detect(images[0], "warm-up")
    timings: list[float] = []
    for index, image in enumerate(images):
        start = time.perf_counter()
        detector.detect(image, f"bench-{index}")
        timings.append((time.perf_counter() - start) * 1000.0)
    return statistics.median(timings)
