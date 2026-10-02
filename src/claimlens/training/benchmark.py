"""Median CPU time per image for a detector, after one warm-up call."""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable, Sequence
from pathlib import Path

from claimlens.vision.base import Detector


def benchmark_callable(fn: Callable[[Path], object], images: Sequence[Path]) -> float:
    """Median milliseconds per image for `fn`, after one warm-up call."""
    if not images:
        raise ValueError("benchmark needs at least one image")
    fn(images[0])
    timings: list[float] = []
    for image in images:
        start = time.perf_counter()
        fn(image)
        timings.append((time.perf_counter() - start) * 1000.0)
    return statistics.median(timings)


def benchmark_detector(detector: Detector, images: Sequence[Path]) -> float:
    return benchmark_callable(lambda image: detector.detect(image, "bench"), images)
