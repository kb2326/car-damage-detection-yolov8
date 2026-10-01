from collections.abc import Callable, Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from PIL import Image

from claimlens.events.store import SQLiteEventStore


@pytest.fixture
def fixed_clock() -> Callable[[], datetime]:
    current = datetime(2026, 10, 8, 9, 0, tzinfo=UTC)

    def tick() -> datetime:
        nonlocal current
        current += timedelta(seconds=1)
        return current

    return tick


@pytest.fixture
def store(tmp_path: Path, fixed_clock: Callable[[], datetime]) -> Iterator[SQLiteEventStore]:
    event_store = SQLiteEventStore(tmp_path / "events.db", clock=fixed_clock)
    yield event_store
    event_store.close()


@pytest.fixture
def make_image(tmp_path: Path) -> Callable[..., Path]:
    """Create a solid-colour test image. Different colours give different file hashes."""

    def _make(
        name: str,
        size: tuple[int, int] = (640, 480),
        color: tuple[int, int, int] = (200, 30, 30),
    ) -> Path:
        path = tmp_path / "images" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        image_format = "PNG" if path.suffix.lower() == ".png" else "JPEG"
        Image.new("RGB", size, color).save(path, format=image_format)
        return path

    return _make
