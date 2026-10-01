from collections.abc import Callable, Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

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
