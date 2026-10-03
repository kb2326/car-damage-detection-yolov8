"""A read-only event store, for the public showcase (M8b)."""

import sqlite3
from pathlib import Path
from uuid import uuid4

import pytest

from claimlens.events.envelope import Actor, ActorKind
from claimlens.events.payloads import ClaimReported
from claimlens.events.store import SQLiteEventStore

ACTOR = Actor(kind=ActorKind.SYSTEM, name="test")


def test_a_read_only_store_reads_and_verifies_but_never_writes(tmp_path: Path) -> None:
    path = tmp_path / "claims.db"
    claim = uuid4()
    writer = SQLiteEventStore(path)
    writer.append(claim, ClaimReported(policy_id="P-1001", description="x"), ACTOR)
    writer.close()
    reader = SQLiteEventStore(path, read_only=True)
    assert [e.type for e in reader.load(claim)] == ["ClaimReported"]
    with pytest.raises(sqlite3.OperationalError):
        reader.append(uuid4(), ClaimReported(policy_id="P-1001", description="y"), ACTOR)
    reader.close()


def test_a_missing_read_only_store_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        SQLiteEventStore(tmp_path / "nothing.db", read_only=True)
    assert not (tmp_path / "nothing.db").exists()
