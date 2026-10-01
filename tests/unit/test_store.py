import sqlite3
from pathlib import Path
from uuid import UUID

import pytest

from claimlens.events.envelope import Actor, ActorKind, ChainIntegrityError
from claimlens.events.payloads import ClaimReported, PhotoUploaded
from claimlens.events.store import ClaimNotFoundError, SQLiteEventStore

ACTOR = Actor(kind=ActorKind.SYSTEM, name="test")
CLAIM_A = UUID(int=1)
CLAIM_B = UUID(int=2)


def _report() -> ClaimReported:
    return ClaimReported(policy_id="P-1001", description="scrape")


def _photo(photo_id: str, sha256: str) -> PhotoUploaded:
    return PhotoUploaded(
        photo_id=photo_id,
        sha256=sha256,
        filename=f"{photo_id}.jpg",
        blob_name=f"{sha256}.jpg",
        size_bytes=10,
    )


def test_append_assigns_gapless_sequence_and_links_hashes(store: SQLiteEventStore) -> None:
    first = store.append(CLAIM_A, _report(), ACTOR)
    second = store.append(CLAIM_A, _photo("p1", "a" * 64), ACTOR)
    assert (first.seq, second.seq) == (1, 2)
    assert second.prev_hash == first.hash


def test_sequences_are_per_claim(store: SQLiteEventStore) -> None:
    store.append(CLAIM_A, _report(), ACTOR)
    assert store.append(CLAIM_B, _report(), ACTOR).seq == 1


def test_load_returns_the_appended_events(store: SQLiteEventStore) -> None:
    appended = [
        store.append(CLAIM_A, _report(), ACTOR),
        store.append(CLAIM_A, _photo("p1", "a" * 64), ACTOR),
    ]
    assert store.load(CLAIM_A) == appended


def test_load_unknown_claim_raises(store: SQLiteEventStore) -> None:
    with pytest.raises(ClaimNotFoundError):
        store.load(UUID(int=99))


def test_claim_ids_lists_each_claim_once(store: SQLiteEventStore) -> None:
    store.append(CLAIM_A, _report(), ACTOR)
    store.append(CLAIM_A, _photo("p1", "a" * 64), ACTOR)
    store.append(CLAIM_B, _report(), ACTOR)
    assert sorted(store.claim_ids()) == [CLAIM_A, CLAIM_B]


def test_claims_with_photo_finds_every_claim_using_it(store: SQLiteEventStore) -> None:
    for claim in (CLAIM_A, CLAIM_B):
        store.append(claim, _report(), ACTOR)
        store.append(claim, _photo("p1", "c" * 64), ACTOR)
    assert store.claims_with_photo("c" * 64) == {CLAIM_A, CLAIM_B}
    assert store.claims_with_photo("d" * 64) == set()


def test_tampered_row_is_detected(tmp_path: Path) -> None:
    path = tmp_path / "events.db"
    event_store = SQLiteEventStore(path)
    event_store.append(CLAIM_A, _report(), ACTOR)
    event_store.append(CLAIM_A, _photo("p1", "a" * 64), ACTOR)
    with sqlite3.connect(path) as conn:
        conn.execute("UPDATE events SET data = replace(data, 'P-1001', 'P-9999') WHERE seq = 1")
    with pytest.raises(ChainIntegrityError, match="hash does not match"):
        event_store.load(CLAIM_A)
    event_store.close()


def test_store_persists_across_connections(tmp_path: Path) -> None:
    path = tmp_path / "events.db"
    first = SQLiteEventStore(path)
    first.append(CLAIM_A, _report(), ACTOR)
    first.close()
    second = SQLiteEventStore(path)
    assert len(second.load(CLAIM_A)) == 1
    second.close()


@pytest.mark.parametrize(
    "statement",
    [
        "UPDATE events SET type = 'Edited' WHERE seq = 2",
        "UPDATE events SET hash = 'x' WHERE seq = 2",
    ],
)
def test_edited_index_columns_are_detected(tmp_path: Path, statement: str) -> None:
    path = tmp_path / "events.db"
    event_store = SQLiteEventStore(path)
    event_store.append(CLAIM_A, _report(), ACTOR)
    event_store.append(CLAIM_A, _photo("p1", "a" * 64), ACTOR)
    with sqlite3.connect(path) as conn:
        conn.execute(statement)
    with pytest.raises(ChainIntegrityError, match="index columns"):
        event_store.load(CLAIM_A)
    event_store.close()


def test_photo_lookup_ignores_the_editable_type_column(tmp_path: Path) -> None:
    path = tmp_path / "events.db"
    event_store = SQLiteEventStore(path)
    event_store.append(CLAIM_A, _report(), ACTOR)
    event_store.append(CLAIM_A, _photo("p1", "c" * 64), ACTOR)
    with sqlite3.connect(path) as conn:
        conn.execute("UPDATE events SET type = 'Edited' WHERE seq = 2")
    assert event_store.claims_with_photo("c" * 64) == {CLAIM_A}
    event_store.close()
