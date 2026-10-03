"""The claim memory on LanceDB: upsert, forget, rebuild, and "have we seen this before?"."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any
from uuid import UUID

from claimlens.domain import Frozen
from claimlens.events.projection import fold
from claimlens.events.store import SQLiteEventStore
from claimlens.knowledge.embed import Embedder
from claimlens.memory.records import MemoryRecord, build_record

TABLE = "claim_memory"
# Measured on golden photos: a re-save or resize moves the hash at most 2 bits and a 2% crop at
# most 10, while two different photos were never closer than 18. Heavier crops are not caught.
NEAR_COPY_DISTANCE = 10
_CANDIDATES = 20


class SimilarMemory(Frozen):
    claim_id: str
    reason: str  # near-copy photo, same policy, similar damage
    distance: int | None  # Hamming distance for a near-copy photo
    route: str
    review_action: str
    damage: str
    cost_low: int
    cost_high: int


def _claim_id(value: str) -> str:
    try:
        return str(UUID(value))  # also keeps LanceDB filters injection-safe
    except ValueError:
        raise ValueError(f"not a claim id: {value!r}") from None


def hamming(a: str, b: str) -> int:
    return (int(a, 16) ^ int(b, 16)).bit_count()


def _damage_types(damage: str) -> set[str]:
    return {item.split(" on ")[0].strip() for item in damage.split(";") if item.strip()}


class ClaimMemory:
    def __init__(self, path: Path, embedder: Embedder) -> None:
        import lancedb

        path.mkdir(parents=True, exist_ok=True)
        self._db: Any = lancedb.connect(str(path))
        self._embedder = embedder

    def _table(self) -> Any | None:
        if TABLE not in self._db.table_names():
            return None
        return self._db.open_table(TABLE)

    def upsert(self, record: MemoryRecord) -> None:
        """Write or replace the record; a claim is never in memory twice."""
        claim = _claim_id(record.claim_id)
        row = {
            "claim_id": claim,
            "policy_id": record.policy_id,
            "record": record.model_dump_json(),
            "vector": self._embedder.embed([record.summary])[0],
        }
        table = self._table()
        if table is None:
            self._db.create_table(TABLE, data=[row])
            return
        table.delete(f"claim_id = '{claim}'")
        table.add([row])

    def forget(self, claim_id: str) -> bool:
        """Delete the claim's row, and the old table versions that still hold it."""
        claim = _claim_id(claim_id)
        table = self._table()
        if table is None or self.get(claim) is None:
            return False
        table.delete(f"claim_id = '{claim}'")
        table.optimize(cleanup_older_than=timedelta(0))  # LanceDB keeps old versions otherwise
        return True

    def get(self, claim_id: str) -> MemoryRecord | None:
        claim = _claim_id(claim_id)
        return next((r for r in self.all() if r.claim_id == claim), None)

    def all(self) -> list[MemoryRecord]:
        table = self._table()
        if table is None:
            return []
        rows = table.to_arrow().select(["record"]).to_pylist()
        return [MemoryRecord.model_validate_json(row["record"]) for row in rows]

    def similar(self, record: MemoryRecord, limit: int = 5) -> list[SimilarMemory]:
        """Earlier claims like this one: near-copy photos, then same policy, then similar damage."""
        others = [r for r in self.all() if r.claim_id != record.claim_id]
        found: dict[str, SimilarMemory] = {}

        def add(other: MemoryRecord, reason: str, distance: int | None = None) -> None:
            if other.claim_id not in found:
                found[other.claim_id] = SimilarMemory(
                    claim_id=other.claim_id,
                    reason=reason,
                    distance=distance,
                    route=other.final_route or other.route,
                    review_action=other.review_action,
                    damage=other.damage,
                    cost_low=other.cost_low,
                    cost_high=other.cost_high,
                )

        copies = []
        for other in others:
            distances = [
                hamming(mine, theirs)
                for mine in record.photo_phashes
                for theirs in other.photo_phashes
            ]
            best = min(distances, default=NEAR_COPY_DISTANCE + 1)
            if best <= NEAR_COPY_DISTANCE:
                copies.append((best, other))
        for distance, other in sorted(copies, key=lambda item: item[0]):
            add(other, "near-copy photo", distance)
        for other in others:
            if other.policy_id == record.policy_id:
                add(other, "same policy")
        mine = _damage_types(record.damage)
        table = self._table()
        if mine and table is not None:
            vector = self._embedder.embed([record.summary])[0]
            rows = table.search(vector).limit(_CANDIDATES).to_list()
            for row in rows:
                other = MemoryRecord.model_validate_json(row["record"])
                if other.claim_id != record.claim_id and mine & _damage_types(other.damage):
                    add(other, "similar damage")
        return list(found.values())[:limit]


@dataclass(frozen=True)
class RebuildResult:
    remembered: int
    skipped: list[str]  # "<claim id>: why"


def rebuild(
    memory: ClaimMemory, store: SQLiteEventStore, photo_path: Callable[[str], Path]
) -> RebuildResult:
    """Rebuild memory from every decided claim's log (memory is derived data). A claim that cannot
    be read (say a missing photo) is skipped and reported; the rest carry on."""
    count = 0
    skipped: list[str] = []
    for claim_id in store.claim_ids():
        events = store.load(claim_id)
        state = fold(events)
        if state.decision is None or any(e.type == "MemoryForgotten" for e in events):
            continue
        try:
            memory.upsert(build_record(state, events, photo_path))
        except Exception as exc:
            skipped.append(f"{claim_id}: {type(exc).__name__}: {exc}")
            continue
        count += 1
    return RebuildResult(remembered=count, skipped=skipped)
