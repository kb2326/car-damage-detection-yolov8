"""Integrity checks that raise fraud signals: a photo reused from another claim, exactly (M1) or
as a near-copy found in the claim memory (M7, ADR 0019)."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from claimlens.domain import FraudSignal
from claimlens.events.projection import ClaimState, PhotoStatus
from claimlens.events.store import SQLiteEventStore
from claimlens.memory.index import NEAR_COPY_DISTANCE, hamming
from claimlens.memory.records import phash_of

if TYPE_CHECKING:
    from claimlens.memory.index import ClaimMemory

log = logging.getLogger(__name__)
NEAR_COPY_SCORE = 0.8  # strong evidence, a little below an exact copy (1.0); R1 needs 0.5


def check_integrity(
    state: ClaimState,
    store: SQLiteEventStore,
    memory: ClaimMemory | None = None,
    photo_path: Callable[[str], Path] | None = None,
) -> tuple[FraudSignal, ...]:
    signals, exact = _exact_copies(state, store)
    if memory is not None and photo_path is not None:
        try:
            signals += _near_copies(state, memory, photo_path, skip=exact)
        except Exception:  # memory or a photo unreadable: keep the exact-copy evidence
            log.exception("near-copy check skipped for claim %s", state.claim_id)
    return tuple(signals)


def _exact_copies(state: ClaimState, store: SQLiteEventStore) -> tuple[list[FraudSignal], set[str]]:
    signals: list[FraudSignal] = []
    reused: set[str] = set()
    for photo in state.photos.values():
        other_claims = store.claims_with_photo(photo.sha256) - {state.claim_id}
        if other_claims:
            reused.add(photo.photo_id)
            ids = ", ".join(sorted(str(claim) for claim in other_claims))
            signals.append(
                FraudSignal(
                    kind="photo_reuse",
                    score=1.0,
                    detail=f"Photo {photo.photo_id} also appears in claim(s) {ids}.",
                )
            )
    return signals, reused


def _near_copies(
    state: ClaimState,
    memory: ClaimMemory,
    photo_path: Callable[[str], Path],
    skip: set[str],
) -> list[FraudSignal]:
    """A re-saved, resized or lightly cropped copy of a photo in another remembered claim. A photo
    already reported as an exact copy is not reported twice."""
    others = [r for r in memory.all() if r.claim_id != str(state.claim_id)]
    found: list[FraudSignal] = []
    for photo in state.photos.values():
        if photo.status is PhotoStatus.REJECTED or photo.photo_id in skip:
            continue
        mine = phash_of(photo_path(photo.blob_name))
        matches = [
            (hamming(mine, theirs), record.claim_id)
            for record in others
            for theirs in record.photo_phashes
        ]
        best = min((m for m in matches if m[0] <= NEAR_COPY_DISTANCE), default=None)
        if best is not None:
            distance, claim = best
            found.append(
                FraudSignal(
                    kind="photo_near_copy",
                    score=NEAR_COPY_SCORE,
                    detail=f"Photo {photo.photo_id} is a near-copy ({distance} bits apart) "
                    f"of a photo in claim {claim}.",
                )
            )
    return found
