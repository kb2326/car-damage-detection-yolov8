"""Integrity checks that raise fraud signals. M1: exact photo reuse across claims."""

from __future__ import annotations

from claimlens.domain import FraudSignal
from claimlens.events.projection import ClaimState
from claimlens.events.store import SQLiteEventStore


def check_integrity(state: ClaimState, store: SQLiteEventStore) -> tuple[FraudSignal, ...]:
    signals: list[FraudSignal] = []
    for photo in state.photos.values():
        other_claims = store.claims_with_photo(photo.sha256) - {state.claim_id}
        if other_claims:
            ids = ", ".join(sorted(str(claim) for claim in other_claims))
            signals.append(
                FraudSignal(
                    kind="photo_reuse",
                    score=1.0,
                    detail=f"Photo {photo.photo_id} also appears in claim(s) {ids}.",
                )
            )
    return tuple(signals)
