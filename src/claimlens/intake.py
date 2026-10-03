"""First notice of loss: record the report and the uploaded photos."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from uuid import UUID, uuid4

from claimlens.blobs import BlobStore
from claimlens.events.envelope import Actor, ActorKind
from claimlens.events.payloads import ClaimReported, PhotoUploaded
from claimlens.events.store import SQLiteEventStore

CLAIMANT = Actor(kind=ActorKind.HUMAN, name="claimant")


def submit_claim(
    store: SQLiteEventStore,
    blobs: BlobStore,
    *,
    policy_id: str,
    description: str,
    photo_paths: Sequence[Path],
    claim_id: UUID | None = None,
    allow_no_photos: bool = False,
) -> UUID:
    """Record a claim. `allow_no_photos` is for intake: a claim with no usable photo is still
    filed, and rule R3 sends it to a person."""
    if not photo_paths and not allow_no_photos:
        raise ValueError("a claim needs at least one photo")
    for path in photo_paths:
        if not path.is_file():
            raise FileNotFoundError(f"photo not found: {path}")

    new_id = claim_id or uuid4()
    store.append(new_id, ClaimReported(policy_id=policy_id, description=description), CLAIMANT)
    for index, path in enumerate(photo_paths, start=1):
        blob = blobs.put(path)
        store.append(
            new_id,
            PhotoUploaded(
                photo_id=f"p{index}",
                sha256=blob.sha256,
                filename=path.name,
                blob_name=blob.name,
                size_bytes=path.stat().st_size,
            ),
            CLAIMANT,
        )
    return new_id
