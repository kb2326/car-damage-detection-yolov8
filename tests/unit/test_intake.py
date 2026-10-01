from collections.abc import Callable
from pathlib import Path

import pytest

from claimlens.blobs import BlobStore
from claimlens.events.projection import fold
from claimlens.events.store import SQLiteEventStore
from claimlens.intake import submit_claim


def test_blob_store_is_content_addressed(tmp_path: Path, make_image: Callable[..., Path]) -> None:
    image = make_image("front.JPG")
    blobs = BlobStore(tmp_path / "blobs")
    first = blobs.put(image)
    second = blobs.put(image)
    assert first == second
    assert first.name == f"{first.sha256}.jpg"
    assert blobs.path(first.name).read_bytes() == image.read_bytes()


def test_submit_claim_records_report_and_photos(
    store: SQLiteEventStore, tmp_path: Path, make_image: Callable[..., Path]
) -> None:
    photos = [make_image("front.jpg"), make_image("rear.jpg", color=(10, 20, 30))]
    claim_id = submit_claim(
        store,
        BlobStore(tmp_path / "blobs"),
        policy_id="P-1001",
        description="scrape",
        photo_paths=photos,
    )
    state = fold(store.load(claim_id))
    assert state.policy_id == "P-1001"
    assert sorted(state.photos) == ["p1", "p2"]
    assert state.photos["p2"].filename == "rear.jpg"


def test_submit_claim_requires_a_photo(store: SQLiteEventStore, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least one photo"):
        submit_claim(
            store, BlobStore(tmp_path / "b"), policy_id="P-1001", description="", photo_paths=[]
        )


def test_submit_claim_checks_files_before_writing(store: SQLiteEventStore, tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="photo not found"):
        submit_claim(
            store,
            BlobStore(tmp_path / "b"),
            policy_id="P-1001",
            description="",
            photo_paths=[tmp_path / "missing.jpg"],
        )
    assert store.claim_ids() == []
