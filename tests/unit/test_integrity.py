from collections.abc import Callable
from pathlib import Path

from claimlens.blobs import BlobStore
from claimlens.events.projection import fold
from claimlens.events.store import SQLiteEventStore
from claimlens.intake import submit_claim
from claimlens.integrity import check_integrity


def test_photo_reused_from_another_claim_raises_a_signal(
    store: SQLiteEventStore, tmp_path: Path, make_image: Callable[..., Path]
) -> None:
    blobs = BlobStore(tmp_path / "blobs")
    image = make_image("a.jpg")
    first = submit_claim(store, blobs, policy_id="P-1001", description="", photo_paths=[image])
    second = submit_claim(store, blobs, policy_id="P-1002", description="", photo_paths=[image])

    signals = check_integrity(fold(store.load(second)), store)

    assert len(signals) == 1
    assert signals[0].kind == "photo_reuse"
    assert signals[0].score == 1.0
    assert str(first) in signals[0].detail


def test_same_photo_twice_in_one_claim_is_not_reuse(
    store: SQLiteEventStore, tmp_path: Path, make_image: Callable[..., Path]
) -> None:
    image = make_image("a.jpg")
    claim = submit_claim(
        store,
        BlobStore(tmp_path / "blobs"),
        policy_id="P-1001",
        description="",
        photo_paths=[image, image],
    )
    assert check_integrity(fold(store.load(claim)), store) == ()


def test_fresh_photos_raise_nothing(
    store: SQLiteEventStore, tmp_path: Path, make_image: Callable[..., Path]
) -> None:
    claim = submit_claim(
        store,
        BlobStore(tmp_path / "blobs"),
        policy_id="P-1001",
        description="",
        photo_paths=[make_image("a.jpg")],
    )
    assert check_integrity(fold(store.load(claim)), store) == ()
