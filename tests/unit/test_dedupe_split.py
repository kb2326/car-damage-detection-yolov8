from pathlib import Path

from PIL import Image

from claimlens.data.dedupe import ImageHash, find_clusters, hash_file
from claimlens.data.records import ImageRecord
from claimlens.data.split import assign_splits
from tests.data_helpers import make_pattern_image


def _record(image_id: str, split: str) -> ImageRecord:
    return ImageRecord(
        image_id=image_id,
        source="s",
        source_split=split,
        path=f"{image_id}.jpg",
        width=1,
        height=1,
    )


def test_resized_copy_is_a_near_duplicate_and_new_image_is_not(tmp_path: Path) -> None:
    original = make_pattern_image(tmp_path / "a.jpg", seed=1)
    resized = tmp_path / "a_small.jpg"
    with Image.open(original) as image:
        image.resize((256, 192)).save(resized, quality=80)
    other = make_pattern_image(tmp_path / "b.jpg", seed=2)

    hashes = [hash_file("a", original), hash_file("a_small", resized), hash_file("b", other)]
    clusters = find_clusters(hashes, max_distance=6)

    assert clusters["a"] == clusters["a_small"]
    assert clusters["b"] != clusters["a"]
    assert sorted(set(clusters.values())) == [0, 1]


def test_identical_bytes_cluster_even_with_distance_zero_threshold() -> None:
    hashes = [ImageHash("x", "same", 0b1111), ImageHash("y", "same", 0)]
    assert find_clusters(hashes, max_distance=0) == {"x": 0, "y": 0}


def test_cluster_spanning_train_and_test_goes_to_test() -> None:
    records = [_record("a", "train"), _record("b", "test"), _record("c", "train")]
    clusters = {"a": 0, "b": 0, "c": 1}
    assert assign_splits(records, clusters, protected_clusters=set()) == {
        "a": "test",
        "b": "test",
        "c": "train",
    }


def test_cluster_spanning_train_and_valid_goes_to_valid() -> None:
    records = [_record("a", "train"), _record("b", "valid")]
    assert assign_splits(records, {"a": 0, "b": 0}, set()) == {"a": "valid", "b": "valid"}


def test_protected_cluster_is_excluded() -> None:
    records = [_record("a", "train"), _record("b", "test")]
    assert assign_splits(records, {"a": 0, "b": 1}, protected_clusters={0}) == {
        "a": "excluded",
        "b": "test",
    }
