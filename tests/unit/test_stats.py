from claimlens.data.records import Annotation, ImageRecord
from claimlens.data.stats import dataset_stats, stats_markdown
from claimlens.data.taxonomy import Taxonomy

TAXONOMY = Taxonomy(name="damage", classes=("crack", "dent"))
SQUARE = (0.1, 0.1, 0.5, 0.1, 0.5, 0.5)


def _record(image_id: str, split: str, *labels: str) -> ImageRecord:
    return ImageRecord(
        image_id=image_id,
        source="s",
        source_split=split,
        path="x.jpg",
        width=1,
        height=1,
        annotations=tuple(Annotation(label=label, polygon=SQUARE) for label in labels),
    )


def test_stats_count_images_instances_and_leaks() -> None:
    records = [
        _record("a", "train", "dent", "dent"),
        _record("b", "train", "crack"),
        _record("c", "test", "dent"),
    ]
    splits = {"a": "train", "b": "test", "c": "test"}
    stats = dataset_stats(
        "damage-v1",
        records,
        splits,
        TAXONOMY,
        clusters={"a": 0, "b": 1, "c": 1},
        ignored_labels={"smash": 3},
        warning_counts={"tiny_instance": 2},
    )
    assert stats["images"] == {"train": 1, "test": 2}
    assert stats["instances"] == {
        "train": {"crack": 0, "dent": 2},
        "test": {"crack": 1, "dent": 1},
    }
    assert stats["leaks_prevented"] == 1
    assert stats["clusters"] == 2
    assert stats["ignored_labels"] == {"smash": 3}

    markdown = stats_markdown(stats)
    assert "# Dataset damage-v1" in markdown
    assert "| test | 2 | 1 | 1 |" in markdown
