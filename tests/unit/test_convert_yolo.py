from pathlib import Path

import pytest

from claimlens.data.convert_yolo import convert_yolo_split, parse_yolo_seg_line, read_class_names
from claimlens.data.taxonomy import Taxonomy
from tests.data_helpers import make_pattern_image

TAXONOMY = Taxonomy(name="damage", classes=("crack", "dent"), ignore=("smash",))
NAMES = ["crack", "dent", "smash"]


def test_box_line_becomes_rectangle_polygon() -> None:
    assert parse_yolo_seg_line("1 0.5 0.5 0.2 0.4") == (
        1,
        (0.4, 0.3, 0.6, 0.3, 0.6, 0.7, 0.4, 0.7),
    )


def test_bad_line_raises() -> None:
    with pytest.raises(ValueError, match="expected a box"):
        parse_yolo_seg_line("1 0.5 0.5 0.2 0.4 0.1")


def test_read_class_names_accepts_list_and_mapping(tmp_path: Path) -> None:
    listed = tmp_path / "a.yaml"
    listed.write_text("names: ['crack', 'dent']\n", encoding="utf-8")
    mapped = tmp_path / "b.yaml"
    mapped.write_text("names:\n  1: dent\n  0: crack\n", encoding="utf-8")
    assert read_class_names(listed) == ["crack", "dent"]
    assert read_class_names(mapped) == ["crack", "dent"]


def test_convert_yolo_split_reads_labels_by_name(tmp_path: Path) -> None:
    images = tmp_path / "data" / "raw" / "legacy" / "train" / "images"
    labels = tmp_path / "data" / "raw" / "legacy" / "train" / "labels"
    make_pattern_image(images / "a.jpg", seed=1)
    make_pattern_image(images / "b.png", seed=2)
    labels.mkdir(parents=True)
    (labels / "a.txt").write_text("1 0.1 0.1 0.4 0.1 0.4 0.4\n2 0 0 1 0 1 1\n\n", encoding="utf-8")

    result = convert_yolo_split(
        images,
        labels,
        source="legacy",
        split="train",
        class_names=NAMES,
        taxonomy=TAXONOMY,
        repo_root=tmp_path,
    )

    first, second = result.records
    assert first.image_id == "legacy:a"
    assert (first.width, first.height) == (320, 240)
    assert [a.label for a in first.annotations] == ["dent"]
    assert second.annotations == ()
    assert result.ignored_labels == {"smash": 1}


def test_out_of_range_class_index_names_the_file(tmp_path: Path) -> None:
    images = tmp_path / "images"
    labels = tmp_path / "labels"
    make_pattern_image(images / "a.jpg", seed=1)
    labels.mkdir()
    (labels / "a.txt").write_text("7 0.1 0.1 0.4 0.1 0.4 0.4\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"a\.txt:1: class index 7"):
        convert_yolo_split(
            images,
            labels,
            source="s",
            split="train",
            class_names=NAMES,
            taxonomy=TAXONOMY,
            repo_root=tmp_path,
        )
