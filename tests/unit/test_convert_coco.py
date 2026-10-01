import json
from pathlib import Path

import pytest

from claimlens.data.convert_coco import convert_coco_split
from claimlens.data.taxonomy import Taxonomy, UnknownLabelError
from tests.data_helpers import make_pattern_image

TAXONOMY = Taxonomy(name="damage", classes=("crack", "dent"), ignore=("smash",))


def _coco(
    tmp_path: Path, categories: list[dict[str, object]], annotations: list[dict[str, object]]
) -> Path:
    split_dir = tmp_path / "data" / "raw" / "cardd" / "train"
    make_pattern_image(split_dir / "000581_jpg.rf.abc.jpg", seed=1, size=(200, 100))
    make_pattern_image(split_dir / "other.jpg", seed=2, size=(200, 100))
    payload = {
        "categories": categories,
        "images": [
            {"id": 0, "file_name": "000581_jpg.rf.abc.jpg", "width": 200, "height": 100},
            {"id": 1, "file_name": "other.jpg", "width": 200, "height": 100},
        ],
        "annotations": annotations,
    }
    path = split_dir / "_annotations.coco.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


CATEGORIES: list[dict[str, object]] = [
    {"id": 0, "name": "dent", "supercategory": "none"},
    {"id": 1, "name": "crack", "supercategory": "dent"},
    {"id": 2, "name": "dent", "supercategory": "dent"},
    {"id": 3, "name": "smash", "supercategory": "dent"},
]


def test_coco_polygons_are_normalised_and_labels_mapped(tmp_path: Path) -> None:
    annotations: list[dict[str, object]] = [
        {"id": 0, "image_id": 0, "category_id": 2, "segmentation": [[0, 0, 100, 0, 100, 50]]},
        {"id": 1, "image_id": 0, "category_id": 1, "segmentation": [[0, 0, 200, 0, 200, 100]]},
        {"id": 2, "image_id": 1, "category_id": 3, "segmentation": [[0, 0, 10, 0, 10, 10]]},
    ]
    result = convert_coco_split(
        _coco(tmp_path, CATEGORIES, annotations),
        source="cardd",
        split="train",
        taxonomy=TAXONOMY,
        repo_root=tmp_path,
    )

    first, second = result.records
    assert first.image_id == "cardd:000581_jpg.rf.abc"
    assert first.origin_id == "000581"
    assert first.path == "data/raw/cardd/train/000581_jpg.rf.abc.jpg"
    assert [a.label for a in first.annotations] == ["dent", "crack"]
    assert first.annotations[0].polygon == (0.0, 0.0, 0.5, 0.0, 0.5, 0.5)
    assert second.annotations == ()
    assert second.origin_id is None
    assert result.ignored_labels == {"smash": 1}


def test_coco_unknown_category_raises(tmp_path: Path) -> None:
    categories: list[dict[str, object]] = [{"id": 5, "name": "Headlight-Damage"}]
    annotations: list[dict[str, object]] = [
        {"id": 0, "image_id": 0, "category_id": 5, "segmentation": [[0, 0, 1, 0, 1, 1]]}
    ]
    with pytest.raises(UnknownLabelError, match="Headlight-Damage"):
        convert_coco_split(
            _coco(tmp_path, categories, annotations),
            source="cardd",
            split="train",
            taxonomy=TAXONOMY,
            repo_root=tmp_path,
        )


def test_coco_rle_masks_are_rejected(tmp_path: Path) -> None:
    annotations: list[dict[str, object]] = [
        {
            "id": 7,
            "image_id": 0,
            "category_id": 2,
            "segmentation": {"counts": "abc", "size": [100, 200]},
        }
    ]
    with pytest.raises(ValueError, match="annotation 7 uses RLE"):
        convert_coco_split(
            _coco(tmp_path, CATEGORIES, annotations),
            source="cardd",
            split="train",
            taxonomy=TAXONOMY,
            repo_root=tmp_path,
        )
