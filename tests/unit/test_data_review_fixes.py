"""Regression tests for the M2a whole-branch review findings."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from claimlens.cli import main
from claimlens.data.config import SourceConfig
from claimlens.data.convert_coco import convert_coco_split
from claimlens.data.convert_yolo import convert_yolo_split
from claimlens.data.export import export_yolo_seg
from claimlens.data.records import Annotation, ImageRecord
from claimlens.data.taxonomy import Taxonomy
from tests.data_helpers import make_pattern_image

ROOT = Path(__file__).resolve().parents[2]
TAXONOMY = Taxonomy(name="damage", classes=("dent",))


def test_data_yaml_has_no_machine_specific_path(tmp_path: Path) -> None:
    make_pattern_image(tmp_path / "a.jpg", seed=1)
    record = ImageRecord(
        image_id="s:a", source="s", source_split="train", path="a.jpg", width=320, height=240
    )
    out = tmp_path / "out"
    export_yolo_seg([record], {"s:a": "train"}, TAXONOMY, repo_root=tmp_path, out_dir=out)
    yaml_text = (out / "data.yaml").read_text(encoding="utf-8")
    assert "path:" not in yaml_text
    assert str(tmp_path.as_posix()) not in yaml_text


def test_unknown_split_name_is_rejected() -> None:
    with pytest.raises(ValidationError, match="train"):
        SourceConfig.model_validate(
            {
                "id": "x",
                "kind": "local",
                "format": "yolo-seg",
                "terms": "t",
                "splits": {"val": {"images": "images/val", "labels": "labels/val"}},
            }
        )


def test_empty_coco_segmentation_is_rejected(tmp_path: Path) -> None:
    make_pattern_image(tmp_path / "a.jpg", seed=1, size=(200, 100))
    payload = {
        "categories": [{"id": 1, "name": "dent"}],
        "images": [{"id": 0, "file_name": "a.jpg", "width": 200, "height": 100}],
        "annotations": [
            {"id": 9, "image_id": 0, "category_id": 1, "bbox": [0, 0, 10, 10], "segmentation": []}
        ],
    }
    path = tmp_path / "_annotations.coco.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="annotation 9 has no polygon"):
        convert_coco_split(path, source="s", split="train", taxonomy=TAXONOMY, repo_root=tmp_path)


def test_missing_yolo_labels_folder_is_rejected(tmp_path: Path) -> None:
    make_pattern_image(tmp_path / "images" / "a.jpg", seed=1)
    with pytest.raises(FileNotFoundError, match="labels folder"):
        convert_yolo_split(
            tmp_path / "images",
            tmp_path / "lables",
            source="s",
            split="train",
            class_names=["dent"],
            taxonomy=TAXONOMY,
            repo_root=tmp_path,
        )


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["data", "build", "nope"], "unknown dataset 'nope'"),
        (["data", "fetch", "nope"], "unknown source 'nope'"),
    ],
)
def test_unknown_ids_give_a_clear_error(
    args: list[str], message: str, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["--config", str(ROOT / "config"), *args]) == 2
    assert message in capsys.readouterr().err


def test_missing_config_gives_a_clear_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["--config", str(tmp_path / "nowhere"), "data", "build", "damage-v1"]) == 2
    assert "error:" in capsys.readouterr().err


def test_build_errors_are_reported_not_raised(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config = tmp_path / "config"
    config.mkdir()
    (config / "taxonomy.toml").write_text('[damage]\nclasses = ["dent"]\n', encoding="utf-8")
    (config / "datasets.toml").write_text(
        '[source.toy]\nkind = "local"\nformat = "yolo-seg"\nclass_names_file = "data.yaml"\n'
        'terms = "t"\n[source.toy.splits]\n'
        'train = { images = "train/images", labels = "train/labels" }\n'
        '[dataset.toy-v1]\ntaxonomy = "damage"\nsources = ["toy"]\n',
        encoding="utf-8",
    )
    raw = tmp_path / "data" / "raw" / "toy"
    make_pattern_image(raw / "train" / "images" / "a.jpg", seed=1)
    (raw / "train" / "labels").mkdir(parents=True)
    (raw / "train" / "labels" / "a.txt").write_text("0 0.1 0.1 0.5 0.1 0.5 0.5\n", encoding="utf-8")
    (raw / "data.yaml").write_text("names: ['smashed']\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    assert main(["--config", str(config), "data", "build", "toy-v1"]) == 1
    assert "not in the damage taxonomy" in capsys.readouterr().err


def test_degenerate_annotation_helper_still_exported_cleanly(tmp_path: Path) -> None:
    make_pattern_image(tmp_path / "a.jpg", seed=1)
    record = ImageRecord(
        image_id="s:a",
        source="s",
        source_split="train",
        path="a.jpg",
        width=320,
        height=240,
        annotations=(Annotation(label="dent", polygon=(1.002, 0.1, 1.005, 0.1, 1.004, 0.2)),),
    )
    out = tmp_path / "out"
    export_yolo_seg([record], {"s:a": "train"}, TAXONOMY, repo_root=tmp_path, out_dir=out)
    assert (out / "labels" / "train" / "s__a.txt").read_text(encoding="utf-8") == ""
