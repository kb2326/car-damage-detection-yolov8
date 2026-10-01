from pathlib import Path

from claimlens.data.export import export_yolo_seg
from claimlens.data.records import Annotation, ImageRecord
from claimlens.data.taxonomy import Taxonomy
from tests.data_helpers import make_pattern_image

TAXONOMY = Taxonomy(name="damage", classes=("crack", "dent"))


def _record(tmp_path: Path, image_id: str, split: str, *annotations: Annotation) -> ImageRecord:
    name = image_id.split(":")[1]
    make_pattern_image(tmp_path / "raw" / f"{name}.jpg", seed=len(name))
    return ImageRecord(
        image_id=image_id,
        source="s",
        source_split=split,
        path=f"raw/{name}.jpg",
        width=320,
        height=240,
        annotations=annotations,
    )


def test_export_writes_yolo_folders_and_yaml(tmp_path: Path) -> None:
    square = Annotation(label="dent", polygon=(0.1, 0.1, 0.5, 0.1, 0.5, 0.5))
    records = [
        _record(tmp_path, "s:a", "train", square),
        _record(tmp_path, "s:bb", "valid"),
        _record(tmp_path, "s:ccc", "test"),
    ]
    splits = {"s:a": "train", "s:bb": "valid", "s:ccc": "excluded"}
    out = tmp_path / "processed" / "damage-v1"
    (out / "stale.txt").parent.mkdir(parents=True)
    (out / "stale.txt").write_text("old", encoding="utf-8")

    counts = export_yolo_seg(records, splits, TAXONOMY, repo_root=tmp_path, out_dir=out)

    assert counts == {"train": 1, "val": 1, "excluded": 1}
    assert not (out / "stale.txt").exists()
    assert (out / "images" / "train" / "s__a.jpg").is_file()
    label = (out / "labels" / "train" / "s__a.txt").read_text(encoding="utf-8")
    assert label == "1 0.100000 0.100000 0.500000 0.100000 0.500000 0.500000\n"
    assert (out / "labels" / "val" / "s__bb.txt").read_text(encoding="utf-8") == ""
    yaml_text = (out / "data.yaml").read_text(encoding="utf-8")
    assert "val: images/val" in yaml_text
    assert "  1: dent" in yaml_text


def test_export_skips_degenerate_and_clamps_polygons(tmp_path: Path) -> None:
    records = [
        _record(
            tmp_path,
            "s:a",
            "train",
            Annotation(label="dent", polygon=(0.1, 0.1, 0.2, 0.2)),
            Annotation(label="dent", polygon=(0.1, 0.1, 0.2, 0.2, 0.3, 0.3)),
            Annotation(label="crack", polygon=(0.0, -0.004, 1.005, 0.0, 1.0, 1.0)),
        )
    ]
    out = tmp_path / "out"
    export_yolo_seg(records, {"s:a": "train"}, TAXONOMY, repo_root=tmp_path, out_dir=out)
    label = (out / "labels" / "train" / "s__a.txt").read_text(encoding="utf-8")
    assert label == "0 0.000000 0.000000 1.000000 0.000000 1.000000 1.000000\n"
