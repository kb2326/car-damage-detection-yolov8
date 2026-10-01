from pathlib import Path

import pytest

from claimlens.data.records import (
    Annotation,
    ImageRecord,
    read_records,
    relative_posix,
    write_records,
)


def _record(image_id: str = "src:a") -> ImageRecord:
    return ImageRecord(
        image_id=image_id,
        source="src",
        source_split="train",
        path="data/raw/src/a.jpg",
        width=640,
        height=480,
        annotations=(Annotation(label="dent", polygon=(0.1, 0.1, 0.5, 0.1, 0.5, 0.5)),),
    )


def test_records_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "interim" / "records.jsonl"
    records = [_record("src:a"), _record("src:b")]
    write_records(path, records)
    assert read_records(path) == records


def test_duplicate_image_ids_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    write_records(path, [_record(), _record()])
    with pytest.raises(ValueError, match="duplicate image id src:a"):
        read_records(path)


def test_relative_posix_uses_forward_slashes(tmp_path: Path) -> None:
    target = tmp_path / "data" / "raw" / "x.jpg"
    assert relative_posix(target, tmp_path) == "data/raw/x.jpg"
