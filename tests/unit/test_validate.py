from pathlib import Path

import pytest

from claimlens.data.records import Annotation, ImageRecord
from claimlens.data.taxonomy import Taxonomy
from claimlens.data.validate import Issue, polygon_area, validate_record, validate_records
from tests.data_helpers import make_pattern_image

TAXONOMY = Taxonomy(name="damage", classes=("dent",))
SQUARE = (0.1, 0.1, 0.5, 0.1, 0.5, 0.5, 0.1, 0.5)


def _record(tmp_path: Path, *annotations: Annotation, width: int = 320) -> ImageRecord:
    make_pattern_image(tmp_path / "a.jpg", seed=1)
    return ImageRecord(
        image_id="s:a",
        source="s",
        source_split="train",
        path="a.jpg",
        width=width,
        height=240,
        annotations=annotations,
    )


def _codes(issues: list[Issue]) -> list[str]:
    return sorted(i.code for i in issues)


def test_polygon_area_of_square() -> None:
    assert polygon_area(SQUARE) == pytest.approx(0.16)


def test_clean_record_has_no_issues(tmp_path: Path) -> None:
    record = _record(tmp_path, Annotation(label="dent", polygon=SQUARE))
    assert validate_record(record, tmp_path, TAXONOMY) == []


def test_record_level_errors(tmp_path: Path) -> None:
    record = _record(tmp_path, Annotation(label="crack", polygon=SQUARE), width=999)
    assert _codes(validate_record(record, tmp_path, TAXONOMY)) == ["size_mismatch", "unknown_label"]
    missing = record.model_copy(update={"path": "nope.jpg"})
    assert "missing_file" in _codes(validate_record(missing, tmp_path, TAXONOMY))


def test_annotation_warnings(tmp_path: Path) -> None:
    record = _record(
        tmp_path,
        Annotation(label="dent", polygon=(0.1, 0.1, 0.2, 0.2)),
        Annotation(label="dent", polygon=(0.1, 0.1, 0.2, 0.2, 0.3, 0.3)),
        Annotation(label="dent", polygon=(0.0, 0.0, 1.005, 0.0, 1.005, 1.0)),
        Annotation(label="dent", polygon=(0.1, 0.1, 0.101, 0.1, 0.101, 0.101)),
    )
    assert _codes(validate_record(record, tmp_path, TAXONOMY)) == [
        "clamped",
        "degenerate_polygon",
        "degenerate_polygon",
        "tiny_instance",
    ]


def test_far_out_of_bounds_is_an_error(tmp_path: Path) -> None:
    record = _record(tmp_path, Annotation(label="dent", polygon=(0, 0, 1.5, 0, 1.5, 1)))
    assert _codes(validate_record(record, tmp_path, TAXONOMY)) == ["out_of_bounds"]


def test_report_counts_warnings_and_collects_errors(tmp_path: Path) -> None:
    good = _record(tmp_path)
    bad = good.model_copy(update={"image_id": "s:b", "path": "missing.jpg"})
    report = validate_records([good, bad], tmp_path, TAXONOMY)
    assert not report.ok
    assert report.total_images == 2
    assert [e.code for e in report.errors] == ["missing_file"]
    assert report.warning_counts == {"no_annotations": 2}
