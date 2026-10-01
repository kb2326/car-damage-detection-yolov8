from pathlib import Path

import pytest
from pydantic import ValidationError

from claimlens.data.records import Annotation, ImageRecord
from claimlens.domain import Route
from claimlens.evals.golden import GoldenClaim
from claimlens.review.decisions import (
    GoldenReview,
    PartReview,
    annotation_key,
    apply_golden_review,
    apply_part_review,
    read_review,
    write_review,
)

SQUARE = (0.1, 0.1, 0.5, 0.1, 0.5, 0.5)


def _record(image_id: str, *labels: str) -> ImageRecord:
    return ImageRecord(
        image_id=image_id,
        source="s",
        source_split="test",
        path="x.jpg",
        width=1,
        height=1,
        annotations=tuple(Annotation(label=label, polygon=SQUARE, score=0.5) for label in labels),
    )


def _review(**decisions: str) -> PartReview:
    return PartReview.model_validate({"job_id": "j", "reviewer": "me", "decisions": decisions})


def test_only_approved_annotations_are_kept() -> None:
    records = [_record("a", "door", "hood")]
    review = _review(**{annotation_key("a", 0): "approved", annotation_key("a", 1): "rejected"})
    kept, counts = apply_part_review(records, review)
    assert [a.label for a in kept[0].annotations] == ["door"]
    assert counts == {"approved": 1, "rejected": 1}


def test_unreviewed_image_is_left_out() -> None:
    records = [_record("a", "door", "hood")]
    kept, counts = apply_part_review(records, _review(**{annotation_key("a", 0): "approved"}))
    assert kept == []
    assert counts == {"unreviewed_images": 1}


def test_all_rejected_and_empty_images_are_counted() -> None:
    records = [_record("a", "door"), _record("b")]
    kept, counts = apply_part_review(records, _review(**{annotation_key("a", 0): "rejected"}))
    assert kept == []
    assert counts == {"rejected": 1, "images_all_rejected": 1, "images_without_proposals": 1}


def test_invalid_part_decision_is_rejected() -> None:
    with pytest.raises(ValidationError):
        _review(**{"a#0": "maybe"})


def _case(case_id: str, route: Route = Route.FAST_TRACK) -> GoldenClaim:
    return GoldenClaim(
        case_id=case_id,
        scenario="s",
        policy_id="P-1001",
        description="",
        photos=("x.jpg",),
        expected_route=route,
        label_source="oracle",
    )


def test_golden_review_marks_and_corrects_cases() -> None:
    review = GoldenReview(reviewer="me", decisions={"g1": "agree", "g2": "ADJUSTER_REVIEW"})
    g1, g2, g3 = apply_golden_review([_case("g1"), _case("g2"), _case("g3")], review)
    assert g1.reviewed
    assert g1.expected_route is Route.FAST_TRACK
    assert g2.reviewed
    assert g2.expected_route is Route.ADJUSTER_REVIEW
    assert "FAST_TRACK to ADJUSTER_REVIEW" in g2.notes
    assert not g3.reviewed


def test_unknown_golden_decision_raises() -> None:
    review = GoldenReview(reviewer="me", decisions={"g1": "DENY"})
    with pytest.raises(ValueError, match="DENY"):
        apply_golden_review([_case("g1")], review)


def test_reviews_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "reviews" / "j.json"
    review = _review(**{"a#0": "approved"})
    write_review(path, review)
    assert read_review(path, PartReview) == review
