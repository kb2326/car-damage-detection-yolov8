import pytest
from pydantic import ValidationError

from claimlens.vision.instances import SegInstance
from claimlens.vision.masks import GRID, area, intersection, mask_iou, rasterize

QUARTER = (0.0, 0.0, 0.5, 0.0, 0.5, 0.5, 0.0, 0.5)
RIGHT_QUARTER = (0.5, 0.0, 1.0, 0.0, 1.0, 0.5, 0.5, 0.5)


def test_quarter_square_covers_a_quarter_of_the_grid() -> None:
    assert area(rasterize(QUARTER)) / GRID**2 == pytest.approx(0.25, abs=0.01)


def test_degenerate_polygons_are_empty() -> None:
    assert area(rasterize(())) == 0
    assert area(rasterize((0.1, 0.1, 0.2, 0.2))) == 0


def test_odd_coordinates_are_rejected() -> None:
    with pytest.raises(ValueError, match="pairs"):
        rasterize((0.1, 0.2, 0.3))


def test_identical_masks_have_iou_one_and_disjoint_zero() -> None:
    a = rasterize(QUARTER)
    assert mask_iou(a, a) == 1.0
    assert mask_iou(a, rasterize((0.6, 0.6, 0.9, 0.6, 0.9, 0.9))) == 0.0
    assert mask_iou(rasterize(()), rasterize(())) == 0.0


def test_neighbouring_squares_share_only_their_edge() -> None:
    a, b = rasterize(QUARTER), rasterize(RIGHT_QUARTER)
    assert intersection(a, b) <= GRID // 2 + 1


def test_instances_validate_confidence() -> None:
    with pytest.raises(ValidationError):
        SegInstance(label="dent", confidence=1.5, box_xyxy=(0, 0, 1, 1), polygon_xyn=QUARTER)
