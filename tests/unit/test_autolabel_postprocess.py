import numpy as np
import pytest

from claimlens.autolabel.postprocess import Detection, box_iou, mask_to_polygon, select_detections

PROMPTS = {"car door": "door", "hood": "hood", "headlight": "light"}


def test_box_iou() -> None:
    assert box_iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)
    assert box_iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(1 / 3)
    assert box_iou((0, 0, 1, 1), (2, 2, 3, 3)) == 0.0


def test_merged_phrase_is_dropped() -> None:
    kept, dropped = select_detections(
        [Detection("headlight side mirror", 0.9, (0, 0, 10, 10))], PROMPTS, min_score=0.3
    )
    assert kept == []
    assert dropped == {"ambiguous_phrase": 1}


def test_low_score_and_duplicates_are_dropped() -> None:
    detections = [
        Detection("car door", 0.8, (0, 0, 100, 100)),
        Detection("car door", 0.7, (5, 5, 100, 100)),
        Detection("Car Door ", 0.6, (200, 0, 300, 100)),
        Detection("hood", 0.2, (0, 0, 50, 50)),
        Detection("hood", 0.5, (0, 0, 100, 100)),
    ]
    kept, dropped = select_detections(detections, PROMPTS, min_score=0.3, max_per_group=2)
    assert [(g, d.score) for g, d in kept] == [("door", 0.8), ("door", 0.6), ("hood", 0.5)]
    assert dropped == {"duplicate": 1, "low_score": 1}


def test_max_per_group_caps_detections() -> None:
    detections = [
        Detection("car door", 0.9 - i / 10, (i * 200, 0, i * 200 + 100, 100)) for i in range(3)
    ]
    kept, dropped = select_detections(detections, PROMPTS, min_score=0.3, max_per_group=2)
    assert len(kept) == 2
    assert dropped == {"duplicate": 1}


def test_rectangle_mask_becomes_four_corner_polygon() -> None:
    mask = np.zeros((100, 200), dtype=bool)
    mask[20:60, 50:150] = True
    polygon = mask_to_polygon(mask)
    assert polygon is not None
    xs, ys = polygon[0::2], polygon[1::2]
    assert len(xs) == 4
    assert min(xs) == pytest.approx(0.25)
    assert max(xs) == pytest.approx(149 / 200)
    assert min(ys) == pytest.approx(0.2)
    assert max(ys) == pytest.approx(59 / 100)


def test_empty_mask_gives_no_polygon() -> None:
    assert mask_to_polygon(np.zeros((10, 10), dtype=bool)) is None
