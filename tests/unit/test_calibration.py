from pathlib import Path

import pytest

from claimlens.data.taxonomy import load_part_groups
from claimlens.fusion import calibrate_confidence
from claimlens.training.calibration import (
    ece,
    fit_temperature,
    match_predictions,
    part_agreement,
    read_yolo_labels,
    recommend_threshold,
)
from claimlens.vision.instances import SegInstance

SQUARE = (0.1, 0.1, 0.4, 0.1, 0.4, 0.4, 0.1, 0.4)
OTHER = (0.6, 0.6, 0.9, 0.6, 0.9, 0.9, 0.6, 0.9)


def _pred(label: str, polygon: tuple[float, ...], confidence: float) -> SegInstance:
    return SegInstance(
        label=label, confidence=confidence, box_xyxy=(0, 0, 1, 1), polygon_xyn=polygon
    )


def test_yolo_labels_are_read_with_class_names(tmp_path: Path) -> None:
    path = tmp_path / "a.txt"
    path.write_text("1 0.1 0.1 0.4 0.1 0.4 0.4\n\n0 0.5 0.5 0.6 0.5 0.6 0.6\n", encoding="utf-8")
    truths = read_yolo_labels(path, {0: "crack", 1: "dent"})
    assert [t[0] for t in truths] == ["dent", "crack"]
    assert len(truths[0][1]) == 6


def test_matching_needs_same_class_and_overlap() -> None:
    preds = [_pred("dent", SQUARE, 0.9), _pred("scratch", SQUARE, 0.8), _pred("dent", OTHER, 0.7)]
    pairs = match_predictions(preds, [("dent", SQUARE)])
    assert pairs == [(0.9, True), (0.8, False), (0.7, False)]


def test_each_truth_is_matched_once_highest_confidence_first() -> None:
    preds = [_pred("dent", SQUARE, 0.6), _pred("dent", SQUARE, 0.9)]
    assert match_predictions(preds, [("dent", SQUARE)]) == [(0.9, True), (0.6, False)]


def _synthetic(true_temperature: float) -> list[tuple[float, bool]]:
    pairs: list[tuple[float, bool]] = []
    for step in range(5, 100, 5):
        p = step / 100
        positives = round(calibrate_confidence(p, true_temperature) * 200)
        pairs += [(p, True)] * positives + [(p, False)] * (200 - positives)
    return pairs


def test_fit_recovers_a_known_temperature() -> None:
    assert fit_temperature(_synthetic(2.0)) == pytest.approx(2.0, abs=0.1)
    assert fit_temperature(_synthetic(0.5)) == pytest.approx(0.5, abs=0.1)


def test_fit_needs_predictions() -> None:
    with pytest.raises(ValueError, match="no matched predictions"):
        fit_temperature([])


def test_calibration_lowers_ece() -> None:
    pairs = _synthetic(2.0)
    fitted = fit_temperature(pairs)
    assert ece(pairs, temperature=fitted) < ece(pairs)


def test_threshold_is_the_lowest_meeting_the_precision() -> None:
    pairs = [(0.3, False)] * 5 + [(0.5, True)] * 8 + [(0.5, False)] * 2 + [(0.9, True)] * 10
    assert recommend_threshold(pairs, 1.0, precision=0.80) == pytest.approx(0.35)
    assert recommend_threshold([(0.9, False)], 1.0) is None


GROUPS = load_part_groups(Path(__file__).resolve().parents[2] / "config" / "taxonomy.toml")


def test_part_agreement_counts_per_group() -> None:
    truths = {"img": [("door", SQUARE), ("wheel", OTHER)]}
    preds = {"img": [_pred("front_left_door", SQUARE, 0.9), _pred("hood", OTHER, 0.9)]}
    assert part_agreement(truths, preds, GROUPS) == {"door": (1, 1), "wheel": (0, 1)}


def test_images_without_predictions_count_as_misses() -> None:
    assert part_agreement({"img": [("door", SQUARE)]}, {}, GROUPS) == {"door": (0, 1)}
