import pytest

from claimlens.domain import Route
from claimlens.evals.metrics import compute_triage_metrics

FAST, ADJ, FRAUD = Route.FAST_TRACK, Route.ADJUSTER_REVIEW, Route.FRAUD_REVIEW


def test_metrics_count_accuracy_and_recall() -> None:
    metrics = compute_triage_metrics(
        [(FAST, FAST), (ADJ, ADJ), (ADJ, FAST), (FRAUD, ADJ), (FAST, None)]
    )
    assert (metrics.total, metrics.correct) == (5, 2)
    assert metrics.route_accuracy == pytest.approx(0.4)
    assert (metrics.needs_human, metrics.escalated) == (3, 2)
    assert metrics.escalation_recall == pytest.approx(2 / 3)
    assert metrics.confusion[(ADJ, FAST)] == 1


def test_recall_is_none_without_human_cases() -> None:
    assert compute_triage_metrics([(FAST, FAST)]).escalation_recall is None


def test_empty_input_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least one case"):
        compute_triage_metrics([])
