"""Triage metrics. Escalation recall matters most: claims that need a human must reach one."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from claimlens.domain import Route

HUMAN_ROUTES = frozenset({Route.ADJUSTER_REVIEW, Route.FRAUD_REVIEW})


@dataclass(frozen=True)
class TriageMetrics:
    total: int
    correct: int
    route_accuracy: float
    needs_human: int
    escalated: int
    escalation_recall: float | None
    confusion: Mapping[tuple[Route, Route | None], int]


def compute_triage_metrics(pairs: Sequence[tuple[Route, Route | None]]) -> TriageMetrics:
    """`pairs` are (expected, predicted); predicted is None when the case errored."""
    if not pairs:
        raise ValueError("metrics need at least one case")
    correct = sum(expected == predicted for expected, predicted in pairs)
    human_cases = [predicted for expected, predicted in pairs if expected in HUMAN_ROUTES]
    escalated = sum(predicted in HUMAN_ROUTES for predicted in human_cases)
    return TriageMetrics(
        total=len(pairs),
        correct=correct,
        route_accuracy=correct / len(pairs),
        needs_human=len(human_cases),
        escalated=escalated,
        escalation_recall=escalated / len(human_cases) if human_cases else None,
        confusion=Counter(pairs),
    )
