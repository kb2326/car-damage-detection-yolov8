"""Run golden claims through the full pipeline and report the results."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from claimlens.decision import DecisionConfig, decide
from claimlens.domain import Route
from claimlens.evals.golden import GoldenClaim
from claimlens.evals.metrics import TriageMetrics, compute_triage_metrics
from claimlens.events.projection import ClaimState, fold
from claimlens.intake import submit_claim
from claimlens.workflow import PipelineDeps, process_claim


@dataclass(frozen=True)
class CaseResult:
    case_id: str
    scenario: str
    expected: Route
    predicted: Route | None
    rule_id: str = ""
    reason: str = ""
    error: str = ""
    state: ClaimState | None = None


@dataclass(frozen=True)
class ReportMeta:
    golden_path: str
    model_version: str
    agent_version: str
    decision_policy_version: str
    generated_on: date


def run_case(case: GoldenClaim, deps: PipelineDeps, repo_root: Path) -> CaseResult:
    for prior in case.prior_claims:
        prior_id = submit_claim(
            deps.store,
            deps.blobs,
            policy_id=prior.policy_id,
            description="Earlier claim",
            photo_paths=[repo_root / p for p in prior.photos],
        )
        process_claim(prior_id, deps)
    claim_id = submit_claim(
        deps.store,
        deps.blobs,
        policy_id=case.policy_id,
        description=case.description,
        photo_paths=[repo_root / p for p in case.photos],
    )
    decision = process_claim(claim_id, deps)
    return CaseResult(
        case_id=case.case_id,
        scenario=case.scenario,
        expected=case.expected_route,
        predicted=decision.route,
        rule_id=decision.rule_id,
        reason=decision.reason,
        state=fold(deps.store.load(claim_id)),
    )


def run_triage_eval(
    cases: Sequence[GoldenClaim],
    make_deps: Callable[[Path], PipelineDeps],
    *,
    repo_root: Path,
    workdir: Path,
) -> list[CaseResult]:
    """Each case gets a fresh store so cases cannot affect each other."""
    results: list[CaseResult] = []
    for case in cases:
        case_dir = workdir / case.case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        deps = make_deps(case_dir)
        try:
            results.append(run_case(case, deps, repo_root))
        except Exception as exc:
            results.append(
                CaseResult(
                    case_id=case.case_id,
                    scenario=case.scenario,
                    expected=case.expected_route,
                    predicted=None,
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
        finally:
            deps.store.close()
    return results


def what_if_thresholds(
    results: Sequence[CaseResult], config: DecisionConfig, thresholds: Sequence[float]
) -> list[tuple[float, TriageMetrics]]:
    """Re-decide each finished claim at other confidence thresholds. Agent output is held fixed."""
    table: list[tuple[float, TriageMetrics]] = []
    for threshold in thresholds:
        variant = config.model_copy(update={"min_finding_confidence": threshold})
        pairs = [
            (r.expected, decide(r.state, variant).route if r.state is not None else None)
            for r in results
        ]
        table.append((threshold, compute_triage_metrics(pairs)))
    return table


def render_report(
    cases: Sequence[GoldenClaim],
    results: Sequence[CaseResult],
    metrics: TriageMetrics,
    meta: ReportMeta,
    what_if: Sequence[tuple[float, TriageMetrics]] = (),
) -> str:
    recall = "n/a" if metrics.escalation_recall is None else f"{metrics.escalation_recall:.2f}"
    reviewed = sum(case.reviewed for case in cases)
    lines = [
        f"# Triage evaluation: {meta.golden_path}",
        "",
        f"- Date: {meta.generated_on.isoformat()}",
        f"- Detector: `{meta.model_version}`",
        f"- Agent: `{meta.agent_version}`",
        f"- Decision policy: `{meta.decision_policy_version}`",
        f"- Cases: {metrics.total} ({reviewed} human-reviewed)",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Route accuracy | {metrics.route_accuracy:.2f} ({metrics.correct}/{metrics.total}) |",
        f"| Escalation recall | {recall} ({metrics.escalated}/{metrics.needs_human}) |",
        "",
        "## Confusion matrix (rows: expected, columns: predicted)",
        "",
    ]
    columns: list[Route | None] = [*Route, None]
    header = " | ".join(c.value if c is not None else "ERROR" for c in columns)
    lines.append(f"| expected \\ predicted | {header} |")
    lines.append("|---" * (len(columns) + 1) + "|")
    for expected in Route:
        counts = " | ".join(str(metrics.confusion.get((expected, c), 0)) for c in columns)
        lines.append(f"| {expected.value} | {counts} |")

    by_scenario: dict[str, list[CaseResult]] = defaultdict(list)
    for result in results:
        by_scenario[result.scenario].append(result)
    lines += ["", "## Accuracy by scenario", "", "| Scenario | Correct | Total |", "|---|---|---|"]
    for scenario in sorted(by_scenario):
        group = by_scenario[scenario]
        correct = sum(r.predicted == r.expected for r in group)
        lines.append(f"| {scenario} | {correct} | {len(group)} |")

    misses = [r for r in results if r.predicted != r.expected]
    lines += ["", "## Mismatches", ""]
    if not misses:
        lines.append("None.")
    else:
        lines += [
            "| Case | Scenario | Expected | Predicted | Rule | Reason |",
            "|---|---|---|---|---|---|",
        ]
        for r in misses:
            predicted = r.predicted.value if r.predicted is not None else "ERROR"
            reason = r.error or r.reason
            lines.append(
                f"| {r.case_id} | {r.scenario} | {r.expected.value} | {predicted} "
                f"| {r.rule_id or '-'} | {reason} |"
            )
    if what_if:
        lines += [
            "",
            "## What if: other confidence thresholds (R6)",
            "",
            "Policy unchanged; each claim is re-decided from its final state.",
            "",
            "| Min confidence | Route accuracy | Escalation recall | Correct fast-tracks |",
            "|---|---|---|---|",
        ]
        for threshold, m in what_if:
            rec = "n/a" if m.escalation_recall is None else f"{m.escalation_recall:.2f}"
            fast = m.confusion.get((Route.FAST_TRACK, Route.FAST_TRACK), 0)
            expected_fast = sum(v for (e, _), v in m.confusion.items() if e is Route.FAST_TRACK)
            lines.append(
                f"| {threshold:.2f} | {m.route_accuracy:.2f} | {rec} | {fast} of {expected_fast} |"
            )
    return "\n".join(lines) + "\n"
