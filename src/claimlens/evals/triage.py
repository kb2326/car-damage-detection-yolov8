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
from claimlens.events.envelope import ClaimEvent
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
    llm_calls: int = 0
    tool_calls: int = 0
    llm_cost_usd: float = 0.0
    agent_error: str = ""


def _cell(text: str) -> str:
    """Text safe for one Markdown table cell."""
    return " ".join(text.split()).replace("|", r"\|")


def agent_counts(events: Sequence[ClaimEvent]) -> tuple[int, int, float, str]:
    """Model calls, tool calls, LLM cost and the agent's failure (if any) on one claim's log."""
    llm = [e for e in events if e.type == "LLMCalled"]
    tools = sum(e.type == "ToolCalled" for e in events)
    cost = sum(float(e.payload["cost_usd"]) for e in llm)
    error = next(
        (
            str(e.payload["error"])
            for e in events
            if e.type == "StageFailed" and e.payload.get("stage") == "agent"
        ),
        "",
    )
    return len(llm), tools, cost, error


@dataclass(frozen=True)
class AgentSummary:
    cases: int
    llm_calls_mean: float
    tool_calls_mean: float
    cost_total_usd: float
    cost_mean_usd: float
    cost_max_usd: float
    agent_failures: dict[str, int]


def agent_summary(results: Sequence[CaseResult]) -> AgentSummary:
    n = max(len(results), 1)
    costs = [r.llm_cost_usd for r in results]
    failures: dict[str, int] = {}
    for r in results:
        if r.agent_error:
            kind = r.agent_error.split(":", 1)[0]
            failures[kind] = failures.get(kind, 0) + 1
    return AgentSummary(
        cases=len(results),
        llm_calls_mean=sum(r.llm_calls for r in results) / n,
        tool_calls_mean=sum(r.tool_calls for r in results) / n,
        cost_total_usd=sum(costs),
        cost_mean_usd=sum(costs) / n,
        cost_max_usd=max(costs, default=0.0),
        agent_failures=dict(sorted(failures.items())),
    )


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
    events = deps.store.load(claim_id)
    llm_calls, tool_calls, cost, agent_error = agent_counts(events)
    return CaseResult(
        case_id=case.case_id,
        scenario=case.scenario,
        expected=case.expected_route,
        predicted=decision.route,
        rule_id=decision.rule_id,
        reason=decision.reason,
        state=fold(events),
        llm_calls=llm_calls,
        tool_calls=tool_calls,
        llm_cost_usd=cost,
        agent_error=agent_error,
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
    agent: AgentSummary | None = None,
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
    if agent is not None:
        failures = ", ".join(f"{k}: {v}" for k, v in agent.agent_failures.items()) or "none"
        lines += [
            "",
            "## Agent",
            "",
            "| Measure | Value |",
            "|---|---|",
            f"| Model calls per claim (mean) | {agent.llm_calls_mean:.1f} |",
            f"| Tool calls per claim (mean) | {agent.tool_calls_mean:.1f} |",
            f"| Cost per claim (mean / max) | ${agent.cost_mean_usd:.3f} / "
            f"${agent.cost_max_usd:.3f} |",
            f"| Total cost of this run | ${agent.cost_total_usd:.2f} |",
            f"| Agent failures (sent to a person) | {failures} |",
        ]
        held = [
            r
            for r in results
            if r.rule_id in ("R7", "R8") and r.state is not None and r.state.recommendation
        ]
        if held:
            lines += [
                "",
                "## Claims the agent held back (R7, R8)",
                "",
                "| Case | Expected | Rule | Agent said | Rationale | Open questions |",
                "|---|---|---|---|---|---|",
            ]
            for r in held:
                advice = r.state.recommendation if r.state is not None else None
                assert advice is not None
                questions = " ".join(advice.open_questions) or "-"
                lines.append(
                    f"| {r.case_id} | {r.expected.value} | {r.rule_id} "
                    f"| {advice.route_suggestion.value}, {advice.confidence.value} "
                    f"| {_cell(advice.rationale)} | {_cell(questions)} |"
                )
    return "\n".join(lines) + "\n"
