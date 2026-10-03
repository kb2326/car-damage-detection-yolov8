from collections.abc import Callable
from datetime import date
from pathlib import Path

import pytest

from claimlens.cli import main
from claimlens.decision import DecisionConfig
from claimlens.domain import DamageFinding, Route
from claimlens.evals.golden import GoldenClaim, PriorClaim, load_golden, write_golden
from claimlens.evals.metrics import compute_triage_metrics
from claimlens.evals.triage import (
    CaseResult,
    ReportMeta,
    render_report,
    run_triage_eval,
    what_if_thresholds,
)
from claimlens.events.store import SQLiteEventStore
from tests.fakes import CONFIG_DIR, FakeDetector, dent, make_test_deps


def _cases(image: Path, missing: Path) -> list[GoldenClaim]:
    return [
        GoldenClaim(
            case_id="g001",
            scenario="clean",
            policy_id="P-1001",
            description="",
            photos=(str(image),),
            expected_route=Route.FAST_TRACK,
            label_source="scenario",
            reviewed=True,
        ),
        GoldenClaim(
            case_id="g002",
            scenario="photo_reuse",
            policy_id="P-1002",
            description="",
            photos=(str(image),),
            prior_claims=(PriorClaim(policy_id="P-1001", photos=(str(image),)),),
            expected_route=Route.FRAUD_REVIEW,
            label_source="scenario",
        ),
        GoldenClaim(
            case_id="g003",
            scenario="missing_file",
            policy_id="P-1001",
            description="",
            photos=(str(missing),),
            expected_route=Route.ADJUSTER_REVIEW,
            label_source="scenario",
        ),
    ]


def test_golden_round_trip(tmp_path: Path) -> None:
    cases = _cases(tmp_path / "a.jpg", tmp_path / "b.jpg")
    path = tmp_path / "claims.jsonl"
    write_golden(path, cases)
    assert load_golden(path) == cases


def test_duplicate_case_ids_are_rejected(tmp_path: Path) -> None:
    case = _cases(tmp_path / "a.jpg", tmp_path / "b.jpg")[0]
    path = tmp_path / "claims.jsonl"
    write_golden(path, [case, case])
    with pytest.raises(ValueError, match="duplicate case id g001"):
        load_golden(path)


def test_run_triage_eval_scores_each_case(tmp_path: Path, make_image: Callable[..., Path]) -> None:
    cases = _cases(make_image("a.jpg"), tmp_path / "nope.jpg")
    detector = FakeDetector()

    results = run_triage_eval(
        cases,
        lambda case_dir: make_test_deps(case_dir, SQLiteEventStore(case_dir / "c.db"), detector),
        repo_root=tmp_path,
        workdir=tmp_path / "work",
    )

    assert [r.predicted for r in results] == [Route.FAST_TRACK, Route.FRAUD_REVIEW, None]
    assert "photo not found" in results[2].error

    metrics = compute_triage_metrics([(r.expected, r.predicted) for r in results])
    meta = ReportMeta(
        golden_path="claims.jsonl",
        model_version=detector.model_version,
        agent_version="stub-v0",
        decision_policy_version="decision-policy-v0",
        generated_on=date(2026, 10, 18),
    )
    report = render_report(cases, results, metrics, meta)
    assert "| Route accuracy | 0.67 (2/3) |" in report
    assert "Cases: 3 (1 human-reviewed)" in report
    assert "| g003 | missing_file | ADJUSTER_REVIEW | ERROR |" in report


def test_eval_triage_command_writes_a_report(
    tmp_path: Path, make_image: Callable[..., Path], capsys: pytest.CaptureFixture[str]
) -> None:
    golden = tmp_path / "claims.jsonl"
    write_golden(golden, _cases(make_image("a.jpg"), tmp_path / "nope.jpg")[:2])
    report = tmp_path / "reports" / "triage.md"

    code = main(
        [
            "--config",
            str(CONFIG_DIR),
            "eval-triage",
            "--golden",
            str(golden),
            "--report",
            str(report),
        ],
        detector_factory=lambda *_: FakeDetector(),
    )

    assert code == 0
    assert report.is_file()
    assert "Route accuracy: 1.00" in capsys.readouterr().out


def _config(threshold: float = 0.40) -> DecisionConfig:
    return DecisionConfig(
        version="t",
        fraud_score_threshold=0.8,
        max_fast_track_cost_usd=3000,
        min_finding_confidence=threshold,
    )


def test_what_if_counts_errored_cases_as_errors() -> None:
    errored = CaseResult(
        case_id="g1", scenario="s", expected=Route.FAST_TRACK, predicted=None, error="boom"
    )
    ((threshold, metrics),) = what_if_thresholds([errored], _config(), [0.25])
    assert threshold == 0.25
    assert metrics == compute_triage_metrics([(Route.FAST_TRACK, None)])


class _LowConfidenceDetector(FakeDetector):
    def detect(self, image_path: Path, photo_id: str) -> list[DamageFinding]:
        return [dent(photo_id, confidence=0.30)]


def test_what_if_redecides_from_the_final_state(
    tmp_path: Path, make_image: Callable[..., Path]
) -> None:
    case = _cases(make_image("a.jpg"), tmp_path / "nope.jpg")[0]
    results = run_triage_eval(
        [case],
        lambda case_dir: make_test_deps(
            case_dir, SQLiteEventStore(case_dir / "c.db"), _LowConfidenceDetector()
        ),
        repo_root=tmp_path,
        workdir=tmp_path / "work",
    )
    assert results[0].state is not None
    assert results[0].predicted is Route.ADJUSTER_REVIEW
    low, strict = what_if_thresholds(results, _config(), [0.25, 0.55])
    assert low[1].confusion.get((Route.FAST_TRACK, Route.FAST_TRACK), 0) == 1
    assert strict[1].confusion.get((Route.FAST_TRACK, Route.ADJUSTER_REVIEW), 0) == 1
    metrics = compute_triage_metrics([(r.expected, r.predicted) for r in results])
    meta = ReportMeta(
        golden_path="claims.jsonl",
        model_version="fake",
        agent_version="stub-v0",
        decision_policy_version="t",
        generated_on=date(2026, 10, 2),
    )
    report = render_report([case], results, metrics, meta, what_if=[low, strict])
    assert "## What if" in report
    assert "| 0.25 | 1.00 | n/a | 1 of 1 |" in report


def test_agent_summary_counts_calls_costs_and_failures() -> None:
    from claimlens.evals.triage import agent_summary

    results = [
        CaseResult(
            "a",
            "s",
            Route.FAST_TRACK,
            Route.FAST_TRACK,
            llm_calls=3,
            tool_calls=2,
            llm_cost_usd=0.04,
        ),
        CaseResult(
            "b",
            "s",
            Route.FAST_TRACK,
            Route.ADJUSTER_REVIEW,
            rule_id="R2",
            reason="Processing failed at: agent.",
            llm_calls=6,
            tool_calls=5,
            llm_cost_usd=0.08,
            agent_error="AgentFailed: step limit of 6 model calls reached",
        ),
    ]
    summary = agent_summary(results)
    assert summary.cases == 2
    assert summary.llm_calls_mean == 4.5
    assert summary.tool_calls_mean == 3.5
    assert summary.cost_total_usd == pytest.approx(0.12)
    assert summary.cost_mean_usd == pytest.approx(0.06)
    assert summary.cost_max_usd == pytest.approx(0.08)
    assert summary.agent_failures == {"AgentFailed": 1}


def test_report_gains_an_agent_section() -> None:
    from claimlens.evals.triage import agent_summary

    case = GoldenClaim(
        case_id="a",
        scenario="s",
        policy_id="P-1001",
        description="",
        photos=("x.jpg",),
        expected_route=Route.FAST_TRACK,
        label_source="t",
    )
    results = [
        CaseResult(
            "a",
            "s",
            Route.FAST_TRACK,
            Route.FAST_TRACK,
            rule_id="R9",
            llm_calls=3,
            tool_calls=2,
            llm_cost_usd=0.041,
        )
    ]
    meta = ReportMeta("g.jsonl", "m", "triage-agent-v1+triage/v1", "p", date(2026, 10, 3))
    metrics = compute_triage_metrics([(Route.FAST_TRACK, Route.FAST_TRACK)])
    report = render_report([case], results, metrics, meta, agent=agent_summary(results))
    assert "## Agent" in report
    assert "| Model calls per claim (mean) | 3.0 |" in report
    assert "| Cost per claim (mean / max) | $0.041 / $0.041 |" in report
    assert "| Agent failures (sent to a person) | none |" in report
    assert "## Agent" not in render_report([case], results, metrics, meta)


def test_case_results_count_llm_and_tool_events(tmp_path: Path) -> None:
    from uuid import uuid4

    from claimlens.evals.triage import agent_counts
    from claimlens.events.envelope import Actor, ActorKind
    from claimlens.events.payloads import ClaimReported, LLMCalled, StageFailed, ToolCalled

    store = SQLiteEventStore(tmp_path / "c.db")
    claim = uuid4()
    system = Actor(kind=ActorKind.SYSTEM, name="t")
    store.append(claim, ClaimReported(policy_id="P-1001", description=""), system)
    for cost in (0.01, 0.02):
        store.append(
            claim,
            LLMCalled(
                request_id="r",
                model="m",
                prompt_id="triage/v1",
                input_tokens=1,
                output_tokens=1,
                cost_usd=cost,
                cached=False,
                outcome="ok",
            ),
            system,
        )
    store.append(
        claim,
        ToolCalled(server="s", tool="t", profile="triage", input_sha256="x", outcome="ok"),
        system,
    )
    store.append(claim, StageFailed(stage="agent", error="AgentFailed: step limit"), system)
    events = store.load(claim)
    store.close()
    llm_calls, tool_calls, cost, error = agent_counts(events)
    assert (llm_calls, tool_calls, error) == (2, 1, "AgentFailed: step limit")
    assert cost == pytest.approx(0.03)


def test_report_explains_claims_the_agent_held_back() -> None:
    from uuid import uuid4

    from claimlens.domain import AgentRecommendation, Confidence
    from claimlens.evals.triage import agent_summary
    from claimlens.events.projection import ClaimState

    case = GoldenClaim(
        case_id="a",
        scenario="s",
        policy_id="P-1001",
        description="",
        photos=("x.jpg",),
        expected_route=Route.FAST_TRACK,
        label_source="t",
    )
    state = ClaimState(claim_id=uuid4(), policy_id="P-1001", description="")
    state.recommendation = AgentRecommendation(
        route_suggestion=Route.FAST_TRACK,
        confidence=Confidence.MEDIUM,
        rationale="Damage | matches the story.",
        citations=(),
        open_questions=("Was the second photo | taken the same day?",),
    )
    held = CaseResult("a", "s", Route.FAST_TRACK, Route.ADJUSTER_REVIEW, rule_id="R7", state=state)
    meta = ReportMeta("g.jsonl", "m", "agent", "p", date(2026, 10, 3))
    metrics = compute_triage_metrics([(Route.FAST_TRACK, Route.ADJUSTER_REVIEW)])
    report = render_report([case], [held], metrics, meta, agent=agent_summary([held]))
    assert "## Claims the agent held back (R7, R8)" in report
    assert (
        r"| a | FAST_TRACK | R7 | FAST_TRACK, medium | Damage \| matches the story. "
        r"| Was the second photo \| taken the same day? |"
    ) in report
