from collections.abc import Callable
from datetime import date
from pathlib import Path

import pytest

from claimlens.cli import main
from claimlens.domain import Route
from claimlens.evals.golden import GoldenClaim, PriorClaim, load_golden, write_golden
from claimlens.evals.metrics import compute_triage_metrics
from claimlens.evals.triage import ReportMeta, render_report, run_triage_eval
from claimlens.events.store import SQLiteEventStore
from tests.fakes import CONFIG_DIR, FakeDetector, make_test_deps


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
        detector_factory=lambda _: FakeDetector(),
    )

    assert code == 0
    assert report.is_file()
    assert "Route accuracy: 1.00" in capsys.readouterr().out
