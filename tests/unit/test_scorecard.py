from datetime import date
from pathlib import Path

import pytest

from claimlens.evals.scorecard import GateConfig, Scorecard, check_gate, fingerprints

CONFIG = GateConfig(
    max_drop={
        "route_accuracy": 0.02,
        "narrative_catch_rate": 0.05,
        "benign_pass_rate": 0.10,
        "judge_pass_rate": 0.05,
    },
    must_equal_one=("escalation_recall", "citation_validity"),
    max_cost_increase=0.25,
)
PRINTS = {"prompts/triage/v1.md": "aaa", "config/agent.toml": "bbb"}
METRICS = {
    "escalation_recall": 1.0,
    "citation_validity": 1.0,
    "route_accuracy": 0.80,
    "narrative_catch_rate": 0.90,
    "benign_pass_rate": 0.80,
    "judge_pass_rate": 0.85,
    "cost_mean_usd": 0.040,
}


def _card(**metrics: float | None) -> Scorecard:
    return Scorecard(
        created_on=date(2026, 10, 2),
        golden="evals/golden/v2/claims.jsonl",
        versions={"agent": "triage-agent-v1+triage/v1"},
        metrics={**METRICS, **metrics},
        fingerprints=dict(PRINTS),
    )


def test_equal_to_baseline_passes() -> None:
    assert check_gate(_card(), _card(), PRINTS, CONFIG) == []


def test_stale_scorecard_fails() -> None:
    on_disk = {**PRINTS, "prompts/triage/v1.md": "changed"}
    assert check_gate(_card(), _card(), on_disk, CONFIG) == [
        "stale: prompts/triage/v1.md changed since the evaluation ran; re-run eval-triage"
    ]


def test_missing_or_new_fingerprinted_file_is_stale() -> None:
    on_disk = {"config/agent.toml": "bbb", "prompts/triage/v2.md": "ccc"}
    problems = check_gate(_card(), _card(), on_disk, CONFIG)
    assert "stale: prompts/triage/v1.md is missing" in problems
    assert (
        "stale: prompts/triage/v2.md is new since the evaluation ran; re-run eval-triage"
        in problems
    )


def test_safety_metrics_must_be_exactly_one() -> None:
    assert check_gate(_card(escalation_recall=0.99), _card(), PRINTS, CONFIG) == [
        "safety: escalation_recall is 0.99, must be 1.00"
    ]
    assert check_gate(_card(citation_validity=None), _card(), PRINTS, CONFIG) == [
        "safety: citation_validity was not measured"
    ]


def test_regression_beyond_the_margin_fails() -> None:
    assert check_gate(_card(route_accuracy=0.79), _card(), PRINTS, CONFIG) == []
    assert check_gate(_card(route_accuracy=0.77), _card(), PRINTS, CONFIG) == [
        "regression: route_accuracy fell from 0.80 to 0.77 (allowed drop 0.02)"
    ]


def test_cost_increase_beyond_the_margin_fails() -> None:
    assert check_gate(_card(cost_mean_usd=0.060), _card(), PRINTS, CONFIG) == [
        "cost: mean cost per claim rose from $0.040 to $0.060 (allowed +25%)"
    ]


def test_an_unvalidated_judge_metric_is_skipped() -> None:
    assert (
        check_gate(_card(judge_pass_rate=None), _card(judge_pass_rate=None), PRINTS, CONFIG) == []
    )


def test_fingerprints_cover_the_files_that_change_results(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    prints = fingerprints(root, root / "evals" / "golden" / "v1" / "claims.jsonl")
    for path in (
        "config/decision_policy.toml",
        "config/rate_card.toml",
        "config/models.toml",
        "config/agent.toml",
        "config/llm.toml",
        "prompts/triage/v1.md",
        "knowledge/policies/standard.md",
        "evals/golden/v1/claims.jsonl",
    ):
        assert path in prints
        assert len(prints[path]) == 64
    assert prints == fingerprints(root, root / "evals" / "golden" / "v1" / "claims.jsonl")


def test_fusion_taxonomy_and_judge_prompt_are_fingerprinted() -> None:
    root = Path(__file__).resolve().parents[2]
    prints = fingerprints(root, root / "evals" / "golden" / "v1" / "claims.jsonl")
    assert "config/taxonomy.toml" in prints
    assert "config/agents.toml" in prints
    assert "prompts/judge/v1.md" in prints


def test_gate_config_loads() -> None:
    from claimlens.evals.scorecard import load_gate_config

    root = Path(__file__).resolve().parents[2]
    config = load_gate_config(root / "config" / "eval_gate.toml")
    assert config.must_equal_one == ("escalation_recall", "citation_validity")
    assert config.max_drop["route_accuracy"] == 0.02


def _repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    for rel, text in {
        "config/agent.toml": "a",
        "prompts/triage/v1.md": "p",
        "evals/golden/v2/claims.jsonl": "{}\n",
    }.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text, encoding="utf-8")
    return root


def _write_card(root: Path, name: str, **metrics: float | None) -> Path:
    golden = root / "evals" / "golden" / "v2" / "claims.jsonl"
    card = Scorecard(
        created_on=date(2026, 10, 3),
        golden="evals/golden/v2/claims.jsonl",
        versions={},
        metrics={**METRICS, **metrics},
        fingerprints=fingerprints(root, golden),
    )
    path = root / "evals" / "scorecards" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(card.model_dump_json(indent=1), encoding="utf-8")
    return path


def test_eval_gate_command(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from claimlens.cli import main

    root = _repo(tmp_path)
    _write_card(root, "current.json")
    _write_card(root, "baseline.json")
    config = Path(__file__).resolve().parents[2] / "config"
    args = ["--config", str(config), "eval-gate", "--root", str(root)]
    assert main(args) == 0
    assert "eval gate: ok" in capsys.readouterr().out
    (root / "prompts" / "triage" / "v1.md").write_text("changed", encoding="utf-8")
    assert main(args) == 1
    assert "stale: prompts/triage/v1.md changed" in capsys.readouterr().out


def test_eval_gate_with_a_missing_golden_file_is_stale_not_a_crash(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from claimlens.cli import main

    root = _repo(tmp_path)
    _write_card(root, "current.json")
    _write_card(root, "baseline.json")
    (root / "evals" / "golden" / "v2" / "claims.jsonl").unlink()
    config = Path(__file__).resolve().parents[2] / "config"
    assert main(["--config", str(config), "eval-gate", "--root", str(root)]) == 1
    assert "stale: evals/golden/v2/claims.jsonl is missing" in capsys.readouterr().out


def test_eval_gate_without_a_scorecard_fails_clearly(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from claimlens.cli import main

    config = Path(__file__).resolve().parents[2] / "config"
    assert main(["--config", str(config), "eval-gate", "--root", str(tmp_path)]) == 1
    assert "no scorecard" in capsys.readouterr().out


def test_scorecard_from_results() -> None:
    from claimlens.domain import Route
    from claimlens.evals.agent_metrics import AgentQuality
    from claimlens.evals.metrics import compute_triage_metrics
    from claimlens.evals.scorecard import scorecard_metrics
    from claimlens.evals.triage import AgentSummary

    triage = compute_triage_metrics([(Route.FAST_TRACK, Route.FAST_TRACK)])
    summary = AgentSummary(1, 2.0, 1.0, 0.01, 0.01, 0.01, {}, {"R9": 1})
    quality = AgentQuality(1.0, 0.5, 0.8, 0.9, 0.0)
    metrics = scorecard_metrics(triage, summary, quality)
    assert metrics == {
        "route_accuracy": 1.0,
        "escalation_recall": None,
        "citation_validity": 1.0,
        "citation_hit_rate": 0.5,
        "narrative_catch_rate": 0.8,
        "benign_pass_rate": 0.9,
        "agent_failure_rate": 0.0,
        "cost_mean_usd": 0.01,
        "judge_pass_rate": None,
    }


def test_fingerprints_accept_a_relative_golden_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _repo(tmp_path)
    monkeypatch.chdir(root)
    prints = fingerprints(Path.cwd(), Path("evals/golden/v2/claims.jsonl"))
    assert "evals/golden/v2/claims.jsonl" in prints
