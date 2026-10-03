"""A committed scorecard, and the gate that compares it with the baseline and the files on disk."""

from __future__ import annotations

import hashlib
import tomllib
from collections.abc import Mapping
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING

from claimlens.domain import Frozen

if TYPE_CHECKING:
    from claimlens.evals.agent_metrics import AgentQuality
    from claimlens.evals.metrics import TriageMetrics
    from claimlens.evals.triage import AgentSummary

FINGERPRINTED: tuple[str, ...] = (
    "config/decision_policy.toml",
    "config/rate_card.toml",
    "config/models.toml",
    "config/agent.toml",
    "config/llm.toml",
    "config/policies.toml",
    "config/taxonomy.toml",
    "config/agents.toml",
    "prompts/triage/*.md",
    "prompts/judge/*.md",
    "knowledge/policies/*.md",
)


def fingerprints(repo_root: Path, golden: Path) -> dict[str, str]:
    repo_root = repo_root.resolve()
    golden = (golden if golden.is_absolute() else repo_root / golden).resolve()
    found = {p.resolve() for pattern in FINGERPRINTED for p in repo_root.glob(pattern)}
    paths = found | ({golden} if golden.is_file() else set())  # a missing file reads as stale
    return {
        p.relative_to(repo_root).as_posix(): hashlib.sha256(
            p.read_bytes().replace(b"\r\n", b"\n")  # the same hash on Windows and Linux
        ).hexdigest()
        for p in sorted(paths)
    }


class Scorecard(Frozen):
    created_on: date
    golden: str
    versions: dict[str, str]
    metrics: dict[str, float | None]
    fingerprints: dict[str, str]


class GateConfig(Frozen):
    max_drop: dict[str, float]
    must_equal_one: tuple[str, ...]
    max_cost_increase: float


def load_gate_config(path: Path) -> GateConfig:
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    return GateConfig(
        max_drop=data["max_drop"],
        must_equal_one=tuple(data["must_equal_one"]),
        max_cost_increase=data["max_cost_increase"],
    )


def check_gate(
    current: Scorecard, baseline: Scorecard, on_disk: Mapping[str, str], config: GateConfig
) -> list[str]:
    problems: list[str] = []
    for path, digest in current.fingerprints.items():
        if path not in on_disk:
            problems.append(f"stale: {path} is missing")
        elif on_disk[path] != digest:
            problems.append(f"stale: {path} changed since the evaluation ran; re-run eval-triage")
    problems.extend(
        f"stale: {path} is new since the evaluation ran; re-run eval-triage"
        for path in on_disk
        if path not in current.fingerprints
    )
    for name in config.must_equal_one:
        value = current.metrics.get(name)
        if value is None:
            problems.append(f"safety: {name} was not measured")
        elif value < 1.0:
            problems.append(f"safety: {name} is {value:.2f}, must be 1.00")
    for name, margin in config.max_drop.items():
        now, before = current.metrics.get(name), baseline.metrics.get(name)
        if now is None or before is None:
            continue
        if now < before - margin - 1e-9:
            problems.append(
                f"regression: {name} fell from {before:.2f} to {now:.2f} "
                f"(allowed drop {margin:.2f})"
            )
    now_cost, before_cost = (
        current.metrics.get("cost_mean_usd"),
        baseline.metrics.get("cost_mean_usd"),
    )
    if now_cost and before_cost and now_cost > before_cost * (1 + config.max_cost_increase) + 1e-9:
        problems.append(
            f"cost: mean cost per claim rose from ${before_cost:.3f} to ${now_cost:.3f} "
            f"(allowed +{config.max_cost_increase:.0%})"
        )
    return problems


def scorecard_metrics(
    triage: TriageMetrics,
    agent: AgentSummary | None,
    quality: AgentQuality | None,
    list_cost_mean: float | None = None,
) -> dict[str, float | None]:
    """The metrics the gate compares; the judge's pass rate is added once it is validated."""
    return {
        "route_accuracy": triage.route_accuracy,
        "escalation_recall": triage.escalation_recall,
        "citation_validity": quality.citation_validity if quality else None,
        "citation_hit_rate": quality.citation_hit_rate if quality else None,
        "narrative_catch_rate": quality.narrative_catch_rate if quality else None,
        "benign_pass_rate": quality.benign_pass_rate if quality else None,
        "agent_failure_rate": quality.agent_failure_rate if quality else None,
        # List price from token counts, so response-cache hits do not lower it.
        "cost_mean_usd": list_cost_mean
        if list_cost_mean is not None
        else (agent.cost_mean_usd if agent else None),
        "judge_pass_rate": None,
    }


def run_eval_gate(repo_root: Path, current: Path, baseline: Path, config: GateConfig) -> int:
    """Print each problem and return 1, or print `eval gate: ok` and return 0."""
    for path in (current, baseline):
        if not path.is_file():
            print(f"eval gate: no scorecard at {path}; run eval-triage --scorecard first")
            return 1
    now = Scorecard.model_validate_json(current.read_text(encoding="utf-8"))
    before = Scorecard.model_validate_json(baseline.read_text(encoding="utf-8"))
    on_disk = fingerprints(repo_root, repo_root / now.golden)
    problems = check_gate(now, before, on_disk, config)
    for problem in problems:
        print(problem)
    if problems:
        return 1
    print("eval gate: ok")
    return 0
