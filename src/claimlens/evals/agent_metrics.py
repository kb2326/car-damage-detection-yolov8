"""Agent quality scored by code (no LLM): citations, narrative cases, failures."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from claimlens.domain import Route
from claimlens.evals.golden import GoldenClaim
from claimlens.evals.triage import CaseResult


@dataclass(frozen=True)
class AgentQuality:
    citation_validity: float | None
    citation_hit_rate: float | None
    narrative_catch_rate: float | None
    benign_pass_rate: float | None
    agent_failure_rate: float | None


def _rate(hits: int, total: int) -> float | None:
    return None if total == 0 else hits / total


def agent_quality(
    cases: Sequence[GoldenClaim],
    results: Sequence[CaseResult],
    wording_of_policy: Mapping[str, str],
    wording_of_clause: Callable[[str], str | None],
) -> AgentQuality:
    by_id = {r.case_id: r for r in results}
    cited = valid = hit_total = hits = esc_total = caught = benign_total = passed = failed = 0
    for case in cases:
        result = by_id.get(case.case_id)
        if result is None:
            continue
        rec = result.state.recommendation if result.state is not None else None
        failed += bool(result.agent_error)
        if rec is not None:
            wording = wording_of_policy.get(case.policy_id)
            cited += len(rec.policy_citations)
            valid += sum(
                wording is not None and wording_of_clause(c) == wording
                for c in rec.policy_citations
            )
        if not case.narrative:
            continue
        if case.expected_route is Route.FAST_TRACK:
            benign_total += 1
            passed += rec is not None and rec.route_suggestion is Route.FAST_TRACK
        else:
            esc_total += 1
            caught += rec is None or rec.route_suggestion is not Route.FAST_TRACK
            if case.expected_citations and rec is not None:
                hit_total += 1
                hits += bool(set(case.expected_citations) & set(rec.policy_citations))
    return AgentQuality(
        citation_validity=_rate(valid, cited),
        citation_hit_rate=_rate(hits, hit_total),
        narrative_catch_rate=_rate(caught, esc_total),
        benign_pass_rate=_rate(passed, benign_total),
        agent_failure_rate=_rate(failed, len(results)),
    )
