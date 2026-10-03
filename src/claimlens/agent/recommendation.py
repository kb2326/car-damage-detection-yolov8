"""The agent's structured answer, and the checks code applies before trusting it."""

from __future__ import annotations

from collections.abc import Callable, Collection, Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from claimlens.domain import AgentRecommendation, Confidence, Route

SUBMIT = "submit_recommendation"


class Recommendation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    route_suggestion: Route
    confidence: Confidence
    rationale: str = Field(max_length=1200)
    evidence_ids: list[str] = Field(default_factory=list, max_length=20)
    policy_citations: list[str] = Field(default_factory=list, max_length=10)
    open_questions: list[str] = Field(default_factory=list, max_length=5)


SUBMIT_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": SUBMIT,
        "description": (
            "Submit your final recommendation. Call this exactly once, when you are done. "
            "Cite evidence by its label (E1) and policy wording by clause id (STD-8.5)."
        ),
        "parameters": Recommendation.model_json_schema(),
    },
}


def check_recommendation(
    args: Mapping[str, Any],
    *,
    evidence_ids: Collection[str],
    wording: str | None,
    wording_of: Callable[[str], str | None],
) -> list[str]:
    """Problems that make the recommendation unusable; empty when it can be trusted."""
    try:
        rec = Recommendation.model_validate(dict(args))
    except ValidationError as exc:
        return [f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()[:5]]
    problems: list[str] = []
    if rec.policy_citations and wording is None:
        problems.append("no policy wording is on file, so no clause may be cited")
    else:
        for clause in rec.policy_citations:
            actual = wording_of(clause)
            if actual is None:
                problems.append(f"clause {clause} does not exist")
            elif actual != wording:
                problems.append(
                    f"clause {clause} is from the {actual} wording; this policy uses {wording}"
                )
    problems.extend(
        f"evidence {e} is not in the evidence list"
        for e in rec.evidence_ids
        if e not in evidence_ids
    )
    if not rec.rationale.strip():
        problems.append("give a reason in rationale")
    return problems


def to_agent_recommendation(rec: Recommendation, ids: Mapping[str, str]) -> AgentRecommendation:
    return AgentRecommendation(
        route_suggestion=rec.route_suggestion,
        confidence=rec.confidence,
        rationale=rec.rationale.strip(),
        citations=tuple(ids[e] for e in rec.evidence_ids),
        policy_citations=tuple(rec.policy_citations),
        open_questions=tuple(q for q in rec.open_questions if q.strip()),
    )
