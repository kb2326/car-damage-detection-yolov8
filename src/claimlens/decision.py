"""Deterministic decision policy: the rules that make the final routing call (spec section 11)."""

from __future__ import annotations

import tomllib
from pathlib import Path

from pydantic import Field

from claimlens.domain import Confidence, Decision, Frozen, Route
from claimlens.events.projection import ClaimState


class DecisionConfig(Frozen):
    version: str
    fraud_score_threshold: float = Field(ge=0.0, le=1.0)
    max_fast_track_cost_usd: int = Field(ge=0)
    min_finding_confidence: float = Field(ge=0.0, le=1.0)


def load_decision_config(path: Path) -> DecisionConfig:
    return DecisionConfig.model_validate(tomllib.loads(path.read_text(encoding="utf-8")))


def decide(state: ClaimState, config: DecisionConfig) -> Decision:
    """Apply rules in order; the first match wins. No rule can deny a claim."""

    def result(route: Route, rule_id: str, reason: str) -> Decision:
        return Decision(route=route, rule_id=rule_id, reason=reason, policy_version=config.version)

    flagged = [s for s in state.fraud_signals if s.score >= config.fraud_score_threshold]
    if flagged:
        return result(Route.FRAUD_REVIEW, "R1", " ".join(s.detail for s in flagged))
    if state.failures:
        stages = ", ".join(sorted({f.stage for f in state.failures}))
        return result(Route.ADJUSTER_REVIEW, "R2", f"Processing failed at: {stages}.")
    if not state.accepted_photos:
        return result(Route.ADJUSTER_REVIEW, "R3", "No usable photos were provided.")
    if state.coverage is None or not state.coverage.confirmed:
        return result(Route.ADJUSTER_REVIEW, "R4", "Coverage could not be confirmed.")
    limit = config.max_fast_track_cost_usd
    if state.cost_estimate is None or state.cost_estimate.high > limit:
        return result(
            Route.ADJUSTER_REVIEW, "R5", f"Estimate missing or above the ${limit:,} limit."
        )
    threshold = config.min_finding_confidence
    uncertain = [f for f in state.findings if f.confidence < threshold]
    if uncertain:
        return result(
            Route.ADJUSTER_REVIEW,
            "R6",
            f"{len(uncertain)} finding(s) below confidence {threshold:.2f}.",
        )
    rec = state.recommendation
    if rec is None or rec.confidence is Confidence.LOW or rec.open_questions:
        return result(
            Route.ADJUSTER_REVIEW, "R7", "The triage agent is unsure or has open questions."
        )
    if rec.route_suggestion is not Route.FAST_TRACK:
        return result(
            Route.ADJUSTER_REVIEW, "R8", "The triage agent did not recommend fast-tracking."
        )
    return result(Route.FAST_TRACK, "R9", "All checks passed.")
