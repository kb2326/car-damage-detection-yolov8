"""Placeholder triage agent for M1. Replaced by the LLM triage agent in M5."""

from __future__ import annotations

from claimlens.domain import AgentRecommendation, Confidence, Route
from claimlens.events.projection import ClaimState


class StubTriageAgent:
    agent_version = "stub-v0"

    def recommend(self, state: ClaimState) -> AgentRecommendation:
        citations = tuple(state.detection_event_ids.values())
        if not state.findings:
            return AgentRecommendation(
                route_suggestion=Route.ADJUSTER_REVIEW,
                confidence=Confidence.LOW,
                rationale="No damage was detected in the accepted photos.",
                citations=citations,
                open_questions=("Is there damage that the photos do not show?",),
            )
        damage_types = ", ".join(sorted({f.damage_type.value for f in state.findings}))
        return AgentRecommendation(
            route_suggestion=Route.FAST_TRACK,
            confidence=Confidence.MEDIUM,
            rationale=f"Detected {len(state.findings)} damage finding(s): {damage_types}.",
            citations=citations,
        )
