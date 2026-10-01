"""Triage agents. Every agent advises; the decision policy decides."""

from __future__ import annotations

from typing import Protocol

from claimlens.domain import AgentRecommendation
from claimlens.events.projection import ClaimState


class TriageAgent(Protocol):
    @property
    def agent_version(self) -> str: ...

    def recommend(self, state: ClaimState) -> AgentRecommendation: ...
