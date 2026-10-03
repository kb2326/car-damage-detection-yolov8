"""The LLM triage agent: implements `TriageAgent` with a LangGraph graph on our gateway."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from claimlens.agent.chat_model import GatewayChatModel
from claimlens.agent.config import AgentConfig
from claimlens.agent.evidence import render_evidence
from claimlens.agent.graph import build_graph, run_graph
from claimlens.agent.recommendation import (
    Recommendation,
    check_recommendation,
    to_agent_recommendation,
)
from claimlens.agent.tools import load_tools
from claimlens.domain import AgentRecommendation
from claimlens.events.projection import ClaimState
from claimlens.knowledge.index import PolicyIndex
from claimlens.llm.gateway import Gateway
from claimlens.llm.prompts import Prompt
from claimlens.mcp.base import ScopedServer
from claimlens.policy import PolicyRepository

SEARCH = "search_policy_clauses"


class LangGraphTriageAgent:
    def __init__(
        self,
        gateway: Gateway,
        config: AgentConfig,
        prompt: Prompt,
        servers: Sequence[ScopedServer],
        policies: PolicyRepository,
        index: PolicyIndex | Callable[[], PolicyIndex],
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._gateway = gateway
        self._config = config
        self._prompt = prompt
        self._servers = servers
        self._policies = policies
        self._index = index
        self._clock = clock

    @property
    def agent_version(self) -> str:
        return f"{self._config.version}+{self._prompt.id}"

    def _wording_of(self, clause_id: str) -> str | None:
        index = self._index() if callable(self._index) else self._index
        clause = index.get(clause_id)
        return None if clause is None else clause.wording

    def recommend(self, state: ClaimState) -> AgentRecommendation:
        """Raises on any failure; the workflow records it and rule R2 routes to a person."""
        record = self._policies.get_record(state.policy_id)
        wording = None if record is None else record.wording
        if wording is not None:
            # Fail before any model call if the policy wording cannot be searched.
            _ = self._index() if callable(self._index) else self._index
        evidence = render_evidence(state)
        allow = [t for t in self._config.tools if wording is not None or t != SEARCH]
        tools = load_tools(
            self._servers,
            allow=allow,
            bound={"claim_id": str(state.claim_id), "policy_id": state.policy_id},
        )
        model = GatewayChatModel(
            gateway=self._gateway,
            tier=self._config.tier,
            claim_id=str(state.claim_id),
            prompt_id=self._prompt.id,
            max_tokens=self._config.max_tokens,
        )

        def check(args: Mapping[str, Any]) -> list[str]:
            return check_recommendation(
                args, evidence_ids=evidence.ids, wording=wording, wording_of=self._wording_of
            )

        graph = build_graph(model, tools, check, self._config, self._clock)
        accepted = run_graph(graph, self._prompt.text, evidence.text, self._config, self._clock)
        return to_agent_recommendation(Recommendation.model_validate(accepted), evidence.ids)
