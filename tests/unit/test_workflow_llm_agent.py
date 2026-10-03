"""An LLM agent failure must send the claim to a person (rule R2), never fast-track it."""

from collections.abc import Callable
from pathlib import Path

import pytest

from claimlens.agent.graph import AgentFailed
from claimlens.domain import AgentRecommendation, Route
from claimlens.events.projection import ClaimState, fold
from claimlens.events.store import SQLiteEventStore
from claimlens.intake import submit_claim
from claimlens.llm.types import BudgetExceeded, InvalidModelOutput, LLMUnavailable
from claimlens.workflow import process_claim
from tests.fakes import FakeDetector, make_test_deps


class Broken:
    agent_version = "broken-agent"

    def __init__(self, error: Exception) -> None:
        self._error = error

    def recommend(self, state: ClaimState) -> AgentRecommendation:
        raise self._error


@pytest.mark.parametrize(
    "error",
    [
        BudgetExceeded("LLM cap of $0.10 reached for claim c1"),
        LLMUnavailable("all LLM models are unavailable; route to a person"),
        InvalidModelOutput("model output failed validation twice"),
        AgentFailed("step limit of 6 model calls reached"),
    ],
)
def test_llm_agent_failures_route_to_adjuster_review(
    store: SQLiteEventStore,
    tmp_path: Path,
    make_image: Callable[..., Path],
    error: Exception,
) -> None:
    deps = make_test_deps(tmp_path, store, FakeDetector(), agent=Broken(error))
    claim_id = submit_claim(
        store,
        deps.blobs,
        policy_id="P-1001",
        description="scrape",
        photo_paths=[make_image("a.jpg")],
    )
    decision = process_claim(claim_id, deps)
    assert decision.route is Route.ADJUSTER_REVIEW
    assert decision.rule_id == "R2"
    failures = fold(store.load(claim_id)).failures
    assert any(f.stage == "agent" and type(error).__name__ in f.error for f in failures)
