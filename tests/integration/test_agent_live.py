"""One real claim through the real agent. Opt-in: CLAIMLENS_LIVE=1 (spends a few cents)."""

import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(os.environ.get("CLAIMLENS_LIVE") != "1", reason="live LLM test")


def test_live_agent_escalates_a_delivery_claim_and_cites_the_exclusion(tmp_path: Path) -> None:
    from claimlens.agent.factory import build_llm_agent
    from claimlens.domain import Route
    from tests.unit.test_langgraph_agent import _state

    root = Path(__file__).resolve().parents[2]
    agent = build_llm_agent(root / "config", root, tmp_path / "claims.db")
    rec = agent.recommend(_state(description="I hit a pole while delivering pizza for my job."))
    print(rec.model_dump_json(indent=1))
    assert rec.route_suggestion is Route.ADJUSTER_REVIEW
    assert "STD-8.5" in rec.policy_citations
