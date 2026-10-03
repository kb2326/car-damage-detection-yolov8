from datetime import date
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest

from claimlens.agent.config import load_agent_config
from claimlens.agent.graph import AgentFailed
from claimlens.agent.langgraph_agent import LangGraphTriageAgent
from claimlens.agent.recommendation import SUBMIT
from claimlens.domain import Confidence, Coverage, Route
from claimlens.events.projection import ClaimState
from claimlens.knowledge.clauses import load_wordings
from claimlens.knowledge.embed import FakeEmbedder
from claimlens.knowledge.index import IndexMissingError, PolicyIndex, build_index
from claimlens.llm.budget import Budget
from claimlens.llm.cache import ResponseCache
from claimlens.llm.config import load_llm_config
from claimlens.llm.gateway import Gateway
from claimlens.llm.prompts import load_prompt
from claimlens.llm.provider import FakeProvider, ProviderReply
from claimlens.llm.types import ToolCall
from claimlens.mcp.claims_system import build_claims_system
from claimlens.mcp.policy_admin import build_policy_admin
from claimlens.mcp.profiles import load_profiles
from claimlens.policy import load_policies
from tests.mcp_helpers import MemoryAudit

ROOT = Path(__file__).resolve().parents[2]
LLM = load_llm_config(ROOT / "config" / "llm.toml")
CONFIG = load_agent_config(ROOT / "config" / "agent.toml")
PROMPT = load_prompt(ROOT / "prompts", "triage", "v1")
POLICIES = load_policies(ROOT / "config" / "policies.toml")
TRIAGE = load_profiles(ROOT / "config" / "agents.toml")["triage"]


def _call(name: str, args: dict[str, Any], id_: str) -> ProviderReply:
    return ProviderReply("", 500, 50, tool_calls=(ToolCall(id=id_, name=name, arguments=args),))


@pytest.fixture
def index(tmp_path: Path) -> PolicyIndex:
    build_index(
        load_wordings(ROOT / "knowledge" / "policies"), FakeEmbedder(), tmp_path / "lancedb"
    )
    return PolicyIndex.open(tmp_path / "lancedb", FakeEmbedder())


def _agent(
    tmp_path: Path, script: list[ProviderReply | Exception], index: Any
) -> tuple[LangGraphTriageAgent, FakeProvider, MemoryAudit]:
    fake, audit = FakeProvider(script), MemoryAudit()
    gateway = Gateway(
        LLM,
        fake,
        Budget(tmp_path / "b.sqlite", LLM.limits, today=lambda: date(2026, 10, 2)),
        ResponseCache(tmp_path / "c.sqlite"),
        lambda call: None,
        sleep=lambda s: None,
    )
    servers = [
        build_policy_admin(TRIAGE, POLICIES, audit, lambda: index() if callable(index) else index),
        build_claims_system(TRIAGE, tmp_path / "claims.db", audit),
    ]
    return LangGraphTriageAgent(gateway, CONFIG, PROMPT, servers, POLICIES, index), fake, audit


def _state(
    policy_id: str = "P-1001", description: str = "Hit a pole while delivering pizza."
) -> ClaimState:
    state = ClaimState(claim_id=uuid4(), policy_id=policy_id, description=description)
    state.detection_event_ids = {"p1": "evt-abc"}
    state.coverage = Coverage(
        policy_id=policy_id, found=True, active=True, collision=True, deductible=250
    )
    return state


ESCALATE = {
    "route_suggestion": "ADJUSTER_REVIEW",
    "confidence": "high",
    "rationale": "The description says the car was used for deliveries; STD-8.5 excludes that.",
    "evidence_ids": ["E1"],
    "policy_citations": ["STD-8.5"],
    "open_questions": [],
}


def test_agent_searches_then_recommends_with_citations(tmp_path: Path, index: PolicyIndex) -> None:
    agent, fake, audit = _agent(
        tmp_path,
        [
            _call("search_policy_clauses", {"query": "delivery commercial use"}, "t1"),
            _call(SUBMIT, ESCALATE, "t2"),
        ],
        index,
    )
    rec = agent.recommend(_state())
    assert rec.route_suggestion is Route.ADJUSTER_REVIEW
    assert rec.confidence is Confidence.HIGH
    assert rec.policy_citations == ("STD-8.5",)
    assert rec.citations == ("evt-abc",)
    assert [r.tool for r in audit.records] == ["search_policy_clauses"]
    assert agent.agent_version == "triage-agent-v1+triage/v1"
    assert fake.calls[0]["system"] == PROMPT.text


def test_search_is_bound_to_the_claimants_policy(tmp_path: Path, index: PolicyIndex) -> None:
    agent, fake, _ = _agent(
        tmp_path,
        [
            _call("search_policy_clauses", {"query": "rental car", "policy_id": "P-1003"}, "t1"),
            _call(SUBMIT, {**ESCALATE, "policy_citations": []}, "t2"),
        ],
        index,
    )
    agent.recommend(_state())
    result = fake.calls[1]["messages"][-1].tool_results[0].content
    assert "STD-" in result
    assert "PRM-" not in result
    schema = next(
        t for t in fake.calls[0]["tools"] if t.name == "search_policy_clauses"
    ).input_schema
    assert "policy_id" not in schema["properties"]


def test_a_clause_from_another_wording_is_rejected_then_fails(
    tmp_path: Path, index: PolicyIndex
) -> None:
    wrong = {**ESCALATE, "policy_citations": ["PRM-8.5"]}
    agent, _, _ = _agent(tmp_path, [_call(SUBMIT, wrong, "t1"), _call(SUBMIT, wrong, "t2")], index)
    with pytest.raises(AgentFailed, match="premium wording"):
        agent.recommend(_state())


def test_unknown_policy_gets_no_search_tool_and_no_citations(
    tmp_path: Path, index: PolicyIndex
) -> None:
    answer = {**ESCALATE, "policy_citations": [], "rationale": "No policy is on file."}
    agent, fake, _ = _agent(tmp_path, [_call(SUBMIT, answer, "t1")], index)
    state = _state(policy_id="P-0000")
    state.coverage = Coverage(
        policy_id="P-0000", found=False, active=False, collision=False, deductible=0
    )
    assert agent.recommend(state).policy_citations == ()
    assert "search_policy_clauses" not in {t.name for t in fake.calls[0]["tools"]}


def test_a_missing_index_fails_to_a_person(tmp_path: Path) -> None:
    def missing() -> PolicyIndex:
        raise IndexMissingError("no policy index: run `claimlens knowledge build`")

    agent, _, _ = _agent(
        tmp_path, [_call(SUBMIT, ESCALATE, "t1"), _call(SUBMIT, ESCALATE, "t2")], missing
    )
    with pytest.raises((AgentFailed, IndexMissingError), match=r"knowledge build|index"):
        agent.recommend(_state())


def test_build_llm_agent_wires_config_prompt_and_tools(tmp_path: Path, index: PolicyIndex) -> None:
    from claimlens.agent.factory import build_llm_agent

    fake = FakeProvider([_call(SUBMIT, {**ESCALATE, "policy_citations": []}, "t1")])
    gateway = Gateway(
        LLM,
        fake,
        Budget(tmp_path / "b.sqlite", LLM.limits, today=lambda: date(2026, 10, 2)),
        ResponseCache(tmp_path / "c.sqlite"),
        lambda call: None,
        sleep=lambda s: None,
    )
    agent = build_llm_agent(
        ROOT / "config", ROOT, tmp_path / "claims.db", gateway=gateway, index=index
    )
    assert agent.agent_version == "triage-agent-v1+triage/v2"
    agent.recommend(_state())
    assert {t.name for t in fake.calls[0]["tools"]} == {*CONFIG.tools, SUBMIT}
    assert fake.calls[0]["max_tokens"] == CONFIG.max_tokens


def test_a_failed_policy_search_cannot_end_in_fast_track(tmp_path: Path) -> None:
    """Review C1: the index is missing, the search errors, and the model submits FAST_TRACK."""

    def missing() -> PolicyIndex:
        raise IndexMissingError("no policy index: run `claimlens knowledge build`")

    fast = {
        "route_suggestion": "FAST_TRACK",
        "confidence": "high",
        "rationale": "Looks fine.",
        "evidence_ids": ["E1"],
        "policy_citations": [],
        "open_questions": [],
    }
    agent, _, _ = _agent(
        tmp_path,
        [
            _call("search_policy_clauses", {"query": "racing exclusion"}, "t1"),
            _call(SUBMIT, fast, "t2"),
            _call(SUBMIT, fast, "t3"),
        ],
        missing,
    )
    with pytest.raises((AgentFailed, IndexMissingError)):
        agent.recommend(_state(description="Crashed during a track day race."))
