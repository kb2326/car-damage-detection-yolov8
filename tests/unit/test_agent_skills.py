"""The triage agent loads approved skills and records which ones it used."""

from datetime import date
from pathlib import Path
from typing import Any

from claimlens.agent.config import load_agent_config
from claimlens.agent.langgraph_agent import LangGraphTriageAgent
from claimlens.agent.recommendation import SUBMIT
from claimlens.llm.budget import Budget
from claimlens.llm.cache import ResponseCache
from claimlens.llm.gateway import Gateway
from claimlens.llm.prompts import load_prompt
from claimlens.llm.provider import FakeProvider, ProviderReply
from claimlens.mcp.claims_system import build_claims_system
from claimlens.mcp.policy_admin import build_policy_admin
from claimlens.skills import Skill
from tests.mcp_helpers import MemoryAudit
from tests.unit.test_langgraph_agent import (
    ESCALATE,
    LLM,
    POLICIES,
    TRIAGE,
    _call,
    _state,
    index,  # noqa: F401
)

ROOT = Path(__file__).resolve().parents[2]
GLASS = Skill(
    name="glass-claims",
    description="Main damage is glass.",
    version=1,
    approved_by="Owner",
    approved_on="2026-10-03",
    body="1. Check which glass.\n2. Check the wording.",
)


def _agent(
    tmp_path: Path, script: list[ProviderReply | Exception], idx: Any, skills: dict[str, Skill]
) -> tuple[LangGraphTriageAgent, FakeProvider]:
    fake, audit = FakeProvider(script), MemoryAudit()
    gateway = Gateway(
        LLM,
        fake,
        Budget(tmp_path / "b.sqlite", LLM.limits, today=lambda: date(2026, 10, 3)),
        ResponseCache(tmp_path / "c.sqlite"),
        lambda call: None,
        sleep=lambda s: None,
    )
    servers = [
        build_policy_admin(TRIAGE, POLICIES, audit, lambda: idx),
        build_claims_system(TRIAGE, tmp_path / "claims.db", audit),
    ]
    config = load_agent_config(ROOT / "config" / "agent.toml")
    prompt = load_prompt(ROOT / "prompts", config.prompt_name, config.prompt_version)
    agent = LangGraphTriageAgent(gateway, config, prompt, servers, POLICIES, idx, skills=skills)
    return agent, fake


def test_the_repo_agent_uses_prompt_v2_and_the_skills_folder() -> None:
    config = load_agent_config(ROOT / "config" / "agent.toml")
    assert (config.prompt_name, config.prompt_version) == ("triage", "v2")
    assert config.skills_dir == "skills"


def test_approved_skills_are_listed_and_loadable(tmp_path: Path, index: Any) -> None:  # noqa: F811
    answer = {**ESCALATE, "policy_citations": []}
    agent, fake = _agent(
        tmp_path,
        [_call("load_skill", {"name": "glass-claims"}, "t1"), _call(SUBMIT, answer, "t2")],
        index,
        {"glass-claims": GLASS},
    )
    rec = agent.recommend(_state())
    system = fake.calls[0]["system"]
    assert "Approved procedures" in system
    assert "- glass-claims: Main damage is glass." in system
    assert "load_skill" in {t.name for t in fake.calls[0]["tools"]}
    result = fake.calls[1]["messages"][-1].tool_results[0]
    assert not result.is_error
    assert result.content.startswith('<procedure name="glass-claims" version="1">')
    assert "1. Check which glass." in result.content
    assert rec.skills_used == ("glass-claims@v1",)


def test_an_unknown_skill_is_a_plain_answer_not_an_error(tmp_path: Path, index: Any) -> None:  # noqa: F811
    answer = {**ESCALATE, "policy_citations": []}
    agent, fake = _agent(
        tmp_path,
        [_call("load_skill", {"name": "nope"}, "t1"), _call(SUBMIT, answer, "t2")],
        index,
        {"glass-claims": GLASS},
    )
    rec = agent.recommend(_state())
    result = fake.calls[1]["messages"][-1].tool_results[0]
    assert not result.is_error
    assert "No approved procedure named 'nope'" in result.content
    assert "glass-claims" in result.content
    assert rec.skills_used == ()


def test_no_skills_means_no_tool_and_no_list(tmp_path: Path, index: Any) -> None:  # noqa: F811
    answer = {**ESCALATE, "policy_citations": []}
    agent, fake = _agent(tmp_path, [_call(SUBMIT, answer, "t1")], index, {})
    agent.recommend(_state())
    assert "Approved procedures" not in fake.calls[0]["system"]
    assert "load_skill" not in {t.name for t in fake.calls[0]["tools"]}


def test_the_factory_loads_only_approved_repo_skills(tmp_path: Path, index: Any) -> None:  # noqa: F811
    from claimlens.agent.factory import build_llm_agent
    from claimlens.skills import load_skills

    gateway = Gateway(
        LLM,
        FakeProvider([]),
        Budget(tmp_path / "b.sqlite", LLM.limits, today=lambda: date(2026, 10, 3)),
        ResponseCache(tmp_path / "c.sqlite"),
        lambda call: None,
        sleep=lambda s: None,
    )
    agent = build_llm_agent(
        ROOT / "config", ROOT, tmp_path / "claims.db", gateway=gateway, index=index
    )
    assert set(agent.skills) == set(load_skills(ROOT / "skills"))
    assert agent.agent_version == "triage-agent-v1+triage/v2"
