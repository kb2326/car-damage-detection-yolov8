from pathlib import Path

import pytest
from langchain_core.tools import ToolException

from claimlens.agent.tools import load_tools
from claimlens.mcp.base import ScopedServer
from claimlens.mcp.policy_admin import build_policy_admin
from claimlens.mcp.profiles import load_profiles
from claimlens.policy import load_policies
from tests.mcp_helpers import MemoryAudit

ROOT = Path(__file__).resolve().parents[2]
PROFILES = load_profiles(ROOT / "config" / "agents.toml")
POLICIES = load_policies(ROOT / "config" / "policies.toml")


def _policy_server(profile: str = "triage") -> tuple[ScopedServer, MemoryAudit]:
    audit = MemoryAudit()
    return build_policy_admin(PROFILES[profile], POLICIES, audit), audit


def test_only_allowed_tools_are_loaded() -> None:
    server, _ = _policy_server()
    tools = load_tools([server], allow=["get_coverage"], bound={})
    assert [t.name for t in tools] == ["get_coverage"]


def test_bound_arguments_are_hidden_and_filled_in() -> None:
    server, audit = _policy_server()
    (tool,) = load_tools([server], allow=["get_coverage"], bound={"policy_id": "P-1001"})
    assert "policy_id" not in tool.args
    result = tool.invoke({})
    assert '"deductible"' in result
    assert '"found": true' in result
    assert audit.records[-1].tool == "get_coverage"


def test_the_model_cannot_override_a_bound_argument() -> None:
    server, _ = _policy_server()
    (tool,) = load_tools([server], allow=["get_coverage"], bound={"policy_id": "P-1001"})
    assert tool.invoke({"policy_id": "P-9999"}) == tool.invoke({})


def test_tool_errors_are_tool_exceptions() -> None:
    server, _ = _policy_server()
    (tool,) = load_tools([server], allow=["search_policy_clauses"], bound={"policy_id": "P-1001"})
    with pytest.raises(ToolException, match="not configured"):
        tool.invoke({"query": "rental car"})


def test_scopes_still_apply() -> None:
    server, _ = _policy_server("intake")  # intake may not search clauses
    assert load_tools([server], allow=["search_policy_clauses"], bound={}) == []
