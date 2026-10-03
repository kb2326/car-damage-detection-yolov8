from datetime import date
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from claimlens.agent.chat_model import GatewayChatModel, to_gateway_messages
from claimlens.llm.budget import Budget
from claimlens.llm.cache import ResponseCache
from claimlens.llm.config import load_llm_config
from claimlens.llm.gateway import Gateway
from claimlens.llm.provider import FakeProvider, ProviderReply
from claimlens.llm.types import BudgetExceeded, Message, ToolCall, ToolResult

ROOT = Path(__file__).resolve().parents[2]
CONFIG = load_llm_config(ROOT / "config" / "llm.toml")
CALL = ToolCall(id="t1", name="get_policy", arguments={"policy_id": "P-1001"})
LOOKUP = {
    "type": "function",
    "function": {
        "name": "get_policy",
        "description": "Look up a policy.",
        "parameters": {"type": "object", "properties": {"policy_id": {"type": "string"}}},
    },
}


def _model(
    tmp_path: Path, script: list[ProviderReply | Exception]
) -> tuple[GatewayChatModel, FakeProvider]:
    fake = FakeProvider(script)
    gateway = Gateway(
        CONFIG,
        fake,
        Budget(tmp_path / "b.sqlite", CONFIG.limits, today=lambda: date(2026, 10, 2)),
        ResponseCache(tmp_path / "c.sqlite"),
        lambda call: None,
        sleep=lambda s: None,
    )
    return GatewayChatModel(
        gateway=gateway, tier="strong", claim_id="c1", prompt_id="triage/v1"
    ), fake


def test_messages_convert_to_gateway_messages() -> None:
    system, messages = to_gateway_messages(
        [
            SystemMessage("You are a triage agent."),
            HumanMessage("evidence"),
            AIMessage(
                content="",
                tool_calls=[{"name": "get_policy", "args": {"policy_id": "P-1001"}, "id": "t1"}],
            ),
            ToolMessage(content='{"found": true}', tool_call_id="t1"),
            ToolMessage(content="boom", tool_call_id="t2", status="error"),
        ]
    )
    assert system == "You are a triage agent."
    assert messages == [
        Message(role="user", content="evidence"),
        Message(role="assistant", tool_calls=(CALL,)),
        Message(
            role="user",
            tool_results=(
                ToolResult(tool_call_id="t1", content='{"found": true}'),
                ToolResult(tool_call_id="t2", content="boom", is_error=True),
            ),
        ),
    ]


def test_invoke_returns_tool_calls_usage_and_cost(tmp_path: Path) -> None:
    model, fake = _model(tmp_path, [ProviderReply("", 1000, 100, tool_calls=(CALL,))])
    reply = model.bind_tools([LOOKUP]).invoke([SystemMessage("sys"), HumanMessage("evidence")])
    assert reply.tool_calls[0]["name"] == "get_policy"
    assert reply.tool_calls[0]["args"] == {"policy_id": "P-1001"}
    assert reply.tool_calls[0]["id"] == "t1"
    assert reply.usage_metadata == {
        "input_tokens": 1000,
        "output_tokens": 100,
        "total_tokens": 1100,
    }
    assert reply.response_metadata["model_name"] == "claude-sonnet-5-5"
    assert reply.response_metadata["cost_usd"] == pytest.approx(0.003)
    assert fake.calls[0]["system"] == "sys"
    assert [t.name for t in fake.calls[0]["tools"]] == ["get_policy"]


def test_bind_tools_returns_a_copy_and_passes_tool_choice(tmp_path: Path) -> None:
    model, fake = _model(tmp_path, [ProviderReply("", 1, 1, tool_calls=(CALL,))])
    bound = model.bind_tools([LOOKUP], tool_choice="get_policy")
    assert model.specs == ()
    bound.invoke([HumanMessage("evidence")])
    assert fake.calls[0]["tool_choice"] == "get_policy"


def test_gateway_errors_are_not_swallowed(tmp_path: Path) -> None:
    model, _ = _model(tmp_path, [ProviderReply("x", 1, 1)])
    huge = "word " * 400_000
    with pytest.raises(BudgetExceeded):
        model.invoke([HumanMessage(huge)])
