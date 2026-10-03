from datetime import date
from pathlib import Path

import pytest
from pydantic import BaseModel, ValidationError

from claimlens.llm.budget import Budget
from claimlens.llm.cache import ResponseCache
from claimlens.llm.config import load_llm_config
from claimlens.llm.gateway import Gateway
from claimlens.llm.log import LLMCall
from claimlens.llm.provider import FakeProvider, ProviderReply
from claimlens.llm.types import LLMRequest, Message, ToolCall, ToolResult, ToolSpec

ROOT = Path(__file__).resolve().parents[2]
CONFIG = load_llm_config(ROOT / "config" / "llm.toml")
LOOKUP = ToolSpec(
    name="get_policy",
    description="Look up a policy.",
    input_schema={"type": "object", "properties": {"policy_id": {"type": "string"}}},
)
CALL = ToolCall(id="t1", name="get_policy", arguments={"policy_id": "P-1001"})


def _gateway(
    tmp_path: Path, script: list[ProviderReply | Exception]
) -> tuple[Gateway, FakeProvider]:
    fake = FakeProvider(script)
    calls: list[LLMCall] = []
    gateway = Gateway(
        CONFIG,
        fake,
        Budget(tmp_path / "b.sqlite", CONFIG.limits, today=lambda: date(2026, 10, 2)),
        ResponseCache(tmp_path / "c.sqlite"),
        calls.append,
        sleep=lambda s: None,
    )
    return gateway, fake


def _ask(**kw: object) -> LLMRequest:
    return LLMRequest(messages=[Message(role="user", content="evidence")], tier="strong", **kw)


def test_tool_calls_come_back_from_the_gateway(tmp_path: Path) -> None:
    gateway, fake = _gateway(tmp_path, [ProviderReply("", 100, 20, tool_calls=(CALL,))])
    response = gateway.generate(_ask(tools=(LOOKUP,)))
    assert response.tool_calls == (CALL,)
    assert fake.calls[0]["tools"] == (LOOKUP,)
    assert fake.calls[0]["tool_choice"] is None


def test_tool_choice_reaches_the_provider(tmp_path: Path) -> None:
    gateway, fake = _gateway(tmp_path, [ProviderReply("", 1, 1, tool_calls=(CALL,))])
    gateway.generate(_ask(tools=(LOOKUP,), tool_choice="get_policy"))
    assert fake.calls[0]["tool_choice"] == "get_policy"


def test_tool_choice_must_name_a_tool() -> None:
    with pytest.raises(ValidationError, match="tool_choice"):
        _ask(tools=(LOOKUP,), tool_choice="other")


def test_tools_and_output_schema_cannot_be_combined() -> None:
    class Out(BaseModel):
        x: int

    with pytest.raises(ValidationError, match="cannot be combined"):
        _ask(tools=(LOOKUP,), output_schema=Out)


def test_a_message_needs_content_or_tool_blocks() -> None:
    with pytest.raises(ValidationError, match="empty message"):
        Message(role="user")
    Message(role="user", tool_results=(ToolResult(tool_call_id="t1", content="{}"),))
    Message(role="assistant", tool_calls=(CALL,))


def test_tool_calls_are_cached(tmp_path: Path) -> None:
    gateway, fake = _gateway(tmp_path, [ProviderReply("", 100, 20, tool_calls=(CALL,))])
    first = gateway.generate(_ask(tools=(LOOKUP,)))
    second = gateway.generate(_ask(tools=(LOOKUP,)))
    assert second.cached
    assert second.tool_calls == first.tool_calls
    assert len(fake.calls) == 1


def test_different_tools_do_not_share_a_cache_entry(tmp_path: Path) -> None:
    other = LOOKUP.model_copy(update={"name": "get_coverage"})
    gateway, fake = _gateway(tmp_path, [ProviderReply("a", 1, 1), ProviderReply("b", 1, 1)])
    gateway.generate(_ask(tools=(LOOKUP,)))
    gateway.generate(_ask(tools=(other,)))
    assert len(fake.calls) == 2


def test_tool_results_in_the_conversation_reach_the_provider(tmp_path: Path) -> None:
    gateway, fake = _gateway(tmp_path, [ProviderReply("done", 1, 1)])
    messages = [
        Message(role="user", content="evidence"),
        Message(role="assistant", tool_calls=(CALL,)),
        Message(
            role="user", tool_results=(ToolResult(tool_call_id="t1", content='{"found": true}'),)
        ),
    ]
    gateway.generate(LLMRequest(messages=messages, tier="strong", tools=(LOOKUP,)))
    assert fake.calls[0]["messages"] == messages
