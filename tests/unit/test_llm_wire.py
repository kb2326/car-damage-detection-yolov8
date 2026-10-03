from types import SimpleNamespace

from claimlens.llm.types import Message, ToolCall, ToolResult, ToolSpec
from claimlens.llm.wire import (
    from_anthropic_content,
    to_anthropic_messages,
    to_anthropic_tool_choice,
    to_anthropic_tools,
)

CALL = ToolCall(id="t1", name="get_policy", arguments={"policy_id": "P-1001"})


def test_plain_messages_stay_plain() -> None:
    assert to_anthropic_messages([Message(role="user", content="hi")]) == [
        {"role": "user", "content": "hi"}
    ]


def test_assistant_tool_calls_become_tool_use_blocks() -> None:
    out = to_anthropic_messages(
        [Message(role="assistant", content="Checking.", tool_calls=(CALL,))]
    )
    assert out == [
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Checking."},
                {
                    "type": "tool_use",
                    "id": "t1",
                    "name": "get_policy",
                    "input": {"policy_id": "P-1001"},
                },
            ],
        }
    ]


def test_tool_results_come_first_in_a_user_message() -> None:
    message = Message(
        role="user",
        content="Now submit.",
        tool_results=(ToolResult(tool_call_id="t1", content="boom", is_error=True),),
    )
    assert to_anthropic_messages([message])[0]["content"] == [
        {"type": "tool_result", "tool_use_id": "t1", "content": "boom", "is_error": True},
        {"type": "text", "text": "Now submit."},
    ]


def test_tools_and_tool_choice() -> None:
    spec = ToolSpec(name="get_policy", description="Look up.", input_schema={"type": "object"})
    assert to_anthropic_tools([spec]) == [
        {"name": "get_policy", "description": "Look up.", "input_schema": {"type": "object"}}
    ]
    assert to_anthropic_tool_choice(None) is None
    assert to_anthropic_tool_choice("get_policy") == {"type": "tool", "name": "get_policy"}


def test_reply_blocks_are_split_into_text_and_tool_calls() -> None:
    blocks = [
        SimpleNamespace(type="text", text="Let me check. "),
        SimpleNamespace(type="tool_use", id="t1", name="get_policy", input={"policy_id": "P-1001"}),
        SimpleNamespace(type="thinking", thinking="..."),
    ]
    assert from_anthropic_content(blocks) == ("Let me check. ", (CALL,))
