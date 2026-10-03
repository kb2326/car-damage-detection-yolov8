"""Pure mapping between gateway types and Anthropic's content-block format (no SDK import)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from claimlens.llm.types import Message, ToolCall, ToolSpec


def to_anthropic_messages(messages: Sequence[Message]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for m in messages:
        if not m.tool_calls and not m.tool_results:
            out.append({"role": m.role, "content": m.content})
            continue
        # A tool_result must come before any text in the same user message.
        blocks: list[dict[str, Any]] = [
            {
                "type": "tool_result",
                "tool_use_id": r.tool_call_id,
                "content": r.content,
                "is_error": r.is_error,
            }
            for r in m.tool_results
        ]
        if m.content:
            blocks.append({"type": "text", "text": m.content})
        blocks.extend(
            {"type": "tool_use", "id": c.id, "name": c.name, "input": c.arguments}
            for c in m.tool_calls
        )
        out.append({"role": m.role, "content": blocks})
    return out


def to_anthropic_tools(tools: Sequence[ToolSpec]) -> list[dict[str, Any]]:
    return [
        {"name": t.name, "description": t.description, "input_schema": t.input_schema}
        for t in tools
    ]


def to_anthropic_tool_choice(name: str | None) -> dict[str, Any] | None:
    return None if name is None else {"type": "tool", "name": name}


def from_anthropic_content(blocks: Sequence[Any]) -> tuple[str, tuple[ToolCall, ...]]:
    text = "".join(b.text for b in blocks if b.type == "text")
    calls = tuple(
        ToolCall(id=b.id, name=b.name, arguments=dict(b.input))
        for b in blocks
        if b.type == "tool_use"
    )
    return text, calls
