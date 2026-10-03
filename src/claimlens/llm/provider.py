"""Provider protocol and a scripted fake for tests (no network, no key)."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from claimlens.llm.types import Message, ToolCall, ToolSpec


@dataclass(frozen=True)
class ProviderReply:
    text: str
    input_tokens: int
    output_tokens: int
    tool_calls: tuple[ToolCall, ...] = ()


def billable_input_tokens(
    input_tokens: int, cache_creation: int | None, cache_read: int | None
) -> int:
    """Input-token equivalents: cache writes cost 1.25x and cache reads 0.1x the input price."""
    return (
        input_tokens + math.ceil(1.25 * (cache_creation or 0)) + math.ceil(0.1 * (cache_read or 0))
    )


class ProviderTransientError(Exception):
    """Worth retrying: timeout, connection, rate limit, overload, server error."""


class ProviderFatalError(Exception):
    """Not worth retrying or falling back: authentication, permission, bad request."""


class Provider(Protocol):
    def complete(
        self,
        model: str,
        system: str,
        messages: Sequence[Message],
        max_tokens: int,
        tools: Sequence[ToolSpec] = (),
        tool_choice: str | None = None,
    ) -> ProviderReply: ...


@dataclass
class FakeProvider:
    script: list[ProviderReply | Exception]
    calls: list[dict[str, Any]] = field(default_factory=list)

    def complete(
        self,
        model: str,
        system: str,
        messages: Sequence[Message],
        max_tokens: int,
        tools: Sequence[ToolSpec] = (),
        tool_choice: str | None = None,
    ) -> ProviderReply:
        self.calls.append(
            {
                "model": model,
                "system": system,
                "messages": list(messages),
                "max_tokens": max_tokens,
                "tools": tuple(tools),
                "tool_choice": tool_choice,
            }
        )
        assert self.script, "FakeProvider script ran out"
        step = self.script.pop(0)
        if isinstance(step, Exception):
            raise step
        return step
