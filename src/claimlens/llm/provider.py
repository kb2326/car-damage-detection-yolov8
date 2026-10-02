"""Provider protocol and a scripted fake for tests (no network, no key)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from claimlens.llm.types import Message


@dataclass(frozen=True)
class ProviderReply:
    text: str
    input_tokens: int
    output_tokens: int


class ProviderTransientError(Exception):
    """Worth retrying: timeout, connection, rate limit, overload, server error."""


class ProviderFatalError(Exception):
    """Not worth retrying or falling back: authentication, permission, bad request."""


class Provider(Protocol):
    def complete(
        self, model: str, system: str, messages: Sequence[Message], max_tokens: int
    ) -> ProviderReply: ...


@dataclass
class FakeProvider:
    script: list[ProviderReply | Exception]
    calls: list[dict[str, Any]] = field(default_factory=list)

    def complete(
        self, model: str, system: str, messages: Sequence[Message], max_tokens: int
    ) -> ProviderReply:
        self.calls.append(
            {"model": model, "system": system, "messages": list(messages), "max_tokens": max_tokens}
        )
        assert self.script, "FakeProvider script ran out"
        step = self.script.pop(0)
        if isinstance(step, Exception):
            raise step
        return step
