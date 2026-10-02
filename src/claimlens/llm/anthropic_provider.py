"""Claude through the official Anthropic SDK. The only module that imports `anthropic`."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import anthropic

from claimlens.llm.provider import (
    ProviderFatalError,
    ProviderReply,
    ProviderTransientError,
    billable_input_tokens,
)
from claimlens.llm.types import Message

_TRANSIENT: tuple[type[Exception], ...] = (
    anthropic.APITimeoutError,
    anthropic.APIConnectionError,
    anthropic.RateLimitError,
    anthropic.OverloadedError,
    anthropic.InternalServerError,
    anthropic.ServiceUnavailableError,
)
_FATAL: tuple[type[Exception], ...] = (
    anthropic.AuthenticationError,
    anthropic.PermissionDeniedError,
    anthropic.BadRequestError,
    anthropic.NotFoundError,
)


class AnthropicProvider:
    def __init__(self, api_key: str, timeout_s: float = 60.0) -> None:
        # The gateway owns retries, so the SDK must not retry on its own.
        self._client: Any = anthropic.Anthropic(api_key=api_key, timeout=timeout_s, max_retries=0)

    def complete(
        self, model: str, system: str, messages: Sequence[Message], max_tokens: int
    ) -> ProviderReply:
        system_blocks = (
            [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
            if system
            else []
        )
        try:
            response = self._client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=system_blocks,
                messages=[{"role": m.role, "content": m.content} for m in messages],
            )
        except _TRANSIENT as exc:
            raise ProviderTransientError(f"{type(exc).__name__}") from None
        except _FATAL as exc:
            raise ProviderFatalError(
                f"{type(exc).__name__}: {getattr(exc, 'status_code', '')}"
            ) from None
        except anthropic.APIError as exc:
            # Anything unclassified (413, 422, 409, validation of the response, ...) stops here.
            raise ProviderFatalError(f"{type(exc).__name__}") from None
        text = "".join(block.text for block in response.content if block.type == "text")
        return ProviderReply(
            text=text,
            input_tokens=billable_input_tokens(
                int(response.usage.input_tokens),
                getattr(response.usage, "cache_creation_input_tokens", None),
                getattr(response.usage, "cache_read_input_tokens", None),
            ),
            output_tokens=int(response.usage.output_tokens),
        )
