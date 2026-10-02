"""Requests, responses and errors of the LLM gateway."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from claimlens.domain import Frozen


class Message(Frozen):
    role: Literal["user", "assistant"]
    content: str


class LLMRequest(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    messages: list[Message] = Field(min_length=1)
    tier: str
    system: str = ""
    max_tokens: int | None = None
    output_schema: type[BaseModel] | None = None
    claim_id: str | None = None
    prompt_id: str | None = None


class LLMResponse(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    text: str
    parsed: BaseModel | None
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    cached: bool
    latency_ms: int
    attempts: int


class LLMError(Exception):
    """Base class for gateway errors; messages never contain secrets."""


class BudgetExceeded(LLMError):  # noqa: N818 - spec name
    pass


class LLMUnavailable(LLMError):  # noqa: N818 - spec name
    pass


class InvalidModelOutput(LLMError):  # noqa: N818 - spec name
    pass
