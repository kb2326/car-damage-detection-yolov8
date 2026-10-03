"""Requests, responses and errors of the LLM gateway."""

from __future__ import annotations

from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from claimlens.domain import Frozen


class ToolSpec(Frozen):
    name: str
    description: str
    input_schema: dict[str, Any]


class ToolCall(Frozen):
    id: str
    name: str
    arguments: dict[str, Any]


class ToolResult(Frozen):
    tool_call_id: str
    content: str
    is_error: bool = False


class Message(Frozen):
    role: Literal["user", "assistant"]
    content: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    tool_results: tuple[ToolResult, ...] = ()

    @model_validator(mode="after")
    def _not_empty(self) -> Self:
        if not (self.content or self.tool_calls or self.tool_results):
            raise ValueError("empty message: give content, tool_calls or tool_results")
        return self


class LLMRequest(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    messages: list[Message] = Field(min_length=1)
    tier: str
    system: str = ""
    max_tokens: int | None = None
    output_schema: type[BaseModel] | None = None
    claim_id: str | None = None
    prompt_id: str | None = None
    tools: tuple[ToolSpec, ...] = ()
    tool_choice: str | None = None

    @model_validator(mode="after")
    def _tools_ok(self) -> Self:
        if self.tools and self.output_schema is not None:
            raise ValueError("tools and output_schema cannot be combined")
        if self.tool_choice is not None and self.tool_choice not in {t.name for t in self.tools}:
            raise ValueError(f"tool_choice {self.tool_choice!r} is not one of the tools")
        return self


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
    tool_calls: tuple[ToolCall, ...] = ()


class LLMError(Exception):
    """Base class for gateway errors; messages never contain secrets."""


class BudgetExceeded(LLMError):  # noqa: N818 - spec name
    pass


class LLMUnavailable(LLMError):  # noqa: N818 - spec name
    pass


class InvalidModelOutput(LLMError):  # noqa: N818 - spec name
    pass
