"""LangChain chat model backed by the ClaimLens LLM gateway (caps, cache and log apply)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.utils.function_calling import convert_to_openai_tool

from claimlens.llm.types import LLMRequest, Message, ToolCall, ToolResult, ToolSpec


def to_gateway_messages(messages: Sequence[BaseMessage]) -> tuple[str, list[Message]]:
    """Split out the system text and convert the rest; consecutive tool results are merged."""
    system: list[str] = []
    out: list[Message] = []
    for m in messages:
        if isinstance(m, SystemMessage):
            system.append(str(m.content))
        elif isinstance(m, HumanMessage):
            out.append(Message(role="user", content=str(m.content)))
        elif isinstance(m, AIMessage):
            calls = tuple(
                ToolCall(id=str(c["id"]), name=c["name"], arguments=dict(c["args"]))
                for c in m.tool_calls
            )
            out.append(Message(role="assistant", content=str(m.content), tool_calls=calls))
        elif isinstance(m, ToolMessage):
            result = ToolResult(
                tool_call_id=m.tool_call_id, content=str(m.content), is_error=m.status == "error"
            )
            last = out[-1] if out else None
            if last is not None and last.role == "user" and last.tool_results and not last.content:
                out[-1] = last.model_copy(update={"tool_results": (*last.tool_results, result)})
            else:
                out.append(Message(role="user", tool_results=(result,)))
        else:
            raise TypeError(f"unsupported message type {type(m).__name__}")
    return "\n\n".join(system), out


class GatewayChatModel(BaseChatModel):
    gateway: Any
    tier: str
    claim_id: str | None = None
    prompt_id: str | None = None
    max_tokens: int | None = None
    specs: tuple[ToolSpec, ...] = ()
    forced: str | None = None

    @property
    def _llm_type(self) -> str:
        return "claimlens-gateway"

    def bind_tools(
        self, tools: Sequence[Any], *, tool_choice: str | None = None, **kwargs: Any
    ) -> Any:
        specs = []
        for tool in tools:
            fn = convert_to_openai_tool(tool)["function"]
            specs.append(
                ToolSpec(
                    name=fn["name"],
                    description=fn.get("description", ""),
                    input_schema=fn.get("parameters") or {"type": "object", "properties": {}},
                )
            )
        return self.model_copy(update={"specs": tuple(specs), "forced": tool_choice})

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        system, converted = to_gateway_messages(messages)
        response = self.gateway.generate(
            LLMRequest(
                messages=converted,
                tier=self.tier,
                system=system,
                max_tokens=self.max_tokens,
                tools=self.specs,
                tool_choice=self.forced,
                claim_id=self.claim_id,
                prompt_id=self.prompt_id,
            )
        )
        message = AIMessage(
            content=response.text,
            tool_calls=[
                {"name": c.name, "args": c.arguments, "id": c.id, "type": "tool_call"}
                for c in response.tool_calls
            ],
            usage_metadata={
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "total_tokens": response.input_tokens + response.output_tokens,
            },
            response_metadata={
                "model_name": response.model,
                "cost_usd": response.cost_usd,
                "cached": response.cached,
            },
        )
        return ChatResult(generations=[ChatGeneration(message=message)])
