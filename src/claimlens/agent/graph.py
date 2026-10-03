"""The triage agent as a LangGraph graph: agent -> tools -> agent ... -> finalize."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from claimlens.agent.config import AgentConfig
from claimlens.agent.recommendation import SUBMIT, SUBMIT_TOOL

Check = Callable[[Mapping[str, Any]], list[str]]
NUDGE = "Call the submit_recommendation tool now with your recommendation. Do not reply in text."


class AgentFailed(Exception):  # noqa: N818 - reads as an outcome, like the gateway's errors
    """The agent could not produce a trustworthy recommendation; the claim goes to a person."""


class TriageState(MessagesState):
    steps: int
    repairs: int
    deadline: float
    recommendation: dict[str, Any] | None


def build_graph(
    model: Any,
    tools: Sequence[BaseTool],
    check: Check,
    config: AgentConfig,
    clock: Callable[[], float],
) -> Any:
    def agent(state: TriageState) -> dict[str, Any]:
        if state["steps"] >= config.max_steps:
            raise AgentFailed(f"step limit of {config.max_steps} model calls reached")
        if clock() > state["deadline"]:
            raise AgentFailed(f"time limit of {config.max_seconds:g} s reached")
        last = state["steps"] == config.max_steps - 1
        bound = model.bind_tools([*tools, SUBMIT_TOOL], tool_choice=SUBMIT if last else None)
        return {"messages": [bound.invoke(state["messages"])], "steps": state["steps"] + 1}

    def after_agent(state: TriageState) -> str:
        reply = state["messages"][-1]
        names = [c["name"] for c in reply.tool_calls] if isinstance(reply, AIMessage) else []
        if SUBMIT in names:
            return "finalize"
        return "tools" if names else "nudge"

    def nudge(state: TriageState) -> dict[str, Any]:
        if state["repairs"] >= config.max_repairs:
            raise AgentFailed("the model did not submit a recommendation")
        return {"messages": [HumanMessage(NUDGE)], "repairs": state["repairs"] + 1}

    def finalize(state: TriageState) -> dict[str, Any]:
        reply = state["messages"][-1]
        assert isinstance(reply, AIMessage)
        submit = next(c for c in reply.tool_calls if c["name"] == SUBMIT)
        # Every tool call needs a result, or the next model call is refused.
        skipped = [
            ToolMessage(
                content="Not run: submit_recommendation was called in the same turn.",
                tool_call_id=c["id"],
            )
            for c in reply.tool_calls
            if c["id"] != submit["id"]
        ]
        problems = check(submit["args"])
        tool_failed = any(
            isinstance(m, ToolMessage) and m.status == "error" and m.name != SUBMIT
            for m in state["messages"]
        )
        if tool_failed and submit["args"].get("route_suggestion") == "FAST_TRACK":
            # A failed lookup means something went unchecked, so a person must look.
            problems.append(
                "a tool failed during this review, so FAST_TRACK is not allowed; "
                "recommend ADJUSTER_REVIEW and say what could not be checked"
            )
        if not problems:
            accepted = ToolMessage(content="Accepted.", tool_call_id=submit["id"], name=SUBMIT)
            return {"messages": [*skipped, accepted], "recommendation": dict(submit["args"])}
        if state["repairs"] >= config.max_repairs:
            raise AgentFailed("recommendation rejected: " + "; ".join(problems))
        rejected = ToolMessage(
            content="Rejected: " + "; ".join(problems) + ". Fix this and submit again.",
            tool_call_id=submit["id"],
            name=SUBMIT,
            status="error",
        )
        return {"messages": [*skipped, rejected], "repairs": state["repairs"] + 1}

    def after_finalize(state: TriageState) -> str:
        return END if state["recommendation"] is not None else "agent"

    graph = StateGraph(TriageState)
    graph.add_node("agent", agent)
    graph.add_node("tools", ToolNode(list(tools), handle_tool_errors=True))
    graph.add_node("nudge", nudge)
    graph.add_node("finalize", finalize)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent", after_agent, {"tools": "tools", "nudge": "nudge", "finalize": "finalize"}
    )
    graph.add_edge("tools", "agent")
    graph.add_edge("nudge", "agent")
    graph.add_conditional_edges("finalize", after_finalize, {END: END, "agent": "agent"})
    return graph.compile()


def run_graph(
    graph: Any, system: str, evidence: str, config: AgentConfig, clock: Callable[[], float]
) -> dict[str, Any]:
    start: TriageState = {
        "messages": [SystemMessage(system), HumanMessage(evidence)],
        "steps": 0,
        "repairs": 0,
        "deadline": clock() + config.max_seconds,
        "recommendation": None,
    }
    try:
        # Backstop only: each step is at most agent + tools/nudge/finalize.
        out = graph.invoke(start, config={"recursion_limit": 3 * config.max_steps + 6})
    except GraphRecursionError:
        raise AgentFailed("graph recursion limit reached") from None
    recommendation: dict[str, Any] = out["recommendation"]
    return recommendation
