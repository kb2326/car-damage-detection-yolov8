"""The intake conversation as a LangGraph graph: agent -> act -> ask (pause) -> agent ... -> finish.

The graph pauses with `interrupt()` whenever the agent speaks to the customer, so a conversation
can stop for days and resume from a checkpoint. Facts and photos are checked in code; the agent
only collects. It never decides, prices or promises.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import interrupt

from claimlens.intake_agent.config import IntakeConfig
from claimlens.intake_agent.facts import FACTS, missing, validate_fact
from claimlens.intake_agent.photos import coach_photo

ASK, PHOTO, FINISH = "ask_customer", "request_photo", "finish_intake"
FACT, LOOKUP = "record_fact", "lookup_policy"
CUSTOMER_FACING = {ASK, PHOTO}
FALLBACK = "Could you tell me a little more, please?"
NEUTRAL = "Thank you, I've noted that. A claims handler will review everything and get back to you."
# The intake agent must never promise cover, payment or an outcome.
NO_PROMISES = re.compile(
    r"you(?:'re| are) covered|\bapproved\b|will be paid|we(?:'ll| will) pay|guarantee",
    re.IGNORECASE,
)
_TAG = re.compile(r"<\s*/?\s*customer_message[^>]*>", re.IGNORECASE)


def _tool(name: str, description: str, properties: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": list(properties),
            },
        },
    }


TOOLS: list[dict[str, Any]] = [
    _tool(
        FACT,
        "Record one fact the customer told you. Names: "
        + "; ".join(f"{k} ({v})" for k, v in FACTS.items()),
        {"name": {"type": "string"}, "value": {"type": "string"}},
    ),
    _tool(LOOKUP, "Check that a policy number exists.", {"policy_id": {"type": "string"}}),
    _tool(
        ASK,
        "Say something to the customer and wait for the reply.",
        {"message": {"type": "string"}},
    ),
    _tool(
        PHOTO,
        "Ask the customer for one photo and wait for it. Kinds: overview (the whole car, "
        "damaged side), damage_closeup, plate.",
        {"kind": {"type": "string"}, "message": {"type": "string"}},
    ),
    _tool(
        FINISH, "Finish when every fact and photo is collected.", {"summary": {"type": "string"}}
    ),
]


class IntakeState(MessagesState):
    facts: dict[str, str]
    photos: dict[str, str]  # kind -> photo path
    gaps: dict[str, str]  # kind -> why no usable photo
    retakes: dict[str, int]
    turns: int
    steps: int
    pending: dict[str, Any] | None  # the customer-facing or finish call waiting for an answer
    outgoing: str  # the last message for the customer
    transcript: list[dict[str, str]]  # customer-facing exchange, for the transcript hash
    claim_id: str | None
    forced: bool
    session_id: str


def initial_state(system: str, kickoff: str, session_id: str = "") -> dict[str, Any]:
    from langchain_core.messages import HumanMessage, SystemMessage

    return {
        "messages": [SystemMessage(system), HumanMessage(kickoff)],
        "facts": {},
        "photos": {},
        "gaps": {},
        "retakes": {},
        "turns": 0,
        "steps": 0,
        "pending": None,
        "outgoing": "",
        "transcript": [],
        "claim_id": None,
        "forced": False,
        "session_id": session_id,
    }


def _speak(text: str) -> str:
    return NEUTRAL if NO_PROMISES.search(text) else text


def build_intake_graph(
    model: Any,
    lookup_policy: Callable[[str], str],
    config: IntakeConfig,
    submit: Callable[[IntakeState], str],
    today: Callable[[], date],
    checkpointer: Any = None,
) -> Any:
    settings = config  # LangGraph passes the run's settings to a node parameter named `config`

    def agent(state: IntakeState, config: RunnableConfig) -> dict[str, Any]:
        n = f"{state['turns']}-{state['steps']}"
        if state["steps"] >= settings.max_steps_per_turn:
            fallback = {"name": ASK, "args": {"message": FALLBACK}, "id": f"fallback-{n}"}
            return {"messages": [AIMessage(content="", tool_calls=[fallback])]}
        # Each session spends against its own cap (the gateway's per-claim cap).
        session = config.get("configurable", {}).get("thread_id")
        session_model = model.model_copy(update={"claim_id": f"intake-{session}"})
        reply = session_model.bind_tools(TOOLS).invoke(state["messages"])
        if not reply.tool_calls:  # plain text is treated as a question to the customer
            text = str(reply.content).strip() or FALLBACK
            call = {"name": ASK, "args": {"message": text}, "id": f"text-{n}"}
            reply = AIMessage(content="", tool_calls=[call])
        return {"messages": [reply], "steps": state["steps"] + 1}

    def act(state: IntakeState) -> dict[str, Any]:
        """Run fact and lookup calls; keep at most one customer-facing or finish call pending."""
        reply = state["messages"][-1]
        assert isinstance(reply, AIMessage)
        facts = dict(state["facts"])
        answers: list[ToolMessage] = []
        pending: dict[str, Any] | None = None
        for call in reply.tool_calls:
            name, args, call_id = call["name"], call["args"], str(call["id"])
            if name in CUSTOMER_FACING or name == FINISH:
                if pending is None:
                    pending = {"name": name, "args": dict(args), "id": call_id}
                else:
                    answers.append(_error(call_id, name, "one question at a time; not sent"))
                continue
            try:
                if name == FACT:
                    value = validate_fact(str(args.get("name")), str(args.get("value")), today())
                    if args.get("name") == "policy_id" and '"found":true' not in _compact(
                        lookup_policy(value)
                    ):
                        raise ValueError(f"policy {value} was not found; ask the customer to check")
                    facts[str(args["name"])] = value
                    answers.append(
                        ToolMessage(
                            f"Recorded {args['name']} = {value}", tool_call_id=call_id, name=name
                        )
                    )
                elif name == LOOKUP:
                    result = lookup_policy(str(args.get("policy_id", "")))
                    answers.append(ToolMessage(result, tool_call_id=call_id, name=name))
                else:
                    answers.append(_error(call_id, name, f"unknown tool {name}"))
            except Exception as exc:
                answers.append(_error(call_id, name, str(exc)))
        return {"messages": answers, "facts": facts, "pending": pending}

    def after_act(state: IntakeState) -> str:
        pending = state["pending"]
        if pending is None:
            return "agent"
        return "finish" if pending["name"] == FINISH else "ask"

    def ask(state: IntakeState) -> dict[str, Any]:
        call = state["pending"]
        assert call is not None
        if state["turns"] >= config.max_turns:
            return {"forced": True}
        kind = call["args"].get("kind") if call["name"] == PHOTO else None
        if kind is not None and kind not in config.photo_kinds:
            kinds = ", ".join(config.photo_kinds)
            return {
                "messages": [_error(call["id"], call["name"], f"unknown photo kind; use {kinds}")],
                "pending": None,
            }
        message = _speak(str(call["args"].get("message", "")).strip() or FALLBACK)
        reply = interrupt({"message": message, "photo_kind": kind})
        text = _TAG.sub("", str(reply.get("text", ""))).strip()
        photo = reply.get("photo")
        photos, gaps, retakes = dict(state["photos"]), dict(state["gaps"]), dict(state["retakes"])
        lines = [f"<customer_message>{text}</customer_message>"]
        if photo:
            target = kind or f"extra_{len(photos) + 1}"
            check = coach_photo(Path(str(photo)), config)
            if check.ok:
                photos[target] = str(photo)
                lines.append(f"Photo received for {target}: accepted.")
            elif kind is None:
                lines.append(
                    f"A photo was sent without a request; it was not usable ({check.reason})."
                )
            else:
                retakes[kind] = retakes.get(kind, 0) + 1
                if retakes[kind] > config.max_retakes:
                    gaps[kind] = f"{check.reason} after {config.max_retakes} retakes"
                    lines.append(
                        f"Photo for {kind}: rejected, {check.reason}. No retakes left: "
                        "move on without it."
                    )
                else:
                    lines.append(
                        f"Photo for {kind}: rejected, {check.reason} "
                        f"(retake {retakes[kind]} of {config.max_retakes}). Ask for a retake."
                    )
        elif kind is not None:
            lines.append(f"No photo was attached for {kind}.")
        exchange = {"agent": message, "customer": text, "photo": str(photo or "")}
        return {
            "messages": [ToolMessage("\n".join(lines), tool_call_id=call["id"], name=call["name"])],
            "photos": photos,
            "gaps": gaps,
            "retakes": retakes,
            "turns": state["turns"] + 1,
            "steps": 0,
            "pending": None,
            "outgoing": message,
            "transcript": [*state["transcript"], exchange],
        }

    def after_ask(state: IntakeState) -> str:
        return "finish" if state["forced"] else "agent"

    def finish(state: IntakeState) -> dict[str, Any]:
        call = state["pending"]
        assert call is not None
        need = missing(state["facts"], state["photos"], state["gaps"], config.photo_kinds)
        if need and not state["forced"]:
            text = "Cannot finish yet. Still missing: " + ", ".join(need)
            return {"messages": [_error(call["id"], call["name"], text)], "pending": None}
        claim_id = submit(state)
        outgoing = (
            f"Thank you, that's everything I need. Your claim number is {claim_id}. "
            "A claims handler will review it and be in touch."
        )
        done = ToolMessage(
            f"Submitted as claim {claim_id}.", tool_call_id=call["id"], name=call["name"]
        )
        return {"messages": [done], "pending": None, "claim_id": claim_id, "outgoing": outgoing}

    def after_finish(state: IntakeState) -> str:
        return END if state["claim_id"] else "agent"

    graph = StateGraph(IntakeState)
    graph.add_node("agent", agent)
    graph.add_node("act", act)
    graph.add_node("ask", ask)
    graph.add_node("finish", finish)
    graph.add_edge(START, "agent")
    graph.add_edge("agent", "act")
    graph.add_conditional_edges(
        "act", after_act, {"agent": "agent", "ask": "ask", "finish": "finish"}
    )
    graph.add_conditional_edges("ask", after_ask, {"agent": "agent", "finish": "finish"})
    graph.add_conditional_edges("finish", after_finish, {END: END, "agent": "agent"})
    return graph.compile(checkpointer=checkpointer)


def _error(call_id: str, name: str, text: str) -> ToolMessage:
    return ToolMessage(text, tool_call_id=call_id, name=name, status="error")


def _compact(text: str) -> str:
    try:
        return json.dumps(json.loads(text), separators=(",", ":"))
    except ValueError:
        return text.replace(" ", "")
