from datetime import date
from pathlib import Path
from typing import Any

import pytest
from langchain_core.tools import StructuredTool, ToolException

from claimlens.agent.chat_model import GatewayChatModel
from claimlens.agent.config import load_agent_config
from claimlens.agent.graph import AgentFailed, build_graph, run_graph
from claimlens.agent.recommendation import SUBMIT
from claimlens.llm.budget import Budget
from claimlens.llm.cache import ResponseCache
from claimlens.llm.config import load_llm_config
from claimlens.llm.gateway import Gateway
from claimlens.llm.provider import FakeProvider, ProviderReply
from claimlens.llm.types import ToolCall

ROOT = Path(__file__).resolve().parents[2]
LLM = load_llm_config(ROOT / "config" / "llm.toml")
CONFIG = load_agent_config(ROOT / "config" / "agent.toml")
ANSWER = {"route_suggestion": "FAST_TRACK", "confidence": "high", "rationale": "Consistent."}


def _call(name: str, args: dict[str, Any], id_: str) -> ProviderReply:
    return ProviderReply("", 500, 50, tool_calls=(ToolCall(id=id_, name=name, arguments=args),))


def _search(query: str) -> str:
    if query == "boom":
        raise ToolException("index offline")
    return '{"clauses": [{"clause_id": "STD-8.5"}]}'


SEARCH = StructuredTool.from_function(
    func=_search,
    name="search_policy_clauses",
    description="Search the policy wording.",
    args_schema={
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    },
)


def _run(
    tmp_path: Path,
    script: list[ProviderReply | Exception],
    check: Any = lambda args: [],
    clock: Any = lambda: 0.0,
) -> tuple[dict[str, Any], FakeProvider]:
    fake = FakeProvider(script)
    gateway = Gateway(
        LLM,
        fake,
        Budget(tmp_path / "b.sqlite", LLM.limits, today=lambda: date(2026, 10, 2)),
        ResponseCache(tmp_path / "c.sqlite"),
        lambda call: None,
        sleep=lambda s: None,
    )
    model = GatewayChatModel(
        gateway=gateway, tier="strong", claim_id="c1", max_tokens=CONFIG.max_tokens
    )
    graph = build_graph(model, [SEARCH], check, CONFIG, clock)
    return run_graph(graph, "system", "evidence", CONFIG, clock), fake


def test_tool_call_then_submit(tmp_path: Path) -> None:
    out, fake = _run(
        tmp_path,
        [_call("search_policy_clauses", {"query": "delivery"}, "t1"), _call(SUBMIT, ANSWER, "t2")],
    )
    assert out == ANSWER
    assert len(fake.calls) == 2
    second = fake.calls[1]["messages"]
    assert second[-1].tool_results[0].content == '{"clauses": [{"clause_id": "STD-8.5"}]}'
    assert {t.name for t in fake.calls[0]["tools"]} == {"search_policy_clauses", SUBMIT}


def test_a_tool_error_goes_back_to_the_model(tmp_path: Path) -> None:
    escalate = {**ANSWER, "route_suggestion": "ADJUSTER_REVIEW", "rationale": "Search failed."}
    out, fake = _run(
        tmp_path,
        [_call("search_policy_clauses", {"query": "boom"}, "t1"), _call(SUBMIT, escalate, "t2")],
    )
    assert out == escalate
    result = fake.calls[1]["messages"][-1].tool_results[0]
    assert result.is_error
    assert "index offline" in result.content


def test_plain_text_gets_one_nudge(tmp_path: Path) -> None:
    out, fake = _run(
        tmp_path, [ProviderReply("I think fast-track.", 500, 20), _call(SUBMIT, ANSWER, "t1")]
    )
    assert out == ANSWER
    assert "submit_recommendation" in fake.calls[1]["messages"][-1].content


def test_two_plain_text_replies_fail(tmp_path: Path) -> None:
    with pytest.raises(AgentFailed, match="did not submit"):
        _run(tmp_path, [ProviderReply("a", 1, 1), ProviderReply("b", 1, 1)])


def test_a_rejected_answer_can_be_repaired_once(tmp_path: Path) -> None:
    verdicts = [["clause STD-99.9 does not exist"], []]
    out, fake = _run(
        tmp_path,
        [_call(SUBMIT, ANSWER, "t1"), _call(SUBMIT, ANSWER, "t2")],
        check=lambda args: verdicts.pop(0),
    )
    assert out == ANSWER
    rejected = fake.calls[1]["messages"][-1].tool_results[0]
    assert rejected.is_error
    assert "STD-99.9" in rejected.content


def test_a_second_rejection_fails(tmp_path: Path) -> None:
    with pytest.raises(AgentFailed, match="rejected"):
        _run(
            tmp_path,
            [_call(SUBMIT, ANSWER, "t1"), _call(SUBMIT, ANSWER, "t2")],
            check=lambda args: ["clause STD-99.9 does not exist"],
        )


def test_submit_with_another_tool_in_the_same_turn_answers_both(tmp_path: Path) -> None:
    both = ProviderReply(
        "",
        500,
        50,
        tool_calls=(
            ToolCall(id="t1", name="search_policy_clauses", arguments={"query": "x"}),
            ToolCall(id="t2", name=SUBMIT, arguments=ANSWER),
        ),
    )
    verdicts = [["evidence E7 is not in the evidence list"], []]
    out, fake = _run(tmp_path, [both, _call(SUBMIT, ANSWER, "t3")], check=lambda a: verdicts.pop(0))
    assert out == ANSWER
    answered = {r.tool_call_id for r in fake.calls[1]["messages"][-1].tool_results}
    assert answered == {"t1", "t2"}


def test_the_last_step_forces_submit(tmp_path: Path) -> None:
    searches: list[ProviderReply | Exception] = [
        _call("search_policy_clauses", {"query": f"q{i}"}, f"t{i}")
        for i in range(CONFIG.max_steps - 1)
    ]
    out, fake = _run(tmp_path, [*searches, _call(SUBMIT, ANSWER, "last")])
    assert out == ANSWER
    assert [c["tool_choice"] for c in fake.calls] == [None] * (CONFIG.max_steps - 1) + [SUBMIT]


def test_step_limit(tmp_path: Path) -> None:
    searches: list[ProviderReply | Exception] = [
        _call("search_policy_clauses", {"query": f"q{i}"}, f"t{i}") for i in range(CONFIG.max_steps)
    ]
    with pytest.raises(AgentFailed, match="step limit"):
        _run(tmp_path, searches)


def test_time_limit(tmp_path: Path) -> None:
    ticks = iter([0.0, 0.0, 1000.0, 1000.0, 1000.0])
    with pytest.raises(AgentFailed, match="time limit"):
        _run(
            tmp_path,
            [_call("search_policy_clauses", {"query": "x"}, "t1"), _call(SUBMIT, ANSWER, "t2")],
            clock=lambda: next(ticks),
        )


def test_fast_track_is_refused_after_a_tool_error(tmp_path: Path) -> None:
    fast = {**ANSWER, "route_suggestion": "FAST_TRACK"}
    escalate = {**ANSWER, "route_suggestion": "ADJUSTER_REVIEW", "rationale": "Search failed."}
    out, fake = _run(
        tmp_path,
        [
            _call("search_policy_clauses", {"query": "boom"}, "t1"),
            _call(SUBMIT, fast, "t2"),
            _call(SUBMIT, escalate, "t3"),
        ],
    )
    assert out == escalate
    rejected = fake.calls[2]["messages"][-1].tool_results[0]
    assert rejected.is_error
    assert "a tool failed" in rejected.content
