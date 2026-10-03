"""Intake sessions: start, reply, pause and resume across a restart, and the hand-over."""

import sqlite3
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from langgraph.checkpoint.sqlite import SqliteSaver

from claimlens.agent.chat_model import GatewayChatModel
from claimlens.agent.evidence import render_evidence
from claimlens.events.projection import fold
from claimlens.events.store import SQLiteEventStore
from claimlens.intake_agent.graph import ASK, FINISH, PHOTO, IntakeState, build_intake_graph
from claimlens.intake_agent.session import IntakeSessions, pipeline_submitter, transcript_sha256
from claimlens.llm.budget import Budget
from claimlens.llm.cache import ResponseCache
from claimlens.llm.gateway import Gateway
from claimlens.llm.log import LLMCall
from claimlens.llm.provider import FakeProvider, ProviderReply
from tests.fakes import FakeDetector, make_test_deps
from tests.unit.test_intake_graph import CONFIG, LLM, TODAY, call, lookup, photo, record_all

SCRIPT: list[ProviderReply | Exception] = [
    call(ASK, {"message": "Your policy number?"}, "a1"),
    record_all(),
    call(PHOTO, {"kind": "overview", "message": "Whole car"}, "a2"),
    call(PHOTO, {"kind": "damage_closeup", "message": "Close-up"}, "a3"),
    call(PHOTO, {"kind": "plate", "message": "Plate"}, "a4"),
    call(FINISH, {"summary": "Bollard."}, "a5"),
]


def _sessions(
    tmp_path: Path,
    fake: FakeProvider,
    submit: Callable[[IntakeState], str],
    log: list[LLMCall] | None = None,
) -> tuple[IntakeSessions, sqlite3.Connection]:
    gateway = Gateway(
        LLM,
        fake,
        Budget(tmp_path / "b.sqlite", LLM.limits, today=lambda: TODAY),
        ResponseCache(tmp_path / "c.sqlite"),
        (log.append if log is not None else lambda c: None),
        sleep=lambda s: None,
    )
    model = GatewayChatModel(gateway=gateway, tier="fast", max_tokens=400)
    conn = sqlite3.connect(str(tmp_path / "intake.sqlite"), check_same_thread=False)
    graph = build_intake_graph(
        model, lookup, CONFIG, submit, lambda: TODAY, checkpointer=SqliteSaver(conn)
    )
    return IntakeSessions(graph, system="system prompt"), conn


def _talk(sessions: IntakeSessions, tmp_path: Path, session_id: str) -> Any:
    sessions.reply(session_id, "P-1001, I reversed into a bollard yesterday")
    for kind in ("overview", "closeup", "plate"):
        turn = sessions.reply(session_id, "here", Path(photo(tmp_path / f"{kind}.png")))
    return turn


def test_a_session_runs_from_start_to_claim(tmp_path: Path) -> None:
    sessions, conn = _sessions(tmp_path, FakeProvider(list(SCRIPT)), lambda s: "claim-1")
    first = sessions.start()
    assert first.message == "Your policy number?"
    assert first.claim_id is None
    last = _talk(sessions, tmp_path, first.session_id)
    assert last.claim_id == "claim-1"
    assert "claim-1" in last.message
    assert sessions.pending(first.session_id) is None
    conn.close()


def test_a_paused_session_resumes_after_a_restart(tmp_path: Path) -> None:
    fake = FakeProvider(list(SCRIPT))
    sessions, conn = _sessions(tmp_path, fake, lambda s: "claim-2")
    sid = sessions.start().session_id
    sessions.reply(sid, "P-1001, I reversed into a bollard yesterday")
    conn.close()  # the process stops while the agent waits for the overview photo

    again, conn2 = _sessions(tmp_path, fake, lambda s: "claim-2")
    waiting = again.pending(sid)
    assert waiting is not None
    assert waiting.photo_kind == "overview"
    assert again.open_sessions() == [sid]
    turn = again.reply(sid, "here", Path(photo(tmp_path / "o.png")))
    assert turn.photo_kind == "damage_closeup"
    again.reply(sid, "here", Path(photo(tmp_path / "c.png")))
    assert again.reply(sid, "here", Path(photo(tmp_path / "p.png"))).claim_id == "claim-2"
    assert again.open_sessions() == []
    conn2.close()


def test_replying_to_a_finished_or_unknown_session_is_an_error(tmp_path: Path) -> None:
    sessions, conn = _sessions(tmp_path, FakeProvider(list(SCRIPT)), lambda s: "claim-3")
    with pytest.raises(ValueError, match="no open intake session"):
        sessions.reply("nope", "hello")
    conn.close()


def test_each_session_has_its_own_spending_cap(tmp_path: Path) -> None:
    log: list[LLMCall] = []
    sessions, conn = _sessions(tmp_path, FakeProvider(list(SCRIPT)), lambda s: "c", log)
    sid = sessions.start().session_id
    assert log[0].claim_id == f"intake-{sid}"
    conn.close()


def test_the_transcript_hash_is_stable() -> None:
    exchange = [{"agent": "Policy?", "customer": "P-1001", "photo": ""}]
    assert transcript_sha256(exchange) == transcript_sha256([dict(exchange[0])])
    assert transcript_sha256(exchange) != transcript_sha256([])


def test_the_pipeline_submitter_files_a_decided_claim_with_the_facts(
    tmp_path: Path, store: SQLiteEventStore
) -> None:
    deps = make_test_deps(tmp_path, store, FakeDetector())
    submit = pipeline_submitter(lambda: deps, CONFIG, process=True)
    sessions, conn = _sessions(tmp_path, FakeProvider(list(SCRIPT)), submit)
    sid = sessions.start().session_id
    claim_id = _talk(sessions, tmp_path, sid).claim_id
    conn.close()
    events = store.load(__import__("uuid").UUID(claim_id))
    types = [e.type for e in events]
    assert "IntakeCompleted" in types
    assert types[-1] == "RouteDecided"
    state = fold(events)
    assert state.intake is not None
    assert state.intake.session_id == sid
    assert state.intake.photo_kinds == {"overview": "p1", "damage_closeup": "p2", "plate": "p3"}
    assert state.intake.turns == 4
    assert "- driving_for_work: no" in render_evidence(state).text
    assert state.description == "Reversed into a bollard."


def test_a_claim_with_no_usable_photo_is_still_filed(
    tmp_path: Path, store: SQLiteEventStore
) -> None:
    deps = make_test_deps(tmp_path, store, FakeDetector())
    submit = pipeline_submitter(lambda: deps, CONFIG, process=True)
    state: dict[str, Any] = {
        "facts": {"policy_id": "P-1001", "what_happened": "Hit a post."},
        "photos": {},
        "gaps": {"overview": "too dark after 2 retakes"},
        "retakes": {"overview": 3},
        "turns": 5,
        "transcript": [],
        "session_id": "s-empty",
    }
    claim_id = submit(state)  # type: ignore[arg-type]
    decision = fold(store.load(__import__("uuid").UUID(claim_id))).decision
    assert decision is not None
    assert decision.rule_id == "R3"


def test_build_intake_wires_the_real_policy_lookup(tmp_path: Path, store: SQLiteEventStore) -> None:
    from claimlens.intake_agent.graph import FACT
    from claimlens.intake_agent.session import build_intake

    root = Path(__file__).resolve().parents[2]
    fake = FakeProvider(
        [
            call(ASK, {"message": "Policy?"}, "a1"),
            call(FACT, {"name": "policy_id", "value": "P-1003"}, "a2"),
            call(ASK, {"message": "Thanks. What happened?"}, "a3"),
        ]
    )
    gateway = Gateway(
        LLM,
        fake,
        Budget(tmp_path / "b.sqlite", LLM.limits, today=lambda: TODAY),
        ResponseCache(tmp_path / "c.sqlite"),
        lambda c: None,
        sleep=lambda s: None,
    )
    deps = make_test_deps(tmp_path, store, FakeDetector())
    sessions = build_intake(
        root / "config", root, tmp_path / "intake.sqlite", lambda: deps, gateway=gateway
    )
    first = sessions.start()
    assert first.message == "Policy?"
    assert fake.calls[0]["system"].startswith("You are the ClaimLens claims assistant.")
    turn = sessions.reply(first.session_id, "P-1003")
    assert turn.message == "Thanks. What happened?"
    assert "Recorded policy_id = P-1003" in fake.calls[2]["messages"][-1].tool_results[0].content
