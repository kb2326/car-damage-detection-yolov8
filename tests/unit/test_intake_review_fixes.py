"""Tests for the M6a review findings."""

import json
import sqlite3
from pathlib import Path
from uuid import UUID

import pytest
from langgraph.checkpoint.sqlite import SqliteSaver

from claimlens.agent.chat_model import GatewayChatModel
from claimlens.events.projection import fold
from claimlens.events.store import SQLiteEventStore
from claimlens.intake_agent.commands import run_chat
from claimlens.intake_agent.graph import ASK, PHOTO, build_intake_graph, promises
from claimlens.intake_agent.photos import coach_photo
from claimlens.intake_agent.session import IntakeSessions, pipeline_submitter, policy_found_only
from claimlens.llm.budget import Budget
from claimlens.llm.cache import ResponseCache
from claimlens.llm.gateway import Gateway
from claimlens.llm.provider import FakeProvider, ProviderReply
from claimlens.llm.types import BudgetExceeded, LLMUnavailable
from tests.fakes import FakeDetector, make_test_deps
from tests.unit.test_intake_graph import (
    CONFIG,
    LLM,
    TODAY,
    Harness,
    asked,
    call,
    lookup,
    photo,
)


def _sessions(
    tmp_path: Path,
    script: list[ProviderReply | Exception],
    submit: object = None,
    stage: object = None,
) -> tuple[IntakeSessions, sqlite3.Connection]:
    gateway = Gateway(
        LLM,
        FakeProvider(script),
        Budget(tmp_path / "b.sqlite", LLM.limits, today=lambda: TODAY),
        ResponseCache(tmp_path / "c.sqlite"),
        lambda c: None,
        sleep=lambda s: None,
    )
    model = GatewayChatModel(gateway=gateway, tier="fast", max_tokens=400)
    conn = sqlite3.connect(str(tmp_path / "intake.sqlite"), check_same_thread=False)
    kwargs = {"stage": stage} if stage is not None else {}
    graph = build_intake_graph(
        model,
        lookup,
        CONFIG,
        submit or (lambda s: "claim-x"),  # type: ignore[arg-type]
        lambda: TODAY,
        checkpointer=SqliteSaver(conn),
        **kwargs,  # type: ignore[arg-type]
    )
    return IntakeSessions(graph, system="system"), conn


# 1. An error after a resume must not lose the session.
def test_a_session_that_errored_can_be_resumed(tmp_path: Path) -> None:
    script: list[ProviderReply | Exception] = [
        call(ASK, {"message": "Policy?"}, "a1"),
        LLMUnavailable("all LLM models are unavailable; route to a person"),
        call(ASK, {"message": "Thanks, what happened?"}, "a2"),
    ]
    sessions, conn = _sessions(tmp_path, script)
    sid = sessions.start().session_id
    with pytest.raises(LLMUnavailable):
        sessions.reply(sid, "P-1001")
    assert sid in sessions.open_sessions()
    turn = sessions.resume(sid)
    assert turn is not None
    assert turn.message == "Thanks, what happened?"
    conn.close()


def test_the_chat_survives_an_llm_outage(tmp_path: Path) -> None:
    script: list[ProviderReply | Exception] = [
        call(ASK, {"message": "Policy?"}, "a1"),
        LLMUnavailable("all LLM models are unavailable; route to a person"),
    ]
    sessions, conn = _sessions(tmp_path, script)
    lines = iter(["P-1001"])
    out: list[str] = []
    turn = run_chat(sessions, lambda: next(lines), out.append)
    conn.close()
    assert turn is not None
    assert any("saved" in line and "claimlens intake --session" in line for line in out)


# 4. Hitting the session's spending cap hands the claim to a person.
def test_the_spending_cap_hands_the_claim_over(tmp_path: Path) -> None:
    filed: list[dict[str, object]] = []
    script: list[ProviderReply | Exception] = [
        call(ASK, {"message": "Policy?"}, "a1"),
        BudgetExceeded("LLM cap of $0.10 reached for claim intake-x"),
    ]

    def submit(state: object) -> str:
        filed.append(dict(state))  # type: ignore[call-overload]
        return "c-9"

    sessions, conn = _sessions(tmp_path, script, submit=submit)
    sid = sessions.start().session_id
    turn = sessions.reply(sid, "P-1001")
    conn.close()
    assert turn.claim_id == "c-9"
    assert filed[0]["handover"] == "spending cap reached"
    assert "pass what we have to a claims handler" in turn.message


# 2. Photos are kept when accepted, so moving the original later does not matter.
def test_an_accepted_photo_is_kept_before_the_original_moves(tmp_path: Path) -> None:
    kept: list[Path] = []

    def stage(path: Path) -> Path:
        copy = tmp_path / "kept" / path.name
        copy.parent.mkdir(exist_ok=True)
        copy.write_bytes(path.read_bytes())
        kept.append(copy)
        return copy

    script: list[ProviderReply | Exception] = [
        call(PHOTO, {"kind": "overview", "message": "Whole car"}, "a1"),
        call(ASK, {"message": "Thanks"}, "a2"),
    ]
    sessions, conn = _sessions(tmp_path, script, stage=stage)
    sid = sessions.start().session_id
    original = Path(photo(tmp_path / "o.png"))
    sessions.reply(sid, "here", original)
    original.unlink()
    state = sessions._graph.get_state({"configurable": {"thread_id": sid}}).values
    conn.close()
    assert state["photos"]["overview"] == str(kept[0])
    assert kept[0].is_file()


def test_handing_over_twice_files_one_claim(tmp_path: Path, store: SQLiteEventStore) -> None:
    deps = make_test_deps(tmp_path, store, FakeDetector())
    submit = pipeline_submitter(lambda: deps, CONFIG, process=True)
    state: dict[str, object] = {
        "facts": {"policy_id": "P-1001", "what_happened": "Hit a post."},
        "photos": {"overview": photo(tmp_path / "o.png")},
        "gaps": {},
        "retakes": {},
        "turns": 3,
        "transcript": [],
        "session_id": "same-session",
        "handover": "",
    }
    first = submit(state)  # type: ignore[arg-type]
    second = submit(state)  # type: ignore[arg-type]
    assert first == second
    events = store.load(UUID(first))
    assert [e.type for e in events].count("ClaimReported") == 1
    assert [e.type for e in events].count("IntakeCompleted") == 1


# 3. No promises or denials reach the customer, and the agent never sees cover details.
@pytest.mark.parametrize(
    "text",
    [
        "You\u2019re covered for this.",
        "You're not covered for collision.",
        "Your damage is covered.",
        "Your policy covers this.",
        "We will cover the repair.",
        "Your excess is $500.",
        "The repair should cost around $800.",
        "Your claim will be accepted.",
        "I'm afraid we have to decline this claim.",
        "Your claim is denied.",
    ],
)
def test_the_filter_catches_promises_denials_and_prices(text: str) -> None:
    assert promises(text)


def test_a_promise_is_not_sent_and_the_agent_rephrases(tmp_path: Path) -> None:
    h = Harness(
        tmp_path,
        [
            call(ASK, {"message": "Good news, your policy covers this!"}, "a1"),
            call(ASK, {"message": "Thanks. When did it happen?"}, "a2"),
        ],
    )
    assert asked(h.start())["message"] == "Thanks. When did it happen?"
    refused = h.result(1)[0]
    assert refused.is_error
    assert "not sent" in refused.content


def test_the_agent_only_learns_whether_a_policy_exists() -> None:
    full = json.dumps(
        {
            "found": True,
            "policy_id": "P-1001",
            "status": "active",
            "collision": True,
            "deductible": 250,
        }
    )
    assert json.loads(policy_found_only(full)) == {"found": True}
    assert json.loads(policy_found_only("not json")) == {"found": False}


# 5. A broken image is a retake reason, not a crash.
def test_a_truncated_image_is_not_readable(tmp_path: Path) -> None:
    good = Path(photo(tmp_path / "g.jpg"))
    cut = tmp_path / "cut.jpg"
    cut.write_bytes(good.read_bytes()[: good.stat().st_size // 2])
    check = coach_photo(cut, CONFIG)
    assert not check.ok
    assert check.reason == "not a readable image"


# Minors.
def test_a_later_good_photo_clears_the_gap(tmp_path: Path) -> None:
    script: list[ProviderReply | Exception] = [
        call(PHOTO, {"kind": "plate", "message": f"Plate {i}"}, f"a{i}") for i in range(4)
    ]
    script.append(call(ASK, {"message": "Done"}, "a9"))
    h = Harness(tmp_path, script)
    h.start()
    for i in range(3):
        h.reply("try", photo(tmp_path / f"d{i}.png", dark=True))
    h.reply("better", photo(tmp_path / "ok.png"))
    values = h.graph.get_state(h.config).values
    assert "plate" in values["photos"]
    assert values["gaps"] == {}


def test_long_customer_messages_are_capped(tmp_path: Path) -> None:
    h = Harness(
        tmp_path,
        [call(ASK, {"message": "What happened?"}, "a1"), call(ASK, {"message": "OK"}, "a2")],
    )
    h.start()
    h.reply("x" * 10_000)
    assert len(h.result(1)[0].content) < 2_200


def test_a_forced_finish_is_recorded_and_worded_honestly(
    tmp_path: Path, store: SQLiteEventStore
) -> None:
    deps = make_test_deps(tmp_path, store, FakeDetector())
    submit = pipeline_submitter(lambda: deps, CONFIG, process=False)
    limited = CONFIG.model_copy(update={"max_turns": 1})
    h = Harness(
        tmp_path,
        [call(ASK, {"message": "Policy?"}, "a1"), call(ASK, {"message": "More?"}, "a2")],
        config=limited,
    )
    h.graph = build_intake_graph(
        h.graph.nodes
        and GatewayChatModel(
            gateway=Gateway(
                LLM,
                h.fake,
                Budget(tmp_path / "b3.sqlite", LLM.limits, today=lambda: TODAY),
                ResponseCache(tmp_path / "c3.sqlite"),
                lambda c: None,
                sleep=lambda s: None,
            ),
            tier="fast",
        ),
        lookup,
        limited,
        submit,
        lambda: TODAY,
        checkpointer=__import__("langgraph.checkpoint.memory").checkpoint.memory.InMemorySaver(),
    )
    h.start()
    out = h.reply("P-1001")
    intake = fold(store.load(UUID(out["claim_id"]))).intake
    assert intake is not None
    assert intake.handover == "turn limit reached"
    assert "pass what we have to a claims handler" in out["outgoing"]


def test_the_terminal_ignores_empty_lines_and_odd_commands(tmp_path: Path) -> None:
    script: list[ProviderReply | Exception] = [
        call(ASK, {"message": "Policy?"}, "a1"),
        call(ASK, {"message": "Thanks"}, "a2"),
    ]
    sessions, conn = _sessions(tmp_path, script)
    lines = iter(["", "   ", "/photograph", "P-1001"])
    out: list[str] = []

    def read() -> str:
        try:
            return next(lines)
        except StopIteration:
            raise EOFError from None

    run_chat(sessions, read, out.append)
    conn.close()
    assert out.count("ClaimLens: Policy?") == 1
    assert any("Unknown command" in line for line in out)
    assert "ClaimLens: Thanks" in out


def test_ctrl_c_pauses(tmp_path: Path) -> None:
    sessions, conn = _sessions(tmp_path, [call(ASK, {"message": "Policy?"}, "a1")])

    def interrupt() -> str:
        raise KeyboardInterrupt

    out: list[str] = []
    run_chat(sessions, interrupt, out.append)
    conn.close()
    assert "Paused" in out[-1]


def test_keep_photo_copies_by_content(tmp_path: Path) -> None:
    from claimlens.intake_agent.session import keep_photo

    stage = keep_photo(tmp_path / "kept")
    original = Path(photo(tmp_path / "o.PNG"))
    first = stage(original)
    assert first.parent == tmp_path / "kept"
    assert first.suffix == ".png"
    assert first.read_bytes() == original.read_bytes()
    assert stage(original) == first
