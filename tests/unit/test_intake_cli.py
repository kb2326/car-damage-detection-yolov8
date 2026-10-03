"""The `claimlens intake` terminal chat, driven with scripted input."""

from collections.abc import Iterator
from pathlib import Path

import pytest

from claimlens.intake_agent.commands import run_chat
from claimlens.intake_agent.graph import ASK, PHOTO
from claimlens.llm.provider import FakeProvider
from tests.unit.test_intake_graph import call, photo
from tests.unit.test_intake_session import SCRIPT, _sessions


def _reader(lines: list[str]) -> "Iterator[str]":
    yield from lines


def _io(lines: list[str]) -> tuple["object", list[str]]:
    it = iter(lines)
    out: list[str] = []

    def read() -> str:
        try:
            return next(it)
        except StopIteration:
            raise EOFError from None

    return read, out


def test_a_full_chat_files_a_claim(tmp_path: Path) -> None:
    sessions, conn = _sessions(tmp_path, FakeProvider(list(SCRIPT)), lambda s: "claim-9")
    read, out = _io(
        [
            "P-1001, reversed into a bollard yesterday",
            f"/photo {photo(tmp_path / 'a b.png')}",
            f'/photo "{photo(tmp_path / "c.png")}"',
            f"/photo {photo(tmp_path / 'p.png')}",
        ]
    )
    turn = run_chat(sessions, read, out.append)  # type: ignore[arg-type]
    conn.close()
    assert turn is not None
    assert turn.claim_id == "claim-9"
    assert out[0] == "ClaimLens: Your policy number?"
    assert out[-1].startswith("ClaimLens: Thank you, that's everything I need.")


def test_quit_pauses_and_the_session_resumes(tmp_path: Path) -> None:
    fake = FakeProvider(list(SCRIPT))
    sessions, conn = _sessions(tmp_path, fake, lambda s: "claim-10")
    read, out = _io(["P-1001, reversed into a bollard yesterday", "/quit"])
    paused = run_chat(sessions, read, out.append)  # type: ignore[arg-type]
    assert paused is not None
    assert paused.claim_id is None
    assert any("claimlens intake --session" in line for line in out)
    conn.close()

    again, conn2 = _sessions(tmp_path, fake, lambda s: "claim-10")
    read2, out2 = _io([f"/photo {photo(tmp_path / f'{k}.png')}" for k in ("o", "c", "p")])
    done = run_chat(again, read2, out2.append, session_id=paused.session_id)  # type: ignore[arg-type]
    conn2.close()
    assert done is not None
    assert done.claim_id == "claim-10"
    assert out2[0] == "ClaimLens: Whole car"


def test_end_of_input_pauses(tmp_path: Path) -> None:
    sessions, conn = _sessions(tmp_path, FakeProvider(list(SCRIPT)), lambda s: "x")
    read, out = _io([])
    turn = run_chat(sessions, read, out.append)  # type: ignore[arg-type]
    conn.close()
    assert turn is not None
    assert turn.claim_id is None
    assert "Paused" in out[-1]


def test_a_missing_photo_file_is_a_message_not_a_crash(tmp_path: Path) -> None:
    script = [
        call(PHOTO, {"kind": "overview", "message": "Whole car"}, "a1"),
        call(ASK, {"message": "Thanks"}, "a2"),
    ]
    sessions, conn = _sessions(tmp_path, FakeProvider(list(script)), lambda s: "x")
    read, out = _io(["/photo C:/no/such/file.jpg", f"/photo {photo(tmp_path / 'o.png')}"])
    run_chat(sessions, read, out.append)  # type: ignore[arg-type]
    conn.close()
    assert any("can't find that file" in line for line in out)
    assert out[-2] == "ClaimLens: Thanks"


def test_resuming_an_unknown_session_says_so(tmp_path: Path) -> None:
    sessions, conn = _sessions(tmp_path, FakeProvider([]), lambda s: "x")
    read, out = _io([])
    assert run_chat(sessions, read, out.append, session_id="nope") is None  # type: ignore[arg-type]
    conn.close()
    assert "No open intake session" in out[0]


def test_the_intake_command_lists_sessions_and_reports_a_missing_key(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from claimlens.cli import main
    from claimlens.llm.types import LLMUnavailable

    sessions, conn = _sessions(tmp_path, FakeProvider(list(SCRIPT)), lambda s: "x")
    sid = sessions.start().session_id
    assert main(["intake", "--list"], intake_factory=lambda args: sessions) == 0
    assert sid in capsys.readouterr().out
    conn.close()

    def no_key(args: object) -> object:
        raise LLMUnavailable("set ANTHROPIC_API_KEY in .env to use the LLM gateway")

    assert main(["intake"], intake_factory=no_key) == 2  # type: ignore[arg-type]
    assert "ANTHROPIC_API_KEY" in capsys.readouterr().err
