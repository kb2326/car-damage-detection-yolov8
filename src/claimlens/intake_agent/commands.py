"""`claimlens intake`: chat with the intake agent in the terminal."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from claimlens.intake_agent.session import AgentTurn, IntakeSessions

IntakeFactory = Callable[[argparse.Namespace], IntakeSessions]
HELP = "Type your answer. /photo <path> sends a photo, /quit pauses (you can resume later)."


def add_intake_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    intake = sub.add_parser("intake", help="report a claim by chatting with the intake agent")
    intake.add_argument("--session", default=None, help="resume a paused session")
    intake.add_argument("--list", action="store_true", help="list paused sessions")
    intake.add_argument(
        "--no-process", action="store_true", help="file the claim but do not run the pipeline"
    )


def _photo_path(rest: str) -> Path:
    return Path(rest.strip().strip('"').strip("'"))


def run_chat(
    sessions: IntakeSessions,
    read: Callable[[], str],
    write: Callable[[str], None],
    session_id: str | None = None,
) -> AgentTurn | None:
    """Talk until the claim is filed or the customer pauses. Returns the last turn."""
    from claimlens.llm.types import LLMError

    if session_id is None:
        turn = sessions.start()
    else:
        resumed = sessions.resume(session_id)
        if resumed is None:
            write(f"No open intake session {session_id}. Use --list to see paused sessions.")
            return None
        turn = resumed
    write(f"ClaimLens: {turn.message}")
    while turn.claim_id is None:
        try:
            line = read().strip()
        except (EOFError, KeyboardInterrupt):
            line = "/quit"
        if not line:
            continue  # an empty line costs nothing
        if line == "/quit":
            write(f"Paused. Resume with: claimlens intake --session {turn.session_id}")
            return turn
        photo: Path | None = None
        text = line
        if line == "/photo" or line.startswith("/photo "):
            photo = _photo_path(line[len("/photo") :])
            if not photo.is_file():
                write(f"ClaimLens: I can't find that file ({photo}). Please check the path.")
                continue
            text = ""
        elif line.startswith("/"):
            write("Unknown command. Use /photo <path> to send a photo or /quit to pause.")
            continue
        try:
            turn = sessions.reply(turn.session_id, text, photo)
        except LLMError:
            write(
                "ClaimLens: Sorry, something went wrong on our side. Your answers are saved. "
                f"Please try again later with: claimlens intake --session {turn.session_id}"
            )
            return turn
        write(f"ClaimLens: {turn.message}")
    return turn


def run_intake_command(args: argparse.Namespace, factory: IntakeFactory) -> int:
    from claimlens.llm.types import LLMError

    try:
        sessions = factory(args)
    except ImportError:
        print("error: the intake agent needs: uv sync --group agent", file=sys.stderr)
        return 2
    except LLMError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.list:
        open_ids = sessions.open_sessions()
        print("\n".join(open_ids) if open_ids else "No paused intake sessions.")
        return 0
    print(HELP)
    turn = run_chat(sessions, input, print, session_id=args.session)
    if turn is None:
        return 2
    if turn.claim_id is not None:
        _print_outcome(args, turn.claim_id)
    return 0


def _print_outcome(args: argparse.Namespace, claim_id: str) -> None:
    from uuid import UUID

    from claimlens.events.projection import fold
    from claimlens.events.store import SQLiteEventStore

    store = SQLiteEventStore(args.db)
    try:
        state = fold(store.load(UUID(claim_id)))
    finally:
        store.close()
    if state.decision is not None:
        print(f"(Back office) Route: {state.decision.route.value} ({state.decision.rule_id})")


def default_factory(make_deps: Callable[[argparse.Namespace], Any]) -> IntakeFactory:
    """The real intake: the repo's config, var/intake.sqlite, and the CLI's pipeline settings."""

    def build(args: argparse.Namespace) -> IntakeSessions:
        from claimlens.intake_agent.session import build_intake

        return build_intake(
            args.config,
            Path.cwd(),
            Path("var/intake.sqlite"),
            lambda: make_deps(args),
            process=not args.no_process,
        )

    return build
