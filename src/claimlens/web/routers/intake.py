"""The customer's chat with the intake agent. Every call runs on the intake worker thread."""

from __future__ import annotations

from collections.abc import Callable
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from claimlens.intake_agent.session import AgentTurn, IntakeSessions
from claimlens.llm.types import LLMError
from claimlens.web.app import get_services
from claimlens.web.schemas import ChatTurn
from claimlens.web.services import WebServices
from claimlens.web.uploads import UploadError, check_upload

router = APIRouter(prefix="/api/intake", tags=["intake"])
Services = Annotated[WebServices, Depends(get_services)]
SAVED = "Sorry, something went wrong on our side. Your answers are saved; please try again shortly."
NO_CHAT = "No open chat with that reference."


def _chat_turn(services: WebServices, turn: AgentTurn) -> ChatTurn:
    """The one place a turn reaches the customer: a turn that filed the claim starts the
    pipeline (the runner ignores a claim it is already processing)."""
    if turn.claim_id is not None:
        services.runner.start(UUID(turn.claim_id))
    return ChatTurn(
        session_id=turn.session_id,
        message=turn.message,
        photo_kind=turn.photo_kind,
        claim_id=turn.claim_id,
    )


def _available(services: WebServices) -> None:
    if services.intake is None:
        raise HTTPException(503, "The intake chat is not available.")


def _current(sessions: IntakeSessions, session_id: str) -> AgentTurn | None:
    """What a returning customer should see: the waiting question, a stalled chat restarted from
    its last saved step, or the hand-over message of a finished chat. None if unknown."""
    turn = sessions.resume(session_id)
    if turn is not None:
        return turn
    saved = sessions.values(session_id)
    if saved.get("claim_id"):
        return AgentTurn(session_id, str(saved.get("outgoing", "")), None, str(saved["claim_id"]))
    return None


def _call[T](services: WebServices, work: Callable[[], T]) -> T:
    try:
        return services.intake_worker.call(work)
    except LLMError:
        raise HTTPException(503, SAVED) from None


@router.post("", status_code=201, summary="Start a claim chat")
def start(services: Services) -> ChatTurn:
    _available(services)
    turn: AgentTurn = _call(services, lambda: services.sessions().start())
    return _chat_turn(services, turn)


@router.get("/{session_id}", summary="Where a chat stands: the question it waits on, or its claim")
def show(session_id: str, services: Services) -> ChatTurn:
    _available(services)
    turn: AgentTurn | None = _call(services, lambda: _current(services.sessions(), session_id))
    if turn is None:
        raise HTTPException(404, NO_CHAT)
    return _chat_turn(services, turn)


@router.post("/{session_id}/reply", summary="Answer, with an optional photo")
def reply(
    session_id: str,
    services: Services,
    text: Annotated[str, Form(max_length=4000)] = "",
    photo: Annotated[UploadFile | None, File()] = None,
) -> ChatTurn:
    _available(services)

    def is_open() -> bool:
        sessions = services.sessions()
        return sessions.pending(session_id) is not None or sessions.stalled(session_id)

    if not _call(services, is_open):  # nothing is saved for a chat that does not exist
        raise HTTPException(404, NO_CHAT)
    kept = None
    if photo is not None:
        limit_mb = services.settings.max_upload_mb
        limit = limit_mb * 1_000_000
        try:
            data = photo.file.read(limit + 1)
            kept = check_upload(data, limit, services.upload_dir, limit_mb=limit_mb)
        except UploadError as exc:
            raise HTTPException(422, str(exc)) from None

    def answer() -> AgentTurn | None:
        sessions = services.sessions()
        if sessions.pending(session_id) is not None:
            return sessions.reply(session_id, text, kept)
        # A step failed after the last answer was saved (an LLM outage): carry on from there.
        return sessions.resume(session_id)

    turn: AgentTurn | None = _call(services, answer)
    if turn is None:
        raise HTTPException(404, NO_CHAT)
    return _chat_turn(services, turn)
