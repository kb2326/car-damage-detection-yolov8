"""The customer's chat with the intake agent. Every call runs on the intake worker thread."""

from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from claimlens.intake_agent.session import AgentTurn
from claimlens.llm.types import LLMError
from claimlens.web.app import get_services
from claimlens.web.schemas import ChatTurn
from claimlens.web.services import WebServices
from claimlens.web.uploads import UploadError, check_upload

router = APIRouter(prefix="/api/intake", tags=["intake"])
Services = Annotated[WebServices, Depends(get_services)]
SAVED = "Sorry, something went wrong on our side. Your answers are saved; please try again shortly."
NO_CHAT = "No open chat with that reference."


def _chat_turn(turn: AgentTurn) -> ChatTurn:
    return ChatTurn(
        session_id=turn.session_id,
        message=turn.message,
        photo_kind=turn.photo_kind,
        claim_id=turn.claim_id,
    )


def _available(services: WebServices) -> None:
    if services.intake is None:
        raise HTTPException(503, "The intake chat is not available.")


@router.post("", status_code=201, summary="Start a claim chat")
def start(services: Services) -> ChatTurn:
    _available(services)
    try:
        turn = services.intake_worker.call(lambda: services.sessions().start())
    except LLMError:
        raise HTTPException(503, SAVED) from None
    return _chat_turn(turn)


@router.get("/{session_id}", summary="The question a chat is waiting on")
def show(session_id: str, services: Services) -> ChatTurn:
    _available(services)
    try:
        turn = services.intake_worker.call(lambda: services.sessions().resume(session_id))
    except LLMError:
        raise HTTPException(503, SAVED) from None
    if turn is None:
        raise HTTPException(404, NO_CHAT)
    return _chat_turn(turn)


@router.post("/{session_id}/reply", summary="Answer, with an optional photo")
def reply(
    session_id: str,
    services: Services,
    text: Annotated[str, Form(max_length=4000)] = "",
    photo: Annotated[UploadFile | None, File()] = None,
) -> ChatTurn:
    _available(services)
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
        if sessions.pending(session_id) is None:
            return None
        return sessions.reply(session_id, text, kept)

    try:
        turn = services.intake_worker.call(answer)
    except LLMError:
        raise HTTPException(503, SAVED) from None
    if turn is None:
        raise HTTPException(404, NO_CHAT)
    if turn.claim_id is not None:
        services.runner.start(UUID(turn.claim_id))
    return _chat_turn(turn)
