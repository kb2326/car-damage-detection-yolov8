"""Claims: list, detail, photos, resume."""

from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

from claimlens.events.envelope import ChainIntegrityError, ClaimEvent
from claimlens.events.projection import fold
from claimlens.events.store import ClaimNotFoundError, SQLiteEventStore
from claimlens.web.app import get_services
from claimlens.web.schemas import ClaimDetail, ClaimSummary
from claimlens.web.services import WebServices
from claimlens.web.views import claim_detail, claim_summary

router = APIRouter(prefix="/api/claims", tags=["claims"])
Services = Annotated[WebServices, Depends(get_services)]
LOG_FAILED = "The audit log failed verification. Nobody can act on this claim."


def parse_claim_id(claim_id: str) -> UUID:
    try:
        return UUID(claim_id)
    except ValueError:
        raise HTTPException(404, "No such claim.") from None


def load_claim(store: SQLiteEventStore, claim_id: UUID) -> list[ClaimEvent]:
    try:
        return store.load(claim_id)
    except ClaimNotFoundError:
        raise HTTPException(404, "No such claim.") from None
    except ChainIntegrityError:
        raise HTTPException(409, LOG_FAILED) from None


@router.get("", summary="All claims, newest first")
def list_claims(services: Services) -> list[ClaimSummary]:
    rows: list[ClaimSummary] = []
    with services.open_store() as store:
        for claim_id in store.claim_ids():
            try:
                events = store.load(claim_id)
            except ChainIntegrityError:
                rows.append(
                    ClaimSummary(
                        claim_id=str(claim_id), policy_id="?", filed_at="", status="log_failed"
                    )
                )
                continue
            rows.append(claim_summary(fold(events), events, services.runner.running(claim_id)))
    return sorted(rows, key=lambda r: r.filed_at, reverse=True)


@router.get("/{claim_id}", summary="One claim, built from its events")
def get_claim(claim_id: str, services: Services) -> ClaimDetail:
    cid = parse_claim_id(claim_id)
    with services.open_store() as store:
        events = load_claim(store, cid)
    state = fold(events)
    return claim_detail(
        state,
        events,
        running=services.runner.running(cid),
        error=services.runner.error(cid),
        similar=services.similar(state, events),
    )


@router.get("/{claim_id}/photos/{photo_id}", summary="A photo of this claim")
def get_photo(claim_id: str, photo_id: str, services: Services) -> FileResponse:
    cid = parse_claim_id(claim_id)
    with services.open_store() as store:
        state = fold(load_claim(store, cid))
    photo = state.photos.get(photo_id)
    if photo is None:
        raise HTTPException(404, "No such photo.")
    return FileResponse(services.blobs.path(photo.blob_name))  # blob names keep their suffix


@router.post("/{claim_id}/resume", status_code=202, summary="Finish processing a stopped claim")
def resume_claim(claim_id: str, services: Services) -> dict[str, str]:
    cid = parse_claim_id(claim_id)
    with services.open_store() as store:
        state = fold(load_claim(store, cid))
    if state.decision is not None:
        raise HTTPException(409, "This claim is already decided.")
    if not services.runner.start(cid):
        raise HTTPException(409, "This claim is already being processed.")
    return {"detail": "Processing resumed."}
