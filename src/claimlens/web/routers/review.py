"""Review: the only place a person decides, including the only way to deny."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from pydantic import ValidationError

from claimlens.events.payloads import HumanReviewed
from claimlens.events.projection import fold
from claimlens.review_queue import record_review
from claimlens.web.app import get_services
from claimlens.web.routers.claims import load_claim, parse_claim_id
from claimlens.web.schemas import ReviewRequest
from claimlens.web.services import WebServices

router = APIRouter(prefix="/api/claims", tags=["review"])
Services = Annotated[WebServices, Depends(get_services)]


@router.post("/{claim_id}/review", status_code=201, summary="Record what a reviewer decided")
def review_claim(claim_id: str, body: ReviewRequest, services: Services) -> dict[str, int]:
    cid = parse_claim_id(claim_id)
    try:
        review = HumanReviewed(
            reviewer=body.reviewer.strip(),
            action=body.action,
            final_route=body.final_route,
            note=body.note,
        )
    except ValidationError as exc:
        message = str(exc.errors()[0]["msg"]).removeprefix("Value error, ")
        raise HTTPException(422, message[:1].upper() + message[1:] + ".") from None
    with services.open_store() as store:
        if fold(load_claim(store, cid)).decision is None:
            raise HTTPException(409, "This claim has not been decided yet.")
        seq = record_review(
            store, cid, review, memory=services.memory, photo_path=services.blobs.path
        )
    return {"seq": seq}
