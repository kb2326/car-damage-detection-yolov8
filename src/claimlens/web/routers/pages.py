"""Server-rendered pages. All customer text goes through Jinja2 autoescaping."""

from __future__ import annotations

from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse

from claimlens.events.projection import fold
from claimlens.web.app import get_services, templates
from claimlens.web.routers.claims import list_claims, load_claim, parse_claim_id
from claimlens.web.services import WebServices
from claimlens.web.views import claim_detail

router = APIRouter(include_in_schema=False)
Services = Annotated[WebServices, Depends(get_services)]


@router.get("/", response_class=HTMLResponse)
def chat_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "chat.html", {"title": "File a claim"})


@router.get("/claims", response_class=HTMLResponse)
def claims_page(
    request: Request, services: Services, filter: Literal["all", "review", "fraud"] = "all"
) -> HTMLResponse:
    rows = list_claims(services)
    if filter == "review":  # the same rule as `claimlens queue`
        rows = [r for r in rows if r.needs_review]
    elif filter == "fraud":  # the route that counts now, after any override
        rows = [r for r in rows if r.final_route == "FRAUD_REVIEW"]
    return templates.TemplateResponse(
        request, "claims.html", {"title": "Claims", "rows": rows, "filter": filter}
    )


@router.get("/claims/{claim_id}", response_class=HTMLResponse)
def claim_page(request: Request, claim_id: str, services: Services) -> HTMLResponse:
    try:
        cid = parse_claim_id(claim_id)
        with services.open_store() as store:
            events = load_claim(store, cid)
    except HTTPException as exc:
        name = "log_failed.html" if exc.status_code == 409 else "not_found.html"
        missing = {"title": "Claim", "claim_id": claim_id}
        return templates.TemplateResponse(request, name, missing, status_code=exc.status_code)
    state = fold(events)
    detail = claim_detail(
        state,
        events,
        running=services.runner.running(cid),
        error=services.runner.error(cid),
        similar=services.similar(state, events),
    )
    context: dict[str, object] = {
        "title": f"Claim {claim_id[:8]}",
        "c": detail,
        "poll": services.settings.poll_seconds,
    }
    return templates.TemplateResponse(request, "claim.html", context)
