"""The FastAPI app: API routers, pages, static files and plain error messages."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from claimlens.web.services import WebServices

HERE = Path(__file__).parent


def get_services(request: Request) -> WebServices:
    services: WebServices = request.app.state.services
    return services


def create_app(services: WebServices) -> FastAPI:
    from claimlens.web.routers import claims

    app = FastAPI(
        title="ClaimLens",
        summary="Local prototype: file a claim by chat, follow it, review it.",
        version="0.1.0",
    )
    app.state.services = services
    app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")
    app.include_router(claims.router)

    @app.exception_handler(RequestValidationError)
    async def _invalid(request: Request, exc: RequestValidationError) -> JSONResponse:
        errors = exc.errors()
        first = errors[0] if errors else {}
        where = ".".join(str(p) for p in first.get("loc", ())[1:]) or "request"
        message = f"Invalid {where}: {first.get('msg', 'bad value')}."
        return JSONResponse({"detail": message}, status_code=422)

    return app
