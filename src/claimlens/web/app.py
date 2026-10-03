"""The FastAPI app: API routers, pages, static files and plain error messages."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from pathlib import Path
from urllib.parse import urlsplit

from fastapi import FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.trustedhost import TrustedHostMiddleware

from claimlens.web.services import WebServices

HERE = Path(__file__).parent
templates = Jinja2Templates(directory=HERE / "templates")  # autoescapes .html
_SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}
SHOWCASE_REFUSAL = "This is a read-only showcase. Run ClaimLens locally to file and review claims."
_CSP = (
    "default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; "
    "script-src 'self'; frame-ancestors 'self' https://huggingface.co"
)
_DOCS = ("/docs", "/openapi.json", "/redoc")  # Swagger UI loads its assets from a CDN


def get_services(request: Request) -> WebServices:
    services: WebServices = request.app.state.services
    return services


def create_app(services: WebServices) -> FastAPI:
    from claimlens.web.routers import claims, intake, pages, review

    app = FastAPI(
        title="ClaimLens",
        summary="Local prototype: file a claim by chat, follow it, review it.",
        version="0.1.0",
    )
    app.state.services = services
    allowed = set(services.settings.allowed_hosts)
    showcase = services.settings.showcase

    @app.middleware("http")
    async def _read_only_and_headers(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        """Showcase: no request may change anything, whatever route it reaches. Always: standard
        security headers."""
        if showcase and request.method not in _SAFE_METHODS:
            response: Response = JSONResponse({"detail": SHOWCASE_REFUSAL}, 403)
        else:
            response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "same-origin"
        if not request.url.path.startswith(_DOCS):
            response.headers["Content-Security-Policy"] = _CSP
        return response

    @app.middleware("http")
    async def _same_origin(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        """A web page elsewhere must not change anything here (filing, replying, reviewing)."""
        origin = request.headers.get("origin")
        if (
            request.method not in _SAFE_METHODS
            and origin
            and urlsplit(origin).hostname not in allowed
        ):
            return JSONResponse({"detail": "Cross-site requests are not allowed."}, 403)
        return await call_next(request)

    # Outermost: a request for any other host name (DNS rebinding) never reaches the app.
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=sorted(allowed))
    app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")
    app.include_router(claims.router)
    app.include_router(review.router)
    app.include_router(intake.router)
    app.include_router(pages.router)

    @app.exception_handler(RequestValidationError)
    async def _invalid(request: Request, exc: RequestValidationError) -> JSONResponse:
        errors = exc.errors()
        first = errors[0] if errors else {}
        where = ".".join(str(p) for p in first.get("loc", ())[1:]) or "request"
        message = f"Invalid {where}: {first.get('msg', 'bad value')}."
        return JSONResponse({"detail": message}, status_code=422)

    return app
