"""`claimlens serve`: the real wiring for the web prototype."""

from __future__ import annotations

import argparse
import sys
import threading
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any
from uuid import UUID

from claimlens.blobs import BlobStore
from claimlens.events.store import SQLiteEventStore
from claimlens.intake import submit_claim
from claimlens.intake_agent.session import IntakeSessions
from claimlens.web.runner import ClaimRunner
from claimlens.web.services import WebServices, pipeline_processor
from claimlens.web.settings import load_web_settings
from claimlens.web.workers import Worker
from claimlens.workflow import PipelineDeps

LOCAL_HOSTS = {"127.0.0.1", "localhost"}
SEED_STORIES = (
    "Reversed into a bollard in a car park.",
    "Another car clipped my rear bumper at low speed.",
    "Same dent as before, I think it got worse.",
)


def build_services(
    args: argparse.Namespace, detector_factory: Callable[..., Any], *, inline: bool = False
) -> WebServices:
    """Services from the CLI's settings. The detector and agent are built on the claims worker
    (first use), the intake chat on the intake worker, so their SQLite connections stay there."""
    from claimlens.cli import _make_agent, _make_detector, _open_memory, make_deps
    from claimlens.memory.commands import MEMORY_PATH

    settings = load_web_settings(args.config / "web.toml")
    blobs = BlobStore(args.blobs)
    memory = _open_memory(args, MEMORY_PATH)
    detectors: list[Any] = []

    def deps_for(store: SQLiteEventStore) -> PipelineDeps:
        if not detectors:
            detectors.append(_make_detector(args, detector_factory))
        agent = _make_agent(args, args.db)
        return make_deps(store, blobs, args.config, detectors[0], agent, memory)

    services = WebServices(
        db_path=args.db,
        blobs=blobs,
        settings=settings,
        runner=ClaimRunner(Worker("claims", inline=inline), pipeline_processor(args.db, deps_for)),
        intake_worker=Worker("intake", inline=inline),
        intake=None,
        upload_dir=Path("var/web-uploads"),
        memory=memory,
    )
    services.intake = _intake_factory(args, services)
    return services


def _intake_factory(
    args: argparse.Namespace, services: WebServices
) -> Callable[[], IntakeSessions]:
    def build() -> IntakeSessions:
        from claimlens.intake_agent.config import load_intake_config
        from claimlens.intake_agent.graph import IntakeState
        from claimlens.intake_agent.session import build_intake, file_intake_claim

        config = load_intake_config(args.config / "intake.toml")

        def submit(state: IntakeState) -> str:  # files the claim; the runner processes it
            store = SQLiteEventStore(services.db_path)
            try:
                return str(file_intake_claim(store, services.blobs, state, config))
            finally:
                store.close()

        def unused() -> PipelineDeps:
            raise RuntimeError("the web app files claims through its own submit")

        return build_intake(
            args.config, Path.cwd(), Path("var/intake.sqlite"), unused, submit=submit
        )

    return build


def seed_claims(services: WebServices, photos: Sequence[Path]) -> list[UUID]:
    """Three form-filed claims so the adjuster screens are not empty; the third reuses the
    first photo, which shows the reused-photo check. Each claim is filed after the one before
    it is decided, as real claims arrive, so only the later copy is the reuse."""
    if not photos:
        return []
    chosen = [photos[0], photos[1 % len(photos)], photos[0]]
    claims = []
    for story, photo in zip(SEED_STORIES, chosen, strict=True):
        with services.open_store() as store:
            claim = submit_claim(
                store, services.blobs, policy_id="P-1001", description=story, photo_paths=[photo]
            )
        claims.append(claim)
        services.runner.start(claim)
        while services.runner.running(claim):
            time.sleep(0.05)
    return claims


def run_serve(
    args: argparse.Namespace,
    detector_factory: Callable[..., Any],
    run: Callable[..., None] | None = None,
) -> int:
    settings = load_web_settings(args.config / "web.toml")
    host = args.host or settings.host
    if host not in LOCAL_HOSTS:
        print(
            "error: the prototype is local only (127.0.0.1) until sign-in exists", file=sys.stderr
        )
        return 2
    args.agent = "stub" if args.stub_agent else "llm"
    args.memory = not args.no_memory
    try:
        import uvicorn

        from claimlens.web.app import create_app
    except ImportError:
        print("error: the web app needs: uv sync --group web --group agent", file=sys.stderr)
        return 2
    services = build_services(args, detector_factory)
    if args.seed:  # in the background, so the app opens at once
        photos = sorted(Path("tests/fixtures/images").glob("*.jpg"))
        threading.Thread(target=seed_claims, args=(services, photos), daemon=True).start()
    port = args.port or settings.port
    print(f"ClaimLens is running on http://{host}:{port} (Ctrl+C to stop)")
    (run or uvicorn.run)(create_app(services), host=host, port=port)
    return 0
