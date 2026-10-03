import argparse
from pathlib import Path
from typing import Any

import pytest

from claimlens.cli import main
from claimlens.events.projection import fold
from claimlens.events.store import SQLiteEventStore
from tests.fakes import CONFIG_DIR, FakeDetector
from tests.unit.web.helpers import image


def _base(tmp_path: Path) -> list[str]:
    return [
        "--db",
        str(tmp_path / "c.db"),
        "--blobs",
        str(tmp_path / "b"),
        "--config",
        str(CONFIG_DIR),
    ]


def test_serve_starts_uvicorn_locally(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    started: dict[str, Any] = {}

    def fake_run(app: Any, **kwargs: Any) -> None:
        started.update(kwargs, app=app)

    monkeypatch.setattr("uvicorn.run", fake_run)
    monkeypatch.chdir(tmp_path)
    code = main(
        [*_base(tmp_path), "serve", "--stub-agent", "--no-memory", "--port", "8123"],
        detector_factory=lambda *_: FakeDetector(),
    )
    assert code == 0
    assert started["host"] == "127.0.0.1"
    assert started["port"] == 8123
    assert started["app"].title == "ClaimLens"


def test_serve_refuses_a_public_host(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = main(
        [*_base(tmp_path), "serve", "--host", "0.0.0.0"],
        detector_factory=lambda *_: FakeDetector(),
    )
    assert code == 2
    assert "local only" in capsys.readouterr().err


def test_seed_files_three_claims_and_the_reused_photo_is_flagged(tmp_path: Path) -> None:
    from claimlens.web.serve import build_services, seed_claims

    args = argparse.Namespace(
        db=tmp_path / "c.db",
        blobs=tmp_path / "b",
        config=CONFIG_DIR,
        detector=None,
        weights=None,
        agent="stub",
        memory=False,
        embedder_factory=None,
        agent_factory=None,
        llm_daily_cap=None,
    )
    services = build_services(args, lambda *_: FakeDetector(), inline=True)
    photos = [image(tmp_path / "s1.png", (200, 30, 30)), image(tmp_path / "s2.png", (30, 30, 200))]
    claims = seed_claims(services, photos)
    assert len(claims) == 3
    store = SQLiteEventStore(tmp_path / "c.db")
    decisions = [fold(store.load(c)).decision for c in claims]
    store.close()
    routes = [d.route.value if d is not None else None for d in decisions]
    assert routes[2] == "FRAUD_REVIEW"  # the third reuses the first photo
    assert "FRAUD_REVIEW" not in routes[:2]
    assert None not in routes
