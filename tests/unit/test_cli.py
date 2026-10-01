import sqlite3
from collections.abc import Callable
from pathlib import Path
from uuid import UUID

import pytest

from claimlens.cli import main
from tests.fakes import CONFIG_DIR, FakeDetector


def _cli(tmp_path: Path, *args: str) -> int:
    base = ["--db", str(tmp_path / "claims.db"), "--blobs", str(tmp_path / "blobs")]
    return main(
        [*base, "--config", str(CONFIG_DIR), *args], detector_factory=lambda _: FakeDetector()
    )


def _run_claim(
    tmp_path: Path, make_image: Callable[..., Path], capsys: pytest.CaptureFixture[str]
) -> str:
    code = _cli(
        tmp_path, "run", "--policy", "P-1001", "--description", "scrape", str(make_image("a.jpg"))
    )
    assert code == 0
    out = capsys.readouterr().out
    claim_line = next(line for line in out.splitlines() if line.startswith("Claim: "))
    return claim_line.removeprefix("Claim: ")


def test_run_prints_the_decision(
    tmp_path: Path, make_image: Callable[..., Path], capsys: pytest.CaptureFixture[str]
) -> None:
    code = _cli(tmp_path, "run", "--policy", "P-1001", str(make_image("a.jpg")))
    out = capsys.readouterr().out
    assert code == 0
    assert "Route: FAST_TRACK (R9)" in out
    assert "Findings: 1 (dent 0.90)" in out
    assert "Cost estimate: $150-$400 USD" in out


def test_show_prints_the_audit_trail(
    tmp_path: Path, make_image: Callable[..., Path], capsys: pytest.CaptureFixture[str]
) -> None:
    claim_id = _run_claim(tmp_path, make_image, capsys)
    assert _cli(tmp_path, "show", claim_id) == 0
    out = capsys.readouterr().out
    assert "Audit trail:" in out
    assert "RouteDecided" in out
    assert "agent:stub-v0" in out


def test_verify_reports_a_valid_chain(
    tmp_path: Path, make_image: Callable[..., Path], capsys: pytest.CaptureFixture[str]
) -> None:
    claim_id = _run_claim(tmp_path, make_image, capsys)
    assert _cli(tmp_path, "verify", claim_id) == 0
    assert "Chain OK: 9 events" in capsys.readouterr().out


def test_verify_detects_tampering(
    tmp_path: Path, make_image: Callable[..., Path], capsys: pytest.CaptureFixture[str]
) -> None:
    claim_id = _run_claim(tmp_path, make_image, capsys)
    with sqlite3.connect(tmp_path / "claims.db") as conn:
        conn.execute("UPDATE events SET data = replace(data, 'P-1001', 'P-1002') WHERE seq = 1")
    assert _cli(tmp_path, "verify", claim_id) == 1
    assert "failed verification" in capsys.readouterr().err


def test_unknown_claim_returns_2(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert _cli(tmp_path, "show", str(UUID(int=5))) == 2
    assert "no events found" in capsys.readouterr().err
