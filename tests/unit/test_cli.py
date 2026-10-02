import sqlite3
from collections.abc import Callable
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from claimlens.cli import DetectorSpec, main
from tests.fakes import CONFIG_DIR, FakeDetector, SimulatedCrashError


def _cli(tmp_path: Path, *args: str, detector: FakeDetector | None = None) -> int:
    base = ["--db", str(tmp_path / "claims.db"), "--blobs", str(tmp_path / "blobs")]
    chosen = detector or FakeDetector()
    return main([*base, "--config", str(CONFIG_DIR), *args], detector_factory=lambda *_: chosen)


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


def test_resume_finishes_a_claim_that_crashed_mid_run(
    tmp_path: Path, make_image: Callable[..., Path], capsys: pytest.CaptureFixture[str]
) -> None:
    photo = str(make_image("a.jpg"))
    with pytest.raises(SimulatedCrashError):
        _cli(tmp_path, "run", "--policy", "P-1001", photo, detector=FakeDetector(crash_on="p1"))
    submitted = capsys.readouterr().err
    claim_id = submitted.split("Submitted claim ")[1].split()[0]

    assert _cli(tmp_path, "resume", claim_id) == 0

    out = capsys.readouterr().out
    assert f"Claim: {claim_id}" in out
    assert "Route: FAST_TRACK (R9)" in out


def test_missing_champion_weights_are_a_clear_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    def missing(spec: DetectorSpec) -> FakeDetector:
        raise FileNotFoundError(f"model weights not found: {spec.weights}")

    golden = tmp_path / "claims.jsonl"
    golden.write_text("", encoding="utf-8")
    code = main(
        [
            "--config",
            str(CONFIG_DIR),
            "eval-triage",
            "--golden",
            str(golden),
            "--report",
            str(tmp_path / "r.md"),
        ],
        detector_factory=missing,
    )
    assert code == 1
    assert "model weights not found" in capsys.readouterr().err


def test_bad_what_if_value_is_a_usage_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit):
        main(["eval-triage", "--golden", "g.jsonl", "--report", "r.md", "--what-if", "abc"])
    assert "--what-if" in capsys.readouterr().err


def test_approve_payment_prints_a_token_for_decided_claims(
    tmp_path: Path,
    make_image: Callable[..., Path],
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CLAIMLENS_APPROVAL_SECRET", "s3cret")
    claim = _run_claim(tmp_path, make_image, capsys)
    base = ["--db", str(tmp_path / "claims.db"), "--blobs", str(tmp_path / "blobs")]
    assert main([*base, "approve-payment", claim, "300"]) == 0
    assert len(capsys.readouterr().out.strip()) > 40
    assert main([*base, "approve-payment", str(uuid4()), "300"]) == 2
