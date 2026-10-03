import argparse
import sqlite3
from collections.abc import Callable
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from claimlens.cli import DetectorSpec, main
from claimlens.domain import AgentRecommendation, Confidence, Route
from claimlens.events.projection import ClaimState
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


class _AdvisingAgent:
    agent_version = "test-llm-agent"

    def recommend(self, state: ClaimState) -> AgentRecommendation:
        return AgentRecommendation(
            route_suggestion=Route.ADJUSTER_REVIEW,
            confidence=Confidence.HIGH,
            rationale="Delivery use is excluded.",
            citations=(),
            policy_citations=("STD-8.5",),
        )


def test_run_with_the_llm_agent_uses_the_agent_factory(
    tmp_path: Path, make_image: Callable[..., Path], capsys: pytest.CaptureFixture[str]
) -> None:
    seen: list[Path] = []

    def factory(args: argparse.Namespace, store_path: Path) -> _AdvisingAgent:
        seen.append(store_path)
        return _AdvisingAgent()

    base = ["--db", str(tmp_path / "claims.db"), "--blobs", str(tmp_path / "blobs")]
    code = main(
        [
            *base,
            "--config",
            str(CONFIG_DIR),
            "--agent",
            "llm",
            "run",
            "--policy",
            "P-1001",
            str(make_image("a.jpg")),
        ],
        detector_factory=lambda *_: FakeDetector(),
        agent_factory=factory,
    )
    assert code == 0
    assert seen == [tmp_path / "claims.db"]
    assert "Route: ADJUSTER_REVIEW (R8)" in capsys.readouterr().out


def test_llm_agent_without_a_key_exits_with_a_clear_message(
    tmp_path: Path, make_image: Callable[..., Path], capsys: pytest.CaptureFixture[str]
) -> None:
    from claimlens.llm.types import LLMUnavailable

    def factory(args: argparse.Namespace, store_path: Path) -> _AdvisingAgent:
        raise LLMUnavailable("set ANTHROPIC_API_KEY in .env to use the LLM gateway")

    base = ["--db", str(tmp_path / "claims.db"), "--blobs", str(tmp_path / "blobs")]
    code = main(
        [
            *base,
            "--config",
            str(CONFIG_DIR),
            "--agent",
            "llm",
            "run",
            "--policy",
            "P-1001",
            str(make_image("a.jpg")),
        ],
        detector_factory=lambda *_: FakeDetector(),
        agent_factory=factory,
    )
    assert code == 2
    assert "ANTHROPIC_API_KEY" in capsys.readouterr().err
    assert not (tmp_path / "blobs").exists()  # nothing was submitted


def test_default_agent_is_the_stub(
    tmp_path: Path, make_image: Callable[..., Path], capsys: pytest.CaptureFixture[str]
) -> None:
    def factory(args: argparse.Namespace, store_path: Path) -> _AdvisingAgent:
        raise AssertionError("the LLM agent must not be built by default")

    base = ["--db", str(tmp_path / "claims.db"), "--blobs", str(tmp_path / "blobs")]
    code = main(
        [*base, "--config", str(CONFIG_DIR), "run", "--policy", "P-1001", str(make_image("a.jpg"))],
        detector_factory=lambda *_: FakeDetector(),
        agent_factory=factory,
    )
    assert code == 0


def test_eval_triage_can_run_a_subset(tmp_path: Path) -> None:
    from claimlens.cli import _select_cases
    from claimlens.evals.golden import load_golden

    golden = load_golden(CONFIG_DIR.parent / "evals" / "golden" / "v1" / "claims.jsonl")
    cases = tmp_path / "cases.txt"
    cases.write_text("g002\n# comment\n\ng001\n", encoding="utf-8")
    assert [c.case_id for c in _select_cases(golden, cases)] == ["g001", "g002"]
    cases.write_text("nope\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown case id"):
        _select_cases(golden, cases)


def test_dev20_is_a_valid_stratified_subset() -> None:
    from claimlens.cli import _select_cases
    from claimlens.evals.golden import load_golden

    root = CONFIG_DIR.parent / "evals" / "golden" / "v1"
    subset = _select_cases(load_golden(root / "claims.jsonl"), root / "dev20.txt")
    routes = [c.expected_route.value for c in subset]
    assert len(subset) == 20
    assert (routes.count("ADJUSTER_REVIEW"), routes.count("FAST_TRACK")) == (12, 6)
    assert routes.count("FRAUD_REVIEW") == 2


def test_llm_daily_cap_reaches_the_agent_factory(
    tmp_path: Path, make_image: Callable[..., Path]
) -> None:
    caps: list[float | None] = []

    def factory(args: argparse.Namespace, store_path: Path) -> _AdvisingAgent:
        caps.append(args.llm_daily_cap)
        return _AdvisingAgent()

    golden = tmp_path / "g.jsonl"
    golden.write_text(
        '{"case_id":"x1","scenario":"s","policy_id":"P-1001","description":"d",'
        f'"photos":["{make_image("a.jpg").as_posix()}"],"expected_route":"ADJUSTER_REVIEW",'
        '"label_source":"t"}\n',
        encoding="utf-8",
    )
    code = main(
        [
            "--config",
            str(CONFIG_DIR),
            "--agent",
            "llm",
            "eval-triage",
            "--golden",
            str(golden),
            "--report",
            str(tmp_path / "r.md"),
            "--llm-daily-cap",
            "5",
        ],
        detector_factory=lambda *_: FakeDetector(),
        agent_factory=factory,
    )
    assert code == 0
    assert caps == [5.0]
    report = (tmp_path / "r.md").read_text(encoding="utf-8")
    assert "## Agent" in report
    assert "| Citation validity |" in report


def test_review_claim_and_queue(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from tests.fakes import decided_claim

    store, claim = decided_claim(tmp_path, Route.ADJUSTER_REVIEW)
    store.close()
    db = ["--db", str(tmp_path / "claims.db")]
    assert main([*db, "queue"]) == 0
    out = capsys.readouterr().out
    assert str(claim) in out
    assert "R7" in out
    assert main([*db, "review-claim", str(claim), "--deny", "--reviewer", "sam"]) == 2
    assert "--note" in capsys.readouterr().err
    assert main([*db, "review-claim", str(claim), "--approve", "--reviewer", "sam"]) == 0
    assert main([*db, "queue"]) == 0
    assert "No claims are waiting" in capsys.readouterr().out
