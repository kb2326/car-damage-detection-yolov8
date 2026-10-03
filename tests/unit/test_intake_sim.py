"""Simulated customers: run trials against the real intake sessions and score pass^k."""

import json
from datetime import date
from pathlib import Path

import pytest

from claimlens.evals.intake_sim import (
    Persona,
    PhotoFixtures,
    Trial,
    load_personas,
    pass_at_k,
    render_intake_report,
    run_trial,
    score,
)
from claimlens.intake_agent.graph import ASK, FINISH, PHOTO
from claimlens.llm.provider import FakeProvider, ProviderReply
from tests.unit.test_intake_graph import TODAY, call, calls, photo
from tests.unit.test_intake_session import _sessions

ROOT = Path(__file__).resolve().parents[2]
PERSONA = Persona(
    persona_id="courier",
    style="cooperative",
    story="I clipped a post while delivering parcels for work.",
    policy_id="P-1001",
    days_ago=1,
    facts={"driving_for_work": "yes", "other_party_involved": "no", "injuries": "no"},
)
FACTS = {
    "policy_id": "P-1001",
    "incident_date": "yesterday",
    "location": "high street",
    "what_happened": "Clipped a post while delivering parcels.",
    "object_hit": "a post",
    "other_party_involved": "no",
    "driver_is_policyholder": "yes",
    "driving_for_work": "yes",
    "injuries": "no",
}


def _script(facts: dict[str, str]) -> list:  # type: ignore[type-arg]
    return [
        call(ASK, {"message": "Policy number?"}, "a1"),
        calls([("record_fact", {"name": k, "value": v}, f"f-{k}") for k, v in facts.items()]),
        call(PHOTO, {"kind": "overview", "message": "Whole car"}, "a2"),
        call(PHOTO, {"kind": "damage_closeup", "message": "Close-up"}, "a3"),
        call(PHOTO, {"kind": "plate", "message": "Plate"}, "a4"),
        call(FINISH, {"summary": "Post."}, "a5"),
    ]


def _fixtures(tmp_path: Path) -> PhotoFixtures:
    return PhotoFixtures(
        good=Path(photo(tmp_path / "good.png")), dark=Path(photo(tmp_path / "dark.png", dark=True))
    )


def _run(tmp_path: Path, facts: dict[str, str], persona: Persona = PERSONA) -> Trial:
    captured: list[dict[str, object]] = []

    def submit(state: object) -> str:
        captured.append(dict(state))  # type: ignore[call-overload]
        return "c1"

    sessions, conn = _sessions(tmp_path, FakeProvider(_script(facts)), submit)
    replies: list[str] = []

    def customer(history: list[dict[str, str]]) -> str:
        replies.append(history[-1]["text"])
        return "P-1001, I clipped a post yesterday while delivering parcels."

    trial = run_trial(persona, sessions, customer, _fixtures(tmp_path), today=TODAY)
    conn.close()
    assert replies[0] == "Policy number?"
    return trial


def test_a_correct_intake_passes(tmp_path: Path) -> None:
    trial = _run(tmp_path, FACTS)
    assert trial.passed
    assert trial.failures == []
    assert trial.turns == 4


def test_a_wrong_fact_fails_the_trial(tmp_path: Path) -> None:
    trial = _run(tmp_path, {**FACTS, "driving_for_work": "no"})
    assert not trial.passed
    assert trial.failures == ["driving_for_work: got 'no', expected 'yes'"]


def test_the_date_is_checked_against_days_ago() -> None:
    state = {
        "facts": {**FACTS, "incident_date": "2026-10-01"},
        "photos": {"overview": "a", "damage_closeup": "b", "plate": "c"},
        "gaps": {},
        "handover": "",
    }
    assert score(state, PERSONA, TODAY) == [
        "incident_date: got '2026-10-01', expected '2026-10-02'"
    ]


def test_a_handover_or_missing_photo_fails() -> None:
    state = {
        "facts": {**FACTS, "incident_date": "2026-10-02"},
        "photos": {"overview": "a"},
        "gaps": {"plate": "too dark after 2 retakes"},
        "handover": "turn limit reached",
    }
    failures = score(state, PERSONA, TODAY)
    assert "handed over: turn limit reached" in failures
    assert "photo missing: damage_closeup" in failures
    assert "photo missing: plate" in failures


def test_a_police_report_is_required_when_the_truth_needs_one() -> None:
    injured = PERSONA.model_copy(
        update={"facts": {**PERSONA.facts, "injuries": "yes"}, "police_report": True}
    )
    state = {
        "facts": {**FACTS, "incident_date": "2026-10-02", "injuries": "yes"},
        "photos": {"overview": "a", "damage_closeup": "b", "plate": "c"},
        "gaps": {},
        "handover": "",
    }
    assert score(state, injured, TODAY) == ["police_report: not recorded"]


def test_photo_trouble_sends_a_dark_photo_first(tmp_path: Path) -> None:
    trouble = PERSONA.model_copy(update={"photo_trouble": True})
    script: list[ProviderReply | Exception] = [
        call(PHOTO, {"kind": "overview", "message": "Whole car"}, "a1"),
        call(PHOTO, {"kind": "overview", "message": "Too dark, again please"}, "a2"),
        call(ASK, {"message": "Thanks"}, "a3"),
    ]
    sessions, conn = _sessions(tmp_path, FakeProvider(script), lambda s: "c")
    fake_customer_turns: list[str] = []

    def customer(history: list[dict[str, str]]) -> str:
        fake_customer_turns.append(history[-1]["text"])
        raise RuntimeError("stop")  # end the trial after the photos

    with pytest.raises(RuntimeError):
        run_trial(trouble, sessions, customer, _fixtures(tmp_path), today=TODAY)
    conn.close()
    assert fake_customer_turns == ["Thanks"]


def test_a_stuck_trial_ends_at_the_turn_limit(tmp_path: Path) -> None:
    script: list[ProviderReply | Exception] = [
        call(ASK, {"message": f"Question {i}?"}, f"a{i}") for i in range(6)
    ]
    sessions, conn = _sessions(tmp_path, FakeProvider(script), lambda s: "c")
    trial = run_trial(
        PERSONA, sessions, lambda h: "I don't know", _fixtures(tmp_path), today=TODAY, max_turns=5
    )
    conn.close()
    assert not trial.passed
    assert trial.failures == ["did not finish within 5 turns"]


def test_pass_at_k() -> None:
    def t(pid: str, ok: bool) -> Trial:
        return Trial(persona_id=pid, passed=ok, failures=[], turns=1, handover="")

    trials = [t("a", True), t("a", True), t("b", True), t("b", False)]
    assert pass_at_k(trials) == (0.75, 0.5)


def test_the_repo_personas_are_valid() -> None:
    personas = load_personas(ROOT / "evals" / "intake" / "personas.jsonl")
    assert len(personas) == 10
    styles = [p.style for p in personas]
    assert styles.count("cooperative") == 3
    assert styles.count("story-changer") == 2
    assert sum(p.photo_trouble for p in personas) == 2
    assert len({p.persona_id for p in personas}) == 10


def test_the_report_shows_pass_rates_and_failures() -> None:
    trials = [
        Trial(persona_id="a", passed=True, failures=[], turns=9, handover=""),
        Trial(
            persona_id="a",
            passed=False,
            failures=["injuries: got 'no', expected 'yes'"],
            turns=12,
            handover="",
        ),
    ]
    report = render_intake_report(trials, k=2, cost_usd=0.21, generated_on=date(2026, 10, 3))
    assert "pass^2" in report
    assert "| a | 1 of 2 | no |" in report
    assert "injuries: got 'no', expected 'yes'" in report
    assert "$0.21" in report
    json.dumps({"ok": True})


def test_eval_intake_without_a_key_exits_2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from claimlens.cli import main

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr("claimlens.llm.factory.read_secret", lambda *a, **k: None)
    code = main(
        [
            "--config",
            str(ROOT / "config"),
            "eval-intake",
            "--report",
            str(tmp_path / "r.md"),
            "--personas",
            str(ROOT / "evals" / "intake" / "personas.jsonl"),
            "--only",
            "coop-bollard",
        ]
    )
    assert code == 2
    assert "ANTHROPIC_API_KEY" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("style", "extra", "error"),
    [
        ("story-changer", {}, "wrong_days_ago"),
        ("photo-trouble", {}, "photo_trouble"),
    ],
)
def test_a_persona_style_needs_its_flag(
    tmp_path: Path, style: str, extra: dict[str, object], error: str
) -> None:
    row = {
        "persona_id": "p",
        "style": style,
        "story": "I hit a post.",
        "policy_id": "P-1001",
        "days_ago": 1,
        "facts": {},
        **extra,
    }
    path = tmp_path / "personas.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match=error):
        load_personas(path)
