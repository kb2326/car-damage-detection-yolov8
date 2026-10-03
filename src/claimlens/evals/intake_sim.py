"""Simulated customers for the intake agent: an LLM plays a customer with a hidden true story;
the harness scores whether intake recorded it, and pass^k asks for k correct runs out of k."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Literal

from claimlens.domain import Frozen
from claimlens.intake_agent.session import IntakeSessions

PHOTO_KINDS = ("overview", "damage_closeup", "plate")
SCORED_FACTS = ("driving_for_work", "other_party_involved", "injuries")
Style = Literal["cooperative", "vague", "over-sharer", "story-changer", "photo-trouble"]


class Persona(Frozen):
    persona_id: str
    style: Style
    story: str
    policy_id: str
    days_ago: int  # the true incident date, counted back from the run's date
    facts: dict[str, str]  # the true yes/no answers (SCORED_FACTS)
    location: str = ""
    object_hit: str = ""
    police_report: bool = False  # the truth needs a police report reference
    police_reference: str = ""
    photo_trouble: bool = False  # sends a dark photo first
    wrong_days_ago: int | None = None  # a story-changer first gives this, then corrects it


@dataclass(frozen=True)
class PhotoFixtures:
    good: Path
    dark: Path


@dataclass(frozen=True)
class Trial:
    persona_id: str
    passed: bool
    failures: list[str]
    turns: int
    handover: str


def load_personas(path: Path) -> list[Persona]:
    lines = path.read_text(encoding="utf-8").splitlines()
    personas = [Persona.model_validate_json(line) for line in lines if line.strip()]
    for persona in personas:
        unknown = set(persona.facts) - set(SCORED_FACTS)
        if unknown:
            raise ValueError(f"{persona.persona_id}: unknown facts {sorted(unknown)}")
    return personas


def score(state: Mapping[str, Any], persona: Persona, today: date) -> list[str]:
    """What intake got wrong for this persona (empty: the trial passes)."""
    failures: list[str] = []
    if state.get("handover"):
        failures.append(f"handed over: {state['handover']}")
    facts: Mapping[str, str] = state["facts"]
    expected = {
        "policy_id": persona.policy_id,
        "incident_date": (today - timedelta(days=persona.days_ago)).isoformat(),
        **{name: persona.facts[name] for name in SCORED_FACTS if name in persona.facts},
    }
    for name, want in expected.items():
        got = facts.get(name)
        if got is None:
            failures.append(f"{name}: not recorded")
        elif got != want:
            failures.append(f"{name}: got {got!r}, expected {want!r}")
    if persona.police_report and "police_report" not in facts:
        failures.append("police_report: not recorded")
    failures += [f"photo missing: {k}" for k in PHOTO_KINDS if k not in state["photos"]]
    return failures


def run_trial(
    persona: Persona,
    sessions: IntakeSessions,
    customer: Callable[[list[dict[str, str]]], str],
    photos: PhotoFixtures,
    *,
    today: date,
    max_turns: int = 40,
) -> Trial:
    """One conversation. The harness answers photo requests; the customer model answers the rest."""
    turn = sessions.start()
    history = [{"role": "agent", "text": turn.message}]
    dark_sent = False
    for _ in range(max_turns):
        if turn.claim_id is not None:
            break
        if turn.photo_kind is not None:
            if persona.photo_trouble and not dark_sent:
                dark_sent = True
                turn = sessions.reply(turn.session_id, "Here it is.", photos.dark)
            else:
                turn = sessions.reply(turn.session_id, "Here it is.", photos.good)
        else:
            text = customer(history)
            history.append({"role": "customer", "text": text})
            turn = sessions.reply(turn.session_id, text)
        history.append({"role": "agent", "text": turn.message})
    values = sessions.values(turn.session_id)
    if turn.claim_id is None:
        failures = [f"did not finish within {max_turns} turns"]
    else:
        failures = score(values, persona, today)
    return Trial(
        persona_id=persona.persona_id,
        passed=not failures,
        failures=failures,
        turns=int(values.get("turns", 0)),
        handover=str(values.get("handover", "")),
    )


def persona_prompt(template: str, persona: Persona, today: date) -> str:
    when = today - timedelta(days=persona.days_ago)
    facts = "\n".join(f"- {k.replace('_', ' ')}: {v}" for k, v in persona.facts.items())
    lines = [
        template.strip(),
        "",
        "Your situation (the hidden truth; never quote this list):",
        f"- policy number: {persona.policy_id}",
        f"- what happened: {persona.story}",
        f"- date: {when.isoformat()} ({persona.days_ago} day(s) before today, {today.isoformat()})",
        f"- where: {persona.location or 'you can describe it briefly'}",
        f"- what was hit: {persona.object_hit or 'see the story'}",
        "- the policyholder (you) was driving",
        facts,
        f"- police report reference: {persona.police_reference or 'none'}",
        "",
        f"Your style: {persona.style}.",
    ]
    if persona.wrong_days_ago is not None:
        lines.append(
            f"The first time you mention the date, say it happened {persona.wrong_days_ago} "
            "days ago. In your next message, correct yourself to the true date."
        )
    return "\n".join(lines)


def customer_model(
    gateway: Any, template: str, persona: Persona, today: date
) -> Callable[[list[dict[str, str]]], str]:
    """The customer, played by the fast tier. Plain text only; no tools."""
    from claimlens.llm.types import LLMRequest, Message

    system = persona_prompt(template, persona, today)

    def reply(history: list[dict[str, str]]) -> str:
        messages = [
            Message(role="user" if h["role"] == "agent" else "assistant", content=h["text"])
            for h in history
        ]
        response = gateway.generate(
            LLMRequest(messages=messages, tier="fast", system=system, max_tokens=200)
        )
        return str(response.text).strip() or "Sorry, could you repeat that?"

    return reply


def pass_at_k(trials: Sequence[Trial]) -> tuple[float, float]:
    """pass@1 (share of all trials) and pass^k (share of personas with every trial passed)."""
    by_persona: dict[str, list[bool]] = defaultdict(list)
    for trial in trials:
        by_persona[trial.persona_id].append(trial.passed)
    if not trials:
        return 0.0, 0.0
    first = sum(t.passed for t in trials) / len(trials)
    every = sum(all(v) for v in by_persona.values()) / len(by_persona)
    return first, every


def render_intake_report(
    trials: Sequence[Trial], *, k: int, cost_usd: float, generated_on: date
) -> str:
    first, every = pass_at_k(trials)
    lines = [
        "# Intake evaluation: simulated customers",
        "",
        f"- Date: {generated_on.isoformat()}",
        f"- Trials per customer: {k}",
        f"- pass@1: **{first:.2f}**  pass^{k}: **{every:.2f}** (target 0.70)",
        f"- Cost of this run: ${cost_usd:.2f}",
        "",
        "| Customer | Passed | All passed | Mean turns | What went wrong |",
        "|---|---|---|---|---|",
    ]
    by_persona: dict[str, list[Trial]] = defaultdict(list)
    for trial in trials:
        by_persona[trial.persona_id].append(trial)
    for persona_id, group in by_persona.items():
        passed = sum(t.passed for t in group)
        problems = sorted({f for t in group for f in t.failures})
        mean_turns = sum(t.turns for t in group) / len(group)
        everything = "yes" if passed == len(group) else "no"
        lines.append(
            f"| {persona_id} | {passed} of {len(group)} | {everything} "
            f"| {mean_turns:.1f} | {'; '.join(problems) or '-'} |"
        )
    return "\n".join(lines) + "\n"


def write_fixture_photos(folder: Path) -> PhotoFixtures:
    """A sharp, well-lit test photo and a dark one (the coaching check rejects it)."""
    import random

    from PIL import Image

    folder.mkdir(parents=True, exist_ok=True)
    rng = random.Random(7)
    good = Image.new("RGB", (640, 480))
    good.putdata([(rng.randrange(256),) * 3 for _ in range(640 * 480)])
    good.save(folder / "good.png")
    Image.new("RGB", (640, 480), (5, 5, 5)).save(folder / "dark.png")
    return PhotoFixtures(good=folder / "good.png", dark=folder / "dark.png")


def trial_summary(trials: Sequence[Trial]) -> str:
    return json.dumps([t.__dict__ for t in trials], indent=1)
