"""The red-team suite: a catalogue of attacks mapped to the OWASP agentic top 10 (M7).

Each attack points at a test that runs in "hijacked" mode (the fake LLM obeys the attacker), so a
pass means the code controls hold whatever the model says. `claimlens eval-redteam` runs them and
writes a report.
"""

from __future__ import annotations

import tomllib
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any

from claimlens.domain import Frozen

ASI_IDS: tuple[str, ...] = tuple(f"ASI{n:02d}" for n in range(1, 11))
ASI_NAMES = {
    "ASI01": "Agent goal hijack",
    "ASI02": "Tool misuse",
    "ASI03": "Identity and privilege abuse",
    "ASI04": "Supply chain",
    "ASI05": "Unexpected code execution",
    "ASI06": "Memory and context poisoning",
    "ASI07": "Insecure inter-agent communication",
    "ASI08": "Cascading failures",
    "ASI09": "Human-agent trust exploitation",
    "ASI10": "Rogue agents",
}
NOT_APPLICABLE = {
    "ASI07": "ClaimLens has no agent-to-agent channel; its agents never talk to each other."
}
CATALOGUE = Path("evals/redteam/attacks.toml")


class Attack(Frozen):
    id: str
    asi: str
    surface: str
    payload: str
    expect: str
    test: str  # "path::test_name", or "ci:<check>" for a check that runs in CI


def load_attacks(path: Path) -> list[Attack]:
    rows = tomllib.loads(path.read_text(encoding="utf-8")).get("attack", [])
    attacks = [Attack.model_validate(row) for row in rows]
    for attack in attacks:
        if attack.asi not in ASI_IDS:
            raise ValueError(f"{attack.id}: unknown risk {attack.asi}")
    return attacks


def coverage(attacks: list[Attack]) -> dict[str, int]:
    return dict(Counter(a.asi for a in attacks))


def run_attacks(attacks: list[Attack], root: Path) -> dict[str, str]:
    """Run each attack's test with pytest: "held" (passed), "broken" (failed) or "missing"."""
    import pytest

    outcomes: dict[str, str] = {}

    class Collector:
        def pytest_runtest_logreport(self, report: Any) -> None:
            if report.when == "call" or report.failed:
                node = report.nodeid.split("[", 1)[0]
                if report.failed:
                    outcomes[node] = "broken"
                else:
                    outcomes.setdefault(node, "held")

    nodes = sorted({a.test for a in attacks if not a.test.startswith("ci:")})
    pytest.main(["-q", "--no-cov", "-p", "no:cacheprovider", *nodes], plugins=[Collector()])
    results: dict[str, str] = {}
    for attack in attacks:
        if attack.test.startswith("ci:"):
            results[attack.id] = "checked in CI"
        else:
            results[attack.id] = outcomes.get(attack.test, "missing")
    return results


def render_redteam_report(attacks: list[Attack], results: dict[str, str], today: date) -> str:
    held = [a for a in attacks if results.get(a.id) in ("held", "checked in CI")]
    counts = coverage(attacks)
    lines = [
        f"# Red-team suite ({today.isoformat()})",
        "",
        "Hijacked mode: the fake LLM does what each attack asks; a pass means the code controls",
        "held. Mapped to the OWASP Top 10 for Agentic Applications.",
        "",
        f"**{len(held)} of {len(attacks)} attacks held.**"
        + (" Every attack held." if len(held) == len(attacks) else ""),
        "",
        "## Coverage",
        "",
        "| Risk | Name | Attacks |",
        "|---|---|---|",
    ]
    for asi in ASI_IDS:
        shown = (
            f"not applicable: {NOT_APPLICABLE[asi]}"
            if asi in NOT_APPLICABLE
            else counts.get(asi, 0)
        )
        lines.append(f"| {asi} | {ASI_NAMES[asi]} | {shown} |")
    lines += [
        "",
        "## Attacks",
        "",
        "| Id | Risk | Surface | Attack | Control | Result |",
        "|---|---|---|---|---|---|",
    ]
    for a in attacks:
        result = results.get(a.id, "missing")
        lines.append(f"| {a.id} | {a.asi} | {a.surface} | {a.payload} | {a.expect} | {result} |")
    return "\n".join(lines) + "\n"


def run_redteam_command(report: Path, root: Path, today: date | None = None) -> int:
    attacks = load_attacks(root / CATALOGUE)
    import claimlens.evals.redteam as module  # looked up at call time, so tests can replace it

    results = module.run_attacks(attacks, root)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        render_redteam_report(attacks, results, today or date.today()), encoding="utf-8"
    )
    broken = [a.id for a in attacks if results[a.id] not in ("held", "checked in CI")]
    print(f"{len(attacks) - len(broken)} of {len(attacks)} attacks held; report: {report}")
    for attack_id in broken:
        print(f"NOT HELD: {attack_id} ({results[attack_id]})")
    return 1 if broken else 0
