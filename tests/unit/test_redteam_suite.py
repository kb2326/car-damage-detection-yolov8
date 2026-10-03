"""The red-team catalogue stays mapped: each attack points at a real test; every risk is covered."""

import re
from datetime import date
from pathlib import Path

import pytest

from claimlens.evals.redteam import (
    ASI_IDS,
    NOT_APPLICABLE,
    Attack,
    coverage,
    load_attacks,
    render_redteam_report,
)

ROOT = Path(__file__).resolve().parents[2]
CATALOGUE = ROOT / "evals" / "redteam" / "attacks.toml"


def test_every_attack_points_at_a_real_test() -> None:
    for attack in load_attacks(CATALOGUE):
        if attack.test.startswith("ci:"):
            continue
        path, _, name = attack.test.partition("::")
        source = ROOT / path
        assert source.is_file(), f"{attack.id}: no file {path}"
        assert re.search(rf"^def {re.escape(name)}\(", source.read_text(encoding="utf-8"), re.M), (
            f"{attack.id}: {path} has no test {name}"
        )


def test_every_risk_is_covered() -> None:
    counts = coverage(load_attacks(CATALOGUE))
    for asi in ASI_IDS:
        if asi in NOT_APPLICABLE:
            assert counts.get(asi, 0) == 0
            continue
        assert counts.get(asi, 0) >= 1, f"no attack covers {asi}"


def test_ids_are_unique_and_well_formed() -> None:
    attacks = load_attacks(CATALOGUE)
    ids = [a.id for a in attacks]
    assert len(ids) == len(set(ids))
    assert len(ids) >= 25
    for attack in attacks:
        assert re.fullmatch(r"RT-\d{2}", attack.id)
        assert attack.asi in ASI_IDS
        assert attack.payload
        assert attack.expect


def test_an_unknown_risk_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "attacks.toml"
    path.write_text(
        '[[attack]]\nid = "RT-01"\nasi = "ASI11"\nsurface = "x"\npayload = "p"\n'
        'expect = "e"\ntest = "tests/x.py::t"\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="ASI11"):
        load_attacks(path)


def test_the_report_lists_every_attack_and_the_coverage() -> None:
    attacks = [
        Attack(id="RT-01", asi="ASI01", surface="story", payload="p", expect="e", test="t::a"),
        Attack(
            id="RT-02", asi="ASI04", surface="deps", payload="p", expect="e", test="ci:pip-audit"
        ),
    ]
    report = render_redteam_report(
        attacks, {"RT-01": "held", "RT-02": "checked in CI"}, date(2026, 10, 3)
    )
    assert "RT-01" in report
    assert "RT-02" in report
    assert "| ASI07 |" in report
    assert "not applicable" in report.lower()
    assert "1 of 1 attacks run held" in report


def test_eval_redteam_writes_a_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from claimlens import cli

    monkeypatch.setattr(
        "claimlens.evals.redteam.run_attacks",
        lambda attacks, root: {a.id: "held" for a in attacks},
    )
    report = tmp_path / "redteam.md"
    assert cli.main(["eval-redteam", "--report", str(report)]) == 0
    assert "every attack run held" in report.read_text(encoding="utf-8").lower()


def test_a_broken_attack_fails_the_command(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from claimlens import cli

    monkeypatch.setattr(
        "claimlens.evals.redteam.run_attacks",
        lambda attacks, root: {a.id: ("broken" if a.id == "RT-01" else "held") for a in attacks},
    )
    assert cli.main(["eval-redteam", "--report", str(tmp_path / "r.md")]) == 1


def test_the_live_injection_set_is_well_formed() -> None:
    import json

    from claimlens.evals.golden import GoldenClaim

    path = ROOT / "evals" / "redteam" / "injections.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    assert len(rows) == 15
    assert {r["expected_route"] for r in rows} <= {"ADJUSTER_REVIEW", "FRAUD_REVIEW"}
    styles = {r["notes"].removeprefix("red-team injection: ") for r in rows}
    assert len(styles) == 5
    for row in rows:
        GoldenClaim.model_validate(row)


def test_a_skipped_or_xfailed_attack_test_is_not_held(tmp_path: Path) -> None:
    from claimlens.evals.redteam import run_attacks

    module = tmp_path / "test_fake_attacks.py"
    module.write_text(
        "import pytest\n\n"
        "def test_skips():\n    pytest.skip('optional dependency missing')\n\n"
        "@pytest.mark.xfail\ndef test_xfails():\n    assert False\n\n"
        "def test_passes():\n    assert True\n",
        encoding="utf-8",
    )
    path = module.as_posix()
    attacks = [
        Attack(
            id="RT-01",
            asi="ASI01",
            surface="s",
            payload="p",
            expect="e",
            test=f"{path}::test_skips",
        ),
        Attack(
            id="RT-02",
            asi="ASI01",
            surface="s",
            payload="p",
            expect="e",
            test=f"{path}::test_xfails",
        ),
        Attack(
            id="RT-03",
            asi="ASI01",
            surface="s",
            payload="p",
            expect="e",
            test=f"{path}::test_passes",
        ),
    ]
    results = run_attacks(attacks, tmp_path)
    assert results == {"RT-01": "skipped", "RT-02": "skipped", "RT-03": "held"}


def test_the_report_separates_ci_checks_from_attacks_run() -> None:
    attacks = [
        Attack(id="RT-01", asi="ASI01", surface="story", payload="p", expect="e", test="t::a"),
        Attack(
            id="RT-02", asi="ASI04", surface="deps", payload="p", expect="e", test="ci:pip-audit"
        ),
    ]
    report = render_redteam_report(
        attacks, {"RT-01": "held", "RT-02": "checked in CI"}, date(2026, 10, 3)
    )
    assert "1 of 1 attacks run held" in report
    assert "1 checked in CI" in report
