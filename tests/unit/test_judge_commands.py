import json
from datetime import date
from pathlib import Path

import pytest

from claimlens.cli import main
from claimlens.evals.judge_export import save_case
from claimlens.llm.budget import Budget
from claimlens.llm.cache import ResponseCache
from claimlens.llm.config import load_llm_config
from claimlens.llm.gateway import Gateway
from claimlens.llm.provider import FakeProvider, ProviderReply
from tests.unit.test_judge import ITEM

ROOT = Path(__file__).resolve().parents[2]
LLM = load_llm_config(ROOT / "config" / "llm.toml")


def _answer(ok: bool) -> ProviderReply:
    body = {
        "grounded": ok,
        "grounded_reason": "r",
        "citation_relevant": True,
        "citation_relevant_reason": "r",
        "actionable": True,
        "actionable_reason": "r",
    }
    return ProviderReply(json.dumps(body), 900, 80)


def _run_dir(tmp_path: Path, n: int) -> Path:
    run = tmp_path / "run"
    for i in range(n):
        prefix = "n" if i % 2 else "g"
        # Distinct evidence per case, so the response cache does not answer one from another.
        item = ITEM.model_copy(
            update={"case_id": f"{prefix}{i:03d}", "evidence": f"{ITEM.evidence} (case {i})"}
        )
        save_case(run, item)
    return run


def _factory(tmp_path: Path, script: list[ProviderReply]):  # type: ignore[no-untyped-def]
    def make(per_day_usd: float | None) -> Gateway:
        return Gateway(
            LLM,
            FakeProvider(list[ProviderReply | Exception](script)),
            Budget(tmp_path / "b.sqlite", LLM.limits, today=lambda: date(2026, 10, 3)),
            ResponseCache(tmp_path / "c.sqlite"),
            lambda call: None,
            sleep=lambda s: None,
        )

    return make


def test_judge_run_writes_judgements(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    run = _run_dir(tmp_path, 4)
    out = tmp_path / "judgements.jsonl"
    script = [_answer(True), _answer(False), _answer(True), _answer(True)]
    code = main(
        ["judge", "run", "--run-dir", str(run), "--out", str(out)],
        judge_gateway_factory=_factory(tmp_path, script),
    )
    assert code == 0
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert [r["verdict"] for r in rows] == ["pass", "fail", "pass", "pass"]
    assert "Judge pass rate: 0.75 (3/4)" in capsys.readouterr().out


def test_export_then_agreement(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    run = _run_dir(tmp_path, 6)
    judgements = tmp_path / "judgements.jsonl"
    script = [_answer(i % 3 != 0) for i in range(6)]
    main(
        ["judge", "run", "--run-dir", str(run), "--out", str(judgements)],
        judge_gateway_factory=_factory(tmp_path, script),
    )
    page = tmp_path / "labels.html"
    assert (
        main(
            [
                "judge",
                "export",
                "--judgements",
                str(judgements),
                "--run-dir",
                str(run),
                "--out",
                str(page),
            ]
        )
        == 0
    )
    sample = json.loads((tmp_path / "sample.json").read_text(encoding="utf-8"))
    assert len(sample) == 6
    verdicts = {
        json.loads(line)["case_id"]: json.loads(line)["verdict"]
        for line in judgements.read_text(encoding="utf-8").splitlines()
    }
    labels = tmp_path / "judge-labels.json"
    labels.write_text(
        json.dumps({"labeller": "owner", "labels": {i: {"label": verdicts[i]} for i in sample}}),
        encoding="utf-8",
    )
    code = main(["judge", "agreement", "--judgements", str(judgements), "--labels", str(labels)])
    out = capsys.readouterr().out
    assert code == 0
    assert "Agreement: 1.00" in out
    assert "kappa 1.00" in out


def test_agreement_refuses_labels_for_other_items(tmp_path: Path) -> None:
    run = _run_dir(tmp_path, 4)
    judgements = tmp_path / "judgements.jsonl"
    main(
        ["judge", "run", "--run-dir", str(run), "--out", str(judgements)],
        judge_gateway_factory=_factory(tmp_path, [_answer(True)] * 4),
    )
    main(
        [
            "judge",
            "export",
            "--judgements",
            str(judgements),
            "--run-dir",
            str(run),
            "--out",
            str(tmp_path / "p.html"),
        ]
    )
    labels = tmp_path / "judge-labels.json"
    labels.write_text(json.dumps({"labeller": "o", "labels": {"zzz": {"label": "pass"}}}))
    with pytest.raises(ValueError, match="missing labels"):
        main(["judge", "agreement", "--judgements", str(judgements), "--labels", str(labels)])


def test_agreement_below_the_bar_exits_1(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run = _run_dir(tmp_path, 4)
    judgements = tmp_path / "judgements.jsonl"
    main(
        ["judge", "run", "--run-dir", str(run), "--out", str(judgements)],
        judge_gateway_factory=_factory(tmp_path, [_answer(True), _answer(False)] * 2),
    )
    main(
        [
            "judge",
            "export",
            "--judgements",
            str(judgements),
            "--run-dir",
            str(run),
            "--out",
            str(tmp_path / "p.html"),
        ]
    )
    sample = json.loads((tmp_path / "sample.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in judgements.read_text(encoding="utf-8").splitlines()]
    flipped = {r["case_id"]: "fail" if r["verdict"] == "pass" else "pass" for r in rows}
    labels = tmp_path / "judge-labels.json"
    labels.write_text(
        json.dumps({"labeller": "o", "labels": {i: {"label": flipped[i]} for i in sample}})
    )
    assert (
        main(["judge", "agreement", "--judgements", str(judgements), "--labels", str(labels)]) == 1
    )
    assert "below the bar" in capsys.readouterr().out


def test_eval_triage_can_save_the_run(tmp_path: Path, make_image) -> None:  # type: ignore[no-untyped-def]
    from tests.fakes import CONFIG_DIR, FakeDetector
    from tests.unit.test_cli import _AdvisingAgent

    golden = tmp_path / "g.jsonl"
    golden.write_text(
        '{"case_id":"x1","scenario":"s","policy_id":"P-1001","description":"d",'
        f'"photos":["{make_image("a.jpg").as_posix()}"],"expected_route":"ADJUSTER_REVIEW",'
        '"label_source":"t"}\n',
        encoding="utf-8",
    )
    run = tmp_path / "run"
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
            "--save-run",
            str(run),
        ],
        detector_factory=lambda *_: FakeDetector(),
        agent_factory=lambda args, path: _AdvisingAgent(),
    )
    assert code == 0
    saved = json.loads((run / "x1.json").read_text(encoding="utf-8"))
    assert set(saved) == {"case_id", "evidence", "recommendation", "clauses"}
    assert saved["clauses"]["STD-8.5"].startswith("We do not cover damage")
    assert "<claimant_description>d</claimant_description>" in saved["evidence"]


def test_a_validated_judge_adds_its_pass_rate_to_the_scorecard(tmp_path: Path) -> None:
    from claimlens.evals.scorecard import Scorecard

    run = _run_dir(tmp_path, 4)
    judgements = tmp_path / "judgements.jsonl"
    main(
        ["judge", "run", "--run-dir", str(run), "--out", str(judgements)],
        judge_gateway_factory=_factory(tmp_path, [_answer(True), _answer(False)] * 2),
    )
    main(
        [
            "judge",
            "export",
            "--judgements",
            str(judgements),
            "--run-dir",
            str(run),
            "--out",
            str(tmp_path / "p.html"),
        ]
    )
    sample = json.loads((tmp_path / "sample.json").read_text(encoding="utf-8"))
    rows = {
        json.loads(line)["case_id"]: json.loads(line)["verdict"]
        for line in judgements.read_text(encoding="utf-8").splitlines()
    }
    labels = tmp_path / "judge-labels.json"
    labels.write_text(
        json.dumps({"labeller": "o", "labels": {i: {"label": rows[i]} for i in sample}})
    )
    card = tmp_path / "current.json"
    card.write_text(
        Scorecard(
            created_on=date(2026, 10, 3),
            golden="g",
            versions={},
            metrics={"judge_pass_rate": None},
            fingerprints={},
        ).model_dump_json(),
        encoding="utf-8",
    )
    args = [
        "judge",
        "agreement",
        "--judgements",
        str(judgements),
        "--labels",
        str(labels),
        "--scorecard",
        str(card),
    ]
    assert main(args) == 0
    saved = Scorecard.model_validate_json(card.read_text(encoding="utf-8"))
    assert saved.metrics["judge_pass_rate"] == 0.5
    assert saved.versions["judge"] == "judge/v1"
