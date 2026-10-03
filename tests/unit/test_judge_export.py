from pathlib import Path

import pytest

from claimlens.evals.judge import JudgeAnswer, Judgement
from claimlens.evals.judge_export import render_label_page, sample_for_labelling
from tests.unit.test_judge import ITEM

ANSWER = JudgeAnswer(
    grounded=True,
    grounded_reason="",
    citation_relevant=True,
    citation_relevant_reason="",
    actionable=True,
    actionable_reason="",
)


def _judgements(n: int) -> list[Judgement]:
    return [
        Judgement(
            case_id=f"c{i:03d}",
            verdict="pass" if i % 3 else "fail",
            answer=ANSWER,
            judge_version="judge/v1",
            cost_usd=0.0,
        )
        for i in range(n)
    ]


def test_sample_is_deterministic_stratified_and_sized() -> None:
    judgements = _judgements(150)
    narrative = {f"c{i:03d}" for i in range(97, 150)}
    first = sample_for_labelling(judgements, narrative)
    assert first == sample_for_labelling(judgements, narrative)
    assert len(first) == 50
    assert len(set(first)) == 50
    fails = sum(1 for j in judgements if j.case_id in first and j.verdict == "fail")
    assert fails >= 15  # judge-fail items are over-sampled so both classes are well represented
    assert sum(1 for c in first if c in narrative) >= 20


def test_sample_takes_everything_when_fewer_than_n() -> None:
    assert len(sample_for_labelling(_judgements(12), set())) == 12


def test_label_page_hides_the_judge_and_escapes_text() -> None:
    item = ITEM.model_copy(update={"evidence": "<script>alert(1)</script>"})
    page = render_label_page([item])
    assert "<script>alert(1)</script>" not in page
    assert "&lt;script&gt;" in page
    assert "verdict" not in page.lower()
    assert "grounded" not in page.lower()
    assert "Download labels" in page
    assert item.case_id in page


def test_saved_runs_round_trip_to_judge_items(tmp_path: Path) -> None:
    from claimlens.evals.judge_export import load_run, save_case

    save_case(tmp_path, ITEM)
    assert load_run(tmp_path) == [ITEM]
    raw = (tmp_path / "n001.json").read_text(encoding="utf-8")
    assert "expected" not in raw


def test_labels_file_is_parsed_and_checked(tmp_path: Path) -> None:
    from claimlens.evals.judge_export import load_labels

    path = tmp_path / "labels.json"
    path.write_text(
        '{"labeller": "owner", "labels": {"a": {"label": "pass", "note": ""},'
        ' "b": {"label": "fail", "note": "made-up fact"}}}',
        encoding="utf-8",
    )
    assert load_labels(path) == {"a": "pass", "b": "fail"}
    path.write_text('{"labeller": "x", "labels": {"a": {"label": "maybe"}}}', encoding="utf-8")
    with pytest.raises(ValueError, match="pass or fail"):
        load_labels(path)


def test_each_export_keeps_labels_in_its_own_browser_storage() -> None:
    other = ITEM.model_copy(update={"case_id": "n009"})
    first, second = render_label_page([ITEM]), render_label_page([other])
    key = 'const KEY = "claimlens-labels-'
    assert first.split(key)[1][:12] != second.split(key)[1][:12]
    assert render_label_page([ITEM]).split(key)[1][:12] == first.split(key)[1][:12]
