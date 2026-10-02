from datetime import date
from pathlib import Path

import pytest

from claimlens.training.manifest import ModelReport, SplitMetrics, write_json
from claimlens.training.select import (
    load_model_reports,
    load_models_config,
    render_model_report,
    select_champion,
    write_models_config,
)


def _split(mask_map50: float) -> SplitMetrics:
    return SplitMetrics(
        box_map50=0.6,
        box_map50_95=0.4,
        mask_map50=mask_map50,
        mask_map50_95=0.3,
        per_class_mask_map50={"dent": 0.5, "scratch": 0.4},
    )


def _report(run: str, val: float, test: float, status: str = "complete") -> ModelReport:
    return ModelReport(
        run=run,
        base_model=f"{run}.pt",
        commit="c",
        dataset="damage-v1",
        dataset_md5="abc.dir",
        status=status,
        epochs_run=50,
        best_epoch=40,
        val=_split(val),
        test=_split(test),
        mlflow_run_id="id",
        model_version="1",
        cpu_ms_per_image=120.0,
    )


def test_champion_is_chosen_on_validation_not_test() -> None:
    chosen = select_champion([_report("n", val=0.50, test=0.70), _report("s", val=0.60, test=0.40)])
    assert chosen.run == "s"


def test_incomplete_runs_are_never_chosen() -> None:
    reports = [_report("n", 0.5, 0.5), _report("s", 0.9, 0.9, status="incomplete")]
    assert select_champion(reports).run == "n"


def test_no_complete_run_is_an_error() -> None:
    with pytest.raises(ValueError, match="no complete run"):
        select_champion([_report("s", 0.9, 0.9, status="incomplete")])


def test_models_config_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "models.toml"
    assert load_models_config(path) is None
    written = write_models_config(path, _report("s", 0.6, 0.5))
    assert written.damage.weights == "models/damage/s/best.pt"
    assert load_models_config(path) == written


def test_reports_are_loaded_in_name_order(tmp_path: Path) -> None:
    write_json(tmp_path / "s.json", _report("s", 0.6, 0.5))
    write_json(tmp_path / "n.json", _report("n", 0.5, 0.5))
    assert [r.run for r in load_model_reports(tmp_path)] == ["n", "s"]


def test_report_names_the_champion_and_both_runs() -> None:
    reports = [_report("n", 0.50, 0.48), _report("s", 0.60, 0.55)]
    text = render_model_report(reports, reports[1], date(2026, 10, 2))
    assert "Champion: `s`" in text
    assert "| n |" in text
    assert "| s |" in text
    assert "0.550" in text
    assert "chosen on validation" in text.lower()
    assert "| dent |" in text
