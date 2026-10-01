import json
from collections import Counter
from pathlib import Path

import pytest

from claimlens.autolabel.job import load_autolabel_jobs, run_autolabel, select_sample
from claimlens.cli import main
from claimlens.data.records import Annotation, ImageRecord, write_records
from claimlens.data.taxonomy import load_part_groups
from tests.data_helpers import make_pattern_image

ROOT = Path(__file__).resolve().parents[2]
DOOR = Annotation(label="door", polygon=(0.1, 0.1, 0.5, 0.1, 0.5, 0.5), score=0.6)


class FakeLabeller:
    model_version = "fake-labeller"

    def __init__(self) -> None:
        self.seen: list[str] = []

    def label(self, image_path: Path) -> tuple[list[Annotation], Counter[str]]:
        self.seen.append(image_path.name)
        return [DOOR], Counter({"ambiguous_phrase": 1})


def _record(name: str, split: str) -> ImageRecord:
    return ImageRecord(
        image_id=f"s:{name}",
        source="s",
        source_split=split,
        path=f"img/{name}.jpg",
        width=320,
        height=240,
    )


def _repo(tmp_path: Path) -> list[ImageRecord]:
    records = [_record(f"t{i}", "test") for i in range(5)] + [_record("a", "train")]
    for record in records:
        make_pattern_image(tmp_path / record.path, seed=len(record.image_id))
    interim = tmp_path / "data" / "interim" / "damage-v1"
    write_records(interim / "records.jsonl", records)
    splits = {r.image_id: r.source_split for r in records}
    (interim / "splits.json").write_text(json.dumps(splits), encoding="utf-8")
    return records


def test_repo_job_config_loads_and_maps_to_part_groups() -> None:
    job = load_autolabel_jobs(ROOT / "config" / "autolabel.toml")["fusion-eval-v1"]
    groups = load_part_groups(ROOT / "config" / "taxonomy.toml")
    assert job.sample_size == 100
    assert set(job.prompts.values()) <= set(groups.classes)


def test_sample_is_deterministic_and_respects_split() -> None:
    records = [_record(f"t{i}", "test") for i in range(10)] + [_record("a", "train")]
    splits = {r.image_id: r.source_split for r in records}
    first = select_sample(records, splits, split="test", size=4, seed=1)
    again = select_sample(records, splits, split="test", size=4, seed=1)
    assert first == again
    assert len(first) == 4
    assert all(r.source_split == "test" for r in first)
    assert [r.image_id for r in first] == sorted(r.image_id for r in first)


def test_run_autolabel_writes_proposals_and_report(tmp_path: Path) -> None:
    _repo(tmp_path)
    job = load_autolabel_jobs(ROOT / "config" / "autolabel.toml")["fusion-eval-v1"]
    job = job.model_copy(update={"sample_size": 3})
    labeller = FakeLabeller()

    report = run_autolabel(
        job,
        repo_root=tmp_path,
        labeller=labeller,
        part_groups=load_part_groups(ROOT / "config" / "taxonomy.toml"),
    )

    assert len(labeller.seen) == 3
    proposals = tmp_path / "data" / "interim" / "fusion-eval-v1" / "autolabels.jsonl"
    assert len(proposals.read_text(encoding="utf-8").splitlines()) == 3
    assert report["model_version"] == "fake-labeller"
    assert report["proposals"] == {"door": 3}
    assert report["dropped"] == {"ambiguous_phrase": 3}
    assert (tmp_path / "reports" / "data" / "fusion-eval-v1-autolabel.json").is_file()


def test_labeller_output_outside_part_groups_is_rejected(tmp_path: Path) -> None:
    _repo(tmp_path)
    job = load_autolabel_jobs(ROOT / "config" / "autolabel.toml")["fusion-eval-v1"]

    class BadLabeller(FakeLabeller):
        def label(self, image_path: Path) -> tuple[list[Annotation], Counter[str]]:
            return [DOOR.model_copy(update={"label": "fender"})], Counter()

    with pytest.raises(ValueError, match="fender"):
        run_autolabel(
            job,
            repo_root=tmp_path,
            labeller=BadLabeller(),
            part_groups=load_part_groups(ROOT / "config" / "taxonomy.toml"),
        )


def test_autolabel_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _repo(tmp_path)
    monkeypatch.chdir(tmp_path)
    code = main(
        ["--config", str(ROOT / "config"), "data", "autolabel", "fusion-eval-v1"],
        labeller_factory=lambda _job: FakeLabeller(),
    )
    assert code == 0
    assert "Auto-labelled 5 images" in capsys.readouterr().out


def test_autolabel_unknown_job(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--config", str(ROOT / "config"), "data", "autolabel", "nope"]) == 2
    assert "unknown auto-label job 'nope'" in capsys.readouterr().err
