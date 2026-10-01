import importlib.util
import json
from pathlib import Path

import pytest

from claimlens.cli import main
from claimlens.data.records import Annotation, ImageRecord, read_records, write_records
from claimlens.domain import Route
from claimlens.evals.golden import GoldenClaim, load_golden, write_golden
from claimlens.review.decisions import GoldenReview, PartReview, write_review

ROOT = Path(__file__).resolve().parents[2]
SQUARE = (0.1, 0.1, 0.5, 0.1, 0.5, 0.5)


def test_apply_parts_writes_the_eval_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    record = ImageRecord(
        image_id="s:a",
        source="s",
        source_split="test",
        path="a.jpg",
        width=1,
        height=1,
        annotations=(
            Annotation(label="door", polygon=SQUARE, score=0.7),
            Annotation(label="hood", polygon=SQUARE, score=0.4),
        ),
    )
    write_records(tmp_path / "data" / "interim" / "fusion-eval-v1" / "autolabels.jsonl", [record])
    write_review(
        tmp_path / "reviews" / "fusion-eval-v1.json",
        PartReview(
            job_id="fusion-eval-v1",
            reviewer="me",
            decisions={"s:a#0": "approved", "s:a#1": "rejected"},
        ),
    )
    monkeypatch.chdir(tmp_path)

    assert main(["--config", str(ROOT / "config"), "review", "apply", "parts"]) == 0

    kept = read_records(tmp_path / "data" / "processed" / "fusion-eval-v1" / "parts.jsonl")
    assert [a.label for a in kept[0].annotations] == ["door"]
    report_path = tmp_path / "reports" / "data" / "fusion-eval-v1-review.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["counts"] == {"approved": 1, "rejected": 1}
    assert report["images"] == 1
    assert "1 approved" in capsys.readouterr().out


def test_apply_golden_marks_reviewed_cases(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    golden = tmp_path / "evals" / "golden" / "v1" / "claims.jsonl"
    case = GoldenClaim(
        case_id="g1",
        scenario="s",
        policy_id="P-1001",
        description="",
        photos=("x.jpg",),
        expected_route=Route.FAST_TRACK,
        label_source="oracle",
    )
    write_golden(golden, [case])
    write_review(
        tmp_path / "reviews" / "golden-v1.json",
        GoldenReview(reviewer="me", decisions={"g1": "agree"}),
    )
    monkeypatch.chdir(tmp_path)

    assert main(["review", "apply", "golden", "--golden", str(golden)]) == 0
    assert load_golden(golden)[0].reviewed


def test_apply_without_a_review_file_is_a_clear_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    assert main(["review", "apply", "parts"]) == 1
    assert "error:" in capsys.readouterr().err


@pytest.mark.skipif(
    importlib.util.find_spec("fiftyone") is not None, reason="FiftyOne is installed"
)
def test_review_launch_without_fiftyone(capsys: pytest.CaptureFixture[str]) -> None:
    golden = ROOT / "evals" / "golden" / "v0" / "claims.jsonl"
    assert main(["review", "launch", "golden", "--golden", str(golden)]) == 2
    assert "uv sync --group review" in capsys.readouterr().err
