import json
from pathlib import Path

import pytest

from claimlens.cli import main
from claimlens.knowledge.embed import FakeEmbedder
from tests.fakes import CONFIG_DIR

pytest.importorskip("lancedb")
ROOT = Path(__file__).resolve().parents[2]


def _run(tmp_path: Path, *args: str) -> int:
    return main(
        [
            "--config",
            str(CONFIG_DIR),
            "knowledge",
            *args,
            "--root",
            str(ROOT),
            "--index",
            str(tmp_path / "idx"),
        ],
        embedder_factory=FakeEmbedder,
    )


def test_build_then_search_with_a_policy(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _run(tmp_path, "build") == 0
    assert "clauses" in capsys.readouterr().out
    assert _run(tmp_path, "search", "rental car after a collision", "--policy", "P-1004") == 0
    out = capsys.readouterr().out
    assert "BAS-6.1" in out


def test_eval_writes_a_report(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert _run(tmp_path, "build") == 0
    report = tmp_path / "report.md"
    assert _run(tmp_path, "eval", "--out", str(report)) == 0
    text = report.read_text(encoding="utf-8")
    assert "recall@5" in text
    assert (
        len(
            json.loads(
                "["
                + ",".join(
                    (ROOT / "evals/knowledge/questions.jsonl")
                    .read_text(encoding="utf-8")
                    .split("\n")[:-1]
                )
                + "]"
            )
        )
        == 10
    )


def test_search_without_an_index_is_a_clear_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _run(tmp_path, "search", "anything") == 1
    assert "knowledge build" in capsys.readouterr().err
