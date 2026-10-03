"""The workflow writes memory after deciding; reviews update it; the CLI manages it."""

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from uuid import UUID

import pytest

from claimlens.events.payloads import HumanReviewed, ReviewAction
from claimlens.events.projection import fold
from claimlens.events.store import SQLiteEventStore
from claimlens.intake import submit_claim
from claimlens.knowledge.embed import FakeEmbedder
from claimlens.memory.index import ClaimMemory
from claimlens.review_queue import record_review
from claimlens.workflow import PipelineDeps, process_claim
from tests.fakes import FakeDetector, make_test_deps


def _deps(tmp_path: Path, store: SQLiteEventStore) -> tuple[PipelineDeps, ClaimMemory]:
    memory = ClaimMemory(tmp_path / "memory", FakeEmbedder())
    return replace(make_test_deps(tmp_path, store, FakeDetector()), memory=memory), memory


def _claim(deps: PipelineDeps, make_image: Callable[..., Path], name: str = "a.jpg") -> UUID:
    return submit_claim(
        deps.store, deps.blobs, policy_id="P-1001", description="x", photo_paths=[make_image(name)]
    )


def _types(store: SQLiteEventStore, claim: UUID) -> list[str]:
    return [e.type for e in store.load(claim)]


def test_a_decided_claim_is_remembered(
    tmp_path: Path, store: SQLiteEventStore, make_image: Callable[..., Path]
) -> None:
    deps, memory = _deps(tmp_path, store)
    claim = _claim(deps, make_image)
    process_claim(claim, deps)
    assert _types(store, claim)[-2:] == ["RouteDecided", "MemoryWritten"]
    record = memory.get(str(claim))
    assert record is not None
    assert record.route == "FAST_TRACK"


def test_processing_again_writes_nothing_new(
    tmp_path: Path, store: SQLiteEventStore, make_image: Callable[..., Path]
) -> None:
    deps, _ = _deps(tmp_path, store)
    claim = _claim(deps, make_image)
    process_claim(claim, deps)
    process_claim(claim, deps)
    assert _types(store, claim).count("MemoryWritten") == 1


def test_a_missing_memory_write_is_filled_in_on_the_next_run(
    tmp_path: Path, store: SQLiteEventStore, make_image: Callable[..., Path]
) -> None:
    deps, memory = _deps(tmp_path, store)
    claim = _claim(deps, make_image)
    process_claim(claim, replace(deps, memory=None))  # decided, but memory was not written
    assert memory.get(str(claim)) is None
    process_claim(claim, deps)
    assert memory.get(str(claim)) is not None


def test_a_failing_memory_never_changes_the_decision(
    tmp_path: Path, store: SQLiteEventStore, make_image: Callable[..., Path]
) -> None:
    class Broken(ClaimMemory):
        def upsert(self, record: object) -> None:
            raise OSError("disk full")

    deps = replace(
        make_test_deps(tmp_path, store, FakeDetector()),
        memory=Broken(tmp_path / "m", FakeEmbedder()),
    )
    claim = _claim(deps, make_image)
    decision = process_claim(claim, deps)
    state = fold(store.load(claim))
    assert decision.rule_id == "R9"
    assert state.decision == decision
    assert any(f.stage == "memory" and "disk full" in f.error for f in state.failures)


def test_a_review_updates_memory(
    tmp_path: Path, store: SQLiteEventStore, make_image: Callable[..., Path]
) -> None:
    deps, memory = _deps(tmp_path, store)
    claim = _claim(deps, make_image)
    process_claim(claim, deps)
    review = HumanReviewed(reviewer="sam", action=ReviewAction.DENY, note="staged")
    record_review(store, claim, review, memory=memory, photo_path=deps.blobs.path)
    record = memory.get(str(claim))
    assert record is not None
    assert record.review_action == "deny"
    assert _types(store, claim).count("MemoryWritten") == 2


def test_the_memory_commands(
    tmp_path: Path,
    store: SQLiteEventStore,
    make_image: Callable[..., Path],
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from claimlens.cli import main

    deps = make_test_deps(tmp_path, store, FakeDetector())
    claim = _claim(deps, make_image)
    process_claim(claim, deps)
    store.close()
    monkeypatch.chdir(tmp_path)
    base = ["--db", str(tmp_path / "events.db"), "--blobs", str(tmp_path / "blobs")]
    embed = {"embedder_factory": FakeEmbedder}
    assert main([*base, "memory", "rebuild"], **embed) == 0  # type: ignore[arg-type]
    assert "Remembered 1 claim" in capsys.readouterr().out
    assert main([*base, "memory", "show", str(claim)], **embed) == 0  # type: ignore[arg-type]
    assert "FAST_TRACK" in capsys.readouterr().out
    assert main([*base, "memory", "forget", str(claim)], **embed) == 0  # type: ignore[arg-type]
    assert main([*base, "memory", "show", str(claim)], **embed) == 1  # type: ignore[arg-type]
    reopened = SQLiteEventStore(tmp_path / "events.db")
    assert _types(reopened, claim)[-1] == "MemoryForgotten"
    reopened.close()


def test_run_with_memory_remembers_the_claim(
    tmp_path: Path,
    make_image: Callable[..., Path],
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from claimlens.cli import main
    from tests.fakes import CONFIG_DIR

    monkeypatch.chdir(tmp_path)
    base = [
        "--db",
        str(tmp_path / "c.db"),
        "--blobs",
        str(tmp_path / "b"),
        "--config",
        str(CONFIG_DIR),
    ]
    photo = str(make_image("a.jpg"))
    code = main(
        [*base, "--memory", "run", "--policy", "P-1001", photo],
        detector_factory=lambda *_: FakeDetector(),
        embedder_factory=FakeEmbedder,
    )
    assert code == 0
    memory = ClaimMemory(tmp_path / "var" / "memory", FakeEmbedder())
    assert len(memory.all()) == 1
    assert (
        main(
            [*base, "run", "--policy", "P-1001", photo], detector_factory=lambda *_: FakeDetector()
        )
        == 0
    )
    assert len(memory.all()) == 1  # without --memory nothing is remembered
