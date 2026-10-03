"""Near-copy photos from the claim memory are a fraud signal (M7, ADR 0019)."""

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from uuid import UUID

from PIL import Image

from claimlens.domain import Route
from claimlens.events.projection import fold
from claimlens.events.store import SQLiteEventStore
from claimlens.intake import submit_claim
from claimlens.knowledge.embed import FakeEmbedder
from claimlens.memory.index import ClaimMemory
from claimlens.workflow import PipelineDeps, process_claim
from tests.fakes import FakeDetector, make_test_deps
from tests.unit.test_memory_index import scene


def _deps(tmp_path: Path, store: SQLiteEventStore, memory: bool = True) -> PipelineDeps:
    deps = make_test_deps(tmp_path, store, FakeDetector())
    if not memory:
        return deps
    return replace(deps, memory=ClaimMemory(tmp_path / "memory", FakeEmbedder()))


def _claim(deps: PipelineDeps, photo: Path) -> UUID:
    claim = submit_claim(
        deps.store, deps.blobs, policy_id="P-1001", description="x", photo_paths=[photo]
    )
    process_claim(claim, deps)
    return claim


def _signals(store: SQLiteEventStore, claim: UUID) -> list[tuple[str, float, str]]:
    return [(s.kind, s.score, s.detail) for s in fold(store.load(claim)).fraud_signals]


def resaved(source: Path, target: Path) -> Path:
    """Re-saved as a lower-quality JPEG at full size (the quality gate needs 320 px or more)."""
    with Image.open(source) as image:
        image.convert("RGB").save(target, "JPEG", quality=60)
    return target


def _cropped(source: Path, target: Path, share: float = 0.02) -> Path:
    with Image.open(source) as image:
        w, h = image.size
        dx, dy = int(w * share / 2), int(h * share / 2)
        image.crop((dx, dy, w - dx, h - dy)).save(target)
    return target


def test_a_resaved_copy_from_another_claim_goes_to_fraud_review(
    tmp_path: Path, store: SQLiteEventStore
) -> None:
    deps = _deps(tmp_path, store)
    original = scene(tmp_path / "a.png")
    first = _claim(deps, original)
    second = _claim(deps, resaved(original, tmp_path / "b.jpg"))
    ((kind, score, detail),) = _signals(store, second)
    assert kind == "photo_near_copy"
    assert score == 0.8
    assert str(first) in detail
    assert "bits" in detail
    decision = fold(store.load(second)).decision
    assert decision is not None
    assert decision.route is Route.FRAUD_REVIEW
    assert decision.rule_id == "R1"


def test_a_lightly_cropped_copy_is_caught(tmp_path: Path, store: SQLiteEventStore) -> None:
    deps = _deps(tmp_path, store)
    original = scene(tmp_path / "a.png")
    _claim(deps, original)
    second = _claim(deps, _cropped(original, tmp_path / "c.png"))
    assert [k for k, _, _ in _signals(store, second)] == ["photo_near_copy"]


def test_a_different_photo_raises_nothing(tmp_path: Path, store: SQLiteEventStore) -> None:
    deps = _deps(tmp_path, store)
    _claim(deps, scene(tmp_path / "a.png"))
    second = _claim(deps, scene(tmp_path / "d.png", 60, flip=True))
    assert _signals(store, second) == []


def test_a_claims_own_photos_never_match(tmp_path: Path, store: SQLiteEventStore) -> None:
    deps = _deps(tmp_path, store)
    first = _claim(deps, scene(tmp_path / "a.png"))
    assert _signals(store, first) == []


def test_an_exact_copy_reports_only_photo_reuse(tmp_path: Path, store: SQLiteEventStore) -> None:
    deps = _deps(tmp_path, store)
    original = scene(tmp_path / "a.png")
    _claim(deps, original)
    second = _claim(deps, original)
    assert [k for k, _, _ in _signals(store, second)] == ["photo_reuse"]


def test_without_memory_nothing_changes(tmp_path: Path, store: SQLiteEventStore) -> None:
    deps = _deps(tmp_path, store, memory=False)
    original = scene(tmp_path / "a.png")
    _claim(deps, original)
    second = _claim(deps, resaved(original, tmp_path / "b.jpg"))
    assert _signals(store, second) == []


def test_a_forgotten_claim_no_longer_matches(
    tmp_path: Path, store: SQLiteEventStore, make_image: Callable[..., Path]
) -> None:
    deps = _deps(tmp_path, store)
    original = scene(tmp_path / "a.png")
    first = _claim(deps, original)
    assert deps.memory is not None
    deps.memory.forget(str(first))
    second = _claim(deps, resaved(original, tmp_path / "b.jpg"))
    assert _signals(store, second) == []


def test_a_broken_memory_keeps_the_exact_copy_signal(
    tmp_path: Path, store: SQLiteEventStore
) -> None:
    class Broken(ClaimMemory):
        def all(self) -> list:  # type: ignore[type-arg]
            raise OSError("lance table unreadable")

    deps = replace(
        make_test_deps(tmp_path, store, FakeDetector()),
        memory=Broken(tmp_path / "m", FakeEmbedder()),
    )
    original = scene(tmp_path / "a.png")
    _claim(deps, original)
    second = _claim(deps, original)
    assert [k for k, _, _ in _signals(store, second)] == ["photo_reuse"]
    decision = fold(store.load(second)).decision
    assert decision is not None
    assert decision.rule_id == "R1"


def test_a_remembered_claim_does_not_match_itself(tmp_path: Path, store: SQLiteEventStore) -> None:
    from claimlens.integrity import check_integrity

    deps = _deps(tmp_path, store)
    first = _claim(deps, scene(tmp_path / "a.png"))
    assert deps.memory is not None
    assert deps.memory.get(str(first)) is not None  # remembered, so a self-match is possible
    state = fold(store.load(first))
    assert check_integrity(state, store, deps.memory, deps.blobs.path) == ()
