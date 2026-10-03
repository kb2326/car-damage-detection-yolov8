import io
from collections.abc import Callable
from pathlib import Path
from uuid import uuid4

import pytest
from PIL import Image, ImageDraw

from claimlens.events.store import SQLiteEventStore
from claimlens.knowledge.embed import FakeEmbedder
from claimlens.memory.index import NEAR_COPY_DISTANCE, ClaimMemory, rebuild
from claimlens.memory.records import MemoryRecord, phash_of


def scene(path: Path, shade: int = 0, flip: bool = False) -> Path:
    image = Image.new("RGB", (640, 480), (90 + shade, 120, 160))
    draw = ImageDraw.Draw(image)
    draw.rectangle((60, 200, 580, 400), fill=(200, 30, 30))
    draw.ellipse((120, 330, 220, 430), fill=(20, 20, 20))
    draw.ellipse((420, 330, 520, 430), fill=(20, 20, 20))
    draw.rectangle((150, 120, 470, 210), fill=(180, 40, 40))
    if flip:
        image = image.rotate(180)
    image.save(path)
    return path


def resaved(source: Path, target: Path) -> Path:
    with Image.open(source) as image:
        small = image.convert("RGB").resize((384, 288))
    buffer = io.BytesIO()
    small.save(buffer, "JPEG", quality=60)
    target.write_bytes(buffer.getvalue())
    return target


def record(
    phashes: tuple[str, ...] = (),
    policy: str = "P-1001",
    damage: str = "dent on door",
    route: str = "FAST_TRACK",
    claim_id: str | None = None,
) -> MemoryRecord:
    return MemoryRecord(
        claim_id=claim_id or str(uuid4()),
        policy_id=policy,
        decided_on="2026-10-03",
        route=route,
        rule_id="R9",
        damage=damage,
        cost_low=150,
        cost_high=400,
        photo_phashes=phashes,
        summary=f"{damage}; estimate $150-$400; routed {route}",
        source_seq=(5, 7),
    )


@pytest.fixture
def memory(tmp_path: Path) -> ClaimMemory:
    return ClaimMemory(tmp_path / "memory", FakeEmbedder())


def test_upsert_keeps_one_row_per_claim(memory: ClaimMemory) -> None:
    first = record()
    memory.upsert(first)
    memory.upsert(first.model_copy(update={"review_action": "approve"}))
    rows = memory.all()
    assert len(rows) == 1
    assert rows[0].review_action == "approve"
    assert memory.get(first.claim_id) == rows[0]


def test_forget_removes_the_record(memory: ClaimMemory) -> None:
    gone = record()
    memory.upsert(gone)
    assert memory.forget(gone.claim_id)
    assert memory.get(gone.claim_id) is None
    assert not memory.forget(gone.claim_id)


def test_a_bad_claim_id_is_refused(memory: ClaimMemory) -> None:
    with pytest.raises(ValueError, match="not a claim id"):
        memory.get("x' OR '1'='1")


def test_a_resaved_photo_is_found_as_a_near_copy(memory: ClaimMemory, tmp_path: Path) -> None:
    original = scene(tmp_path / "a.png")
    earlier = record(phashes=(phash_of(original),), policy="P-2002", route="ADJUSTER_REVIEW")
    unrelated = record(
        phashes=(phash_of(scene(tmp_path / "b.png", 60, flip=True)),),
        policy="P-1003",
        damage="glass_shatter on glass",
    )
    memory.upsert(earlier)
    memory.upsert(unrelated)
    now = record(phashes=(phash_of(resaved(original, tmp_path / "c.jpg")),), policy="P-1005")
    found = memory.similar(now)
    assert [s.claim_id for s in found] == [earlier.claim_id]
    assert found[0].reason == "near-copy photo"
    assert found[0].distance is not None
    assert found[0].distance <= NEAR_COPY_DISTANCE
    assert found[0].route == "ADJUSTER_REVIEW"


def test_same_policy_and_similar_damage(memory: ClaimMemory) -> None:
    same_policy = record(policy="P-1001", damage="scratch on hood")
    similar_damage = record(policy="P-1003", damage="dent on door")
    other = record(policy="P-1006", damage="glass_shatter on glass")
    for r in (same_policy, similar_damage, other):
        memory.upsert(r)
    now = record(policy="P-1001", damage="dent on door")
    found = {s.claim_id: s.reason for s in memory.similar(now)}
    assert found[same_policy.claim_id] == "same policy"
    assert found[similar_damage.claim_id] == "similar damage"
    assert other.claim_id not in found


def test_the_claim_itself_is_excluded_and_the_limit_applied(memory: ClaimMemory) -> None:
    now = record(policy="P-1001")
    memory.upsert(now)
    for _ in range(8):
        memory.upsert(record(policy="P-1001"))
    found = memory.similar(now, limit=5)
    assert len(found) == 5
    assert now.claim_id not in {s.claim_id for s in found}


def test_rebuild_from_the_event_logs(
    tmp_path: Path, store: SQLiteEventStore, make_image: Callable[..., Path]
) -> None:
    from claimlens.intake import submit_claim
    from claimlens.workflow import process_claim
    from tests.fakes import FakeDetector, make_test_deps

    deps = make_test_deps(tmp_path, store, FakeDetector())
    for n in range(3):
        claim = submit_claim(
            store,
            deps.blobs,
            policy_id="P-1001",
            description="x",
            photo_paths=[make_image(f"{n}.jpg", color=(10 * n, 50, 50))],
        )
        process_claim(claim, deps)
    memory = ClaimMemory(tmp_path / "memory", FakeEmbedder())
    assert rebuild(memory, store, deps.blobs.path) == 3
    assert len(memory.all()) == 3
