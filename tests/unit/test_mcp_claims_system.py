from pathlib import Path
from uuid import UUID, uuid4

from claimlens.domain import BoundingBox, DamageFinding, DamageType
from claimlens.events.envelope import Actor, ActorKind
from claimlens.events.payloads import ClaimReported, DamageDetected
from claimlens.events.projection import fold
from claimlens.events.store import SQLiteEventStore
from claimlens.mcp.claims_system import build_claims_system, event_audit
from claimlens.mcp.profiles import load_profiles
from tests.mcp_helpers import MemoryAudit, call, error_text, tool_names

ROOT = Path(__file__).resolve().parents[2]
PROFILES = load_profiles(ROOT / "config" / "agents.toml")
SYSTEM = Actor(kind=ActorKind.SYSTEM, name="test")


def _claim(store_path: Path, policy: str = "P-1001", part: str | None = "door") -> UUID:
    store = SQLiteEventStore(store_path)
    claim = uuid4()
    store.append(claim, ClaimReported(policy_id=policy, description="scrape"), SYSTEM)
    finding = DamageFinding(
        photo_id="p1",
        damage_type=DamageType.DENT,
        confidence=0.9,
        bbox=BoundingBox(x1=0, y1=0, x2=1, y2=1),
        image_area_fraction=0.1,
        part=part,
    )
    store.append(
        claim, DamageDetected(photo_id="p1", model_version="m", findings=(finding,)), SYSTEM
    )
    store.close()
    return claim


def test_history_returns_state_and_events(tmp_path: Path) -> None:
    db = tmp_path / "claims.db"
    claim = _claim(db)
    history = call(
        build_claims_system(PROFILES["demo"], db, MemoryAudit()),
        "get_claim_history",
        {"claim_id": str(claim)},
    ).structured_content
    assert history is not None
    assert history["found"] is True
    assert history["policy_id"] == "P-1001"
    assert history["findings"] == ["dent on door"]
    assert [e["type"] for e in history["events"]] == ["ClaimReported", "DamageDetected"]


def test_bad_claim_id_is_not_found(tmp_path: Path) -> None:
    server = build_claims_system(PROFILES["demo"], tmp_path / "claims.db", MemoryAudit())
    for claim_id in ("not-a-uuid", str(uuid4())):
        result = call(server, "get_claim_history", {"claim_id": claim_id}).structured_content
        assert result is not None
        assert result["found"] is False


def test_similar_claims_share_a_policy_or_part(tmp_path: Path) -> None:
    db = tmp_path / "claims.db"
    target = _claim(db, "P-1001", "door")
    same_policy = _claim(db, "P-1001", "hood")
    same_part = _claim(db, "P-2002", "door")
    _claim(db, "P-3003", "wheel")
    items = call(
        build_claims_system(PROFILES["triage"], db, MemoryAudit()),
        "find_similar_claims",
        {"claim_id": str(target)},
    ).structured_content
    assert items is not None
    found = {i["claim_id"]: i["reason"] for i in items["items"]}
    assert found == {str(same_policy): "same policy", str(same_part): "same part: door"}


def test_add_note_is_idempotent(tmp_path: Path) -> None:
    db = tmp_path / "claims.db"
    claim = _claim(db)
    server = build_claims_system(PROFILES["triage"], db, MemoryAudit())
    args = {"claim_id": str(claim), "text": "Ask for a wider photo", "idempotency_key": "n1"}
    first = call(server, "add_note", args).structured_content
    second = call(server, "add_note", args).structured_content
    assert first == {"event_seq": 3, "duplicate": False}
    assert second == {"event_seq": 3, "duplicate": True}
    store = SQLiteEventStore(db)
    assert fold(store.load(claim)).notes == ["Ask for a wider photo"]


def test_assign_queue_validates_the_queue(tmp_path: Path) -> None:
    db = tmp_path / "claims.db"
    claim = _claim(db)
    server = build_claims_system(PROFILES["triage"], db, MemoryAudit())
    ok = call(
        server, "assign_queue", {"claim_id": str(claim), "queue": "fraud", "idempotency_key": "q1"}
    )
    assert not ok.is_error
    bad = call(
        server, "assign_queue", {"claim_id": str(claim), "queue": "deny", "idempotency_key": "q2"}
    )
    assert bad.is_error


def test_demo_profile_cannot_write(tmp_path: Path) -> None:
    db = tmp_path / "claims.db"
    claim = _claim(db)
    server = build_claims_system(PROFILES["demo"], db, MemoryAudit())
    assert "add_note" not in tool_names(server)
    result = call(server, "add_note", {"claim_id": str(claim), "text": "x", "idempotency_key": "k"})
    assert "ScopeDenied" in error_text(result)


def test_event_audit_writes_tool_called_to_the_claim(tmp_path: Path) -> None:
    db = tmp_path / "claims.db"
    claim = _claim(db)
    fallback = MemoryAudit()
    server = build_claims_system(PROFILES["triage"], db, event_audit(db, fallback))
    call(server, "get_claim_history", {"claim_id": str(claim)})
    call(server, "get_claim_history", {"claim_id": "not-a-uuid"})
    events = SQLiteEventStore(db).load(claim)
    assert events[-1].type == "ToolCalled"
    assert [r.claim_id for r in fallback.records] == ["not-a-uuid"]


def test_keys_are_scoped_to_the_event_type(tmp_path: Path) -> None:
    db = tmp_path / "claims.db"
    claim = _claim(db)
    server = build_claims_system(PROFILES["triage"], db, MemoryAudit())
    call(server, "add_note", {"claim_id": str(claim), "text": "x", "idempotency_key": "k1"})
    queued = call(
        server, "assign_queue", {"claim_id": str(claim), "queue": "fraud", "idempotency_key": "k1"}
    ).structured_content
    assert queued is not None
    assert queued["duplicate"] is False
    assert fold(SQLiteEventStore(db).load(claim)).queue == "fraud"


def test_similar_claims_come_from_memory_with_outcomes(tmp_path: Path) -> None:
    import json
    from dataclasses import replace

    from PIL import Image

    from claimlens.intake import submit_claim
    from claimlens.knowledge.embed import FakeEmbedder
    from claimlens.memory.index import ClaimMemory
    from claimlens.workflow import process_claim
    from tests.fakes import FakeDetector, make_test_deps
    from tests.unit.test_memory_index import resaved, scene

    db = tmp_path / "claims.db"
    store = SQLiteEventStore(db)
    memory = ClaimMemory(tmp_path / "memory", FakeEmbedder())
    deps = replace(make_test_deps(tmp_path, store, FakeDetector()), memory=memory)
    original = scene(tmp_path / "car.png")
    earlier = submit_claim(
        store,
        deps.blobs,
        policy_id="P-2002",
        description="Bob Smith hit me",
        photo_paths=[original],
    )
    process_claim(earlier, deps)
    copy = resaved(original, tmp_path / "copy.jpg")
    with Image.open(copy) as image:
        assert image.size == (384, 288)
    now = submit_claim(store, deps.blobs, policy_id="P-1005", description="new", photo_paths=[copy])
    store.close()
    server = build_claims_system(
        PROFILES["triage"], db, MemoryAudit(), memory=memory, photo_path=deps.blobs.path
    )
    result = call(server, "find_similar_claims", {"claim_id": str(now)})
    items = json.loads(result.content[0].text)["items"]  # type: ignore[union-attr]
    assert items[0]["claim_id"] == str(earlier)
    assert items[0]["reason"] == "near-copy photo"
    assert items[0]["route"] == "ADJUSTER_REVIEW"  # P-2002 has no collision cover (R4)
    assert items[0]["distance"] <= 10
    assert "Bob" not in result.content[0].text  # type: ignore[union-attr]
