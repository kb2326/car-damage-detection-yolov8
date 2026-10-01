# M1 Walking Skeleton Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A thin but complete ClaimLens pipeline: one command takes a policy number and photos, runs every stage, writes a hash-chained audit log, routes the claim with deterministic rules, and a golden-claims evaluation scores the whole thing.

**Architecture:** An append-only SQLite event log is the single source of truth; `ClaimState` is rebuilt by folding events. A hand-written workflow runs stages (quality gate, perception with the legacy YOLOv8 model, integrity, pricing, coverage, a stub agent) and each stage appends events. A pure `decide()` function applies the versioned decision policy. Stages are idempotent, so re-running a claim resumes where it stopped.

**Tech Stack:** Python 3.12, Pydantic v2, Pillow, SQLite (stdlib), tomllib (stdlib), argparse (stdlib), Ultralytics (optional `vision` dependency group), pytest, mypy `--strict`, ruff, uv.

**Spec:** [`docs/specs/2026-10-01-claimlens-design.md`](../specs/2026-10-01-claimlens-design.md) (sections 4, 6, 11, 13, 17, 18, 20).

## Global Constraints

- Python `>=3.12,<3.13`; all commands run through `uv run …` from the repository root.
- `mypy --strict` and `ruff check` / `ruff format --check` must pass on `src`, `tests` and `scripts`.
- Runtime dependencies are limited to `pydantic>=2.8` and `pillow>=11.0`. `ultralytics>=8.3,<9` lives only in the optional `vision` dependency group; CI never installs it.
- CI must pass with no dataset, no model weights and no API keys. Anything needing them is marked `slow` and skipped when they are absent.
- Outcomes are only `FAST_TRACK`, `ADJUSTER_REVIEW`, `FRAUD_REVIEW`. There is no denial route.
- Every threshold lives in a versioned TOML file under `config/`; code holds no magic numbers for policy.
- Never commit anything under `data/`, `models/` or `var/`.
- Strings and comments in Python files use ASCII only (ruff RUF001-RUF003).
- Every `pytest.raises(ValueError)` passes `match=` (ruff PT011).
- Always run `uv run ruff format .` before `uv run ruff check .`; the formatter wraps long call
  and collection lines that would otherwise fail E501.
- The pre-commit hook may reformat files on commit. If a commit fails because files were modified, `git add` them and commit again.
- Commit messages follow Conventional Commits and end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

1. **The same photo uploaded twice in one claim** must be rejected as a duplicate, priced once, and must not raise a fraud signal (owned by Task 13, `test_duplicate_photo_in_claim_is_rejected_not_fraud`).
2. **A process crash partway through a claim** must resume without duplicating events or re-running finished work (Task 13, `test_resume_after_crash_does_not_repeat_work`).
3. **An edited row in the event database** must stop processing and make `verify` fail (Task 3, `test_tampered_row_is_detected`; Task 14, `test_verify_detects_tampering`).
4. **A model class name that is not a known damage type** must raise an error instead of being silently mislabelled, which was the original project's critical bug (Task 7, `test_normalize_rejects_unknown_name`).
5. **A claim whose every photo is unusable** (too small, corrupt, wrong format) must route to an adjuster without calling the model (Task 13, `test_unusable_photos_route_to_adjuster`).

---

## File structure

```
config/
  policies.toml            fictional policies (Task 8)
  rate_card.toml           fictional repair price ranges (Task 9)
  decision_policy.toml     routing thresholds (Task 11)
src/claimlens/
  domain.py                shared value types: DamageFinding, Route, Decision ... (Task 1)
  events/
    envelope.py            ClaimEvent, Actor, hashing, verify_chain (Task 2)
    payloads.py            one Pydantic model per event type (Task 2)
    store.py               SQLiteEventStore (Task 3)
    projection.py          ClaimState + fold() (Task 4)
  blobs.py                 content-addressed photo storage (Task 5)
  intake.py                submit_claim() (Task 5)
  quality.py               photo quality gate (Task 6)
  vision/
    base.py                Detector protocol, prediction -> finding conversion (Task 7)
    legacy_yolo.py         adapter for the original YOLOv8n weights (Task 7)
  policy.py                PolicyRepository, coverage lookup (Task 8)
  pricing.py               RateCard, estimate_cost() (Task 9)
  integrity.py             cross-claim photo reuse check (Task 10)
  decision.py              DecisionConfig, decide() (Task 11)
  agent/
    __init__.py            TriageAgent protocol (Task 12)
    stub.py                StubTriageAgent (Task 12)
  workflow.py              PipelineDeps, process_claim() (Task 13)
  cli.py                   claimlens run | show | verify | eval-triage (Tasks 14-15)
  evals/
    metrics.py             route accuracy, escalation recall (Task 15)
    golden.py              golden claim format + loader (Task 15)
    triage.py              eval runner + Markdown report (Task 15)
    oracle.py              expected routes from ground-truth labels (Task 16)
scripts/build_golden_v0.py golden claims v0 generator (Task 16)
evals/golden/v0/           claims.jsonl + small unusable-photo assets (Task 16)
evals/reports/             committed evaluation reports (Task 17)
tests/ unit/ integration/ fakes.py conftest.py
```

---

### Task 1: Dependencies and domain types

**Files:**
- Modify: `pyproject.toml`
- Create: `src/claimlens/domain.py`
- Create: `tests/__init__.py`, `tests/unit/__init__.py` (empty)
- Test: `tests/unit/test_domain.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `Frozen` (base model), `DamageType`, `Severity`, `Route`, `Confidence` (StrEnums), `BoundingBox(x1, y1, x2, y2)`, `DamageFinding(photo_id, damage_type, confidence, bbox, image_area_fraction, part=None)`, `FraudSignal(kind, score, detail)`, `CostEstimate(low, high, currency="USD", basis)`, `Coverage(policy_id, found, active, collision, deductible)` with property `confirmed`, `AgentRecommendation(route_suggestion, confidence, rationale, citations, open_questions=())`, `Decision(route, rule_id, reason, policy_version)`.

- [ ] **Step 1: Update `pyproject.toml`**

Replace the `dependencies = []` line and the `[dependency-groups]` and `[tool.mypy]` sections with:

```toml
dependencies = [
  "pillow>=11.0",
  "pydantic>=2.8",
]
```

```toml
[dependency-groups]
dev = [
  "mypy>=1.11",
  "pre-commit>=3.8",
  "pytest>=8.3",
  "pytest-cov>=5.0",
  "pyyaml>=6.0",
  "ruff>=0.6",
  "types-pyyaml>=6.0",
]
vision = [
  "ultralytics>=8.3,<9",
]
```

```toml
[tool.mypy]
python_version = "3.12"
strict = true
files = ["src", "tests", "scripts"]
plugins = ["pydantic.mypy"]

[[tool.mypy.overrides]]
module = ["ultralytics", "ultralytics.*"]
ignore_missing_imports = true

[tool.coverage.run]
omit = ["*/claimlens/vision/legacy_yolo.py"]
```

Create the empty files `tests/__init__.py` and `tests/unit/__init__.py`, and create the `scripts/` directory with an empty `scripts/__init__.py` so mypy has something to check.

Run: `uv sync`
Expected: installs pydantic, pillow, pyyaml, types-pyyaml; no errors.

- [ ] **Step 2: Write the failing test** `tests/unit/test_domain.py`

```python
import pytest
from pydantic import ValidationError

from claimlens.domain import (
    AgentRecommendation,
    BoundingBox,
    Confidence,
    CostEstimate,
    Coverage,
    DamageFinding,
    DamageType,
    Route,
)


def _finding(confidence: float = 0.9) -> DamageFinding:
    return DamageFinding(
        photo_id="p1",
        damage_type=DamageType.DENT,
        confidence=confidence,
        bbox=BoundingBox(x1=0, y1=0, x2=10, y2=10),
        image_area_fraction=0.01,
    )


def test_routes_never_include_a_denial() -> None:
    assert {route.value for route in Route} == {"FAST_TRACK", "ADJUSTER_REVIEW", "FRAUD_REVIEW"}


def test_finding_rejects_confidence_above_one() -> None:
    with pytest.raises(ValidationError):
        _finding(confidence=1.5)


def test_bounding_box_rejects_inverted_corners() -> None:
    with pytest.raises(ValidationError, match="x1 <= x2"):
        BoundingBox(x1=10, y1=0, x2=5, y2=10)


def test_cost_estimate_rejects_low_above_high() -> None:
    with pytest.raises(ValidationError, match="low must not exceed high"):
        CostEstimate(low=500, high=100, basis="test")


def test_coverage_confirmed_requires_found_active_and_collision() -> None:
    def cover(*, found: bool = True, active: bool = True, collision: bool = True) -> Coverage:
        return Coverage(
            policy_id="P", found=found, active=active, collision=collision, deductible=500
        )

    assert cover().confirmed
    assert not cover(found=False).confirmed
    assert not cover(active=False).confirmed
    assert not cover(collision=False).confirmed


def test_damage_type_values_are_snake_case() -> None:
    assert DamageType("glass_shatter") is DamageType.GLASS_SHATTER


def test_recommendation_defaults_to_no_open_questions() -> None:
    rec = AgentRecommendation(
        route_suggestion=Route.FAST_TRACK,
        confidence=Confidence.HIGH,
        rationale="ok",
        citations=("e1",),
    )
    assert rec.open_questions == ()
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_domain.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'claimlens.domain'`

- [ ] **Step 4: Write the implementation** `src/claimlens/domain.py`

```python
"""Core domain types shared across ClaimLens."""

from __future__ import annotations

from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Frozen(BaseModel):
    """Immutable model that rejects unknown fields."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class DamageType(StrEnum):
    CRACK = "crack"
    DENT = "dent"
    GLASS_SHATTER = "glass_shatter"
    LAMP_BROKEN = "lamp_broken"
    SCRATCH = "scratch"
    TIRE_FLAT = "tire_flat"
    SMASH = "smash"


class Severity(StrEnum):
    MINOR = "minor"
    MODERATE = "moderate"
    SEVERE = "severe"


class Route(StrEnum):
    """Possible outcomes. There is deliberately no route that denies a claim."""

    FAST_TRACK = "FAST_TRACK"
    ADJUSTER_REVIEW = "ADJUSTER_REVIEW"
    FRAUD_REVIEW = "FRAUD_REVIEW"


class Confidence(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class BoundingBox(Frozen):
    """Axis-aligned box in pixel coordinates of the original image."""

    x1: float
    y1: float
    x2: float
    y2: float

    @model_validator(mode="after")
    def _ordered(self) -> Self:
        if self.x2 < self.x1 or self.y2 < self.y1:
            raise ValueError("box corners must satisfy x1 <= x2 and y1 <= y2")
        return self


class DamageFinding(Frozen):
    photo_id: str
    damage_type: DamageType
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: BoundingBox
    image_area_fraction: float = Field(ge=0.0, le=1.0)
    part: str | None = None


class FraudSignal(Frozen):
    kind: str
    score: float = Field(ge=0.0, le=1.0)
    detail: str


class CostEstimate(Frozen):
    low: int = Field(ge=0)
    high: int = Field(ge=0)
    currency: str = "USD"
    basis: str

    @model_validator(mode="after")
    def _ordered(self) -> Self:
        if self.low > self.high:
            raise ValueError("low must not exceed high")
        return self


class Coverage(Frozen):
    policy_id: str
    found: bool
    active: bool
    collision: bool
    deductible: int = Field(ge=0)

    @property
    def confirmed(self) -> bool:
        return self.found and self.active and self.collision


class AgentRecommendation(Frozen):
    route_suggestion: Route
    confidence: Confidence
    rationale: str
    citations: tuple[str, ...]
    open_questions: tuple[str, ...] = ()


class Decision(Frozen):
    route: Route
    rule_id: str
    reason: str
    policy_version: str
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_domain.py -v && uv run mypy && uv run ruff check .`
Expected: 7 passed; mypy and ruff report no issues.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock src/claimlens/domain.py tests/__init__.py tests/unit/__init__.py scripts/__init__.py tests/unit/test_domain.py
git commit -m "feat: add core domain types and runtime dependencies"
```

---

### Task 2: Event envelope, hashing and payloads

**Files:**
- Create: `src/claimlens/events/__init__.py` (docstring only: `"""Event-sourced claim state."""`)
- Create: `src/claimlens/events/envelope.py`
- Create: `src/claimlens/events/payloads.py`
- Modify: `docs/specs/2026-10-01-claimlens-design.md` (section 6 event list)
- Test: `tests/unit/test_events.py`

**Interfaces:**
- Consumes: `Frozen`, domain value types (Task 1).
- Produces: `GENESIS_HASH: str`; `ActorKind` (StrEnum: system, agent, human); `Actor(kind, name)`; `ClaimEvent(event_id, claim_id, seq, type, payload, actor, occurred_at, schema_version=1, prev_hash, hash)`; `ChainIntegrityError(claim_id, seq, problem)`; `compute_hash(event) -> str`; `verify_chain(events) -> None`; `EventType` (StrEnum); `Payload` base with `event_type: ClassVar[EventType]`; payload classes `ClaimReported(policy_id, description)`, `PhotoUploaded(photo_id, sha256, filename, blob_name, size_bytes)`, `PhotoAccepted(photo_id, width, height)`, `PhotoRejected(photo_id, reason)`, `DamageDetected(photo_id, model_version, findings)`, `IntegrityChecked(signals)`, `CostEstimated(estimate)`, `PolicyRetrieved(coverage)`, `AgentRecommended(agent_version, recommendation)`, `RouteDecided(decision)`, `StageFailed(stage, error, photo_id=None)`; `PAYLOAD_TYPES: dict[EventType, type[Payload]]`; `parse_payload(event) -> Payload`.

- [ ] **Step 1: Write the failing test** `tests/unit/test_events.py`

```python
from datetime import UTC, datetime
from uuid import UUID

import pytest

from claimlens.events.envelope import (
    GENESIS_HASH,
    Actor,
    ActorKind,
    ChainIntegrityError,
    ClaimEvent,
    compute_hash,
    verify_chain,
)
from claimlens.events.payloads import (
    PAYLOAD_TYPES,
    ClaimReported,
    EventType,
    Payload,
    PhotoAccepted,
    parse_payload,
)

CLAIM = UUID(int=1)
ACTOR = Actor(kind=ActorKind.SYSTEM, name="test")


def _event(seq: int, prev_hash: str, payload: Payload) -> ClaimEvent:
    draft = ClaimEvent(
        event_id=UUID(int=100 + seq),
        claim_id=CLAIM,
        seq=seq,
        type=payload.event_type.value,
        payload=payload.model_dump(mode="json"),
        actor=ACTOR,
        occurred_at=datetime(2026, 10, 8, 12, 0, seq, tzinfo=UTC),
        prev_hash=prev_hash,
        hash="",
    )
    return draft.model_copy(update={"hash": compute_hash(draft)})


def _chain() -> list[ClaimEvent]:
    first = _event(1, GENESIS_HASH, ClaimReported(policy_id="P-1001", description="scrape"))
    second = _event(2, first.hash, PhotoAccepted(photo_id="p1", width=640, height=640))
    return [first, second]


def test_hash_is_deterministic_sha256() -> None:
    first = _chain()[0]
    assert compute_hash(first) == first.hash
    assert len(first.hash) == 64


def test_valid_chain_passes() -> None:
    verify_chain(_chain())


def test_edited_payload_breaks_chain() -> None:
    first, second = _chain()
    tampered = first.model_copy(update={"payload": {"policy_id": "P-9999", "description": "x"}})
    with pytest.raises(ChainIntegrityError, match="hash does not match"):
        verify_chain([tampered, second])


def test_rehashed_edit_is_caught_by_next_event() -> None:
    first, second = _chain()
    edited = first.model_copy(update={"payload": {"policy_id": "P-9999", "description": "x"}})
    rehashed = edited.model_copy(update={"hash": compute_hash(edited)})
    with pytest.raises(ChainIntegrityError, match="prev_hash"):
        verify_chain([rehashed, second])


def test_gap_in_sequence_is_rejected() -> None:
    first, second = _chain()
    with pytest.raises(ChainIntegrityError, match="expected sequence 2"):
        verify_chain([first, second.model_copy(update={"seq": 3})])


def test_parse_payload_round_trips() -> None:
    assert parse_payload(_chain()[0]) == ClaimReported(policy_id="P-1001", description="scrape")


def test_parse_payload_rejects_unknown_type() -> None:
    mystery = _chain()[0].model_copy(update={"type": "Mystery"})
    with pytest.raises(ValueError, match="Mystery"):
        parse_payload(mystery)


def test_every_event_type_has_a_payload_class() -> None:
    assert set(PAYLOAD_TYPES) == set(EventType)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_events.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'claimlens.events'`

- [ ] **Step 3: Write** `src/claimlens/events/envelope.py`

```python
"""Claim event envelope and hash chaining."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from datetime import datetime
from enum import StrEnum
from typing import Any
from uuid import UUID

from pydantic import Field

from claimlens.domain import Frozen

GENESIS_HASH = "0" * 64


class ActorKind(StrEnum):
    SYSTEM = "system"
    AGENT = "agent"
    HUMAN = "human"


class Actor(Frozen):
    kind: ActorKind
    name: str


class ClaimEvent(Frozen):
    event_id: UUID
    claim_id: UUID
    seq: int = Field(ge=1)
    type: str
    payload: dict[str, Any]
    actor: Actor
    occurred_at: datetime
    schema_version: int = 1
    prev_hash: str
    hash: str


class ChainIntegrityError(Exception):
    """Raised when a claim's event log has been altered or is out of order."""

    def __init__(self, claim_id: UUID, seq: int, problem: str) -> None:
        super().__init__(f"claim {claim_id} event #{seq}: {problem}")
        self.claim_id = claim_id
        self.seq = seq


def compute_hash(event: ClaimEvent) -> str:
    """sha256(prev_hash + canonical JSON of every field except `hash`)."""
    body = event.model_dump(mode="json", exclude={"hash"})
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256((event.prev_hash + canonical).encode("utf-8")).hexdigest()


def verify_chain(events: Sequence[ClaimEvent]) -> None:
    prev = GENESIS_HASH
    for expected_seq, event in enumerate(events, start=1):
        if event.seq != expected_seq:
            raise ChainIntegrityError(
                event.claim_id, event.seq, f"expected sequence {expected_seq}"
            )
        if event.prev_hash != prev:
            raise ChainIntegrityError(
                event.claim_id, event.seq, "prev_hash does not match the previous event"
            )
        if compute_hash(event) != event.hash:
            raise ChainIntegrityError(
                event.claim_id, event.seq, "hash does not match event content"
            )
        prev = event.hash
```

- [ ] **Step 4: Write** `src/claimlens/events/payloads.py`

```python
"""One Pydantic model per event type. The payload is the event's business content."""

from __future__ import annotations

from enum import StrEnum
from typing import ClassVar

from claimlens.domain import (
    AgentRecommendation,
    CostEstimate,
    Coverage,
    DamageFinding,
    Decision,
    FraudSignal,
    Frozen,
)
from claimlens.events.envelope import ClaimEvent


class EventType(StrEnum):
    CLAIM_REPORTED = "ClaimReported"
    PHOTO_UPLOADED = "PhotoUploaded"
    PHOTO_ACCEPTED = "PhotoAccepted"
    PHOTO_REJECTED = "PhotoRejected"
    DAMAGE_DETECTED = "DamageDetected"
    INTEGRITY_CHECKED = "IntegrityChecked"
    COST_ESTIMATED = "CostEstimated"
    POLICY_RETRIEVED = "PolicyRetrieved"
    AGENT_RECOMMENDED = "AgentRecommended"
    ROUTE_DECIDED = "RouteDecided"
    STAGE_FAILED = "StageFailed"


class Payload(Frozen):
    event_type: ClassVar[EventType]


class ClaimReported(Payload):
    event_type: ClassVar[EventType] = EventType.CLAIM_REPORTED
    policy_id: str
    description: str


class PhotoUploaded(Payload):
    event_type: ClassVar[EventType] = EventType.PHOTO_UPLOADED
    photo_id: str
    sha256: str
    filename: str
    blob_name: str
    size_bytes: int


class PhotoAccepted(Payload):
    event_type: ClassVar[EventType] = EventType.PHOTO_ACCEPTED
    photo_id: str
    width: int
    height: int


class PhotoRejected(Payload):
    event_type: ClassVar[EventType] = EventType.PHOTO_REJECTED
    photo_id: str
    reason: str


class DamageDetected(Payload):
    event_type: ClassVar[EventType] = EventType.DAMAGE_DETECTED
    photo_id: str
    model_version: str
    findings: tuple[DamageFinding, ...]


class IntegrityChecked(Payload):
    event_type: ClassVar[EventType] = EventType.INTEGRITY_CHECKED
    signals: tuple[FraudSignal, ...]


class CostEstimated(Payload):
    event_type: ClassVar[EventType] = EventType.COST_ESTIMATED
    estimate: CostEstimate


class PolicyRetrieved(Payload):
    event_type: ClassVar[EventType] = EventType.POLICY_RETRIEVED
    coverage: Coverage


class AgentRecommended(Payload):
    event_type: ClassVar[EventType] = EventType.AGENT_RECOMMENDED
    agent_version: str
    recommendation: AgentRecommendation


class RouteDecided(Payload):
    event_type: ClassVar[EventType] = EventType.ROUTE_DECIDED
    decision: Decision


class StageFailed(Payload):
    event_type: ClassVar[EventType] = EventType.STAGE_FAILED
    stage: str
    error: str
    photo_id: str | None = None


PAYLOAD_TYPES: dict[EventType, type[Payload]] = {
    cls.event_type: cls
    for cls in (
        ClaimReported,
        PhotoUploaded,
        PhotoAccepted,
        PhotoRejected,
        DamageDetected,
        IntegrityChecked,
        CostEstimated,
        PolicyRetrieved,
        AgentRecommended,
        RouteDecided,
        StageFailed,
    )
}


def parse_payload(event: ClaimEvent) -> Payload:
    """Validate an event's payload against the model for its type."""
    return PAYLOAD_TYPES[EventType(event.type)].model_validate(event.payload)
```

- [ ] **Step 5: Update the spec's event list**

In `docs/specs/2026-10-01-claimlens-design.md`, section 6, replace the "Core event types" paragraph with:

```markdown
**Core event types:** `ClaimReported`, `PhotoUploaded`, `PhotoAccepted`, `PhotoRejected`,
`PhotoRedacted`, `DamageDetected`, `IntegrityChecked` (carries zero or more fraud signals, so a
clean check is also recorded), `CostEstimated`, `PolicyRetrieved`, `AgentStepCompleted`,
`AgentRecommended`, `HumanReviewRequested`, `RouteDecided`, `StageFailed`, `HumanDecided`,
`HumanOverrode`, `MemoryWritten`, `ClaimClosed`. M1 implements the subset in
`claimlens.events.payloads`.
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_events.py -v && uv run mypy`
Expected: 8 passed; mypy reports no issues.

- [ ] **Step 7: Commit**

```bash
git add src/claimlens/events tests/unit/test_events.py docs/specs/2026-10-01-claimlens-design.md
git commit -m "feat: add hash-chained claim event envelope and typed payloads"
```

---

### Task 3: SQLite event store

**Files:**
- Create: `src/claimlens/events/store.py`
- Create: `tests/conftest.py`
- Create: `docs/adr/0003-event-sourced-claim-state.md`
- Test: `tests/unit/test_store.py`

**Interfaces:**
- Consumes: `ClaimEvent`, `Actor`, `GENESIS_HASH`, `compute_hash`, `verify_chain`, `ChainIntegrityError` (Task 2); `Payload`, `EventType` (Task 2).
- Produces: `utc_now() -> datetime`; `ClaimNotFoundError(LookupError)`; `SQLiteEventStore(path, *, clock=utc_now, new_id=uuid4)` with `append(claim_id: UUID, payload: Payload, actor: Actor) -> ClaimEvent`, `load(claim_id: UUID) -> list[ClaimEvent]` (verifies the chain), `claim_ids() -> list[UUID]`, `claims_with_photo(sha256: str) -> set[UUID]`, `close()`. Test fixtures `fixed_clock` and `store`.

- [ ] **Step 1: Write the shared fixtures** `tests/conftest.py`

```python
from collections.abc import Callable, Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from claimlens.events.store import SQLiteEventStore


@pytest.fixture
def fixed_clock() -> Callable[[], datetime]:
    current = datetime(2026, 10, 8, 9, 0, tzinfo=UTC)

    def tick() -> datetime:
        nonlocal current
        current += timedelta(seconds=1)
        return current

    return tick


@pytest.fixture
def store(tmp_path: Path, fixed_clock: Callable[[], datetime]) -> Iterator[SQLiteEventStore]:
    event_store = SQLiteEventStore(tmp_path / "events.db", clock=fixed_clock)
    yield event_store
    event_store.close()
```

- [ ] **Step 2: Write the failing test** `tests/unit/test_store.py`

```python
import sqlite3
from pathlib import Path
from uuid import UUID

import pytest

from claimlens.events.envelope import Actor, ActorKind, ChainIntegrityError
from claimlens.events.payloads import ClaimReported, PhotoUploaded
from claimlens.events.store import ClaimNotFoundError, SQLiteEventStore

ACTOR = Actor(kind=ActorKind.SYSTEM, name="test")
CLAIM_A = UUID(int=1)
CLAIM_B = UUID(int=2)


def _report() -> ClaimReported:
    return ClaimReported(policy_id="P-1001", description="scrape")


def _photo(photo_id: str, sha256: str) -> PhotoUploaded:
    return PhotoUploaded(
        photo_id=photo_id,
        sha256=sha256,
        filename=f"{photo_id}.jpg",
        blob_name=f"{sha256}.jpg",
        size_bytes=10,
    )


def test_append_assigns_gapless_sequence_and_links_hashes(store: SQLiteEventStore) -> None:
    first = store.append(CLAIM_A, _report(), ACTOR)
    second = store.append(CLAIM_A, _photo("p1", "a" * 64), ACTOR)
    assert (first.seq, second.seq) == (1, 2)
    assert second.prev_hash == first.hash


def test_sequences_are_per_claim(store: SQLiteEventStore) -> None:
    store.append(CLAIM_A, _report(), ACTOR)
    assert store.append(CLAIM_B, _report(), ACTOR).seq == 1


def test_load_returns_the_appended_events(store: SQLiteEventStore) -> None:
    appended = [
        store.append(CLAIM_A, _report(), ACTOR),
        store.append(CLAIM_A, _photo("p1", "a" * 64), ACTOR),
    ]
    assert store.load(CLAIM_A) == appended


def test_load_unknown_claim_raises(store: SQLiteEventStore) -> None:
    with pytest.raises(ClaimNotFoundError):
        store.load(UUID(int=99))


def test_claim_ids_lists_each_claim_once(store: SQLiteEventStore) -> None:
    store.append(CLAIM_A, _report(), ACTOR)
    store.append(CLAIM_A, _photo("p1", "a" * 64), ACTOR)
    store.append(CLAIM_B, _report(), ACTOR)
    assert sorted(store.claim_ids()) == [CLAIM_A, CLAIM_B]


def test_claims_with_photo_finds_every_claim_using_it(store: SQLiteEventStore) -> None:
    for claim in (CLAIM_A, CLAIM_B):
        store.append(claim, _report(), ACTOR)
        store.append(claim, _photo("p1", "c" * 64), ACTOR)
    assert store.claims_with_photo("c" * 64) == {CLAIM_A, CLAIM_B}
    assert store.claims_with_photo("d" * 64) == set()


def test_tampered_row_is_detected(tmp_path: Path) -> None:
    path = tmp_path / "events.db"
    event_store = SQLiteEventStore(path)
    event_store.append(CLAIM_A, _report(), ACTOR)
    event_store.append(CLAIM_A, _photo("p1", "a" * 64), ACTOR)
    with sqlite3.connect(path) as conn:
        conn.execute("UPDATE events SET data = replace(data, 'P-1001', 'P-9999') WHERE seq = 1")
    with pytest.raises(ChainIntegrityError, match="hash does not match"):
        event_store.load(CLAIM_A)
    event_store.close()


def test_store_persists_across_connections(tmp_path: Path) -> None:
    path = tmp_path / "events.db"
    first = SQLiteEventStore(path)
    first.append(CLAIM_A, _report(), ACTOR)
    first.close()
    second = SQLiteEventStore(path)
    assert len(second.load(CLAIM_A)) == 1
    second.close()
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_store.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'claimlens.events.store'`

- [ ] **Step 4: Write** `src/claimlens/events/store.py`

```python
"""Append-only SQLite storage for claim events."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4

from claimlens.events.envelope import (
    GENESIS_HASH,
    Actor,
    ClaimEvent,
    compute_hash,
    verify_chain,
)
from claimlens.events.payloads import EventType, Payload

_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    claim_id TEXT NOT NULL,
    seq INTEGER NOT NULL,
    type TEXT NOT NULL,
    hash TEXT NOT NULL,
    data TEXT NOT NULL,
    PRIMARY KEY (claim_id, seq)
)
"""


def utc_now() -> datetime:
    return datetime.now(UTC)


class ClaimNotFoundError(LookupError):
    def __init__(self, claim_id: UUID) -> None:
        super().__init__(f"no events found for claim {claim_id}")
        self.claim_id = claim_id


class SQLiteEventStore:
    """Events are only ever inserted. Reads verify the hash chain before returning."""

    def __init__(
        self,
        path: Path | str,
        *,
        clock: Callable[[], datetime] = utc_now,
        new_id: Callable[[], UUID] = uuid4,
    ) -> None:
        if isinstance(path, Path):
            path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), isolation_level=None)
        self._conn.execute(_SCHEMA)
        self._clock = clock
        self._new_id = new_id

    def append(self, claim_id: UUID, payload: Payload, actor: Actor) -> ClaimEvent:
        conn = self._conn
        conn.execute("BEGIN IMMEDIATE")
        try:
            row = conn.execute(
                "SELECT seq, hash FROM events WHERE claim_id = ? ORDER BY seq DESC LIMIT 1",
                (str(claim_id),),
            ).fetchone()
            seq, prev_hash = (row[0] + 1, row[1]) if row else (1, GENESIS_HASH)
            draft = ClaimEvent(
                event_id=self._new_id(),
                claim_id=claim_id,
                seq=seq,
                type=payload.event_type.value,
                payload=payload.model_dump(mode="json"),
                actor=actor,
                occurred_at=self._clock(),
                prev_hash=prev_hash,
                hash="",
            )
            event = draft.model_copy(update={"hash": compute_hash(draft)})
            conn.execute(
                "INSERT INTO events (claim_id, seq, type, hash, data) VALUES (?, ?, ?, ?, ?)",
                (str(claim_id), seq, event.type, event.hash, event.model_dump_json()),
            )
            conn.execute("COMMIT")
        except BaseException:
            conn.execute("ROLLBACK")
            raise
        return event

    def load(self, claim_id: UUID) -> list[ClaimEvent]:
        rows = self._conn.execute(
            "SELECT data FROM events WHERE claim_id = ? ORDER BY seq", (str(claim_id),)
        ).fetchall()
        if not rows:
            raise ClaimNotFoundError(claim_id)
        events = [ClaimEvent.model_validate_json(row[0]) for row in rows]
        verify_chain(events)
        return events

    def claim_ids(self) -> list[UUID]:
        rows = self._conn.execute("SELECT DISTINCT claim_id FROM events").fetchall()
        return [UUID(row[0]) for row in rows]

    def claims_with_photo(self, sha256: str) -> set[UUID]:
        rows = self._conn.execute(
            "SELECT DISTINCT claim_id FROM events "
            "WHERE type = ? AND json_extract(data, '$.payload.sha256') = ?",
            (EventType.PHOTO_UPLOADED.value, sha256),
        ).fetchall()
        return {UUID(row[0]) for row in rows}

    def close(self) -> None:
        self._conn.close()
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_store.py -v && uv run mypy`
Expected: 8 passed; mypy reports no issues.

- [ ] **Step 6: Write** `docs/adr/0003-event-sourced-claim-state.md`

```markdown
# ADR 0003: Event-sourced claim state

- **Status:** Accepted
- **Date:** 2026-10-08

## Context

Claims are long-running: a claim may wait days for photos or an adjuster. Every routing decision
must be explainable to an auditor, and agent behaviour must be replayable for debugging and
evaluation.

## Decision

Store each claim as an append-only sequence of typed events in SQLite. Each event records the
SHA-256 of the previous event, forming a hash chain that is verified on every read. Claim state is
never stored; it is rebuilt with `fold(events)`. Workflow and agent steps read state and only
append new events. Corrections are new events, never edits.

## Consequences

- One mechanism gives durability (resume after a crash), working memory, an audit trail and replay.
- Every new piece of state needs an event type and a fold rule.
- Folding the full log on every step is O(n) per step; acceptable at tens of events per claim.
- The hash chain detects tampering but does not prevent someone with database access from
  rewriting the whole chain; anchoring chain heads externally is out of scope for now.
```

- [ ] **Step 7: Commit**

```bash
git add src/claimlens/events/store.py tests/conftest.py tests/unit/test_store.py docs/adr/0003-event-sourced-claim-state.md
git commit -m "feat: add append-only SQLite event store with chain verification"
```

---

### Task 4: Claim state projection

**Files:**
- Create: `src/claimlens/events/projection.py`
- Test: `tests/unit/test_projection.py`

**Interfaces:**
- Consumes: `ClaimEvent` (Task 2), payload classes and `parse_payload` (Task 2), domain types (Task 1).
- Produces: `PhotoStatus` (StrEnum: uploaded, accepted, rejected); `PhotoRecord(photo_id, sha256, filename, blob_name, status=UPLOADED, reject_reason=None)` (dataclass); `ClaimState` (dataclass) with fields `claim_id, policy_id, description, photos: dict[str, PhotoRecord], findings: list[DamageFinding], detection_event_ids: dict[str, str], integrity_checked: bool, fraud_signals: list[FraudSignal], cost_estimate, coverage, recommendation, decision, failures: list[StageFailed], last_seq`, property `accepted_photos -> list[PhotoRecord]`, method `failed(stage: str, photo_id: str | None = None) -> bool`; `fold(events: Sequence[ClaimEvent]) -> ClaimState`.

- [ ] **Step 1: Write the failing test** `tests/unit/test_projection.py`

```python
from uuid import UUID

import pytest

from claimlens.domain import BoundingBox, DamageFinding, DamageType, Decision, Route
from claimlens.events.envelope import Actor, ActorKind
from claimlens.events.payloads import (
    ClaimReported,
    DamageDetected,
    PhotoAccepted,
    PhotoRejected,
    PhotoUploaded,
    RouteDecided,
    StageFailed,
)
from claimlens.events.projection import PhotoStatus, fold
from claimlens.events.store import SQLiteEventStore

CLAIM = UUID(int=7)
ACTOR = Actor(kind=ActorKind.SYSTEM, name="test")
FINDING = DamageFinding(
    photo_id="p1",
    damage_type=DamageType.DENT,
    confidence=0.9,
    bbox=BoundingBox(x1=0, y1=0, x2=64, y2=64),
    image_area_fraction=0.01,
)


def _upload(store: SQLiteEventStore, photo_id: str) -> None:
    store.append(
        CLAIM,
        PhotoUploaded(
            photo_id=photo_id,
            sha256=photo_id * 32,
            filename=f"{photo_id}.jpg",
            blob_name=f"{photo_id}.jpg",
            size_bytes=10,
        ),
        ACTOR,
    )


def test_fold_builds_claim_state(store: SQLiteEventStore) -> None:
    store.append(CLAIM, ClaimReported(policy_id="P-1001", description="scrape"), ACTOR)
    _upload(store, "p1")
    store.append(CLAIM, PhotoAccepted(photo_id="p1", width=640, height=640), ACTOR)
    detected = store.append(
        CLAIM, DamageDetected(photo_id="p1", model_version="m", findings=(FINDING,)), ACTOR
    )

    state = fold(store.load(CLAIM))

    assert state.policy_id == "P-1001"
    assert [photo.photo_id for photo in state.accepted_photos] == ["p1"]
    assert state.findings == [FINDING]
    assert state.detection_event_ids == {"p1": str(detected.event_id)}
    assert state.last_seq == 4
    assert state.decision is None


def test_fold_tracks_rejections_and_failures(store: SQLiteEventStore) -> None:
    store.append(CLAIM, ClaimReported(policy_id="P-1001", description=""), ACTOR)
    _upload(store, "p1")
    _upload(store, "p2")
    store.append(CLAIM, PhotoRejected(photo_id="p1", reason="too small"), ACTOR)
    store.append(CLAIM, StageFailed(stage="perception", photo_id="p2", error="boom"), ACTOR)

    state = fold(store.load(CLAIM))

    assert state.photos["p1"].status is PhotoStatus.REJECTED
    assert state.photos["p1"].reject_reason == "too small"
    assert state.failed("perception", "p2")
    assert not state.failed("perception", "p1")


def test_fold_records_decision(store: SQLiteEventStore) -> None:
    decision = Decision(route=Route.FAST_TRACK, rule_id="R9", reason="ok", policy_version="v0")
    store.append(CLAIM, ClaimReported(policy_id="P-1001", description=""), ACTOR)
    store.append(CLAIM, RouteDecided(decision=decision), ACTOR)
    assert fold(store.load(CLAIM)).decision == decision


def test_fold_requires_claim_reported_first(store: SQLiteEventStore) -> None:
    _upload(store, "p1")
    with pytest.raises(ValueError, match="ClaimReported"):
        fold(store.load(CLAIM))


def test_fold_rejects_an_empty_log() -> None:
    with pytest.raises(ValueError, match="empty"):
        fold([])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_projection.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'claimlens.events.projection'`

- [ ] **Step 3: Write** `src/claimlens/events/projection.py`

```python
"""Rebuild claim state by folding its events. State is a projection, never stored as truth."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from uuid import UUID

from claimlens.domain import (
    AgentRecommendation,
    CostEstimate,
    Coverage,
    DamageFinding,
    Decision,
    FraudSignal,
)
from claimlens.events.envelope import ClaimEvent
from claimlens.events.payloads import (
    AgentRecommended,
    ClaimReported,
    CostEstimated,
    DamageDetected,
    IntegrityChecked,
    Payload,
    PhotoAccepted,
    PhotoRejected,
    PhotoUploaded,
    PolicyRetrieved,
    RouteDecided,
    StageFailed,
    parse_payload,
)


class PhotoStatus(StrEnum):
    UPLOADED = "uploaded"
    ACCEPTED = "accepted"
    REJECTED = "rejected"


@dataclass
class PhotoRecord:
    photo_id: str
    sha256: str
    filename: str
    blob_name: str
    status: PhotoStatus = PhotoStatus.UPLOADED
    reject_reason: str | None = None


@dataclass
class ClaimState:
    claim_id: UUID
    policy_id: str
    description: str
    photos: dict[str, PhotoRecord] = field(default_factory=dict)
    findings: list[DamageFinding] = field(default_factory=list)
    detection_event_ids: dict[str, str] = field(default_factory=dict)
    integrity_checked: bool = False
    fraud_signals: list[FraudSignal] = field(default_factory=list)
    cost_estimate: CostEstimate | None = None
    coverage: Coverage | None = None
    recommendation: AgentRecommendation | None = None
    decision: Decision | None = None
    failures: list[StageFailed] = field(default_factory=list)
    last_seq: int = 0

    @property
    def accepted_photos(self) -> list[PhotoRecord]:
        return [p for p in self.photos.values() if p.status is PhotoStatus.ACCEPTED]

    def failed(self, stage: str, photo_id: str | None = None) -> bool:
        return any(f.stage == stage and f.photo_id == photo_id for f in self.failures)


def fold(events: Sequence[ClaimEvent]) -> ClaimState:
    if not events:
        raise ValueError("cannot fold an empty event list")
    first = parse_payload(events[0])
    if not isinstance(first, ClaimReported):
        raise ValueError("the first event of a claim must be ClaimReported")
    state = ClaimState(
        claim_id=events[0].claim_id, policy_id=first.policy_id, description=first.description
    )
    for event in events[1:]:
        _apply(state, event, parse_payload(event))
    state.last_seq = events[-1].seq
    return state


def _apply(state: ClaimState, event: ClaimEvent, payload: Payload) -> None:
    match payload:
        case PhotoUploaded():
            state.photos[payload.photo_id] = PhotoRecord(
                photo_id=payload.photo_id,
                sha256=payload.sha256,
                filename=payload.filename,
                blob_name=payload.blob_name,
            )
        case PhotoAccepted():
            state.photos[payload.photo_id].status = PhotoStatus.ACCEPTED
        case PhotoRejected():
            photo = state.photos[payload.photo_id]
            photo.status = PhotoStatus.REJECTED
            photo.reject_reason = payload.reason
        case DamageDetected():
            state.findings.extend(payload.findings)
            state.detection_event_ids[payload.photo_id] = str(event.event_id)
        case IntegrityChecked():
            state.integrity_checked = True
            state.fraud_signals.extend(payload.signals)
        case CostEstimated():
            state.cost_estimate = payload.estimate
        case PolicyRetrieved():
            state.coverage = payload.coverage
        case AgentRecommended():
            state.recommendation = payload.recommendation
        case RouteDecided():
            state.decision = payload.decision
        case StageFailed():
            state.failures.append(payload)
        case ClaimReported():
            raise ValueError("ClaimReported may only be the first event of a claim")
        case _:
            raise ValueError(f"no fold rule for event type {event.type}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_projection.py -v && uv run mypy`
Expected: 5 passed; mypy reports no issues.

- [ ] **Step 5: Commit**

```bash
git add src/claimlens/events/projection.py tests/unit/test_projection.py
git commit -m "feat: rebuild claim state by folding events"
```

---

### Task 5: Blob store and claim intake

**Files:**
- Create: `src/claimlens/blobs.py`
- Create: `src/claimlens/intake.py`
- Modify: `tests/conftest.py` (add `make_image` fixture)
- Test: `tests/unit/test_intake.py`

**Interfaces:**
- Consumes: `SQLiteEventStore` (Task 3), `ClaimReported`, `PhotoUploaded` (Task 2), `Actor`, `ActorKind` (Task 2).
- Produces: `StoredBlob(sha256, name)`; `BlobStore(root)` with `put(src: Path) -> StoredBlob` and `path(name: str) -> Path`; `CLAIMANT: Actor`; `submit_claim(store, blobs, *, policy_id, description, photo_paths, claim_id=None) -> UUID` (photo ids are `p1`, `p2`, … in the given order). Fixture `make_image(name, size=(640, 480), color=(200, 30, 30)) -> Path`.

- [ ] **Step 1: Add the `make_image` fixture** to `tests/conftest.py`

Add these imports at the top (merge with the existing ones):

```python
from PIL import Image
```

Append:

```python
@pytest.fixture
def make_image(tmp_path: Path) -> Callable[..., Path]:
    """Create a solid-colour test image. Different colours give different file hashes."""

    def _make(
        name: str,
        size: tuple[int, int] = (640, 480),
        color: tuple[int, int, int] = (200, 30, 30),
    ) -> Path:
        path = tmp_path / "images" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        image_format = "PNG" if path.suffix.lower() == ".png" else "JPEG"
        Image.new("RGB", size, color).save(path, format=image_format)
        return path

    return _make
```

- [ ] **Step 2: Write the failing test** `tests/unit/test_intake.py`

```python
from collections.abc import Callable
from pathlib import Path

import pytest

from claimlens.blobs import BlobStore
from claimlens.events.projection import fold
from claimlens.events.store import SQLiteEventStore
from claimlens.intake import submit_claim


def test_blob_store_is_content_addressed(tmp_path: Path, make_image: Callable[..., Path]) -> None:
    image = make_image("front.JPG")
    blobs = BlobStore(tmp_path / "blobs")
    first = blobs.put(image)
    second = blobs.put(image)
    assert first == second
    assert first.name == f"{first.sha256}.jpg"
    assert blobs.path(first.name).read_bytes() == image.read_bytes()


def test_submit_claim_records_report_and_photos(
    store: SQLiteEventStore, tmp_path: Path, make_image: Callable[..., Path]
) -> None:
    photos = [make_image("front.jpg"), make_image("rear.jpg", color=(10, 20, 30))]
    claim_id = submit_claim(
        store,
        BlobStore(tmp_path / "blobs"),
        policy_id="P-1001",
        description="scrape",
        photo_paths=photos,
    )
    state = fold(store.load(claim_id))
    assert state.policy_id == "P-1001"
    assert sorted(state.photos) == ["p1", "p2"]
    assert state.photos["p2"].filename == "rear.jpg"


def test_submit_claim_requires_a_photo(store: SQLiteEventStore, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least one photo"):
        submit_claim(
            store, BlobStore(tmp_path / "b"), policy_id="P-1001", description="", photo_paths=[]
        )


def test_submit_claim_checks_files_before_writing(store: SQLiteEventStore, tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="photo not found"):
        submit_claim(
            store,
            BlobStore(tmp_path / "b"),
            policy_id="P-1001",
            description="",
            photo_paths=[tmp_path / "missing.jpg"],
        )
    assert store.claim_ids() == []
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_intake.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'claimlens.blobs'`

- [ ] **Step 4: Write** `src/claimlens/blobs.py`

```python
"""Content-addressed storage for uploaded photos."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class StoredBlob:
    sha256: str
    name: str


class BlobStore:
    def __init__(self, root: Path) -> None:
        self._root = root
        root.mkdir(parents=True, exist_ok=True)

    def put(self, src: Path) -> StoredBlob:
        data = src.read_bytes()
        sha256 = hashlib.sha256(data).hexdigest()
        name = f"{sha256}{src.suffix.lower()}"
        destination = self._root / name
        if not destination.exists():
            destination.write_bytes(data)
        return StoredBlob(sha256=sha256, name=name)

    def path(self, name: str) -> Path:
        return self._root / name
```

- [ ] **Step 5: Write** `src/claimlens/intake.py`

```python
"""First notice of loss: record the report and the uploaded photos."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from uuid import UUID, uuid4

from claimlens.blobs import BlobStore
from claimlens.events.envelope import Actor, ActorKind
from claimlens.events.payloads import ClaimReported, PhotoUploaded
from claimlens.events.store import SQLiteEventStore

CLAIMANT = Actor(kind=ActorKind.HUMAN, name="claimant")


def submit_claim(
    store: SQLiteEventStore,
    blobs: BlobStore,
    *,
    policy_id: str,
    description: str,
    photo_paths: Sequence[Path],
    claim_id: UUID | None = None,
) -> UUID:
    if not photo_paths:
        raise ValueError("a claim needs at least one photo")
    for path in photo_paths:
        if not path.is_file():
            raise FileNotFoundError(f"photo not found: {path}")

    new_id = claim_id or uuid4()
    store.append(new_id, ClaimReported(policy_id=policy_id, description=description), CLAIMANT)
    for index, path in enumerate(photo_paths, start=1):
        blob = blobs.put(path)
        store.append(
            new_id,
            PhotoUploaded(
                photo_id=f"p{index}",
                sha256=blob.sha256,
                filename=path.name,
                blob_name=blob.name,
                size_bytes=path.stat().st_size,
            ),
            CLAIMANT,
        )
    return new_id
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_intake.py -v && uv run mypy`
Expected: 4 passed; mypy reports no issues.

- [ ] **Step 7: Commit**

```bash
git add src/claimlens/blobs.py src/claimlens/intake.py tests/conftest.py tests/unit/test_intake.py
git commit -m "feat: add content-addressed blob store and claim submission"
```

---

### Task 6: Photo quality gate

**Files:**
- Create: `src/claimlens/quality.py`
- Test: `tests/unit/test_quality.py`

**Interfaces:**
- Consumes: nothing from earlier tasks (Pillow only).
- Produces: `QualityConfig(min_side_px=320, max_bytes=15*1024*1024, allowed_formats=frozenset({"JPEG","PNG","WEBP"}))`; `QualityResult(ok, reason="", width=0, height=0)`; `check_quality(path: Path, config: QualityConfig) -> QualityResult`.

- [ ] **Step 1: Write the failing test** `tests/unit/test_quality.py`

```python
from collections.abc import Callable
from pathlib import Path

from PIL import Image

from claimlens.quality import QualityConfig, check_quality


def test_accepts_a_normal_photo(make_image: Callable[..., Path]) -> None:
    result = check_quality(make_image("ok.jpg"), QualityConfig())
    assert result.ok
    assert (result.width, result.height) == (640, 480)


def test_rejects_a_small_photo(make_image: Callable[..., Path]) -> None:
    result = check_quality(make_image("tiny.png", size=(100, 100)), QualityConfig())
    assert not result.ok
    assert "too small" in result.reason


def test_rejects_a_file_that_is_not_an_image(tmp_path: Path) -> None:
    path = tmp_path / "note.jpg"
    path.write_text("not an image", encoding="utf-8")
    result = check_quality(path, QualityConfig())
    assert not result.ok
    assert result.reason == "not a readable image"


def test_rejects_an_oversized_file(make_image: Callable[..., Path]) -> None:
    result = check_quality(make_image("big.jpg"), QualityConfig(max_bytes=100))
    assert not result.ok
    assert "limit is 100 bytes" in result.reason


def test_rejects_an_unsupported_format(tmp_path: Path) -> None:
    path = tmp_path / "scan.bmp"
    Image.new("RGB", (640, 480)).save(path, format="BMP")
    result = check_quality(path, QualityConfig())
    assert not result.ok
    assert result.reason == "unsupported format BMP"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_quality.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'claimlens.quality'`

- [ ] **Step 3: Write** `src/claimlens/quality.py`

```python
"""Intake quality gate: reject photos the models cannot use."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image, UnidentifiedImageError

_DEFAULT_FORMATS = frozenset({"JPEG", "PNG", "WEBP"})


@dataclass(frozen=True)
class QualityConfig:
    min_side_px: int = 320
    max_bytes: int = 15 * 1024 * 1024
    allowed_formats: frozenset[str] = _DEFAULT_FORMATS


@dataclass(frozen=True)
class QualityResult:
    ok: bool
    reason: str = ""
    width: int = 0
    height: int = 0


def check_quality(path: Path, config: QualityConfig) -> QualityResult:
    size = path.stat().st_size
    if size > config.max_bytes:
        return QualityResult(
            ok=False, reason=f"file is {size} bytes; the limit is {config.max_bytes} bytes"
        )
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            image_format = image.format or "unknown"
            width, height = image.size
    except (UnidentifiedImageError, OSError, SyntaxError):
        return QualityResult(ok=False, reason="not a readable image")
    if image_format not in config.allowed_formats:
        return QualityResult(ok=False, reason=f"unsupported format {image_format}")
    if min(width, height) < config.min_side_px:
        return QualityResult(
            ok=False,
            reason=(
                f"image too small ({width}x{height}); "
                f"the shortest side must be at least {config.min_side_px}px"
            ),
            width=width,
            height=height,
        )
    return QualityResult(ok=True, width=width, height=height)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_quality.py -v && uv run mypy`
Expected: 5 passed; mypy reports no issues.

- [ ] **Step 5: Commit**

```bash
git add src/claimlens/quality.py tests/unit/test_quality.py
git commit -m "feat: add photo quality gate"
```

---

### Task 7: Vision adapter for the legacy model

**Files:**
- Create: `src/claimlens/vision/__init__.py` (docstring only: `"""Damage perception models."""`)
- Create: `src/claimlens/vision/base.py`
- Create: `src/claimlens/vision/legacy_yolo.py`
- Create: `tests/integration/__init__.py` (empty)
- Test: `tests/unit/test_vision.py`, `tests/integration/test_legacy_yolo.py`

**Interfaces:**
- Consumes: `DamageFinding`, `BoundingBox`, `DamageType` (Task 1).
- Produces: `Detector` protocol (`model_version: str` read-only property; `detect(image_path: Path, photo_id: str) -> list[DamageFinding]`); `normalize_class_name(name: str) -> DamageType`; `findings_from_predictions(*, photo_id, boxes_xyxy, confidences, class_ids, class_names, image_width, image_height) -> list[DamageFinding]`; `LegacyYoloDetector(weights: Path, conf_floor: float = 0.10)` implementing `Detector`.

Design note: the detector keeps low-confidence boxes (floor 0.10) so that uncertainty reaches the decision policy (rule R6) instead of being silently dropped. Class names come from the model file itself, never from a hard-coded list. Passing a file path lets Ultralytics read the image in the colour order it expects, which fixes the legacy RGB/BGR bug.

- [ ] **Step 1: Write the failing test** `tests/unit/test_vision.py`

```python
import pytest

from claimlens.domain import DamageType
from claimlens.vision.base import findings_from_predictions, normalize_class_name

LEGACY_NAMES = {
    0: "crack",
    1: "dent",
    2: "glass shatter",
    3: "lamp broken",
    4: "scratch",
    5: "tire flat",
    6: "smash",
}


def test_normalize_maps_legacy_names() -> None:
    assert normalize_class_name("glass shatter") is DamageType.GLASS_SHATTER
    assert normalize_class_name("Lamp-Broken") is DamageType.LAMP_BROKEN


def test_normalize_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="Front-Windscreen-Damage"):
        normalize_class_name("Front-Windscreen-Damage")


def test_findings_use_the_model_class_names() -> None:
    findings = findings_from_predictions(
        photo_id="p1",
        boxes_xyxy=[[0, 0, 64, 64], [100, 100, 420, 420]],
        confidences=[0.8, 0.3],
        class_ids=[1, 4],
        class_names=LEGACY_NAMES,
        image_width=640,
        image_height=640,
    )
    assert [f.damage_type for f in findings] == [DamageType.DENT, DamageType.SCRATCH]
    assert findings[0].image_area_fraction == pytest.approx(0.01)
    assert findings[1].confidence == pytest.approx(0.3)
    assert findings[0].photo_id == "p1"


def test_boxes_are_clamped_to_the_image() -> None:
    (finding,) = findings_from_predictions(
        photo_id="p1",
        boxes_xyxy=[[-10, -10, 700, 700]],
        confidences=[0.5],
        class_ids=[6],
        class_names=LEGACY_NAMES,
        image_width=640,
        image_height=640,
    )
    assert (finding.bbox.x1, finding.bbox.x2) == (0.0, 640.0)
    assert finding.image_area_fraction == pytest.approx(1.0)


def test_unknown_class_id_raises() -> None:
    with pytest.raises(ValueError, match="class id 9"):
        findings_from_predictions(
            photo_id="p1",
            boxes_xyxy=[[0, 0, 1, 1]],
            confidences=[0.5],
            class_ids=[9],
            class_names=LEGACY_NAMES,
            image_width=640,
            image_height=640,
        )


def test_mismatched_lengths_raise() -> None:
    with pytest.raises(ValueError, match="same length"):
        findings_from_predictions(
            photo_id="p1",
            boxes_xyxy=[[0, 0, 1, 1]],
            confidences=[],
            class_ids=[1],
            class_names=LEGACY_NAMES,
            image_width=640,
            image_height=640,
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_vision.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'claimlens.vision'`

- [ ] **Step 3: Write** `src/claimlens/vision/base.py`

```python
"""Detector interface and conversion from raw model predictions to findings."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Protocol

from claimlens.domain import BoundingBox, DamageFinding, DamageType


class Detector(Protocol):
    @property
    def model_version(self) -> str: ...

    def detect(self, image_path: Path, photo_id: str) -> list[DamageFinding]: ...


def normalize_class_name(name: str) -> DamageType:
    key = name.strip().lower().replace("-", "_").replace(" ", "_")
    try:
        return DamageType(key)
    except ValueError:
        raise ValueError(f"model class {name!r} is not a known damage type") from None


def _clamp(value: float, upper: float) -> float:
    return min(max(float(value), 0.0), upper)


def findings_from_predictions(
    *,
    photo_id: str,
    boxes_xyxy: Sequence[Sequence[float]],
    confidences: Sequence[float],
    class_ids: Sequence[int],
    class_names: Mapping[int, str],
    image_width: int,
    image_height: int,
) -> list[DamageFinding]:
    if not len(boxes_xyxy) == len(confidences) == len(class_ids):
        raise ValueError("boxes, confidences and class ids must have the same length")
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image size must be positive")
    width, height = float(image_width), float(image_height)
    findings: list[DamageFinding] = []
    for box, confidence, class_id in zip(boxes_xyxy, confidences, class_ids, strict=True):
        if class_id not in class_names:
            raise ValueError(f"class id {class_id} is not in the model's class names")
        x1, x2 = _clamp(box[0], width), _clamp(box[2], width)
        y1, y2 = _clamp(box[1], height), _clamp(box[3], height)
        area = max(x2 - x1, 0.0) * max(y2 - y1, 0.0)
        findings.append(
            DamageFinding(
                photo_id=photo_id,
                damage_type=normalize_class_name(class_names[class_id]),
                confidence=float(confidence),
                bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                image_area_fraction=min(area / (width * height), 1.0),
            )
        )
    return findings
```

- [ ] **Step 4: Write** `src/claimlens/vision/legacy_yolo.py`

```python
"""Adapter for the original course model (YOLOv8n, 7 classes). Baseline only.

Requires the optional dependency group: `uv sync --group vision`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from claimlens.domain import DamageFinding
from claimlens.vision.base import findings_from_predictions


class LegacyYoloDetector:
    def __init__(self, weights: Path, conf_floor: float = 0.10) -> None:
        if not weights.is_file():
            raise FileNotFoundError(f"model weights not found: {weights}")
        from ultralytics import YOLO

        self._model: Any = YOLO(str(weights))
        self._conf_floor = conf_floor
        self.model_version = f"legacy-yolov8n:{weights.name}"

    def detect(self, image_path: Path, photo_id: str) -> list[DamageFinding]:
        predictions = self._model.predict(
            source=str(image_path), conf=self._conf_floor, verbose=False
        )
        result = predictions[0]
        height, width = result.orig_shape
        boxes = result.boxes
        return findings_from_predictions(
            photo_id=photo_id,
            boxes_xyxy=boxes.xyxy.tolist(),
            confidences=boxes.conf.tolist(),
            class_ids=[int(c) for c in boxes.cls.tolist()],
            class_names=result.names,
            image_width=int(width),
            image_height=int(height),
        )
```

- [ ] **Step 5: Write the optional integration test** `tests/integration/test_legacy_yolo.py`

```python
import importlib.util
from pathlib import Path

import pytest

from claimlens.domain import DamageType

ROOT = Path(__file__).resolve().parents[2]
WEIGHTS = ROOT / "models" / "legacy" / "yolov8n-cardamage-v6.pt"
SAMPLE = ROOT / "tests" / "fixtures" / "images" / "dent_1.jpg"

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not WEIGHTS.is_file() or importlib.util.find_spec("ultralytics") is None,
        reason="needs models/legacy weights and `uv sync --group vision`",
    ),
]


def test_legacy_detector_returns_known_damage_types() -> None:
    from claimlens.vision.legacy_yolo import LegacyYoloDetector

    detector = LegacyYoloDetector(WEIGHTS)
    findings = detector.detect(SAMPLE, "p1")
    assert detector.model_version == "legacy-yolov8n:yolov8n-cardamage-v6.pt"
    assert all(isinstance(f.damage_type, DamageType) for f in findings)
    assert all(f.photo_id == "p1" for f in findings)
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_vision.py tests/integration -v && uv run mypy`
Expected: 6 passed, 1 skipped (the integration test is skipped without the vision group); mypy reports no issues.

- [ ] **Step 7: Commit**

```bash
git add src/claimlens/vision tests/unit/test_vision.py tests/integration
git commit -m "feat: add detector interface and legacy YOLOv8 adapter using model class names"
```

---

### Task 8: Policy repository

**Files:**
- Create: `config/policies.toml`
- Create: `src/claimlens/policy.py`
- Test: `tests/unit/test_policy.py`

**Interfaces:**
- Consumes: `Coverage`, `Frozen` (Task 1).
- Produces: `PolicyStatus` (StrEnum: active, lapsed); `PolicyRecord(policy_id, status, collision, deductible)`; `PolicyRepository(records)` with `get_coverage(policy_id: str) -> Coverage`; `load_policies(path: Path) -> PolicyRepository`. Config contains `P-1001`…`P-1006` (active, collision), `P-2001` (lapsed), `P-2002` (active, no collision).

- [ ] **Step 1: Write** `config/policies.toml`

```toml
# Fictional policies for development and evaluation. Not real people or insurers.

[[policy]]
policy_id = "P-1001"
status = "active"
collision = true
deductible = 250

[[policy]]
policy_id = "P-1002"
status = "active"
collision = true
deductible = 500

[[policy]]
policy_id = "P-1003"
status = "active"
collision = true
deductible = 500

[[policy]]
policy_id = "P-1004"
status = "active"
collision = true
deductible = 1000

[[policy]]
policy_id = "P-1005"
status = "active"
collision = true
deductible = 500

[[policy]]
policy_id = "P-1006"
status = "active"
collision = true
deductible = 250

[[policy]]
policy_id = "P-2001"
status = "lapsed"
collision = true
deductible = 500

[[policy]]
policy_id = "P-2002"
status = "active"
collision = false
deductible = 500
```

- [ ] **Step 2: Write the failing test** `tests/unit/test_policy.py`

```python
from pathlib import Path

import pytest

from claimlens.policy import PolicyRecord, PolicyRepository, PolicyStatus, load_policies

ROOT = Path(__file__).resolve().parents[2]


def _record(policy_id: str = "P-1") -> PolicyRecord:
    return PolicyRecord(
        policy_id=policy_id, status=PolicyStatus.ACTIVE, collision=True, deductible=500
    )


def test_active_collision_policy_is_confirmed() -> None:
    coverage = PolicyRepository([_record()]).get_coverage("P-1")
    assert coverage.confirmed
    assert coverage.deductible == 500


def test_unknown_policy_is_not_confirmed() -> None:
    coverage = PolicyRepository([]).get_coverage("P-404")
    assert not coverage.found
    assert not coverage.confirmed


def test_duplicate_policy_ids_are_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate policy id P-1"):
        PolicyRepository([_record(), _record()])


def test_repo_config_covers_each_scenario() -> None:
    repo = load_policies(ROOT / "config" / "policies.toml")
    assert repo.get_coverage("P-1001").confirmed
    assert not repo.get_coverage("P-2001").active
    assert not repo.get_coverage("P-2002").collision
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_policy.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'claimlens.policy'`

- [ ] **Step 4: Write** `src/claimlens/policy.py`

```python
"""Mock policy administration system: coverage lookup for fictional policies."""

from __future__ import annotations

import tomllib
from collections.abc import Iterable
from enum import StrEnum
from pathlib import Path

from pydantic import Field

from claimlens.domain import Coverage, Frozen


class PolicyStatus(StrEnum):
    ACTIVE = "active"
    LAPSED = "lapsed"


class PolicyRecord(Frozen):
    policy_id: str
    status: PolicyStatus
    collision: bool
    deductible: int = Field(ge=0)


class PolicyRepository:
    def __init__(self, records: Iterable[PolicyRecord]) -> None:
        self._by_id: dict[str, PolicyRecord] = {}
        for record in records:
            if record.policy_id in self._by_id:
                raise ValueError(f"duplicate policy id {record.policy_id}")
            self._by_id[record.policy_id] = record

    def get_coverage(self, policy_id: str) -> Coverage:
        record = self._by_id.get(policy_id)
        if record is None:
            return Coverage(
                policy_id=policy_id, found=False, active=False, collision=False, deductible=0
            )
        return Coverage(
            policy_id=policy_id,
            found=True,
            active=record.status is PolicyStatus.ACTIVE,
            collision=record.collision,
            deductible=record.deductible,
        )


def load_policies(path: Path) -> PolicyRepository:
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    return PolicyRepository(PolicyRecord.model_validate(item) for item in data.get("policy", []))
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_policy.py -v && uv run mypy`
Expected: 4 passed; mypy reports no issues.

- [ ] **Step 6: Commit**

```bash
git add config/policies.toml src/claimlens/policy.py tests/unit/test_policy.py
git commit -m "feat: add fictional policy repository and coverage lookup"
```

---

### Task 9: Pricing

**Files:**
- Create: `config/rate_card.toml`
- Create: `src/claimlens/pricing.py`
- Test: `tests/unit/test_pricing.py`

**Interfaces:**
- Consumes: `DamageFinding`, `DamageType`, `Severity`, `CostEstimate`, `Frozen` (Task 1).
- Produces: `RateCard(version, minor_max_fraction, moderate_max_fraction, rates)`; `load_rate_card(path: Path) -> RateCard`; `severity_for(fraction: float, card: RateCard) -> Severity`; `estimate_cost(findings: Sequence[DamageFinding], card: RateCard) -> CostEstimate`.

Design note: until part segmentation exists (M3), the same dent seen in two photos would be counted twice. M1 therefore prices each damage type once, at its worst severity across all photos. This is documented in the estimate's `basis` string.

- [ ] **Step 1: Write** `config/rate_card.toml`

```toml
# Fictional USD repair ranges [low, high] for a learning project. Not real market prices.
version = "rate-card-2026.10-v0"

[severity]
# Damage box area as a fraction of the image. Below minor_max is minor, below moderate_max is moderate.
minor_max_fraction = 0.05
moderate_max_fraction = 0.20

[rates.crack]
minor = [200, 500]
moderate = [500, 1500]
severe = [1500, 4000]

[rates.dent]
minor = [150, 400]
moderate = [400, 1200]
severe = [1200, 3500]

[rates.glass_shatter]
minor = [250, 600]
moderate = [600, 1200]
severe = [1200, 2500]

[rates.lamp_broken]
minor = [150, 400]
moderate = [400, 900]
severe = [900, 1800]

[rates.scratch]
minor = [100, 300]
moderate = [300, 800]
severe = [800, 2000]

[rates.tire_flat]
minor = [100, 250]
moderate = [150, 350]
severe = [200, 600]

[rates.smash]
minor = [800, 2000]
moderate = [2000, 5000]
severe = [5000, 12000]
```

- [ ] **Step 2: Write the failing test** `tests/unit/test_pricing.py`

```python
from pathlib import Path

import pytest

from claimlens.domain import BoundingBox, DamageFinding, DamageType, Severity
from claimlens.pricing import estimate_cost, load_rate_card, severity_for

ROOT = Path(__file__).resolve().parents[2]
CARD = load_rate_card(ROOT / "config" / "rate_card.toml")


def _finding(damage_type: DamageType, fraction: float, photo_id: str = "p1") -> DamageFinding:
    return DamageFinding(
        photo_id=photo_id,
        damage_type=damage_type,
        confidence=0.9,
        bbox=BoundingBox(x1=0, y1=0, x2=1, y2=1),
        image_area_fraction=fraction,
    )


def test_severity_bands() -> None:
    assert severity_for(0.01, CARD) is Severity.MINOR
    assert severity_for(0.05, CARD) is Severity.MODERATE
    assert severity_for(0.10, CARD) is Severity.MODERATE
    assert severity_for(0.50, CARD) is Severity.SEVERE


def test_no_findings_cost_nothing() -> None:
    estimate = estimate_cost([], CARD)
    assert (estimate.low, estimate.high) == (0, 0)
    assert "no damage found" in estimate.basis


def test_same_damage_type_is_priced_once_at_its_worst_severity() -> None:
    estimate = estimate_cost(
        [_finding(DamageType.DENT, 0.01), _finding(DamageType.DENT, 0.10, "p2")], CARD
    )
    assert (estimate.low, estimate.high) == (400, 1200)
    assert estimate.basis == "rate-card-2026.10-v0: dent/moderate"


def test_costs_add_up_across_damage_types() -> None:
    estimate = estimate_cost(
        [_finding(DamageType.DENT, 0.01), _finding(DamageType.SCRATCH, 0.01)], CARD
    )
    assert (estimate.low, estimate.high) == (250, 700)


def test_rate_card_must_cover_every_damage_type(tmp_path: Path) -> None:
    text = (ROOT / "config" / "rate_card.toml").read_text(encoding="utf-8")
    broken = text.split("[rates.smash]")[0]
    path = tmp_path / "rate_card.toml"
    path.write_text(broken, encoding="utf-8")
    with pytest.raises(ValueError, match="missing rates for smash"):
        load_rate_card(path)
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_pricing.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'claimlens.pricing'`

- [ ] **Step 4: Write** `src/claimlens/pricing.py`

```python
"""Repair cost range from damage findings and a versioned rate card."""

from __future__ import annotations

import tomllib
from collections.abc import Sequence
from pathlib import Path
from typing import Self

from pydantic import Field, model_validator

from claimlens.domain import CostEstimate, DamageFinding, DamageType, Frozen, Severity

_SEVERITY_ORDER = {Severity.MINOR: 0, Severity.MODERATE: 1, Severity.SEVERE: 2}


class RateCard(Frozen):
    version: str
    minor_max_fraction: float = Field(gt=0.0, lt=1.0)
    moderate_max_fraction: float = Field(gt=0.0, lt=1.0)
    rates: dict[DamageType, dict[Severity, tuple[int, int]]]

    @model_validator(mode="after")
    def _complete(self) -> Self:
        if self.minor_max_fraction >= self.moderate_max_fraction:
            raise ValueError("minor_max_fraction must be below moderate_max_fraction")
        for damage_type in DamageType:
            bands = self.rates.get(damage_type)
            if bands is None or set(bands) != set(Severity):
                raise ValueError(f"rate card is missing rates for {damage_type.value}")
            for low, high in bands.values():
                if not 0 <= low <= high:
                    raise ValueError(f"invalid range for {damage_type.value}: {low}-{high}")
        return self


def load_rate_card(path: Path) -> RateCard:
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    return RateCard.model_validate(
        {"version": data["version"], **data["severity"], "rates": data["rates"]}
    )


def severity_for(fraction: float, card: RateCard) -> Severity:
    if fraction < card.minor_max_fraction:
        return Severity.MINOR
    if fraction < card.moderate_max_fraction:
        return Severity.MODERATE
    return Severity.SEVERE


def estimate_cost(findings: Sequence[DamageFinding], card: RateCard) -> CostEstimate:
    worst: dict[DamageType, Severity] = {}
    for finding in findings:
        severity = severity_for(finding.image_area_fraction, card)
        current = worst.get(finding.damage_type)
        if current is None or _SEVERITY_ORDER[severity] > _SEVERITY_ORDER[current]:
            worst[finding.damage_type] = severity
    if not worst:
        return CostEstimate(low=0, high=0, basis=f"{card.version}: no damage found")
    low = sum(card.rates[t][s][0] for t, s in worst.items())
    high = sum(card.rates[t][s][1] for t, s in worst.items())
    parts = ", ".join(f"{t.value}/{s.value}" for t, s in sorted(worst.items()))
    return CostEstimate(low=low, high=high, basis=f"{card.version}: {parts}")
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_pricing.py -v && uv run mypy`
Expected: 5 passed; mypy reports no issues.

- [ ] **Step 6: Commit**

```bash
git add config/rate_card.toml src/claimlens/pricing.py tests/unit/test_pricing.py
git commit -m "feat: add versioned rate card and cost estimation"
```

---

### Task 10: Integrity check (photo reuse)

**Files:**
- Create: `src/claimlens/integrity.py`
- Test: `tests/unit/test_integrity.py`

**Interfaces:**
- Consumes: `ClaimState` (Task 4), `SQLiteEventStore.claims_with_photo` (Task 3), `FraudSignal` (Task 1).
- Produces: `check_integrity(state: ClaimState, store: SQLiteEventStore) -> tuple[FraudSignal, ...]`. Each reused photo yields `FraudSignal(kind="photo_reuse", score=1.0, detail=...)`.

M1 uses exact byte-level reuse (SHA-256). Perceptual hashing, EXIF checks and synthetic-image detection arrive in M7.

- [ ] **Step 1: Write the failing test** `tests/unit/test_integrity.py`

```python
from collections.abc import Callable
from pathlib import Path

from claimlens.blobs import BlobStore
from claimlens.events.projection import fold
from claimlens.events.store import SQLiteEventStore
from claimlens.intake import submit_claim
from claimlens.integrity import check_integrity


def test_photo_reused_from_another_claim_raises_a_signal(
    store: SQLiteEventStore, tmp_path: Path, make_image: Callable[..., Path]
) -> None:
    blobs = BlobStore(tmp_path / "blobs")
    image = make_image("a.jpg")
    first = submit_claim(store, blobs, policy_id="P-1001", description="", photo_paths=[image])
    second = submit_claim(store, blobs, policy_id="P-1002", description="", photo_paths=[image])

    signals = check_integrity(fold(store.load(second)), store)

    assert len(signals) == 1
    assert signals[0].kind == "photo_reuse"
    assert signals[0].score == 1.0
    assert str(first) in signals[0].detail


def test_same_photo_twice_in_one_claim_is_not_reuse(
    store: SQLiteEventStore, tmp_path: Path, make_image: Callable[..., Path]
) -> None:
    image = make_image("a.jpg")
    claim = submit_claim(
        store,
        BlobStore(tmp_path / "blobs"),
        policy_id="P-1001",
        description="",
        photo_paths=[image, image],
    )
    assert check_integrity(fold(store.load(claim)), store) == ()


def test_fresh_photos_raise_nothing(
    store: SQLiteEventStore, tmp_path: Path, make_image: Callable[..., Path]
) -> None:
    claim = submit_claim(
        store,
        BlobStore(tmp_path / "blobs"),
        policy_id="P-1001",
        description="",
        photo_paths=[make_image("a.jpg")],
    )
    assert check_integrity(fold(store.load(claim)), store) == ()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_integrity.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'claimlens.integrity'`

- [ ] **Step 3: Write** `src/claimlens/integrity.py`

```python
"""Integrity checks that raise fraud signals. M1: exact photo reuse across claims."""

from __future__ import annotations

from claimlens.domain import FraudSignal
from claimlens.events.projection import ClaimState
from claimlens.events.store import SQLiteEventStore


def check_integrity(state: ClaimState, store: SQLiteEventStore) -> tuple[FraudSignal, ...]:
    signals: list[FraudSignal] = []
    for photo in state.photos.values():
        other_claims = store.claims_with_photo(photo.sha256) - {state.claim_id}
        if other_claims:
            ids = ", ".join(sorted(str(claim) for claim in other_claims))
            signals.append(
                FraudSignal(
                    kind="photo_reuse",
                    score=1.0,
                    detail=f"Photo {photo.photo_id} also appears in claim(s) {ids}.",
                )
            )
    return tuple(signals)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_integrity.py -v && uv run mypy`
Expected: 3 passed; mypy reports no issues.

- [ ] **Step 5: Commit**

```bash
git add src/claimlens/integrity.py tests/unit/test_integrity.py
git commit -m "feat: flag photos reused across claims"
```

---

### Task 11: Decision policy

**Files:**
- Create: `config/decision_policy.toml`
- Create: `src/claimlens/decision.py`
- Modify: `docs/specs/2026-10-01-claimlens-design.md` (section 11 rule table)
- Test: `tests/unit/test_decision.py`

**Interfaces:**
- Consumes: `ClaimState`, `PhotoRecord`, `PhotoStatus` (Task 4), domain types (Task 1), `StageFailed` (Task 2).
- Produces: `DecisionConfig(version, fraud_score_threshold, max_fast_track_cost_usd, min_finding_confidence)`; `load_decision_config(path: Path) -> DecisionConfig`; `decide(state: ClaimState, config: DecisionConfig) -> Decision`. Rule ids `R1`–`R9` as listed below.

| Rule | Condition | Route |
|---|---|---|
| R1 | any fraud signal score >= threshold | FRAUD_REVIEW |
| R2 | any stage failed | ADJUSTER_REVIEW |
| R3 | no accepted photos | ADJUSTER_REVIEW |
| R4 | coverage missing or not confirmed | ADJUSTER_REVIEW |
| R5 | no estimate, or estimate high > limit | ADJUSTER_REVIEW |
| R6 | any finding below the confidence threshold | ADJUSTER_REVIEW |
| R7 | no recommendation, agent confidence low, or open questions | ADJUSTER_REVIEW |
| R8 | agent did not suggest FAST_TRACK | ADJUSTER_REVIEW |
| R9 | otherwise | FAST_TRACK |

- [ ] **Step 1: Write** `config/decision_policy.toml`

```toml
# Decision policy thresholds (spec section 11). Versioned with the code.
version = "decision-policy-v0"
fraud_score_threshold = 0.5
max_fast_track_cost_usd = 3000
# Provisional value for the legacy model. Recalibrated from the PR curve in M3.
min_finding_confidence = 0.40
```

- [ ] **Step 2: Write the failing test** `tests/unit/test_decision.py`

```python
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest

from claimlens.decision import DecisionConfig, decide, load_decision_config
from claimlens.domain import (
    AgentRecommendation,
    BoundingBox,
    Confidence,
    CostEstimate,
    Coverage,
    DamageFinding,
    DamageType,
    FraudSignal,
    Route,
)
from claimlens.events.payloads import StageFailed
from claimlens.events.projection import ClaimState, PhotoRecord, PhotoStatus

ROOT = Path(__file__).resolve().parents[2]
CONFIG = DecisionConfig(
    version="test-policy",
    fraud_score_threshold=0.5,
    max_fast_track_cost_usd=3000,
    min_finding_confidence=0.4,
)


def _finding(confidence: float = 0.9) -> DamageFinding:
    return DamageFinding(
        photo_id="p1",
        damage_type=DamageType.DENT,
        confidence=confidence,
        bbox=BoundingBox(x1=0, y1=0, x2=64, y2=64),
        image_area_fraction=0.01,
    )


def _recommendation(**changes: Any) -> AgentRecommendation:
    base = AgentRecommendation(
        route_suggestion=Route.FAST_TRACK,
        confidence=Confidence.MEDIUM,
        rationale="ok",
        citations=("e1",),
    )
    return base.model_copy(update=changes)


def _ready_state(**changes: Any) -> ClaimState:
    state = ClaimState(claim_id=UUID(int=1), policy_id="P-1001", description="")
    state.photos["p1"] = PhotoRecord(
        photo_id="p1",
        sha256="a" * 64,
        filename="p1.jpg",
        blob_name="x.jpg",
        status=PhotoStatus.ACCEPTED,
    )
    state.findings = [_finding()]
    state.integrity_checked = True
    state.cost_estimate = CostEstimate(low=400, high=1200, basis="test")
    state.coverage = Coverage(
        policy_id="P-1001", found=True, active=True, collision=True, deductible=500
    )
    state.recommendation = _recommendation()
    for name, value in changes.items():
        setattr(state, name, value)
    return state


FRAUD = [FraudSignal(kind="photo_reuse", score=1.0, detail="Photo p1 reused.")]
FAILURE = [StageFailed(stage="perception", photo_id="p1", error="boom")]
LAPSED = Coverage(policy_id="P-1001", found=True, active=False, collision=True, deductible=500)


@pytest.mark.parametrize(
    ("changes", "route", "rule"),
    [
        ({}, Route.FAST_TRACK, "R9"),
        ({"fraud_signals": FRAUD, "failures": FAILURE}, Route.FRAUD_REVIEW, "R1"),
        ({"failures": FAILURE}, Route.ADJUSTER_REVIEW, "R2"),
        ({"photos": {}}, Route.ADJUSTER_REVIEW, "R3"),
        ({"coverage": None}, Route.ADJUSTER_REVIEW, "R4"),
        ({"coverage": LAPSED}, Route.ADJUSTER_REVIEW, "R4"),
        (
            {"cost_estimate": CostEstimate(low=2000, high=3500, basis="t")},
            Route.ADJUSTER_REVIEW,
            "R5",
        ),
        ({"findings": [_finding(confidence=0.2)]}, Route.ADJUSTER_REVIEW, "R6"),
        ({"recommendation": None}, Route.ADJUSTER_REVIEW, "R7"),
        (
            {"recommendation": _recommendation(confidence=Confidence.LOW)},
            Route.ADJUSTER_REVIEW,
            "R7",
        ),
        (
            {"recommendation": _recommendation(open_questions=("why?",))},
            Route.ADJUSTER_REVIEW,
            "R7",
        ),
        (
            {"recommendation": _recommendation(route_suggestion=Route.ADJUSTER_REVIEW)},
            Route.ADJUSTER_REVIEW,
            "R8",
        ),
    ],
)
def test_decision_rules(changes: dict[str, Any], route: Route, rule: str) -> None:
    decision = decide(_ready_state(**changes), CONFIG)
    assert (decision.route, decision.rule_id) == (route, rule)
    assert decision.policy_version == "test-policy"


def test_low_fraud_score_does_not_trigger_fraud_review() -> None:
    weak = [FraudSignal(kind="exif_mismatch", score=0.2, detail="minor")]
    assert decide(_ready_state(fraud_signals=weak), CONFIG).route is Route.FAST_TRACK


def test_repo_config_loads() -> None:
    config = load_decision_config(ROOT / "config" / "decision_policy.toml")
    assert config.version == "decision-policy-v0"
    assert config.max_fast_track_cost_usd == 3000
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_decision.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'claimlens.decision'`

- [ ] **Step 4: Write** `src/claimlens/decision.py`

```python
"""Deterministic decision policy: the rules that make the final routing call (spec section 11)."""

from __future__ import annotations

import tomllib
from pathlib import Path

from pydantic import Field

from claimlens.domain import Confidence, Decision, Frozen, Route
from claimlens.events.projection import ClaimState


class DecisionConfig(Frozen):
    version: str
    fraud_score_threshold: float = Field(ge=0.0, le=1.0)
    max_fast_track_cost_usd: int = Field(ge=0)
    min_finding_confidence: float = Field(ge=0.0, le=1.0)


def load_decision_config(path: Path) -> DecisionConfig:
    return DecisionConfig.model_validate(tomllib.loads(path.read_text(encoding="utf-8")))


def decide(state: ClaimState, config: DecisionConfig) -> Decision:
    """Apply rules in order; the first match wins. No rule can deny a claim."""

    def result(route: Route, rule_id: str, reason: str) -> Decision:
        return Decision(route=route, rule_id=rule_id, reason=reason, policy_version=config.version)

    flagged = [s for s in state.fraud_signals if s.score >= config.fraud_score_threshold]
    if flagged:
        return result(Route.FRAUD_REVIEW, "R1", " ".join(s.detail for s in flagged))
    if state.failures:
        stages = ", ".join(sorted({f.stage for f in state.failures}))
        return result(Route.ADJUSTER_REVIEW, "R2", f"Processing failed at: {stages}.")
    if not state.accepted_photos:
        return result(Route.ADJUSTER_REVIEW, "R3", "No usable photos were provided.")
    if state.coverage is None or not state.coverage.confirmed:
        return result(Route.ADJUSTER_REVIEW, "R4", "Coverage could not be confirmed.")
    limit = config.max_fast_track_cost_usd
    if state.cost_estimate is None or state.cost_estimate.high > limit:
        return result(
            Route.ADJUSTER_REVIEW, "R5", f"Estimate missing or above the ${limit:,} limit."
        )
    threshold = config.min_finding_confidence
    uncertain = [f for f in state.findings if f.confidence < threshold]
    if uncertain:
        return result(
            Route.ADJUSTER_REVIEW,
            "R6",
            f"{len(uncertain)} finding(s) below confidence {threshold:.2f}.",
        )
    rec = state.recommendation
    if rec is None or rec.confidence is Confidence.LOW or rec.open_questions:
        return result(
            Route.ADJUSTER_REVIEW, "R7", "The triage agent is unsure or has open questions."
        )
    if rec.route_suggestion is not Route.FAST_TRACK:
        return result(
            Route.ADJUSTER_REVIEW, "R8", "The triage agent did not recommend fast-tracking."
        )
    return result(Route.FAST_TRACK, "R9", "All checks passed.")
```

- [ ] **Step 5: Update the spec's rule table**

In `docs/specs/2026-10-01-claimlens-design.md` section 11, replace the table with the R1–R9 table from this task's header (keeping the sentence above it), adding a column `Rule` with the ids, and note under it: "R2 implements the fail-safe behaviour from section 17."

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_decision.py -v && uv run mypy && uv run ruff check .`
Expected: 14 passed; mypy and ruff report no issues. If ruff flags E501 in the parametrize list, let `uv run ruff format tests/unit/test_decision.py` wrap it.

- [ ] **Step 7: Commit**

```bash
git add config/decision_policy.toml src/claimlens/decision.py tests/unit/test_decision.py docs/specs/2026-10-01-claimlens-design.md
git commit -m "feat: add deterministic decision policy with rules R1-R9"
```

---

### Task 12: Stub triage agent

**Files:**
- Create: `src/claimlens/agent/__init__.py`
- Create: `src/claimlens/agent/stub.py`
- Test: `tests/unit/test_agent_stub.py`

**Interfaces:**
- Consumes: `ClaimState` (Task 4), `AgentRecommendation`, `Confidence`, `Route` (Task 1).
- Produces: `TriageAgent` protocol (`agent_version: str` read-only property; `recommend(state: ClaimState) -> AgentRecommendation`); `StubTriageAgent` with class attribute `agent_version = "stub-v0"`. The LLM triage agent replaces it in M5 behind the same protocol.

- [ ] **Step 1: Write the failing test** `tests/unit/test_agent_stub.py`

```python
from uuid import UUID

from claimlens.agent.stub import StubTriageAgent
from claimlens.domain import BoundingBox, Confidence, DamageFinding, DamageType, Route
from claimlens.events.projection import ClaimState


def _state() -> ClaimState:
    return ClaimState(claim_id=UUID(int=1), policy_id="P-1001", description="")


def test_no_findings_asks_for_human_review() -> None:
    rec = StubTriageAgent().recommend(_state())
    assert rec.route_suggestion is Route.ADJUSTER_REVIEW
    assert rec.confidence is Confidence.LOW
    assert rec.open_questions


def test_findings_suggest_fast_track_with_citations() -> None:
    state = _state()
    state.findings = [
        DamageFinding(
            photo_id="p1",
            damage_type=DamageType.DENT,
            confidence=0.9,
            bbox=BoundingBox(x1=0, y1=0, x2=1, y2=1),
            image_area_fraction=0.01,
        )
    ]
    state.detection_event_ids = {"p1": "event-123"}
    rec = StubTriageAgent().recommend(state)
    assert rec.route_suggestion is Route.FAST_TRACK
    assert rec.confidence is Confidence.MEDIUM
    assert rec.citations == ("event-123",)
    assert "dent" in rec.rationale
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_agent_stub.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'claimlens.agent'`

- [ ] **Step 3: Write** `src/claimlens/agent/__init__.py`

```python
"""Triage agents. Every agent advises; the decision policy decides."""

from __future__ import annotations

from typing import Protocol

from claimlens.domain import AgentRecommendation
from claimlens.events.projection import ClaimState


class TriageAgent(Protocol):
    @property
    def agent_version(self) -> str: ...

    def recommend(self, state: ClaimState) -> AgentRecommendation: ...
```

- [ ] **Step 4: Write** `src/claimlens/agent/stub.py`

```python
"""Placeholder triage agent for M1. Replaced by the LLM triage agent in M5."""

from __future__ import annotations

from claimlens.domain import AgentRecommendation, Confidence, Route
from claimlens.events.projection import ClaimState


class StubTriageAgent:
    agent_version = "stub-v0"

    def recommend(self, state: ClaimState) -> AgentRecommendation:
        citations = tuple(state.detection_event_ids.values())
        if not state.findings:
            return AgentRecommendation(
                route_suggestion=Route.ADJUSTER_REVIEW,
                confidence=Confidence.LOW,
                rationale="No damage was detected in the accepted photos.",
                citations=citations,
                open_questions=("Is there damage that the photos do not show?",),
            )
        damage_types = ", ".join(sorted({f.damage_type.value for f in state.findings}))
        return AgentRecommendation(
            route_suggestion=Route.FAST_TRACK,
            confidence=Confidence.MEDIUM,
            rationale=f"Detected {len(state.findings)} damage finding(s): {damage_types}.",
            citations=citations,
        )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_agent_stub.py -v && uv run mypy`
Expected: 2 passed; mypy reports no issues.

- [ ] **Step 6: Commit**

```bash
git add src/claimlens/agent tests/unit/test_agent_stub.py
git commit -m "feat: add triage agent protocol and M1 stub agent"
```

---

### Task 13: Workflow orchestration

**Files:**
- Create: `src/claimlens/workflow.py`
- Create: `tests/fakes.py`
- Test: `tests/unit/test_workflow.py`

**Interfaces:**
- Consumes: everything from Tasks 2–12.
- Produces: `WORKFLOW: Actor`; `MAX_ATTEMPTS = 2`; `PipelineDeps(store, blobs, detector, policies, rate_card, decision_config, agent, quality=QualityConfig())` (frozen dataclass); `process_claim(claim_id: UUID, deps: PipelineDeps) -> Decision`. Test helpers in `tests/fakes.py`: `ROOT`, `CONFIG_DIR`, `SimulatedCrashError(BaseException)`, `dent(photo_id="p1", confidence=0.9) -> DamageFinding`, `FakeDetector(*, findings=None, fail_photos=None, crash_on=None)` with `.calls: list[str]`, `FailingAgent`, `make_test_deps(workdir, store, detector, agent=None) -> PipelineDeps`.

Stage order and idempotency rules:

| Stage | Runs for | Skips when | Appends |
|---|---|---|---|
| quality | each `UPLOADED` photo | photo already accepted/rejected | `PhotoAccepted` or `PhotoRejected` (duplicate bytes in the same claim are rejected as `duplicate of pN`) |
| perception | each accepted photo | photo already detected, or perception already failed for it | `DamageDetected` or `StageFailed` after 2 attempts |
| integrity | once | `integrity_checked` | `IntegrityChecked` |
| pricing | once | estimate exists | `CostEstimated` |
| coverage | once | coverage exists | `PolicyRetrieved` |
| agent | once | recommendation exists or agent already failed | `AgentRecommended` or `StageFailed` after 2 attempts |
| decision | once | decision exists (then `process_claim` returns it immediately) | `RouteDecided` |

`Exception` subclasses from a stage are caught and recorded; `BaseException` (process death) propagates, which is how the crash test simulates a kill.

- [ ] **Step 1: Write the test helpers** `tests/fakes.py`

```python
"""Test doubles shared across test modules."""

from __future__ import annotations

from pathlib import Path

from claimlens.agent import TriageAgent
from claimlens.agent.stub import StubTriageAgent
from claimlens.blobs import BlobStore
from claimlens.decision import load_decision_config
from claimlens.domain import AgentRecommendation, BoundingBox, DamageFinding, DamageType
from claimlens.events.projection import ClaimState
from claimlens.events.store import SQLiteEventStore
from claimlens.policy import load_policies
from claimlens.pricing import load_rate_card
from claimlens.workflow import PipelineDeps

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "config"


class SimulatedCrashError(BaseException):
    """Stands in for the process dying mid-pipeline. Stage error handling does not catch it."""


def dent(photo_id: str = "p1", confidence: float = 0.9) -> DamageFinding:
    return DamageFinding(
        photo_id=photo_id,
        damage_type=DamageType.DENT,
        confidence=confidence,
        bbox=BoundingBox(x1=0, y1=0, x2=64, y2=64),
        image_area_fraction=0.01,
    )


class FakeDetector:
    model_version = "fake-detector-v1"

    def __init__(
        self,
        *,
        findings: dict[str, list[DamageFinding]] | None = None,
        fail_photos: frozenset[str] | None = None,
        crash_on: str | None = None,
    ) -> None:
        self._findings = findings
        self._fail_photos = fail_photos or frozenset()
        self._crash_on = crash_on
        self.calls: list[str] = []

    def detect(self, image_path: Path, photo_id: str) -> list[DamageFinding]:
        self.calls.append(photo_id)
        if photo_id == self._crash_on:
            raise SimulatedCrashError(photo_id)
        if photo_id in self._fail_photos:
            raise RuntimeError("model server unavailable")
        if self._findings is None:
            return [dent(photo_id)]
        return list(self._findings.get(photo_id, []))


class FailingAgent:
    agent_version = "failing-agent-v1"

    def recommend(self, state: ClaimState) -> AgentRecommendation:
        raise RuntimeError("LLM provider timeout")


def make_test_deps(
    workdir: Path,
    store: SQLiteEventStore,
    detector: FakeDetector,
    agent: TriageAgent | None = None,
) -> PipelineDeps:
    return PipelineDeps(
        store=store,
        blobs=BlobStore(workdir / "blobs"),
        detector=detector,
        policies=load_policies(CONFIG_DIR / "policies.toml"),
        rate_card=load_rate_card(CONFIG_DIR / "rate_card.toml"),
        decision_config=load_decision_config(CONFIG_DIR / "decision_policy.toml"),
        agent=agent or StubTriageAgent(),
    )
```

- [ ] **Step 2: Write the failing test** `tests/unit/test_workflow.py`

```python
import sqlite3
from collections.abc import Callable, Sequence
from pathlib import Path
from uuid import UUID

import pytest

from claimlens.domain import Route
from claimlens.events.envelope import ChainIntegrityError
from claimlens.events.projection import PhotoStatus, fold
from claimlens.events.store import SQLiteEventStore
from claimlens.intake import submit_claim
from claimlens.workflow import PipelineDeps, process_claim
from tests.fakes import FailingAgent, FakeDetector, SimulatedCrashError, make_test_deps


def _submit(deps: PipelineDeps, photos: Sequence[Path], policy_id: str = "P-1001") -> UUID:
    return submit_claim(
        deps.store, deps.blobs, policy_id=policy_id, description="scrape", photo_paths=photos
    )


def _types(store: SQLiteEventStore, claim_id: UUID) -> list[str]:
    return [event.type for event in store.load(claim_id)]


def test_clean_claim_is_fast_tracked_with_full_audit_trail(
    store: SQLiteEventStore, tmp_path: Path, make_image: Callable[..., Path]
) -> None:
    deps = make_test_deps(tmp_path, store, FakeDetector())
    claim_id = _submit(deps, [make_image("a.jpg")])

    decision = process_claim(claim_id, deps)

    assert (decision.route, decision.rule_id) == (Route.FAST_TRACK, "R9")
    assert _types(store, claim_id) == [
        "ClaimReported",
        "PhotoUploaded",
        "PhotoAccepted",
        "DamageDetected",
        "IntegrityChecked",
        "CostEstimated",
        "PolicyRetrieved",
        "AgentRecommended",
        "RouteDecided",
    ]


def test_processing_twice_changes_nothing(
    store: SQLiteEventStore, tmp_path: Path, make_image: Callable[..., Path]
) -> None:
    detector = FakeDetector()
    deps = make_test_deps(tmp_path, store, detector)
    claim_id = _submit(deps, [make_image("a.jpg")])

    first = process_claim(claim_id, deps)
    count = len(store.load(claim_id))
    second = process_claim(claim_id, deps)

    assert first == second
    assert len(store.load(claim_id)) == count
    assert detector.calls == ["p1"]


def test_resume_after_crash_does_not_repeat_work(
    store: SQLiteEventStore, tmp_path: Path, make_image: Callable[..., Path]
) -> None:
    photos = [make_image("a.jpg"), make_image("b.jpg", color=(0, 90, 0))]
    crashing = FakeDetector(crash_on="p2")
    claim_id = _submit(make_test_deps(tmp_path, store, crashing), photos)

    with pytest.raises(SimulatedCrashError):
        process_claim(claim_id, make_test_deps(tmp_path, store, crashing))

    healthy = FakeDetector()
    decision = process_claim(claim_id, make_test_deps(tmp_path, store, healthy))

    assert decision.route is Route.FAST_TRACK
    assert healthy.calls == ["p2"]
    assert _types(store, claim_id).count("DamageDetected") == 2


def test_detector_failure_is_retried_then_routed_to_adjuster(
    store: SQLiteEventStore, tmp_path: Path, make_image: Callable[..., Path]
) -> None:
    detector = FakeDetector(fail_photos=frozenset({"p1"}))
    deps = make_test_deps(tmp_path, store, detector)
    claim_id = _submit(deps, [make_image("a.jpg")])

    decision = process_claim(claim_id, deps)

    assert (decision.route, decision.rule_id) == (Route.ADJUSTER_REVIEW, "R2")
    assert detector.calls == ["p1", "p1"]


def test_duplicate_photo_in_claim_is_rejected_not_fraud(
    store: SQLiteEventStore, tmp_path: Path, make_image: Callable[..., Path]
) -> None:
    image = make_image("a.jpg")
    deps = make_test_deps(tmp_path, store, FakeDetector())
    claim_id = _submit(deps, [image, image])

    decision = process_claim(claim_id, deps)
    state = fold(store.load(claim_id))

    assert decision.route is Route.FAST_TRACK
    assert state.photos["p2"].status is PhotoStatus.REJECTED
    assert state.photos["p2"].reject_reason == "duplicate of p1"
    assert state.fraud_signals == []


def test_photo_reused_across_claims_goes_to_fraud_review(
    store: SQLiteEventStore, tmp_path: Path, make_image: Callable[..., Path]
) -> None:
    image = make_image("a.jpg")
    deps = make_test_deps(tmp_path, store, FakeDetector())
    process_claim(_submit(deps, [image]), deps)

    decision = process_claim(_submit(deps, [image], policy_id="P-1002"), deps)

    assert (decision.route, decision.rule_id) == (Route.FRAUD_REVIEW, "R1")


def test_unusable_photos_route_to_adjuster(
    store: SQLiteEventStore, tmp_path: Path, make_image: Callable[..., Path]
) -> None:
    detector = FakeDetector()
    deps = make_test_deps(tmp_path, store, detector)
    corrupt = tmp_path / "corrupt.jpg"
    corrupt.write_text("not an image", encoding="utf-8")
    claim_id = _submit(deps, [make_image("tiny.png", size=(100, 100)), corrupt])

    decision = process_claim(claim_id, deps)

    assert (decision.route, decision.rule_id) == (Route.ADJUSTER_REVIEW, "R3")
    assert detector.calls == []


def test_lapsed_policy_routes_to_adjuster(
    store: SQLiteEventStore, tmp_path: Path, make_image: Callable[..., Path]
) -> None:
    deps = make_test_deps(tmp_path, store, FakeDetector())
    decision = process_claim(_submit(deps, [make_image("a.jpg")], policy_id="P-2001"), deps)
    assert (decision.route, decision.rule_id) == (Route.ADJUSTER_REVIEW, "R4")


def test_agent_failure_routes_to_adjuster(
    store: SQLiteEventStore, tmp_path: Path, make_image: Callable[..., Path]
) -> None:
    deps = make_test_deps(tmp_path, store, FakeDetector(), agent=FailingAgent())
    decision = process_claim(_submit(deps, [make_image("a.jpg")]), deps)
    assert (decision.route, decision.rule_id) == (Route.ADJUSTER_REVIEW, "R2")


def test_tampered_log_stops_processing(
    store: SQLiteEventStore, tmp_path: Path, make_image: Callable[..., Path]
) -> None:
    deps = make_test_deps(tmp_path, store, FakeDetector())
    claim_id = _submit(deps, [make_image("a.jpg")])
    with sqlite3.connect(tmp_path / "events.db") as conn:
        conn.execute("UPDATE events SET data = replace(data, 'P-1001', 'P-1002') WHERE seq = 1")

    with pytest.raises(ChainIntegrityError):
        process_claim(claim_id, deps)
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_workflow.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'claimlens.workflow'`

- [ ] **Step 4: Write** `src/claimlens/workflow.py`

```python
"""Deterministic claim workflow. Each stage reads folded state and appends events."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from uuid import UUID

from claimlens.agent import TriageAgent
from claimlens.blobs import BlobStore
from claimlens.decision import DecisionConfig, decide
from claimlens.domain import Decision
from claimlens.events.envelope import Actor, ActorKind
from claimlens.events.payloads import (
    AgentRecommended,
    CostEstimated,
    DamageDetected,
    IntegrityChecked,
    Payload,
    PhotoAccepted,
    PhotoRejected,
    PolicyRetrieved,
    RouteDecided,
    StageFailed,
)
from claimlens.events.projection import ClaimState, PhotoStatus, fold
from claimlens.events.store import SQLiteEventStore
from claimlens.integrity import check_integrity
from claimlens.policy import PolicyRepository
from claimlens.pricing import RateCard, estimate_cost
from claimlens.quality import QualityConfig, check_quality
from claimlens.vision.base import Detector

WORKFLOW = Actor(kind=ActorKind.SYSTEM, name="workflow")
MAX_ATTEMPTS = 2


@dataclass(frozen=True)
class PipelineDeps:
    store: SQLiteEventStore
    blobs: BlobStore
    detector: Detector
    policies: PolicyRepository
    rate_card: RateCard
    decision_config: DecisionConfig
    agent: TriageAgent
    quality: QualityConfig = field(default_factory=QualityConfig)


def process_claim(claim_id: UUID, deps: PipelineDeps) -> Decision:
    """Run every unfinished stage, then decide. Safe to call again after any interruption."""
    state = _load(deps, claim_id)
    if state.decision is not None:
        return state.decision
    stages: tuple[Callable[[ClaimState, PipelineDeps], None], ...] = (
        _quality_stage,
        _perception_stage,
        _integrity_stage,
        _pricing_stage,
        _coverage_stage,
        _agent_stage,
    )
    for stage in stages:
        stage(_load(deps, claim_id), deps)
    decision = decide(_load(deps, claim_id), deps.decision_config)
    deps.store.append(claim_id, RouteDecided(decision=decision), WORKFLOW)
    return decision


def _load(deps: PipelineDeps, claim_id: UUID) -> ClaimState:
    return fold(deps.store.load(claim_id))


def _attempt[T](call: Callable[[], T]) -> T | Exception:
    last: Exception = RuntimeError("no attempt was made")
    for _ in range(MAX_ATTEMPTS):
        try:
            return call()
        except Exception as exc:
            last = exc
    return last


def _describe(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"[:300]


def _quality_stage(state: ClaimState, deps: PipelineDeps) -> None:
    seen = {p.sha256: p.photo_id for p in state.photos.values() if p.status is PhotoStatus.ACCEPTED}
    for photo in state.photos.values():
        if photo.status is not PhotoStatus.UPLOADED:
            continue
        payload: Payload
        if photo.sha256 in seen:
            payload = PhotoRejected(
                photo_id=photo.photo_id, reason=f"duplicate of {seen[photo.sha256]}"
            )
        else:
            result = check_quality(deps.blobs.path(photo.blob_name), deps.quality)
            if result.ok:
                payload = PhotoAccepted(
                    photo_id=photo.photo_id, width=result.width, height=result.height
                )
                seen[photo.sha256] = photo.photo_id
            else:
                payload = PhotoRejected(photo_id=photo.photo_id, reason=result.reason)
        deps.store.append(state.claim_id, payload, WORKFLOW)


def _perception_stage(state: ClaimState, deps: PipelineDeps) -> None:
    for photo in state.accepted_photos:
        done = photo.photo_id in state.detection_event_ids
        if done or state.failed("perception", photo.photo_id):
            continue
        image = deps.blobs.path(photo.blob_name)
        outcome = _attempt(partial(deps.detector.detect, image, photo.photo_id))
        payload: Payload
        if isinstance(outcome, Exception):
            payload = StageFailed(
                stage="perception", photo_id=photo.photo_id, error=_describe(outcome)
            )
        else:
            payload = DamageDetected(
                photo_id=photo.photo_id,
                model_version=deps.detector.model_version,
                findings=tuple(outcome),
            )
        deps.store.append(state.claim_id, payload, WORKFLOW)


def _integrity_stage(state: ClaimState, deps: PipelineDeps) -> None:
    if not state.integrity_checked:
        signals = check_integrity(state, deps.store)
        deps.store.append(state.claim_id, IntegrityChecked(signals=signals), WORKFLOW)


def _pricing_stage(state: ClaimState, deps: PipelineDeps) -> None:
    if state.cost_estimate is None:
        estimate = estimate_cost(state.findings, deps.rate_card)
        deps.store.append(state.claim_id, CostEstimated(estimate=estimate), WORKFLOW)


def _coverage_stage(state: ClaimState, deps: PipelineDeps) -> None:
    if state.coverage is None:
        coverage = deps.policies.get_coverage(state.policy_id)
        deps.store.append(state.claim_id, PolicyRetrieved(coverage=coverage), WORKFLOW)


def _agent_stage(state: ClaimState, deps: PipelineDeps) -> None:
    if state.recommendation is not None or state.failed("agent"):
        return
    outcome = _attempt(partial(deps.agent.recommend, state))
    if isinstance(outcome, Exception):
        deps.store.append(
            state.claim_id, StageFailed(stage="agent", error=_describe(outcome)), WORKFLOW
        )
        return
    actor = Actor(kind=ActorKind.AGENT, name=deps.agent.agent_version)
    deps.store.append(
        state.claim_id,
        AgentRecommended(agent_version=deps.agent.agent_version, recommendation=outcome),
        actor,
    )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_workflow.py -v && uv run mypy && uv run ruff check .`
Expected: 10 passed; mypy and ruff report no issues.

- [ ] **Step 6: Run the whole suite**

Run: `uv run pytest`
Expected: all tests pass, 1 skipped (legacy YOLO integration test).

- [ ] **Step 7: Commit**

```bash
git add src/claimlens/workflow.py tests/fakes.py tests/unit/test_workflow.py
git commit -m "feat: add resumable claim workflow wiring every stage"
```

---

### Task 14: Command-line interface

**Files:**
- Create: `src/claimlens/cli.py`
- Modify: `pyproject.toml` (add `[project.scripts]`)
- Modify: `.gitignore` (add `/var/`)
- Modify: `README.md` (Getting started)
- Test: `tests/unit/test_cli.py`

**Interfaces:**
- Consumes: `process_claim`, `PipelineDeps` (Task 13), `submit_claim` (Task 5), loaders (Tasks 8, 9, 11), `StubTriageAgent` (Task 12), `SQLiteEventStore`, `ClaimNotFoundError` (Task 3), `ChainIntegrityError` (Task 2), `Detector` (Task 7).
- Produces: `DetectorFactory = Callable[[Path], Detector]`; `build_parser() -> argparse.ArgumentParser`; `make_deps(store, blobs, config_dir, detector) -> PipelineDeps`; `format_summary(state: ClaimState) -> str`; `format_audit_trail(events) -> str`; `main(argv=None, *, detector_factory=_legacy_detector) -> int`. Exit codes: 0 success, 1 audit log failed verification, 2 unknown claim.

- [ ] **Step 1: Write the failing test** `tests/unit/test_cli.py`

```python
import sqlite3
from collections.abc import Callable
from pathlib import Path
from uuid import UUID

import pytest

from claimlens.cli import main
from tests.fakes import CONFIG_DIR, FakeDetector


def _cli(tmp_path: Path, *args: str) -> int:
    base = ["--db", str(tmp_path / "claims.db"), "--blobs", str(tmp_path / "blobs")]
    return main(
        [*base, "--config", str(CONFIG_DIR), *args], detector_factory=lambda _: FakeDetector()
    )


def _run_claim(
    tmp_path: Path, make_image: Callable[..., Path], capsys: pytest.CaptureFixture[str]
) -> str:
    code = _cli(
        tmp_path, "run", "--policy", "P-1001", "--description", "scrape", str(make_image("a.jpg"))
    )
    assert code == 0
    out = capsys.readouterr().out
    claim_line = next(line for line in out.splitlines() if line.startswith("Claim: "))
    return claim_line.removeprefix("Claim: ")


def test_run_prints_the_decision(
    tmp_path: Path, make_image: Callable[..., Path], capsys: pytest.CaptureFixture[str]
) -> None:
    code = _cli(tmp_path, "run", "--policy", "P-1001", str(make_image("a.jpg")))
    out = capsys.readouterr().out
    assert code == 0
    assert "Route: FAST_TRACK (R9)" in out
    assert "Findings: 1 (dent 0.90)" in out
    assert "Cost estimate: $150-$400 USD" in out


def test_show_prints_the_audit_trail(
    tmp_path: Path, make_image: Callable[..., Path], capsys: pytest.CaptureFixture[str]
) -> None:
    claim_id = _run_claim(tmp_path, make_image, capsys)
    assert _cli(tmp_path, "show", claim_id) == 0
    out = capsys.readouterr().out
    assert "Audit trail:" in out
    assert "RouteDecided" in out
    assert "agent:stub-v0" in out


def test_verify_reports_a_valid_chain(
    tmp_path: Path, make_image: Callable[..., Path], capsys: pytest.CaptureFixture[str]
) -> None:
    claim_id = _run_claim(tmp_path, make_image, capsys)
    assert _cli(tmp_path, "verify", claim_id) == 0
    assert "Chain OK: 9 events" in capsys.readouterr().out


def test_verify_detects_tampering(
    tmp_path: Path, make_image: Callable[..., Path], capsys: pytest.CaptureFixture[str]
) -> None:
    claim_id = _run_claim(tmp_path, make_image, capsys)
    with sqlite3.connect(tmp_path / "claims.db") as conn:
        conn.execute("UPDATE events SET data = replace(data, 'P-1001', 'P-1002') WHERE seq = 1")
    assert _cli(tmp_path, "verify", claim_id) == 1
    assert "failed verification" in capsys.readouterr().err


def test_unknown_claim_returns_2(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert _cli(tmp_path, "show", str(UUID(int=5))) == 2
    assert "no events found" in capsys.readouterr().err
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_cli.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'claimlens.cli'`

- [ ] **Step 3: Write** `src/claimlens/cli.py`

```python
"""Command-line interface: `claimlens run | show | verify`."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from uuid import UUID

from claimlens.agent.stub import StubTriageAgent
from claimlens.blobs import BlobStore
from claimlens.decision import load_decision_config
from claimlens.events.envelope import ChainIntegrityError, ClaimEvent
from claimlens.events.projection import ClaimState, fold
from claimlens.events.store import ClaimNotFoundError, SQLiteEventStore
from claimlens.intake import submit_claim
from claimlens.policy import load_policies
from claimlens.pricing import load_rate_card
from claimlens.vision.base import Detector
from claimlens.workflow import PipelineDeps, process_claim

DEFAULT_WEIGHTS = Path("models/legacy/yolov8n-cardamage-v6.pt")
DetectorFactory = Callable[[Path], Detector]


def _legacy_detector(weights: Path) -> Detector:
    from claimlens.vision.legacy_yolo import LegacyYoloDetector

    return LegacyYoloDetector(weights)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="claimlens", description="ClaimLens claims triage")
    parser.add_argument("--db", type=Path, default=Path("var/claimlens.db"))
    parser.add_argument("--blobs", type=Path, default=Path("var/blobs"))
    parser.add_argument("--config", type=Path, default=Path("config"))
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="submit a claim and process it")
    run.add_argument("--policy", required=True, help="policy number, e.g. P-1001")
    run.add_argument("--description", default="", help="what happened")
    run.add_argument("photos", nargs="+", type=Path, help="photo files")

    show = sub.add_parser("show", help="print a claim's decision and audit trail")
    show.add_argument("claim_id", type=UUID)

    verify = sub.add_parser("verify", help="verify a claim's hash chain")
    verify.add_argument("claim_id", type=UUID)
    return parser


def make_deps(
    store: SQLiteEventStore, blobs: BlobStore, config_dir: Path, detector: Detector
) -> PipelineDeps:
    return PipelineDeps(
        store=store,
        blobs=blobs,
        detector=detector,
        policies=load_policies(config_dir / "policies.toml"),
        rate_card=load_rate_card(config_dir / "rate_card.toml"),
        decision_config=load_decision_config(config_dir / "decision_policy.toml"),
        agent=StubTriageAgent(),
    )


def format_summary(state: ClaimState) -> str:
    lines = [f"Claim: {state.claim_id}"]
    if state.decision is not None:
        lines.append(f"Route: {state.decision.route.value} ({state.decision.rule_id})")
        lines.append(f"Reason: {state.decision.reason}")
    found = ", ".join(f"{f.damage_type.value} {f.confidence:.2f}" for f in state.findings)
    lines.append(f"Findings: {len(state.findings)} ({found or 'none'})")
    if state.cost_estimate is not None:
        est = state.cost_estimate
        lines.append(f"Cost estimate: ${est.low:,}-${est.high:,} {est.currency}")
    for photo in state.photos.values():
        if photo.reject_reason:
            lines.append(f"Rejected {photo.photo_id} ({photo.filename}): {photo.reject_reason}")
    return "\n".join(lines)


def format_audit_trail(events: Sequence[ClaimEvent]) -> str:
    rows = ["Audit trail:"]
    for event in events:
        actor = f"{event.actor.kind.value}:{event.actor.name}"
        rows.append(f"  #{event.seq:<3} {event.type:<18} {actor:<22} {event.hash[:12]}")
    return "\n".join(rows)


def main(
    argv: Sequence[str] | None = None, *, detector_factory: DetectorFactory = _legacy_detector
) -> int:
    args = build_parser().parse_args(argv)
    store = SQLiteEventStore(args.db)
    try:
        if args.command == "run":
            return _run(args, store, detector_factory)
        return _inspect(args, store)
    finally:
        store.close()


def _run(args: argparse.Namespace, store: SQLiteEventStore, factory: DetectorFactory) -> int:
    blobs = BlobStore(args.blobs)
    deps = make_deps(store, blobs, args.config, factory(args.weights))
    claim_id = submit_claim(
        store, blobs, policy_id=args.policy, description=args.description, photo_paths=args.photos
    )
    process_claim(claim_id, deps)
    print(format_summary(fold(store.load(claim_id))))
    return 0


def _inspect(args: argparse.Namespace, store: SQLiteEventStore) -> int:
    try:
        events = store.load(args.claim_id)
    except ClaimNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except ChainIntegrityError as exc:
        print(f"error: audit log failed verification: {exc}", file=sys.stderr)
        return 1
    if args.command == "verify":
        print(f"Chain OK: {len(events)} events")
        return 0
    print(format_summary(fold(events)))
    print(format_audit_trail(events))
    return 0
```

- [ ] **Step 4: Register the command and ignore runtime data**

Add to `pyproject.toml` after the `dependencies` list:

```toml
[project.scripts]
claimlens = "claimlens.cli:main"
```

Add to `.gitignore` under "Experiment & run outputs":

```
/var/
```

Run: `uv sync`
Expected: completes; `uv run claimlens --help` prints the usage with `run`, `show`, `verify`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_cli.py -v && uv run mypy && uv run ruff check .`
Expected: 5 passed; mypy and ruff report no issues.

- [ ] **Step 6: Update `README.md` "Getting started"**

Replace the code block under "Getting started" with:

````markdown
```bash
uv sync                          # Python 3.12 environment with dev tools
uv run pre-commit install        # lint and format checks on every commit
uv run pytest                    # tests (no data, weights or API keys needed)

# Run a claim end to end with the legacy baseline model
uv sync --group vision           # adds Ultralytics (large download)
uv run claimlens run --policy P-1001 --description "Scraped a pole" tests/fixtures/images/dent_1.jpg
uv run claimlens show <claim-id>     # decision and audit trail
uv run claimlens verify <claim-id>   # check the hash chain
```
````

- [ ] **Step 7: Commit**

```bash
git add src/claimlens/cli.py tests/unit/test_cli.py pyproject.toml uv.lock .gitignore README.md
git commit -m "feat: add claimlens CLI with run, show and verify commands"
```

---

### Task 15: Evaluation metrics, golden claim format and triage eval runner

**Files:**
- Create: `src/claimlens/evals/__init__.py` (docstring only: `"""Evaluation harness."""`)
- Create: `src/claimlens/evals/metrics.py`
- Create: `src/claimlens/evals/golden.py`
- Create: `src/claimlens/evals/triage.py`
- Modify: `src/claimlens/cli.py` (add `eval-triage`)
- Test: `tests/unit/test_eval_metrics.py`, `tests/unit/test_eval_triage.py`

**Interfaces:**
- Consumes: `Route` (Task 1), `submit_claim` (Task 5), `process_claim`, `PipelineDeps` (Task 13), CLI pieces (Task 14).
- Produces: `HUMAN_ROUTES`; `TriageMetrics(total, correct, route_accuracy, needs_human, escalated, escalation_recall, confusion)`; `compute_triage_metrics(pairs: Sequence[tuple[Route, Route | None]]) -> TriageMetrics`; `PriorClaim(policy_id, photos)`; `GoldenClaim(case_id, scenario, policy_id, description, photos, expected_route, label_source, prior_claims=(), reviewed=False, notes="")`; `load_golden(path) -> list[GoldenClaim]`; `write_golden(path, cases) -> None`; `CaseResult(case_id, scenario, expected, predicted, rule_id="", reason="", error="")`; `ReportMeta(golden_path, model_version, agent_version, decision_policy_version, generated_on)`; `run_case(case, deps, repo_root) -> CaseResult`; `run_triage_eval(cases, make_deps, *, repo_root, workdir) -> list[CaseResult]`; `render_report(cases, results, metrics, meta) -> str`. CLI: `claimlens eval-triage --golden PATH --report PATH`.

Metric definitions: **route accuracy** = predicted route equals expected route. **Escalation recall** = among cases whose expected route is `ADJUSTER_REVIEW` or `FRAUD_REVIEW`, the share predicted as either of those (sending a fraud case to an adjuster still counts as escalated). A case that errors counts as wrong and not escalated.

- [ ] **Step 1: Write the failing metrics test** `tests/unit/test_eval_metrics.py`

```python
import pytest

from claimlens.domain import Route
from claimlens.evals.metrics import compute_triage_metrics

FAST, ADJ, FRAUD = Route.FAST_TRACK, Route.ADJUSTER_REVIEW, Route.FRAUD_REVIEW


def test_metrics_count_accuracy_and_recall() -> None:
    metrics = compute_triage_metrics(
        [(FAST, FAST), (ADJ, ADJ), (ADJ, FAST), (FRAUD, ADJ), (FAST, None)]
    )
    assert (metrics.total, metrics.correct) == (5, 2)
    assert metrics.route_accuracy == pytest.approx(0.4)
    assert (metrics.needs_human, metrics.escalated) == (3, 2)
    assert metrics.escalation_recall == pytest.approx(2 / 3)
    assert metrics.confusion[(ADJ, FAST)] == 1


def test_recall_is_none_without_human_cases() -> None:
    assert compute_triage_metrics([(FAST, FAST)]).escalation_recall is None


def test_empty_input_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least one case"):
        compute_triage_metrics([])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_eval_metrics.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'claimlens.evals'`

- [ ] **Step 3: Write** `src/claimlens/evals/metrics.py`

```python
"""Triage metrics. Escalation recall matters most: claims that need a human must reach one."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from claimlens.domain import Route

HUMAN_ROUTES = frozenset({Route.ADJUSTER_REVIEW, Route.FRAUD_REVIEW})


@dataclass(frozen=True)
class TriageMetrics:
    total: int
    correct: int
    route_accuracy: float
    needs_human: int
    escalated: int
    escalation_recall: float | None
    confusion: Mapping[tuple[Route, Route | None], int]


def compute_triage_metrics(pairs: Sequence[tuple[Route, Route | None]]) -> TriageMetrics:
    """`pairs` are (expected, predicted); predicted is None when the case errored."""
    if not pairs:
        raise ValueError("metrics need at least one case")
    correct = sum(expected == predicted for expected, predicted in pairs)
    human_cases = [predicted for expected, predicted in pairs if expected in HUMAN_ROUTES]
    escalated = sum(predicted in HUMAN_ROUTES for predicted in human_cases)
    return TriageMetrics(
        total=len(pairs),
        correct=correct,
        route_accuracy=correct / len(pairs),
        needs_human=len(human_cases),
        escalated=escalated,
        escalation_recall=escalated / len(human_cases) if human_cases else None,
        confusion=Counter(pairs),
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_eval_metrics.py -v`
Expected: 3 passed.

- [ ] **Step 5: Write the failing runner test** `tests/unit/test_eval_triage.py`

```python
from collections.abc import Callable
from datetime import date
from pathlib import Path

import pytest

from claimlens.cli import main
from claimlens.domain import Route
from claimlens.evals.golden import GoldenClaim, PriorClaim, load_golden, write_golden
from claimlens.evals.metrics import compute_triage_metrics
from claimlens.evals.triage import ReportMeta, render_report, run_triage_eval
from claimlens.events.store import SQLiteEventStore
from tests.fakes import CONFIG_DIR, FakeDetector, make_test_deps


def _cases(image: Path, missing: Path) -> list[GoldenClaim]:
    return [
        GoldenClaim(
            case_id="g001",
            scenario="clean",
            policy_id="P-1001",
            description="",
            photos=(str(image),),
            expected_route=Route.FAST_TRACK,
            label_source="scenario",
            reviewed=True,
        ),
        GoldenClaim(
            case_id="g002",
            scenario="photo_reuse",
            policy_id="P-1002",
            description="",
            photos=(str(image),),
            prior_claims=(PriorClaim(policy_id="P-1001", photos=(str(image),)),),
            expected_route=Route.FRAUD_REVIEW,
            label_source="scenario",
        ),
        GoldenClaim(
            case_id="g003",
            scenario="missing_file",
            policy_id="P-1001",
            description="",
            photos=(str(missing),),
            expected_route=Route.ADJUSTER_REVIEW,
            label_source="scenario",
        ),
    ]


def test_golden_round_trip(tmp_path: Path) -> None:
    cases = _cases(tmp_path / "a.jpg", tmp_path / "b.jpg")
    path = tmp_path / "claims.jsonl"
    write_golden(path, cases)
    assert load_golden(path) == cases


def test_duplicate_case_ids_are_rejected(tmp_path: Path) -> None:
    case = _cases(tmp_path / "a.jpg", tmp_path / "b.jpg")[0]
    path = tmp_path / "claims.jsonl"
    write_golden(path, [case, case])
    with pytest.raises(ValueError, match="duplicate case id g001"):
        load_golden(path)


def test_run_triage_eval_scores_each_case(tmp_path: Path, make_image: Callable[..., Path]) -> None:
    cases = _cases(make_image("a.jpg"), tmp_path / "nope.jpg")
    detector = FakeDetector()

    results = run_triage_eval(
        cases,
        lambda case_dir: make_test_deps(case_dir, SQLiteEventStore(case_dir / "c.db"), detector),
        repo_root=tmp_path,
        workdir=tmp_path / "work",
    )

    assert [r.predicted for r in results] == [Route.FAST_TRACK, Route.FRAUD_REVIEW, None]
    assert "photo not found" in results[2].error

    metrics = compute_triage_metrics([(r.expected, r.predicted) for r in results])
    meta = ReportMeta(
        golden_path="claims.jsonl",
        model_version=detector.model_version,
        agent_version="stub-v0",
        decision_policy_version="decision-policy-v0",
        generated_on=date(2026, 10, 18),
    )
    report = render_report(cases, results, metrics, meta)
    assert "| Route accuracy | 0.67 (2/3) |" in report
    assert "Cases: 3 (1 human-reviewed)" in report
    assert "| g003 | missing_file | ADJUSTER_REVIEW | ERROR |" in report


def test_eval_triage_command_writes_a_report(
    tmp_path: Path, make_image: Callable[..., Path], capsys: pytest.CaptureFixture[str]
) -> None:
    golden = tmp_path / "claims.jsonl"
    write_golden(golden, _cases(make_image("a.jpg"), tmp_path / "nope.jpg")[:2])
    report = tmp_path / "reports" / "triage.md"

    code = main(
        [
            "--config",
            str(CONFIG_DIR),
            "eval-triage",
            "--golden",
            str(golden),
            "--report",
            str(report),
        ],
        detector_factory=lambda _: FakeDetector(),
    )

    assert code == 0
    assert report.is_file()
    assert "Route accuracy: 1.00" in capsys.readouterr().out
```

- [ ] **Step 6: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_eval_triage.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'claimlens.evals.golden'`

- [ ] **Step 7: Write** `src/claimlens/evals/golden.py`

```python
"""Golden claims: labelled end-to-end test cases stored as JSON Lines."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from claimlens.domain import Frozen, Route


class PriorClaim(Frozen):
    """A claim submitted before the case under test, e.g. to set up photo reuse."""

    policy_id: str
    photos: tuple[str, ...]


class GoldenClaim(Frozen):
    case_id: str
    scenario: str
    policy_id: str
    description: str
    photos: tuple[str, ...]
    expected_route: Route
    label_source: str
    prior_claims: tuple[PriorClaim, ...] = ()
    reviewed: bool = False
    notes: str = ""


def load_golden(path: Path) -> list[GoldenClaim]:
    lines = path.read_text(encoding="utf-8").splitlines()
    cases = [GoldenClaim.model_validate_json(line) for line in lines if line.strip()]
    seen: set[str] = set()
    for case in cases:
        if case.case_id in seen:
            raise ValueError(f"duplicate case id {case.case_id}")
        seen.add(case.case_id)
    return cases


def write_golden(path: Path, cases: Sequence[GoldenClaim]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(case.model_dump_json() + "\n" for case in cases), encoding="utf-8")
```

- [ ] **Step 8: Write** `src/claimlens/evals/triage.py`

```python
"""Run golden claims through the full pipeline and report the results."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from claimlens.domain import Route
from claimlens.evals.golden import GoldenClaim
from claimlens.evals.metrics import TriageMetrics
from claimlens.intake import submit_claim
from claimlens.workflow import PipelineDeps, process_claim


@dataclass(frozen=True)
class CaseResult:
    case_id: str
    scenario: str
    expected: Route
    predicted: Route | None
    rule_id: str = ""
    reason: str = ""
    error: str = ""


@dataclass(frozen=True)
class ReportMeta:
    golden_path: str
    model_version: str
    agent_version: str
    decision_policy_version: str
    generated_on: date


def run_case(case: GoldenClaim, deps: PipelineDeps, repo_root: Path) -> CaseResult:
    for prior in case.prior_claims:
        prior_id = submit_claim(
            deps.store,
            deps.blobs,
            policy_id=prior.policy_id,
            description="Earlier claim",
            photo_paths=[repo_root / p for p in prior.photos],
        )
        process_claim(prior_id, deps)
    claim_id = submit_claim(
        deps.store,
        deps.blobs,
        policy_id=case.policy_id,
        description=case.description,
        photo_paths=[repo_root / p for p in case.photos],
    )
    decision = process_claim(claim_id, deps)
    return CaseResult(
        case_id=case.case_id,
        scenario=case.scenario,
        expected=case.expected_route,
        predicted=decision.route,
        rule_id=decision.rule_id,
        reason=decision.reason,
    )


def run_triage_eval(
    cases: Sequence[GoldenClaim],
    make_deps: Callable[[Path], PipelineDeps],
    *,
    repo_root: Path,
    workdir: Path,
) -> list[CaseResult]:
    """Each case gets a fresh store so cases cannot affect each other."""
    results: list[CaseResult] = []
    for case in cases:
        case_dir = workdir / case.case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        deps = make_deps(case_dir)
        try:
            results.append(run_case(case, deps, repo_root))
        except Exception as exc:
            results.append(
                CaseResult(
                    case_id=case.case_id,
                    scenario=case.scenario,
                    expected=case.expected_route,
                    predicted=None,
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
        finally:
            deps.store.close()
    return results


def render_report(
    cases: Sequence[GoldenClaim],
    results: Sequence[CaseResult],
    metrics: TriageMetrics,
    meta: ReportMeta,
) -> str:
    recall = "n/a" if metrics.escalation_recall is None else f"{metrics.escalation_recall:.2f}"
    reviewed = sum(case.reviewed for case in cases)
    lines = [
        f"# Triage evaluation: {meta.golden_path}",
        "",
        f"- Date: {meta.generated_on.isoformat()}",
        f"- Detector: `{meta.model_version}`",
        f"- Agent: `{meta.agent_version}`",
        f"- Decision policy: `{meta.decision_policy_version}`",
        f"- Cases: {metrics.total} ({reviewed} human-reviewed)",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Route accuracy | {metrics.route_accuracy:.2f} ({metrics.correct}/{metrics.total}) |",
        f"| Escalation recall | {recall} ({metrics.escalated}/{metrics.needs_human}) |",
        "",
        "## Confusion matrix (rows: expected, columns: predicted)",
        "",
    ]
    columns: list[Route | None] = [*Route, None]
    header = " | ".join(c.value if c is not None else "ERROR" for c in columns)
    lines.append(f"| expected \\ predicted | {header} |")
    lines.append("|---" * (len(columns) + 1) + "|")
    for expected in Route:
        counts = " | ".join(str(metrics.confusion.get((expected, c), 0)) for c in columns)
        lines.append(f"| {expected.value} | {counts} |")

    by_scenario: dict[str, list[CaseResult]] = defaultdict(list)
    for result in results:
        by_scenario[result.scenario].append(result)
    lines += ["", "## Accuracy by scenario", "", "| Scenario | Correct | Total |", "|---|---|---|"]
    for scenario in sorted(by_scenario):
        group = by_scenario[scenario]
        correct = sum(r.predicted == r.expected for r in group)
        lines.append(f"| {scenario} | {correct} | {len(group)} |")

    misses = [r for r in results if r.predicted != r.expected]
    lines += ["", "## Mismatches", ""]
    if not misses:
        lines.append("None.")
    else:
        lines += [
            "| Case | Scenario | Expected | Predicted | Rule | Reason |",
            "|---|---|---|---|---|---|",
        ]
        for r in misses:
            predicted = r.predicted.value if r.predicted is not None else "ERROR"
            reason = r.error or r.reason
            lines.append(
                f"| {r.case_id} | {r.scenario} | {r.expected.value} | {predicted} "
                f"| {r.rule_id or '-'} | {reason} |"
            )
    return "\n".join(lines) + "\n"
```

- [ ] **Step 9: Add the `eval-triage` command to** `src/claimlens/cli.py`

Add imports (merge with existing):

```python
import tempfile
from datetime import date

from claimlens.evals.golden import load_golden
from claimlens.evals.metrics import compute_triage_metrics
from claimlens.evals.triage import ReportMeta, render_report, run_triage_eval
```

In `build_parser()`, before `return parser`, add:

```python
    evaluate = sub.add_parser("eval-triage", help="score the pipeline on golden claims")
    evaluate.add_argument("--golden", type=Path, required=True, help="golden claims .jsonl")
    evaluate.add_argument("--report", type=Path, required=True, help="Markdown report to write")
```

Replace `main` with:

```python
def main(
    argv: Sequence[str] | None = None, *, detector_factory: DetectorFactory = _legacy_detector
) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "eval-triage":
        return _eval_triage(args, detector_factory)
    store = SQLiteEventStore(args.db)
    try:
        if args.command == "run":
            return _run(args, store, detector_factory)
        return _inspect(args, store)
    finally:
        store.close()
```

Append:

```python
def _eval_triage(args: argparse.Namespace, factory: DetectorFactory) -> int:
    cases = load_golden(args.golden)
    detector = factory(args.weights)

    def make(case_dir: Path) -> PipelineDeps:
        store = SQLiteEventStore(case_dir / "claims.db")
        return make_deps(store, BlobStore(case_dir / "blobs"), args.config, detector)

    with tempfile.TemporaryDirectory() as workdir:
        results = run_triage_eval(cases, make, repo_root=Path.cwd(), workdir=Path(workdir))
    metrics = compute_triage_metrics([(r.expected, r.predicted) for r in results])
    meta = ReportMeta(
        golden_path=args.golden.as_posix(),
        model_version=detector.model_version,
        agent_version=StubTriageAgent.agent_version,
        decision_policy_version=load_decision_config(args.config / "decision_policy.toml").version,
        generated_on=date.today(),
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(render_report(cases, results, metrics, meta), encoding="utf-8")
    recall = "n/a" if metrics.escalation_recall is None else f"{metrics.escalation_recall:.2f}"
    print(
        f"Cases: {metrics.total}  Route accuracy: {metrics.route_accuracy:.2f}  "
        f"Escalation recall: {recall}"
    )
    print(f"Report written to {args.report}")
    return 0
```

Update the module docstring to `"""Command-line interface: `claimlens run | show | verify | eval-triage`."""`.

- [ ] **Step 10: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_eval_triage.py tests/unit/test_cli.py -v && uv run mypy && uv run ruff check .`
Expected: 9 passed; mypy and ruff report no issues.

- [ ] **Step 11: Commit**

```bash
git add src/claimlens/evals src/claimlens/cli.py tests/unit/test_eval_metrics.py tests/unit/test_eval_triage.py
git commit -m "feat: add golden-claims triage evaluation with Markdown report"
```

---

### Task 16: Ground-truth oracle and golden claims v0

**Files:**
- Create: `src/claimlens/evals/oracle.py`
- Create: `scripts/build_golden_v0.py`
- Create (generated): `evals/golden/v0/claims.jsonl`, `evals/golden/v0/assets/tiny.png`, `evals/golden/v0/assets/not_an_image.jpg`, `evals/golden/v0/assets/thin_strip.png`
- Create: `evals/golden/v0/README.md`
- Test: `tests/unit/test_oracle.py`

**Interfaces:**
- Consumes: `normalize_class_name` (Task 7), `estimate_cost`, `RateCard` (Task 9), `decide`, `DecisionConfig` (Task 11), `ClaimState`, `PhotoRecord`, `PhotoStatus` (Task 4), `GoldenClaim`, `PriorClaim`, `write_golden` (Task 15), `load_policies` (Task 8).
- Produces: `findings_from_yolo_label(text, class_names, photo_id, image_width, image_height) -> list[DamageFinding]`; `oracle_route(findings, coverage, card, config) -> Route`.

How expected routes are labelled: **oracle** cases apply the decision policy to the dataset's ground-truth annotations with a perfect agent, which is the route a correct pipeline should produce. **scenario** cases have routes fixed by design (lapsed policy, reused photo, unusable photo). Both are then spot-checked by a human.

Scenario mix (50 cases, deterministic from the sorted 48 test images):

| Scenario | Cases | Images | Policy | Expected |
|---|---|---|---|---|
| `oracle_single_photo` | 30 | test images 0-29 | P-1001..P-1006 in rotation | oracle |
| `lapsed_policy` | 5 | images 30-34 | P-2001 | ADJUSTER_REVIEW |
| `no_collision_cover` | 5 | images 35-39 | P-2002 | ADJUSTER_REVIEW |
| `photo_reuse` | 5 | images 40-44 (also in a prior claim on P-1001) | P-1002 | FRAUD_REVIEW |
| `unusable_photo` | 3 | the three assets | P-1001 | ADJUSTER_REVIEW |
| `duplicate_in_claim` | 2 | images 45-46, each uploaded twice | P-1001, P-1002 | oracle |

- [ ] **Step 1: Write the failing test** `tests/unit/test_oracle.py`

```python
from pathlib import Path

import pytest

from claimlens.decision import load_decision_config
from claimlens.domain import BoundingBox, Coverage, DamageFinding, DamageType, Route
from claimlens.evals.oracle import findings_from_yolo_label, oracle_route
from claimlens.pricing import load_rate_card

ROOT = Path(__file__).resolve().parents[2]
NAMES = ["crack", "dent", "glass shatter", "lamp broken", "scratch", "tire flat", "smash"]
CARD = load_rate_card(ROOT / "config" / "rate_card.toml")
CONFIG = load_decision_config(ROOT / "config" / "decision_policy.toml")
COVERED = Coverage(policy_id="P-1001", found=True, active=True, collision=True, deductible=250)


def test_polygon_label_becomes_a_pixel_box() -> None:
    (finding,) = findings_from_yolo_label(
        "1 0.1 0.1 0.3 0.1 0.3 0.2 0.1 0.2", NAMES, "p1", 640, 640
    )
    assert finding.damage_type is DamageType.DENT
    assert finding.bbox == BoundingBox(x1=64, y1=64, x2=192, y2=128)
    assert finding.image_area_fraction == pytest.approx(0.02)
    assert finding.confidence == 1.0


def test_box_label_is_supported() -> None:
    (finding,) = findings_from_yolo_label("4 0.5 0.5 0.2 0.1", NAMES, "p1", 640, 640)
    assert finding.damage_type is DamageType.SCRATCH
    assert finding.image_area_fraction == pytest.approx(0.02)


def test_malformed_line_raises() -> None:
    with pytest.raises(ValueError, match="line 1"):
        findings_from_yolo_label("1 0.1 0.2 0.3", NAMES, "p1", 640, 640)


def test_unknown_class_index_raises() -> None:
    with pytest.raises(ValueError, match="class index 9"):
        findings_from_yolo_label("9 0.5 0.5 0.2 0.1", NAMES, "p1", 640, 640)


def _finding(damage_type: DamageType, fraction: float) -> DamageFinding:
    return DamageFinding(
        photo_id="p1",
        damage_type=damage_type,
        confidence=1.0,
        bbox=BoundingBox(x1=0, y1=0, x2=1, y2=1),
        image_area_fraction=fraction,
    )


def test_oracle_fast_tracks_small_covered_damage() -> None:
    assert (
        oracle_route([_finding(DamageType.DENT, 0.01)], COVERED, CARD, CONFIG) is Route.FAST_TRACK
    )


def test_oracle_escalates_expensive_damage() -> None:
    route = oracle_route([_finding(DamageType.SMASH, 0.5)], COVERED, CARD, CONFIG)
    assert route is Route.ADJUSTER_REVIEW


def test_oracle_escalates_when_nothing_is_labelled() -> None:
    assert oracle_route([], COVERED, CARD, CONFIG) is Route.ADJUSTER_REVIEW
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_oracle.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'claimlens.evals.oracle'`

- [ ] **Step 3: Write** `src/claimlens/evals/oracle.py`

```python
"""Expected routes from ground-truth labels: what a perfect perception model would produce."""

from __future__ import annotations

from collections.abc import Sequence
from uuid import UUID

from claimlens.decision import DecisionConfig, decide
from claimlens.domain import (
    AgentRecommendation,
    BoundingBox,
    Confidence,
    Coverage,
    DamageFinding,
    Route,
)
from claimlens.events.projection import ClaimState, PhotoRecord, PhotoStatus
from claimlens.pricing import RateCard, estimate_cost
from claimlens.vision.base import normalize_class_name


def findings_from_yolo_label(
    text: str, class_names: Sequence[str], photo_id: str, image_width: int, image_height: int
) -> list[DamageFinding]:
    """Parse YOLO box (`cls cx cy w h`) or polygon (`cls x1 y1 x2 y2 ...`) lines."""
    findings: list[DamageFinding] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        parts = line.split()
        if not parts:
            continue
        class_index = int(parts[0])
        coords = [float(value) for value in parts[1:]]
        if len(coords) < 4 or len(coords) % 2:
            raise ValueError(f"line {line_no}: expected box or polygon coordinates")
        if not 0 <= class_index < len(class_names):
            raise ValueError(f"line {line_no}: class index {class_index} is out of range")
        if len(coords) == 4:
            cx, cy, w, h = coords
            x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
        else:
            xs, ys = coords[0::2], coords[1::2]
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
        x1, y1 = max(x1, 0.0), max(y1, 0.0)
        x2, y2 = min(x2, 1.0), min(y2, 1.0)
        findings.append(
            DamageFinding(
                photo_id=photo_id,
                damage_type=normalize_class_name(class_names[class_index]),
                confidence=1.0,
                bbox=BoundingBox(
                    x1=x1 * image_width,
                    y1=y1 * image_height,
                    x2=x2 * image_width,
                    y2=y2 * image_height,
                ),
                image_area_fraction=(x2 - x1) * (y2 - y1),
            )
        )
    return findings


def oracle_route(
    findings: Sequence[DamageFinding],
    coverage: Coverage,
    card: RateCard,
    config: DecisionConfig,
) -> Route:
    """Apply the real decision policy to perfect evidence and a perfect agent."""
    state = ClaimState(claim_id=UUID(int=0), policy_id=coverage.policy_id, description="oracle")
    state.photos["p1"] = PhotoRecord(
        photo_id="p1",
        sha256="oracle",
        filename="oracle",
        blob_name="oracle",
        status=PhotoStatus.ACCEPTED,
    )
    state.findings = list(findings)
    state.integrity_checked = True
    state.cost_estimate = estimate_cost(findings, card)
    state.coverage = coverage
    state.recommendation = AgentRecommendation(
        route_suggestion=Route.FAST_TRACK if findings else Route.ADJUSTER_REVIEW,
        confidence=Confidence.HIGH if findings else Confidence.LOW,
        rationale="oracle",
        citations=(),
        open_questions=() if findings else ("no labelled damage",),
    )
    return decide(state, config).route
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_oracle.py -v && uv run mypy`
Expected: 7 passed; mypy reports no issues.

- [ ] **Step 5: Write** `scripts/build_golden_v0.py`

```python
"""Build golden claims v0 from the Roboflow test split. Run from the repository root:

uv run python scripts/build_golden_v0.py
"""

from __future__ import annotations

from pathlib import Path

import yaml
from PIL import Image

from claimlens.decision import load_decision_config
from claimlens.domain import Route
from claimlens.evals.golden import GoldenClaim, PriorClaim, write_golden
from claimlens.evals.oracle import findings_from_yolo_label, oracle_route
from claimlens.policy import load_policies
from claimlens.pricing import load_rate_card

DATASET = Path("data/raw/roboflow-car-damage-v6")
OUT_DIR = Path("evals/golden/v0")
ACTIVE_POLICIES = ["P-1001", "P-1002", "P-1003", "P-1004", "P-1005", "P-1006"]
DESCRIPTION = "Damage reported after a low-speed collision."


def _make_assets(assets: Path) -> list[Path]:
    assets.mkdir(parents=True, exist_ok=True)
    tiny = assets / "tiny.png"
    Image.new("RGB", (100, 100), (120, 120, 120)).save(tiny, format="PNG")
    strip = assets / "thin_strip.png"
    Image.new("RGB", (1200, 200), (90, 90, 90)).save(strip, format="PNG")
    not_image = assets / "not_an_image.jpg"
    not_image.write_text("This file is text, not a photo.\n", encoding="utf-8")
    return [tiny, not_image, strip]


def main() -> int:
    names: list[str] = yaml.safe_load((DATASET / "data.yaml").read_text(encoding="utf-8"))["names"]
    images = sorted((DATASET / "test" / "images").glob("*.jpg"))
    if len(images) < 47:
        raise SystemExit(f"expected at least 47 test images in {DATASET}, found {len(images)}")
    policies = load_policies(Path("config/policies.toml"))
    card = load_rate_card(Path("config/rate_card.toml"))
    config = load_decision_config(Path("config/decision_policy.toml"))

    def expected_for(image: Path, policy_id: str) -> Route:
        label = DATASET / "test" / "labels" / f"{image.stem}.txt"
        with Image.open(image) as opened:
            width, height = opened.size
        text = label.read_text(encoding="utf-8")
        findings = findings_from_yolo_label(text, names, "p1", width, height)
        return oracle_route(findings, policies.get_coverage(policy_id), card, config)

    cases: list[GoldenClaim] = []

    def add(**fields: object) -> None:
        case_id = f"g{len(cases) + 1:03d}"
        cases.append(GoldenClaim.model_validate({"case_id": case_id, **fields}))

    for i in range(30):
        image, policy = images[i], ACTIVE_POLICIES[i % len(ACTIVE_POLICIES)]
        add(
            scenario="oracle_single_photo",
            policy_id=policy,
            description=DESCRIPTION,
            photos=[image.as_posix()],
            expected_route=expected_for(image, policy),
            label_source="oracle",
        )
    for image in images[30:35]:
        add(
            scenario="lapsed_policy",
            policy_id="P-2001",
            description=DESCRIPTION,
            photos=[image.as_posix()],
            expected_route=Route.ADJUSTER_REVIEW,
            label_source="scenario",
        )
    for image in images[35:40]:
        add(
            scenario="no_collision_cover",
            policy_id="P-2002",
            description=DESCRIPTION,
            photos=[image.as_posix()],
            expected_route=Route.ADJUSTER_REVIEW,
            label_source="scenario",
        )
    for image in images[40:45]:
        add(
            scenario="photo_reuse",
            policy_id="P-1002",
            description=DESCRIPTION,
            photos=[image.as_posix()],
            prior_claims=[PriorClaim(policy_id="P-1001", photos=(image.as_posix(),))],
            expected_route=Route.FRAUD_REVIEW,
            label_source="scenario",
        )
    for asset in _make_assets(OUT_DIR / "assets"):
        add(
            scenario="unusable_photo",
            policy_id="P-1001",
            description=DESCRIPTION,
            photos=[asset.as_posix()],
            expected_route=Route.ADJUSTER_REVIEW,
            label_source="scenario",
        )
    for i, image in enumerate(images[45:47]):
        policy = ACTIVE_POLICIES[i]
        add(
            scenario="duplicate_in_claim",
            policy_id=policy,
            description=DESCRIPTION,
            photos=[image.as_posix(), image.as_posix()],
            expected_route=expected_for(image, policy),
            label_source="oracle",
        )

    write_golden(OUT_DIR / "claims.jsonl", cases)
    print(f"Wrote {len(cases)} golden claims to {OUT_DIR / 'claims.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 6: Generate the golden set**

Run: `uv run python scripts/build_golden_v0.py`
Expected: `Wrote 50 golden claims to evals/golden/v0/claims.jsonl`, and three files in `evals/golden/v0/assets/`.

Run: `uv run python -c "from collections import Counter; from pathlib import Path; from claimlens.evals.golden import load_golden; c = load_golden(Path('evals/golden/v0/claims.jsonl')); print(Counter(x.expected_route.value for x in c)); print(Counter(x.scenario for x in c))"`
Expected: 50 cases in total; scenario counts 30/5/5/5/3/2 as in the table above.

- [ ] **Step 7: Write** `evals/golden/v0/README.md`

```markdown
# Golden claims v0

50 end-to-end triage cases built by `scripts/build_golden_v0.py` from the Roboflow car-damage v6
test split (CC BY 4.0). Image paths point into `data/raw/`, which is not in git, so running this
set needs the dataset locally.

| Label source | Meaning |
|---|---|
| `oracle` | Decision policy applied to the ground-truth annotations with a perfect agent: the route a correct pipeline should produce |
| `scenario` | Route fixed by the scenario's design (lapsed policy, missing cover, reused photo, unusable photo) |

`reviewed: true` marks cases a human has checked by looking at the photo and the expected route.

Known limitations: one photo per claim for most cases; severity uses the box area of each
annotation, not the true damaged area; the rate card is fictional.
```

- [ ] **Step 8: Human review (you)**

Open `evals/golden/v0/claims.jsonl` and the photos for cases `g001`, `g004`, `g007`, `g010`, `g013`, `g016`, `g019`, `g022`, `g025` and `g028`. For each, check that the expected route is what you would decide as an adjuster. Tell Claude which cases you agree with; Claude sets `"reviewed": true` on those and records any disagreements in that case's `notes`.

- [ ] **Step 9: Run checks**

Run: `uv run pytest && uv run mypy && uv run ruff check . && uv run ruff format --check .`
Expected: all pass, 1 skipped.

- [ ] **Step 10: Commit**

```bash
git add src/claimlens/evals/oracle.py tests/unit/test_oracle.py scripts/build_golden_v0.py evals/golden/v0
git commit -m "feat: add ground-truth oracle and 50 golden claims (v0)"
```

---

### Task 17: Baseline evaluation and M1 wrap-up

**Files:**
- Create: `evals/reports/2026-10-18-triage-baseline-v0.md` (generated)
- Create: `docs/retros/m1-walking-skeleton.md`
- Modify: `pyproject.toml` (coverage gate)
- Modify: `docs/roadmap.md` (tick M1 items)

**Interfaces:**
- Consumes: everything above.
- Produces: the committed baseline report that M3 must beat, and the M1 retrospective.

- [ ] **Step 1: Install the vision group and run the legacy integration test**

Run: `uv sync --group vision && uv run pytest tests/integration -v`
Expected: `test_legacy_detector_returns_known_damage_types PASSED`.

- [ ] **Step 2: Run the end-to-end demo**

Run: `uv run claimlens run --policy P-1001 --description "Scraped a pole" tests/fixtures/images/dent_1.jpg`
Expected: prints `Claim: <uuid>`, a `Route:` line with a rule id, findings and a cost estimate. Then run `uv run claimlens show <uuid>` and `uv run claimlens verify <uuid>` and confirm `Chain OK: 9 events`.

- [ ] **Step 3: Run the baseline evaluation**

Run: `uv run claimlens eval-triage --golden evals/golden/v0/claims.jsonl --report evals/reports/2026-10-18-triage-baseline-v0.md`
Expected: prints `Cases: 50`, route accuracy and escalation recall, and writes the report. The legacy model is weak, so expect many oracle FAST_TRACK cases to be predicted ADJUSTER_REVIEW (rule R6). Scenario cases should be correct because they do not depend on the model.

- [ ] **Step 4: Add the coverage gate**

Add to `pyproject.toml`:

```toml
[tool.coverage.report]
fail_under = 85
```

Run: `uv run pytest`
Expected: passes and reports total coverage of at least 85%. If it is lower, add tests for the uncovered lines it lists before continuing.

- [ ] **Step 5: Write** `docs/retros/m1-walking-skeleton.md`

Fill in each heading from the actual results:

```markdown
# M1 retrospective: walking skeleton

## What we shipped
## Baseline numbers (from evals/reports/2026-10-18-triage-baseline-v0.md)
## What went well
## What was harder than expected
## What we will change in M2
```

- [ ] **Step 6: Tick the roadmap**

In `docs/roadmap.md`, mark every M1 item `[x]`.

- [ ] **Step 7: Commit, push and open the pull request**

```bash
git add evals/reports pyproject.toml docs/retros docs/roadmap.md
git commit -m "docs: record M1 baseline evaluation and retrospective"
```

Push `feat/m1-walking-skeleton` and open a pull request to `main`, then merge after CI passes.
