# M8a: The Local Web Prototype Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `claimlens serve` opens a local web app where one person files a claim by chat (customer), watches it go through the pipeline, reads the evidence and the agent's reasoning, and reviews it (adjuster).

**Architecture:** A FastAPI app (`src/claimlens/web/`) with server-rendered Jinja2 pages, a typed JSON API and a little plain JavaScript. It adds no business logic: it calls `IntakeSessions`, `submit_claim`, `process_claim`, `record_review`, `fold` and the store's chain check. Everything that holds a SQLite connection (the intake gateway, the triage agent, the pipeline) runs on one dedicated worker thread per area; each request opens and closes its own event store inside the endpoint body.

**Tech Stack:** FastAPI, Jinja2, uvicorn, python-multipart, httpx (`TestClient`), Pillow (upload checks), Playwright (one local browser test).

**Spec:** `docs/specs/2026-10-03-m8a-web-prototype-design.md`

## Global Constraints

- Python 3.12, `from __future__ import annotations`, mypy strict, ruff; commands through `uv run`.
- Binds to `127.0.0.1` only; any other host is refused at start-up (spec section 6).
- Jinja2 autoescaping on; page scripts insert data with `textContent`, never `innerHTML`.
- Uploads: JPEG, PNG or WebP only, checked by content with Pillow; at most `max_upload_mb` (10) from `config/web.toml`.
- API errors: JSON `{"detail": "<plain message>"}` with 404, 409, 422 or 503; never a stack trace, path or secret.
- The customer chat never shows a route, a rule, a price or cover (spec 4.1).
- Only the review form can deny; the web layer never writes a route itself.
- No CarDD or unknown-source photo in any committed or published file (README screenshot: a CC BY 4.0 Roboflow `car-seg` image, credited, or the owner's photo).
- New dependency group `web` (fastapi, jinja2, uvicorn, python-multipart, httpx); CI syncs it. Playwright goes in a separate `browser` group that CI does not sync.
- Tests use `FakeDetector`, `FakeProvider` and `FakeEmbedder`: no paid calls, no weights. Coverage stays at or above 85%.
- Never commit `data/`, `models/`, `var/` or `.env`.

## Review Focus

1. **A browser that reloads mid-chat** expects the same conversation, not a new claim: `GET /api/intake/{session}` returns the waiting question (Task 6 test `test_a_reloaded_chat_continues`).
2. **A double-clicked "send" or "resume"** must not file two claims or run the pipeline twice: the runner refuses a second start while running (Task 3 `test_a_second_start_while_running_is_refused`, Task 4 `test_resume_while_running_is_409`).
3. **A photo whose name says `.jpg` but whose bytes are not an image** must be refused, not passed to the models (Task 6 `test_a_fake_jpeg_is_refused`).
4. **An unknown or malformed claim id in the URL** gives a plain 404, not a 500 (Task 4 `test_a_malformed_claim_id_is_404`).
5. **A photo id from another claim** must not be served (Task 4 `test_a_photo_of_another_claim_is_404`).

---

## File structure

| File | Responsibility |
|---|---|
| `config/web.toml` | Host, port, upload limit, poll interval |
| `src/claimlens/web/__init__.py` | Package marker |
| `src/claimlens/web/settings.py` | `WebSettings`, `load_web_settings` |
| `src/claimlens/web/schemas.py` | Pydantic API models |
| `src/claimlens/web/views.py` | Pure functions: events → view models |
| `src/claimlens/web/workers.py` | `Worker`: one dedicated thread (or inline for tests) |
| `src/claimlens/web/runner.py` | `ClaimRunner`: runs the pipeline for a claim on the claims worker |
| `src/claimlens/web/services.py` | `WebServices`: store opener, blobs, runner, intake, memory |
| `src/claimlens/web/uploads.py` | `check_upload`: type and size checks, saves to the upload folder |
| `src/claimlens/web/app.py` | `create_app(services)`: routers, templates, static files, error handlers |
| `src/claimlens/web/routers/claims.py` | `/api/claims...` list, detail, photo, resume |
| `src/claimlens/web/routers/review.py` | `/api/claims/{id}/review` |
| `src/claimlens/web/routers/intake.py` | `/api/intake...` start, show, reply |
| `src/claimlens/web/routers/pages.py` | `/`, `/claims`, `/claims/{id}` |
| `src/claimlens/web/templates/*.html` | `base`, `chat`, `claims`, `claim`, `not_found`, `log_failed` |
| `src/claimlens/web/static/{app.css,chat.js,claim.js}` | Styles and the two page scripts |
| `src/claimlens/web/serve.py` | `run_serve`: builds the real services from CLI args, seeds, starts uvicorn |
| `src/claimlens/intake_agent/session.py` | Extract `file_intake_claim` from `pipeline_submitter` |
| `src/claimlens/cli.py` | `serve` sub-command |
| `tests/unit/web/` | `helpers.py` and one test module per area |
| `tests/browser/test_web_flow.py` | Playwright end-to-end test (marked `browser`) |
| `docs/adr/0018-web-prototype.md`, `README.md`, `docs/roadmap.md` | Docs |

---

### Task 1: Dependencies, settings, and filing an intake claim without running the pipeline

**Files:**
- Modify: `pyproject.toml` (new `web` and `browser` groups, `browser` marker), `.github/workflows/ci.yml` (sync `--group web`)
- Create: `config/web.toml`, `src/claimlens/web/__init__.py`, `src/claimlens/web/settings.py`
- Modify: `src/claimlens/intake_agent/session.py` (extract `file_intake_claim`)
- Test: `tests/unit/web/__init__.py`, `tests/unit/web/test_settings.py`, `tests/unit/test_intake_session.py` (append)

**Interfaces:**
- Produces: `WebSettings(host: str, port: int, max_upload_mb: int, poll_seconds: float)`, `load_web_settings(path: Path) -> WebSettings`; `file_intake_claim(store: SQLiteEventStore, blobs: BlobStore, state: IntakeState, config: IntakeConfig) -> UUID`.

- [ ] **Step 1: Add the dependency groups and marker**

In `pyproject.toml` `[dependency-groups]` add:

```toml
web = [
  "fastapi>=0.142,<1",
  "httpx>=0.28",
  "jinja2>=3.1",
  "python-multipart>=0.0.20",
  "uvicorn>=0.37",
]
browser = [
  "playwright>=1.55",
]
```

In `[tool.pytest.ini_options]` `markers` add `"browser: end-to-end tests that drive a real browser (local only)",`. In `.github/workflows/ci.yml` change the sync line to
`uv sync --locked --group training --group knowledge --group agent --group tracing --group web`.

Run: `uv lock && uv sync --group training --group knowledge --group agent --group tracing --group web`
Expected: lockfile updated, packages installed.

- [ ] **Step 2: Write the failing tests**

`tests/unit/web/__init__.py`: empty file.

`tests/unit/web/test_settings.py`:

```python
from pathlib import Path

import pytest

from claimlens.web.settings import WebSettings, load_web_settings

ROOT = Path(__file__).resolve().parents[3]


def test_the_repo_settings_load() -> None:
    settings = load_web_settings(ROOT / "config" / "web.toml")
    assert settings == WebSettings(host="127.0.0.1", port=8000, max_upload_mb=10, poll_seconds=1.0)


def test_a_bad_upload_limit_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "web.toml"
    path.write_text("max_upload_mb = 0\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_web_settings(path)
```

Append to `tests/unit/test_intake_session.py`:

```python
def test_file_intake_claim_files_once_and_never_processes(
    tmp_path: Path, store: SQLiteEventStore
) -> None:
    from claimlens.blobs import BlobStore
    from claimlens.intake_agent.session import file_intake_claim

    blobs = BlobStore(tmp_path / "blobs")
    state: Any = {
        "session_id": "s-file",
        "facts": {"policy_id": "P-1001", "what_happened": "Bollard."},
        "photos": {"overview": photo(tmp_path / "o.png")},
        "gaps": {},
        "turns": 4,
        "retakes": {},
        "transcript": [{"role": "user", "text": "hi"}],
    }
    first = file_intake_claim(store, blobs, state, CONFIG)
    again = file_intake_claim(store, blobs, state, CONFIG)
    assert first == again
    types = [e.type for e in store.load(first)]
    assert types.count("IntakeCompleted") == 1
    assert "RouteDecided" not in types
```

- [ ] **Step 3: Run the tests to see them fail**

Run: `uv run pytest tests/unit/web/test_settings.py tests/unit/test_intake_session.py -k "settings or file_intake" -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'claimlens.web'` and `ImportError: cannot import name 'file_intake_claim'`.

- [ ] **Step 4: Implement**

`config/web.toml`:

```toml
# The local web prototype (spec docs/specs/2026-10-03-m8a-web-prototype-design.md).
host = "127.0.0.1"   # local only until sign-in exists (M7/M8b)
port = 8000
max_upload_mb = 10   # per photo
poll_seconds = 1.0   # how often a running claim's page checks for new events
```

`src/claimlens/web/__init__.py`:

```python
"""The local web prototype: a FastAPI app over the existing claim flow (M8a)."""
```

`src/claimlens/web/settings.py`:

```python
"""Settings for the web prototype, from config/web.toml."""

from __future__ import annotations

import tomllib
from pathlib import Path

from pydantic import Field

from claimlens.domain import Frozen


class WebSettings(Frozen):
    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)
    max_upload_mb: int = Field(default=10, ge=1, le=50)
    poll_seconds: float = Field(default=1.0, gt=0, le=10)


def load_web_settings(path: Path) -> WebSettings:
    return WebSettings.model_validate(tomllib.loads(path.read_text(encoding="utf-8")))
```

In `src/claimlens/intake_agent/session.py`, add `from claimlens.blobs import BlobStore` and `from claimlens.events.store import SQLiteEventStore` to the imports, add this function above `pipeline_submitter`, and make `pipeline_submitter` call it:

```python
def file_intake_claim(
    store: SQLiteEventStore, blobs: BlobStore, state: IntakeState, config: IntakeConfig
) -> uuid.UUID:
    """File the collected claim and record IntakeCompleted, once per session. Does not run the
    pipeline: the CLI runs it straight away, the web app on its claims worker."""
    photos = state["photos"]
    kinds = [k for k in config.photo_kinds if k in photos]
    kinds += sorted(k for k in photos if k not in config.photo_kinds)
    facts = state["facts"]
    # One claim per session: a hand-over that is retried after a failure files nothing new.
    claim_id = uuid.uuid5(_NAMESPACE, state["session_id"])
    if claim_id not in store.claim_ids():
        submit_claim(
            store,
            blobs,
            policy_id=facts.get("policy_id", "unknown"),
            description=facts.get("what_happened", ""),
            photo_paths=[Path(photos[k]) for k in kinds],
            claim_id=claim_id,
            allow_no_photos=True,
        )
    if not any(e.type == "IntakeCompleted" for e in store.load(claim_id)):
        store.append(
            claim_id,
            IntakeCompleted(
                session_id=state["session_id"],
                facts=dict(facts),
                photo_kinds={kind: f"p{n}" for n, kind in enumerate(kinds, start=1)},
                photo_gaps=dict(state["gaps"]),
                turns=state["turns"],
                retakes=sum(state["retakes"].values()),
                transcript_sha256=transcript_sha256(state["transcript"]),
                handover=state.get("handover", ""),
            ),
            Actor(kind=ActorKind.AGENT, name=config.version),
        )
    return claim_id


def pipeline_submitter(
    deps_factory: Callable[[], PipelineDeps], config: IntakeConfig, *, process: bool
) -> Callable[[IntakeState], str]:
    """File the collected claim, record IntakeCompleted, and (optionally) run the pipeline."""

    def submit(state: IntakeState) -> str:
        deps = deps_factory()
        claim_id = file_intake_claim(deps.store, deps.blobs, state, config)
        if process:  # process_claim is idempotent: a retry finishes what is missing
            process_claim(claim_id, deps)
        return str(claim_id)

    return submit
```

- [ ] **Step 5: Run the tests**

Run: `uv run pytest tests/unit/web/test_settings.py tests/unit/test_intake_session.py -v`
Expected: PASS (all old intake-session tests still pass).

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock .github/workflows/ci.yml config/web.toml src/claimlens/web src/claimlens/intake_agent/session.py tests/unit/web tests/unit/test_intake_session.py
git commit -m "feat: web settings and filing an intake claim without processing it"
```

---

### Task 2: View models: from a claim's events to what the pages show

**Files:**
- Create: `src/claimlens/web/schemas.py`, `src/claimlens/web/views.py`
- Test: `tests/unit/web/helpers.py`, `tests/unit/web/test_views.py`

**Interfaces:**
- Consumes: `ClaimState`, `ClaimEvent`, `fold`.
- Produces (schemas): `ChatTurn`, `StageStatus`, `Box`, `PhotoView`, `AgentView`, `SimilarView`, `EventItem`, `ClaimSummary`, `ClaimDetail`, `ReviewRequest`, `ClaimStatus` (Literal `"running" | "stopped" | "decided" | "reviewed" | "log_failed"`).
- Produces (views): `claim_status(state: ClaimState, running: bool) -> ClaimStatus`, `stage_timeline(events: Sequence[ClaimEvent], running: bool) -> list[StageStatus]`, `photo_views(state: ClaimState, events: Sequence[ClaimEvent]) -> list[PhotoView]`, `agent_view(state: ClaimState, events: Sequence[ClaimEvent]) -> AgentView | None`, `claim_summary(state, events, running) -> ClaimSummary`, `claim_detail(state, events, *, running: bool, error: str | None, similar: list[SimilarView]) -> ClaimDetail`.

- [ ] **Step 1: Write the test helpers**

`tests/unit/web/helpers.py`:

```python
"""Event logs for web tests, built by running the real pipeline with fakes."""

from __future__ import annotations

from pathlib import Path
from uuid import UUID

from PIL import Image

from claimlens.domain import BoundingBox, DamageFinding, DamageType
from claimlens.events.store import SQLiteEventStore
from claimlens.intake import submit_claim
from claimlens.workflow import PipelineDeps, process_claim
from tests.fakes import FakeDetector, make_test_deps


def image(path: Path, color: tuple[int, int, int] = (200, 30, 30)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (640, 480), color).save(path)
    return path


def finding(photo_id: str = "p1") -> DamageFinding:
    return DamageFinding(
        photo_id=photo_id,
        damage_type=DamageType.DENT,
        confidence=0.9,
        bbox=BoundingBox(x1=64, y1=48, x2=320, y2=240),
        image_area_fraction=0.16,
        part="back_bumper",
        part_area_ratio=0.06,
    )


def deps(tmp_path: Path, store: SQLiteEventStore) -> PipelineDeps:
    detector = FakeDetector(findings={"p1": [finding("p1")]})
    return make_test_deps(tmp_path, store, detector)


def decided_claim(
    tmp_path: Path, store: SQLiteEventStore, color: tuple[int, int, int] = (200, 30, 30)
) -> UUID:
    d = deps(tmp_path, store)
    claim = submit_claim(
        store,
        d.blobs,
        policy_id="P-1001",
        description="Reversed into a bollard <b>slowly</b>",
        photo_paths=[image(tmp_path / "in" / f"{color}.png", color)],
    )
    process_claim(claim, d)
    return claim
```

Note: check `FakeDetector(findings=...)` keys by photo id (see `tests/fakes.py:44`); if the fake keys findings differently, adapt `deps()` to the fake's real signature before Step 3.

- [ ] **Step 2: Write the failing tests**

`tests/unit/web/test_views.py`:

```python
from pathlib import Path

from claimlens.events.projection import fold
from claimlens.events.store import SQLiteEventStore
from claimlens.intake import submit_claim
from claimlens.web.views import (
    agent_view,
    claim_detail,
    claim_status,
    claim_summary,
    photo_views,
    stage_timeline,
)
from tests.unit.web.helpers import decided_claim, deps, image


def _states(timeline: list) -> dict[str, str]:  # type: ignore[type-arg]
    return {s.key: s.state for s in timeline}


def test_a_decided_claim_has_every_pipeline_stage_done(
    tmp_path: Path, store: SQLiteEventStore
) -> None:
    claim = decided_claim(tmp_path, store)
    events = store.load(claim)
    states = _states(stage_timeline(events, running=False))
    for key in ("photos", "damage", "integrity", "pricing", "coverage", "agent", "decision"):
        assert states[key] == "done", key
    assert states["intake"] == "skipped"  # filed by form, not by chat
    assert states["memory"] == "skipped"  # memory is off in this pipeline
    assert states["review"] == "waiting"


def test_a_running_claim_shows_the_next_stage_running(
    tmp_path: Path, store: SQLiteEventStore
) -> None:
    d = deps(tmp_path, store)
    claim = submit_claim(
        store, d.blobs, policy_id="P-1001", description="x", photo_paths=[image(tmp_path / "a.png")]
    )
    states = _states(stage_timeline(store.load(claim), running=True))
    assert states["photos"] == "running"
    assert states["decision"] == "waiting"
    assert claim_status(fold(store.load(claim)), running=False) == "stopped"


def test_boxes_are_percentages_of_the_photo(tmp_path: Path, store: SQLiteEventStore) -> None:
    claim = decided_claim(tmp_path, store)
    events = store.load(claim)
    (view,) = photo_views(fold(events), events)
    (box,) = view.boxes
    assert (box.left, box.top, box.width, box.height) == (10.0, 10.0, 40.0, 40.0)
    assert box.label == "dent · back_bumper · 0.90"
    assert view.status == "accepted"


def test_the_agent_section_and_the_summary(tmp_path: Path, store: SQLiteEventStore) -> None:
    claim = decided_claim(tmp_path, store)
    events = store.load(claim)
    state = fold(events)
    agent = agent_view(state, events)
    assert agent is not None
    assert agent.route_suggestion == state.recommendation.route_suggestion.value  # type: ignore[union-attr]
    summary = claim_summary(state, events, running=False)
    assert summary.status == "decided"
    assert summary.route == state.decision.route.value  # type: ignore[union-attr]
    assert summary.rule_id == state.decision.rule_id  # type: ignore[union-attr]


def test_the_detail_carries_the_audit_trail(tmp_path: Path, store: SQLiteEventStore) -> None:
    claim = decided_claim(tmp_path, store)
    events = store.load(claim)
    detail = claim_detail(fold(events), events, running=False, error=None, similar=[])
    assert detail.chain_ok
    assert detail.event_count == len(events)
    assert [e.type for e in detail.events] == [e.type for e in events]
    assert detail.description == "Reversed into a bollard <b>slowly</b>"  # escaped by templates
    assert detail.cost_low is not None
```

- [ ] **Step 3: Run the tests to see them fail**

Run: `uv run pytest tests/unit/web/test_views.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'claimlens.web.views'`.

- [ ] **Step 4: Implement the schemas**

`src/claimlens/web/schemas.py`:

```python
"""The web API's request and response models (they also feed the page templates)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from claimlens.domain import Route
from claimlens.events.payloads import ReviewAction

ClaimStatus = Literal["running", "stopped", "decided", "reviewed", "log_failed"]
StageState = Literal["done", "running", "failed", "waiting", "skipped"]


class ChatTurn(BaseModel):
    session_id: str
    message: str
    photo_kind: str | None = None  # the photo the assistant is waiting for
    claim_id: str | None = None  # set once the claim is filed


class StageStatus(BaseModel):
    key: str
    label: str
    state: StageState
    detail: str = ""


class Box(BaseModel):
    """A finding's box as percentages of the photo, so it scales with the image."""

    label: str
    left: float
    top: float
    width: float
    height: float


class PhotoView(BaseModel):
    photo_id: str
    kind: str | None
    status: str
    reject_reason: str | None = None
    boxes: list[Box] = Field(default_factory=list)


class AgentView(BaseModel):
    route_suggestion: str
    confidence: str
    rationale: str
    citations: list[str]
    policy_citations: list[str]
    open_questions: list[str]
    skills_used: list[str]
    tools_used: list[str]
    llm_calls: int
    llm_cost_usd: float


class SimilarView(BaseModel):
    claim_id: str
    reason: str
    distance: int | None
    route: str
    damage: str


class EventItem(BaseModel):
    seq: int
    type: str
    actor: str
    occurred_at: str


class ClaimSummary(BaseModel):
    claim_id: str
    policy_id: str
    filed_at: str
    status: ClaimStatus
    route: str | None = None
    rule_id: str | None = None
    review_action: str | None = None


class ClaimDetail(ClaimSummary):
    description: str
    facts: dict[str, str]
    stages: list[StageStatus]
    photos: list[PhotoView]
    damage: list[str]
    cost_low: int | None
    cost_high: int | None
    coverage: str | None
    fraud_signals: list[str]
    agent: AgentView | None
    decision_reason: str | None
    similar: list[SimilarView]
    chain_ok: bool
    event_count: int
    events: list[EventItem]
    error: str | None = None


class ReviewRequest(BaseModel):
    reviewer: str = Field(min_length=1, max_length=80)
    action: ReviewAction
    final_route: Route | None = None
    note: str = Field(default="", max_length=2000)
```

- [ ] **Step 5: Implement the views**

`src/claimlens/web/views.py`:

```python
"""Pure functions from a claim's events to what the pages show. No web, no I/O."""

from __future__ import annotations

from collections.abc import Sequence

from claimlens.events.envelope import ClaimEvent
from claimlens.events.projection import ClaimState, PhotoStatus, fold
from claimlens.web.schemas import (
    AgentView,
    Box,
    ClaimDetail,
    ClaimStatus,
    ClaimSummary,
    EventItem,
    PhotoView,
    SimilarView,
    StageState,
    StageStatus,
)

STAGES: tuple[tuple[str, str], ...] = (
    ("intake", "Intake chat"),
    ("photos", "Photo check"),
    ("damage", "Damage and parts"),
    ("integrity", "Integrity"),
    ("pricing", "Pricing"),
    ("coverage", "Coverage"),
    ("agent", "Triage agent"),
    ("decision", "Decision"),
    ("memory", "Memory"),
    ("review", "Human review"),
)
# Stages done by one event type, and the StageFailed stage name that marks them failed.
_ONE_EVENT = {
    "integrity": ("IntegrityChecked", "integrity"),
    "pricing": ("CostEstimated", "pricing"),
    "coverage": ("PolicyRetrieved", "coverage"),
    "agent": ("AgentRecommended", "agent"),
    "decision": ("RouteDecided", None),
}
# Stages the pipeline itself runs, in order: the first unfinished one is "running".
_PIPELINE = ("photos", "damage", "integrity", "pricing", "coverage", "agent", "decision")


def claim_status(state: ClaimState, running: bool) -> ClaimStatus:
    if running:
        return "running"
    if state.review is not None:
        return "reviewed"
    if state.decision is not None:
        return "decided"
    return "stopped"


def _failures(events: Sequence[ClaimEvent]) -> dict[str, str]:
    return {
        str(e.payload["stage"]): str(e.payload["error"]) for e in events if e.type == "StageFailed"
    }


def _photos_stage(state: ClaimState) -> tuple[StageState, str]:
    if not state.photos:
        return "skipped", "no photos (rule R3 sends the claim to a person)"
    checked = [p for p in state.photos.values() if p.status is not PhotoStatus.UPLOADED]
    if len(checked) < len(state.photos):
        return "waiting", ""
    accepted = len(state.accepted_photos)
    return "done", f"{accepted} accepted, {len(state.photos) - accepted} rejected"


def _damage_stage(state: ClaimState, failures: dict[str, str]) -> tuple[StageState, str]:
    if "perception" in failures:
        return "failed", failures["perception"]
    accepted = state.accepted_photos
    if not state.photos:
        return "skipped", ""
    if accepted and all(p.photo_id in state.detection_event_ids for p in accepted):
        return "done", f"{len(state.findings)} finding(s)"
    if not accepted and all(p.status is PhotoStatus.REJECTED for p in state.photos.values()):
        return "skipped", "no accepted photo"
    return "waiting", ""


def stage_timeline(events: Sequence[ClaimEvent], running: bool) -> list[StageStatus]:
    state = fold(events)
    types = {e.type for e in events}
    failures = _failures(events)
    result: dict[str, tuple[StageState, str]] = {}
    result["intake"] = (
        ("done", f"{state.intake.turns} turns") if state.intake else ("skipped", "filed by form")
    )
    result["photos"] = _photos_stage(state)
    result["damage"] = _damage_stage(state, failures)
    for key, (done_type, failed_stage) in _ONE_EVENT.items():
        if done_type in types:
            detail = ""
            if key == "decision" and state.decision is not None:
                detail = f"{state.decision.route.value} ({state.decision.rule_id})"
            result[key] = ("done", detail)
        elif failed_stage is not None and failed_stage in failures:
            result[key] = ("failed", failures[failed_stage])
        else:
            result[key] = ("waiting", "")
    if "MemoryForgotten" in types:
        result["memory"] = ("skipped", "forgotten")
    elif "MemoryWritten" in types:
        result["memory"] = ("done", "")
    elif "MemoryWriteFailed" in types:
        result["memory"] = ("failed", "the memory write failed; the decision is unchanged")
    elif state.decision is not None:
        result["memory"] = ("skipped", "memory is off")
    else:
        result["memory"] = ("waiting", "")
    result["review"] = ("done", state.review.action.value) if state.review else ("waiting", "")
    if running:
        for key in _PIPELINE:
            if result[key][0] == "waiting":
                result[key] = ("running", "")
                break
    return [
        StageStatus(key=k, label=label, state=result[k][0], detail=result[k][1])
        for k, label in STAGES
    ]


def photo_views(state: ClaimState, events: Sequence[ClaimEvent]) -> list[PhotoView]:
    sizes = {
        str(e.payload["photo_id"]): (int(e.payload["width"]), int(e.payload["height"]))
        for e in events
        if e.type == "PhotoAccepted"
    }
    kinds = {pid: kind for kind, pid in (state.intake.photo_kinds.items() if state.intake else [])}
    views = []
    for photo in state.photos.values():
        boxes = []
        size = sizes.get(photo.photo_id)
        for f in state.findings:
            if f.photo_id != photo.photo_id or size is None:
                continue
            w, h = size
            label = " · ".join(
                [f.damage_type.value, *([f.part] if f.part else []), f"{f.confidence:.2f}"]
            )
            boxes.append(
                Box(
                    label=label,
                    left=round(100 * f.bbox.x1 / w, 2),
                    top=round(100 * f.bbox.y1 / h, 2),
                    width=round(100 * (f.bbox.x2 - f.bbox.x1) / w, 2),
                    height=round(100 * (f.bbox.y2 - f.bbox.y1) / h, 2),
                )
            )
        views.append(
            PhotoView(
                photo_id=photo.photo_id,
                kind=kinds.get(photo.photo_id),
                status=photo.status.value,
                reject_reason=photo.reject_reason,
                boxes=boxes,
            )
        )
    return views


def agent_view(state: ClaimState, events: Sequence[ClaimEvent]) -> AgentView | None:
    rec = state.recommendation
    if rec is None:
        return None
    tools = [str(e.payload["tool"]) for e in events if e.type == "ToolCalled"]
    llm = [e for e in events if e.type == "LLMCalled"]
    return AgentView(
        route_suggestion=rec.route_suggestion.value,
        confidence=str(
            rec.confidence.value if hasattr(rec.confidence, "value") else rec.confidence
        ),
        rationale=rec.rationale,
        citations=list(rec.citations),
        policy_citations=list(rec.policy_citations),
        open_questions=list(rec.open_questions),
        skills_used=list(rec.skills_used),
        tools_used=sorted(set(tools)),
        llm_calls=len(llm),
        llm_cost_usd=round(sum(float(e.payload["cost_usd"]) for e in llm), 4),
    )


def claim_summary(state: ClaimState, events: Sequence[ClaimEvent], running: bool) -> ClaimSummary:
    return ClaimSummary(
        claim_id=str(state.claim_id),
        policy_id=state.policy_id,
        filed_at=events[0].occurred_at.isoformat(timespec="seconds"),
        status=claim_status(state, running),
        route=state.decision.route.value if state.decision else None,
        rule_id=state.decision.rule_id if state.decision else None,
        review_action=state.review.action.value if state.review else None,
    )


def claim_detail(
    state: ClaimState,
    events: Sequence[ClaimEvent],
    *,
    running: bool,
    error: str | None,
    similar: list[SimilarView],
) -> ClaimDetail:
    summary = claim_summary(state, events, running)
    cover = state.coverage
    coverage = None
    if cover is not None:
        coverage = (
            "policy not found"
            if not cover.found
            else f"{'active' if cover.active else 'inactive'}, "
            f"{'collision cover' if cover.collision else 'no collision cover'}, "
            f"${cover.deductible} deductible"
        )
    return ClaimDetail(
        **summary.model_dump(),
        description=state.description,
        facts=dict(state.intake.facts) if state.intake else {},
        stages=stage_timeline(events, running),
        photos=photo_views(state, events),
        damage=[
            f"{f.damage_type.value}{' on ' + f.part if f.part else ''} ({f.confidence:.2f})"
            for f in state.findings
        ],
        cost_low=state.cost_estimate.low if state.cost_estimate else None,
        cost_high=state.cost_estimate.high if state.cost_estimate else None,
        coverage=coverage,
        fraud_signals=[f"{s.kind} ({s.score:.2f}): {s.detail}" for s in state.fraud_signals],
        agent=agent_view(state, events),
        decision_reason=state.decision.reason if state.decision else None,
        similar=similar,
        chain_ok=True,  # the store verified the chain when it loaded these events
        event_count=len(events),
        events=[
            EventItem(
                seq=e.seq,
                type=e.type,
                actor=f"{e.actor.kind.value}:{e.actor.name}",
                occurred_at=e.occurred_at.isoformat(timespec="seconds"),
            )
            for e in events
        ],
        error=error,
    )
```

Before Step 6, open `src/claimlens/domain.py` and check `Confidence`: if it is a `StrEnum`, replace the `confidence=` line with `confidence=rec.confidence.value`; if it is a float, use `confidence=f"{rec.confidence:.2f}"`. Keep whichever matches; do not leave the `hasattr` fallback in.

- [ ] **Step 6: Run the tests**

Run: `uv run pytest tests/unit/web/test_views.py -v && uv run mypy`
Expected: PASS; mypy clean.

- [ ] **Step 7: Commit**

```bash
git add src/claimlens/web/schemas.py src/claimlens/web/views.py tests/unit/web/helpers.py tests/unit/web/test_views.py
git commit -m "feat: web view models built from a claim's events"
```

---

### Task 3: Workers, the claim runner and the services object

**Files:**
- Create: `src/claimlens/web/workers.py`, `src/claimlens/web/runner.py`, `src/claimlens/web/services.py`
- Test: `tests/unit/web/test_runner.py`

**Interfaces:**
- Produces:
  - `Worker(name: str, *, inline: bool = False)` with `submit(fn: Callable[[], T]) -> Future[T]`, `call(fn: Callable[[], T]) -> T`, `shutdown() -> None`.
  - `ClaimRunner(worker: Worker, process: Callable[[UUID], None])` with `start(claim_id: UUID) -> bool` (False when already running), `running(claim_id: UUID) -> bool`, `error(claim_id: UUID) -> str | None`.
  - `WebServices` dataclass: `db_path: Path`, `blobs: BlobStore`, `settings: WebSettings`, `runner: ClaimRunner`, `intake_worker: Worker`, `intake: Callable[[], IntakeSessions] | None`, `memory: ClaimMemory | None`, `upload_dir: Path`; methods `open_store() -> ContextManager[SQLiteEventStore]`, `sessions() -> IntakeSessions` (call only on the intake worker), `similar(state, events) -> list[SimilarView]`.
  - `pipeline_processor(services_db: Path, deps_for: Callable[[SQLiteEventStore], PipelineDeps]) -> Callable[[UUID], None]`.

- [ ] **Step 1: Write the failing tests**

`tests/unit/web/test_runner.py`:

```python
import threading
from pathlib import Path
from uuid import UUID, uuid4

from claimlens.events.store import SQLiteEventStore
from claimlens.intake import submit_claim
from claimlens.web.runner import ClaimRunner
from claimlens.web.services import pipeline_processor
from claimlens.web.workers import Worker
from tests.unit.web.helpers import deps, image


def test_an_inline_worker_runs_at_once() -> None:
    worker = Worker("t", inline=True)
    assert worker.call(lambda: 41 + 1) == 42


def test_a_threaded_worker_keeps_one_thread() -> None:
    worker = Worker("t")
    names = {worker.call(lambda: threading.current_thread().name) for _ in range(3)}
    worker.shutdown()
    assert len(names) == 1


def test_the_runner_processes_a_claim_on_its_worker(tmp_path: Path) -> None:
    db = tmp_path / "events.db"
    store = SQLiteEventStore(db)
    d = deps(tmp_path, store)
    claim = submit_claim(
        store, d.blobs, policy_id="P-1001", description="x", photo_paths=[image(tmp_path / "a.png")]
    )
    store.close()
    process = pipeline_processor(db, lambda s: deps(tmp_path, s))
    runner = ClaimRunner(Worker("claims", inline=True), process)
    assert runner.start(claim)
    assert not runner.running(claim)
    check = SQLiteEventStore(db)
    assert "RouteDecided" in [e.type for e in check.load(claim)]
    check.close()


def test_a_failure_is_kept_as_the_claims_error() -> None:
    def boom(claim_id: UUID) -> None:
        raise RuntimeError("detector crashed")

    runner = ClaimRunner(Worker("claims", inline=True), boom)
    claim = uuid4()
    runner.start(claim)
    assert runner.error(claim) == "RuntimeError: detector crashed"
    assert not runner.running(claim)


def test_a_second_start_while_running_is_refused() -> None:
    gate = threading.Event()
    runner = ClaimRunner(Worker("claims"), lambda claim_id: gate.wait(5) and None)
    claim = uuid4()
    assert runner.start(claim)
    assert runner.running(claim)
    assert not runner.start(claim)
    gate.set()
```

- [ ] **Step 2: Run the tests to see them fail**

Run: `uv run pytest tests/unit/web/test_runner.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'claimlens.web.runner'`.

- [ ] **Step 3: Implement the worker**

`src/claimlens/web/workers.py`:

```python
"""One dedicated thread per area. SQLite connections (the event store, the LLM budget and cache)
must stay on the thread that opened them, so everything that holds one runs here."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor


class Worker:
    def __init__(self, name: str, *, inline: bool = False) -> None:
        self._inline = inline
        self._pool = None if inline else ThreadPoolExecutor(max_workers=1, thread_name_prefix=name)

    def submit[T](self, fn: Callable[[], T]) -> Future[T]:
        if self._pool is not None:
            return self._pool.submit(fn)
        future: Future[T] = Future()
        try:
            future.set_result(fn())
        except Exception as exc:
            future.set_exception(exc)
        return future

    def call[T](self, fn: Callable[[], T]) -> T:
        return self.submit(fn).result()

    def shutdown(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=False, cancel_futures=True)
```

- [ ] **Step 4: Implement the runner**

`src/claimlens/web/runner.py`:

```python
"""Runs the claims pipeline in the background, one claim at a time, on the claims worker."""

from __future__ import annotations

import threading
from collections.abc import Callable
from uuid import UUID

from claimlens.web.workers import Worker


class ClaimRunner:
    def __init__(self, worker: Worker, process: Callable[[UUID], None]) -> None:
        self._worker = worker
        self._process = process
        self._lock = threading.Lock()
        self._running: set[UUID] = set()
        self._errors: dict[UUID, str] = {}

    def start(self, claim_id: UUID) -> bool:
        """Queue the claim; False if it is already queued or running (a double click)."""
        with self._lock:
            if claim_id in self._running:
                return False
            self._running.add(claim_id)
            self._errors.pop(claim_id, None)
        self._worker.submit(lambda: self._run(claim_id))
        return True

    def _run(self, claim_id: UUID) -> None:
        try:
            self._process(claim_id)
        except Exception as exc:  # the claim stays resumable from its log
            with self._lock:
                self._errors[claim_id] = f"{type(exc).__name__}: {exc}"
        finally:
            with self._lock:
                self._running.discard(claim_id)

    def running(self, claim_id: UUID) -> bool:
        with self._lock:
            return claim_id in self._running

    def error(self, claim_id: UUID) -> str | None:
        with self._lock:
            return self._errors.get(claim_id)
```

- [ ] **Step 5: Implement the services**

`src/claimlens/web/services.py`:

```python
"""What the routers need, built once at start-up (the real wiring is in web/serve.py)."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from uuid import UUID

from claimlens.blobs import BlobStore
from claimlens.events.envelope import ClaimEvent
from claimlens.events.projection import ClaimState
from claimlens.events.store import SQLiteEventStore
from claimlens.intake_agent.session import IntakeSessions
from claimlens.memory.index import ClaimMemory
from claimlens.memory.records import build_record
from claimlens.web.runner import ClaimRunner
from claimlens.web.schemas import SimilarView
from claimlens.web.settings import WebSettings
from claimlens.web.workers import Worker
from claimlens.workflow import PipelineDeps, process_claim


def pipeline_processor(
    db_path: Path, deps_for: Callable[[SQLiteEventStore], PipelineDeps]
) -> Callable[[UUID], None]:
    """Process one claim with its own store, opened and closed on the calling (worker) thread."""

    def process(claim_id: UUID) -> None:
        store = SQLiteEventStore(db_path)
        try:
            process_claim(claim_id, deps_for(store))
        finally:
            store.close()

    return process


@dataclass
class WebServices:
    db_path: Path
    blobs: BlobStore
    settings: WebSettings
    runner: ClaimRunner
    intake_worker: Worker
    intake: Callable[[], IntakeSessions] | None
    upload_dir: Path
    memory: ClaimMemory | None = None
    _sessions: list[IntakeSessions] = field(default_factory=list)

    @contextmanager
    def open_store(self) -> Iterator[SQLiteEventStore]:
        """A store for one request: open and close it inside the endpoint body, never in a
        dependency (FastAPI may run a dependency and its endpoint on different threads)."""
        store = SQLiteEventStore(self.db_path)
        try:
            yield store
        finally:
            store.close()

    def sessions(self) -> IntakeSessions:
        """The intake sessions, built on first use. Call only on the intake worker."""
        if self.intake is None:
            raise LookupError("the intake chat is not available")
        if not self._sessions:
            self._sessions.append(self.intake())
        return self._sessions[0]

    def similar(self, state: ClaimState, events: Sequence[ClaimEvent]) -> list[SimilarView]:
        """Similar claims in memory now (read-only). Errors show as none, never break the page."""
        if self.memory is None or not state.photos:
            return []
        try:
            mine = build_record(state, events, self.blobs.path, provisional=True)
            return [
                SimilarView(
                    claim_id=s.claim_id,
                    reason=s.reason,
                    distance=s.distance,
                    route=s.route,
                    damage=s.damage,
                )
                for s in self.memory.similar(mine)
            ]
        except Exception:
            return []
```

- [ ] **Step 6: Run the tests**

Run: `uv run pytest tests/unit/web/test_runner.py -v && uv run mypy`
Expected: PASS; mypy clean.

- [ ] **Step 7: Commit**

```bash
git add src/claimlens/web/workers.py src/claimlens/web/runner.py src/claimlens/web/services.py tests/unit/web/test_runner.py
git commit -m "feat: web workers, background claim runner and services"
```

---

### Task 4: The app factory and the claims API

**Files:**
- Create: `src/claimlens/web/app.py`, `src/claimlens/web/routers/__init__.py`, `src/claimlens/web/routers/claims.py`
- Test: `tests/unit/web/conftest.py`, `tests/unit/web/test_claims_api.py`

**Interfaces:**
- Consumes: `WebServices`, views, schemas.
- Produces: `create_app(services: WebServices) -> FastAPI` (stores services on `app.state.services`); dependency `get_services(request: Request) -> WebServices` in `app.py`; helper `load_claim(store, claim_id) -> list[ClaimEvent]` in `routers/claims.py` raising `HTTPException(404)` or `HTTPException(409)`; endpoints `GET /api/claims`, `GET /api/claims/{claim_id}`, `GET /api/claims/{claim_id}/photos/{photo_id}`, `POST /api/claims/{claim_id}/resume`.
- Fixtures (`tests/unit/web/conftest.py`): `services(tmp_path) -> WebServices` (inline workers, fake pipeline, no intake), `client(services) -> TestClient`.

- [ ] **Step 1: Write the fixtures**

`tests/unit/web/conftest.py`:

```python
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from claimlens.blobs import BlobStore
from claimlens.web.app import create_app
from claimlens.web.runner import ClaimRunner
from claimlens.web.services import WebServices, pipeline_processor
from claimlens.web.settings import WebSettings
from claimlens.web.workers import Worker
from tests.unit.web.helpers import deps


@pytest.fixture
def services(tmp_path: Path) -> WebServices:
    db = tmp_path / "events.db"
    process = pipeline_processor(db, lambda store: deps(tmp_path, store))
    return WebServices(
        db_path=db,
        blobs=BlobStore(tmp_path / "blobs"),
        settings=WebSettings(max_upload_mb=1),
        runner=ClaimRunner(Worker("claims", inline=True), process),
        intake_worker=Worker("intake", inline=True),
        intake=None,
        upload_dir=tmp_path / "uploads",
    )


@pytest.fixture
def client(services: WebServices) -> Iterator[TestClient]:
    with TestClient(create_app(services)) as test_client:
        yield test_client
```

Note: `deps(tmp_path, store)` builds `BlobStore(tmp_path / "blobs")`, the same folder as `services.blobs`.

- [ ] **Step 2: Write the failing tests**

`tests/unit/web/test_claims_api.py`:

```python
import sqlite3
from pathlib import Path
from uuid import uuid4

from fastapi.testclient import TestClient

from claimlens.events.store import SQLiteEventStore
from claimlens.intake import submit_claim
from claimlens.web.services import WebServices
from tests.unit.web.helpers import decided_claim, image


def _decided(
    services: WebServices, tmp_path: Path, color: tuple[int, int, int] = (200, 30, 30)
) -> str:
    with services.open_store() as store:
        return str(decided_claim(tmp_path, store, color))


def test_the_list_and_the_detail(client: TestClient, services: WebServices, tmp_path: Path) -> None:
    claim = _decided(services, tmp_path)
    listed = client.get("/api/claims").json()
    assert [c["claim_id"] for c in listed] == [claim]
    assert listed[0]["status"] == "decided"
    detail = client.get(f"/api/claims/{claim}").json()
    assert detail["chain_ok"] is True
    assert detail["photos"][0]["boxes"][0]["label"].startswith("dent")


def test_an_unknown_claim_is_404(client: TestClient) -> None:
    response = client.get(f"/api/claims/{uuid4()}")
    assert response.status_code == 404
    assert response.json() == {"detail": "No such claim."}


def test_a_malformed_claim_id_is_404(client: TestClient) -> None:
    assert client.get("/api/claims/not-a-uuid").status_code == 404


def test_a_tampered_log_is_409(client: TestClient, services: WebServices, tmp_path: Path) -> None:
    claim = _decided(services, tmp_path)
    with sqlite3.connect(services.db_path) as conn:
        conn.execute("UPDATE events SET data = replace(data, 'P-1001', 'P-1002') WHERE seq = 1")
    response = client.get(f"/api/claims/{claim}")
    assert response.status_code == 409
    assert "failed verification" in response.json()["detail"]
    assert client.get("/api/claims").json()[0]["status"] == "log_failed"


def test_a_photo_is_served_for_its_own_claim(
    client: TestClient, services: WebServices, tmp_path: Path
) -> None:
    claim = _decided(services, tmp_path)
    response = client.get(f"/api/claims/{claim}/photos/p1")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/")


def test_a_photo_of_another_claim_is_404(
    client: TestClient, services: WebServices, tmp_path: Path
) -> None:
    first = _decided(services, tmp_path, (200, 30, 30))
    _decided(services, tmp_path, (30, 200, 30))
    assert client.get(f"/api/claims/{first}/photos/p9").status_code == 404
    assert client.get(f"/api/claims/{first}/photos/..%2F..%2Fevents.db").status_code == 404


def test_resume_finishes_a_stopped_claim(
    client: TestClient, services: WebServices, tmp_path: Path
) -> None:
    with services.open_store() as store:
        claim = submit_claim(
            store,
            services.blobs,
            policy_id="P-1001",
            description="x",
            photo_paths=[image(tmp_path / "in" / "a.png")],
        )
    assert client.get(f"/api/claims/{claim}").json()["status"] == "stopped"
    assert client.post(f"/api/claims/{claim}/resume").status_code == 202
    assert client.get(f"/api/claims/{claim}").json()["status"] == "decided"


def test_resume_of_a_decided_claim_is_409(
    client: TestClient, services: WebServices, tmp_path: Path
) -> None:
    claim = _decided(services, tmp_path)
    assert client.post(f"/api/claims/{claim}/resume").status_code == 409


def test_resume_while_running_is_409(
    client: TestClient, services: WebServices, tmp_path: Path
) -> None:
    with services.open_store() as store:
        claim = submit_claim(
            store,
            services.blobs,
            policy_id="P-1001",
            description="x",
            photo_paths=[image(tmp_path / "in" / "b.png")],
        )
    services.runner._running.add(claim)  # as if a worker were busy with it
    assert client.post(f"/api/claims/{claim}/resume").status_code == 409


def test_the_api_is_documented(client: TestClient) -> None:
    paths = client.get("/openapi.json").json()["paths"]
    assert "/api/claims/{claim_id}" in paths
```

- [ ] **Step 3: Run the tests to see them fail**

Run: `uv run pytest tests/unit/web/test_claims_api.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'claimlens.web.app'`.

- [ ] **Step 4: Implement the app factory**

`src/claimlens/web/app.py`:

```python
"""The FastAPI app: API routers, pages, static files and plain error messages."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from claimlens.web.services import WebServices

HERE = Path(__file__).parent


def get_services(request: Request) -> WebServices:
    services: WebServices = request.app.state.services
    return services


def create_app(services: WebServices) -> FastAPI:
    from claimlens.web.routers import claims

    app = FastAPI(
        title="ClaimLens",
        summary="Local prototype: file a claim by chat, follow it, review it.",
        version="0.1.0",
    )
    app.state.services = services
    app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")
    app.include_router(claims.router)

    @app.exception_handler(RequestValidationError)
    async def _invalid(request: Request, exc: RequestValidationError) -> JSONResponse:
        first = exc.errors()[0] if exc.errors() else {}
        where = ".".join(str(p) for p in first.get("loc", ())[1:]) or "request"
        return JSONResponse({"detail": f"Invalid {where}: {first.get('msg', 'bad value')}."}, 422)

    return app
```

Also create the static folder now so the mount works: `src/claimlens/web/static/app.css` containing a single line `/* ClaimLens styles: filled in by Task 7. */`.

- [ ] **Step 5: Implement the claims router**

`src/claimlens/web/routers/__init__.py`:

```python
"""The web app's routers, one per area."""
```

`src/claimlens/web/routers/claims.py`:

```python
"""Claims: list, detail, photos, resume."""

from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

from claimlens.events.envelope import ChainIntegrityError, ClaimEvent
from claimlens.events.projection import fold
from claimlens.events.store import ClaimNotFoundError, SQLiteEventStore
from claimlens.web.app import get_services
from claimlens.web.schemas import ClaimDetail, ClaimSummary
from claimlens.web.services import WebServices
from claimlens.web.views import claim_detail, claim_summary

router = APIRouter(prefix="/api/claims", tags=["claims"])
Services = Annotated[WebServices, Depends(get_services)]
LOG_FAILED = "The audit log failed verification. Nobody can act on this claim."


def parse_claim_id(claim_id: str) -> UUID:
    try:
        return UUID(claim_id)
    except ValueError:
        raise HTTPException(404, "No such claim.") from None


def load_claim(store: SQLiteEventStore, claim_id: UUID) -> list[ClaimEvent]:
    try:
        return store.load(claim_id)
    except ClaimNotFoundError:
        raise HTTPException(404, "No such claim.") from None
    except ChainIntegrityError:
        raise HTTPException(409, LOG_FAILED) from None


@router.get("", response_model=list[ClaimSummary], summary="All claims, newest first")
def list_claims(services: Services) -> list[ClaimSummary]:
    rows: list[ClaimSummary] = []
    with services.open_store() as store:
        for claim_id in store.claim_ids():
            try:
                events = store.load(claim_id)
            except ChainIntegrityError:
                rows.append(
                    ClaimSummary(
                        claim_id=str(claim_id), policy_id="?", filed_at="", status="log_failed"
                    )
                )
                continue
            rows.append(claim_summary(fold(events), events, services.runner.running(claim_id)))
    return sorted(rows, key=lambda r: r.filed_at, reverse=True)


@router.get("/{claim_id}", response_model=ClaimDetail, summary="One claim, from its events")
def get_claim(claim_id: str, services: Services) -> ClaimDetail:
    cid = parse_claim_id(claim_id)
    with services.open_store() as store:
        events = load_claim(store, cid)
    state = fold(events)
    return claim_detail(
        state,
        events,
        running=services.runner.running(cid),
        error=services.runner.error(cid),
        similar=services.similar(state, events),
    )


@router.get("/{claim_id}/photos/{photo_id}", summary="A photo of this claim")
def get_photo(claim_id: str, photo_id: str, services: Services) -> FileResponse:
    cid = parse_claim_id(claim_id)
    with services.open_store() as store:
        state = fold(load_claim(store, cid))
    photo = state.photos.get(photo_id)
    if photo is None:
        raise HTTPException(404, "No such photo.")
    return FileResponse(services.blobs.path(photo.blob_name))


@router.post("/{claim_id}/resume", status_code=202, summary="Finish processing a stopped claim")
def resume_claim(claim_id: str, services: Services) -> dict[str, str]:
    cid = parse_claim_id(claim_id)
    with services.open_store() as store:
        state = fold(load_claim(store, cid))
    if state.decision is not None:
        raise HTTPException(409, "This claim is already decided.")
    if not services.runner.start(cid):
        raise HTTPException(409, "This claim is already being processed.")
    return {"detail": "Processing resumed."}
```

`FileResponse` sets `content-type` from the blob's suffix; if blobs are stored without a suffix, pass `media_type=` from `PIL.Image.open(path).get_format_mimetype()` instead (check `BlobStore.path` once while implementing).

- [ ] **Step 6: Run the tests**

Run: `uv run pytest tests/unit/web -v && uv run mypy`
Expected: PASS; mypy clean.

- [ ] **Step 7: Commit**

```bash
git add src/claimlens/web/app.py src/claimlens/web/routers src/claimlens/web/static tests/unit/web/conftest.py tests/unit/web/test_claims_api.py
git commit -m "feat: claims API (list, detail, photos, resume)"
```

---

### Task 5: The review API

**Files:**
- Create: `src/claimlens/web/routers/review.py`
- Modify: `src/claimlens/web/app.py` (include the router)
- Test: `tests/unit/web/test_review_api.py`

**Interfaces:**
- Consumes: `ReviewRequest`, `load_claim`, `parse_claim_id`, `record_review(store, claim_id, review, *, memory, photo_path) -> int`.
- Produces: `POST /api/claims/{claim_id}/review` → `201 {"seq": int}`.

- [ ] **Step 1: Write the failing tests**

`tests/unit/web/test_review_api.py`:

```python
import sqlite3
from pathlib import Path

from fastapi.testclient import TestClient

from claimlens.intake import submit_claim
from claimlens.web.services import WebServices
from tests.unit.web.helpers import decided_claim, image


def _decided(services: WebServices, tmp_path: Path) -> str:
    with services.open_store() as store:
        return str(decided_claim(tmp_path, store))


def test_a_person_approves(client: TestClient, services: WebServices, tmp_path: Path) -> None:
    claim = _decided(services, tmp_path)
    body = {"reviewer": "Sam", "action": "approve", "note": "fine"}
    assert client.post(f"/api/claims/{claim}/review", json=body).status_code == 201
    detail = client.get(f"/api/claims/{claim}").json()
    assert detail["status"] == "reviewed"
    assert detail["review_action"] == "approve"


def test_an_override_needs_a_final_route(
    client: TestClient, services: WebServices, tmp_path: Path
) -> None:
    claim = _decided(services, tmp_path)
    body = {"reviewer": "Sam", "action": "override"}
    assert client.post(f"/api/claims/{claim}/review", json=body).status_code == 422
    body["final_route"] = "FRAUD_REVIEW"
    assert client.post(f"/api/claims/{claim}/review", json=body).status_code == 201


def test_a_blank_reviewer_or_unknown_action_is_422(
    client: TestClient, services: WebServices, tmp_path: Path
) -> None:
    claim = _decided(services, tmp_path)
    assert (
        client.post(
            f"/api/claims/{claim}/review", json={"reviewer": "", "action": "approve"}
        ).status_code
        == 422
    )
    assert (
        client.post(
            f"/api/claims/{claim}/review", json={"reviewer": "S", "action": "pay"}
        ).status_code
        == 422
    )


def test_an_undecided_claim_cannot_be_reviewed(
    client: TestClient, services: WebServices, tmp_path: Path
) -> None:
    with services.open_store() as store:
        claim = submit_claim(
            store,
            services.blobs,
            policy_id="P-1001",
            description="x",
            photo_paths=[image(tmp_path / "in" / "u.png")],
        )
    body = {"reviewer": "Sam", "action": "deny"}
    assert client.post(f"/api/claims/{claim}/review", json=body).status_code == 409


def test_a_tampered_claim_cannot_be_reviewed(
    client: TestClient, services: WebServices, tmp_path: Path
) -> None:
    claim = _decided(services, tmp_path)
    with sqlite3.connect(services.db_path) as conn:
        conn.execute("UPDATE events SET data = replace(data, 'P-1001', 'P-1002') WHERE seq = 1")
    body = {"reviewer": "Sam", "action": "approve"}
    assert client.post(f"/api/claims/{claim}/review", json=body).status_code == 409
```

- [ ] **Step 2: Run the tests to see them fail**

Run: `uv run pytest tests/unit/web/test_review_api.py -v`
Expected: FAIL with 404/405 on `/review` (no such route).

- [ ] **Step 3: Implement**

`src/claimlens/web/routers/review.py`:

```python
"""Review: the only place a person decides, including the only way to deny."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from pydantic import ValidationError

from claimlens.events.payloads import HumanReviewed
from claimlens.events.projection import fold
from claimlens.review_queue import record_review
from claimlens.web.app import get_services
from claimlens.web.routers.claims import load_claim, parse_claim_id
from claimlens.web.schemas import ReviewRequest
from claimlens.web.services import WebServices

router = APIRouter(prefix="/api/claims", tags=["review"])
Services = Annotated[WebServices, Depends(get_services)]


@router.post("/{claim_id}/review", status_code=201, summary="Record what a reviewer decided")
def review_claim(claim_id: str, body: ReviewRequest, services: Services) -> dict[str, int]:
    cid = parse_claim_id(claim_id)
    try:
        review = HumanReviewed(
            reviewer=body.reviewer.strip(),
            action=body.action,
            final_route=body.final_route,
            note=body.note,
        )
    except ValidationError as exc:
        raise HTTPException(422, exc.errors()[0]["msg"]) from None
    with services.open_store() as store:
        if fold(load_claim(store, cid)).decision is None:
            raise HTTPException(409, "This claim has not been decided yet.")
        seq = record_review(
            store, cid, review, memory=services.memory, photo_path=services.blobs.path
        )
    return {"seq": seq}
```

In `create_app`, change the import to `from claimlens.web.routers import claims, review` and add `app.include_router(review.router)` after the claims router.

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/unit/web -v && uv run mypy`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/claimlens/web/routers/review.py src/claimlens/web/app.py tests/unit/web/test_review_api.py
git commit -m "feat: review API, the only way a person decides a claim"
```

---

### Task 6: Uploads and the intake chat API

**Files:**
- Create: `src/claimlens/web/uploads.py`, `src/claimlens/web/routers/intake.py`
- Modify: `src/claimlens/web/app.py` (include the router)
- Test: `tests/unit/web/test_uploads.py`, `tests/unit/web/test_intake_api.py`

**Interfaces:**
- Consumes: `IntakeSessions.start/resume/reply`, `AgentTurn`, `file_intake_claim`, `LLMError` (`claimlens.llm.types`), `ClaimRunner.start`.
- Produces: `check_upload(data: bytes, max_bytes: int, folder: Path) -> Path` raising `UploadError(message)`; endpoints `POST /api/intake` → `ChatTurn` (201), `GET /api/intake/{session_id}` → `ChatTurn`, `POST /api/intake/{session_id}/reply` (multipart: `text`, optional `photo`) → `ChatTurn`. Test fixture `chat_services` in `test_intake_api.py`.

- [ ] **Step 1: Write the failing upload tests**

`tests/unit/web/test_uploads.py`:

```python
import io
from pathlib import Path

import pytest
from PIL import Image

from claimlens.web.uploads import UploadError, check_upload


def _png(size: tuple[int, int] = (64, 48)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, (10, 20, 30)).save(buffer, "PNG")
    return buffer.getvalue()


def test_a_png_is_kept_under_its_hash(tmp_path: Path) -> None:
    kept = check_upload(_png(), 1_000_000, tmp_path)
    assert kept.parent == tmp_path
    assert kept.suffix == ".png"
    assert check_upload(_png(), 1_000_000, tmp_path) == kept


def test_a_fake_jpeg_is_refused(tmp_path: Path) -> None:
    with pytest.raises(UploadError, match="JPEG, PNG or WebP"):
        check_upload(b"not an image at all", 1_000_000, tmp_path)


def test_a_gif_is_refused(tmp_path: Path) -> None:
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8)).save(buffer, "GIF")
    with pytest.raises(UploadError, match="JPEG, PNG or WebP"):
        check_upload(buffer.getvalue(), 1_000_000, tmp_path)


def test_a_large_photo_is_refused(tmp_path: Path) -> None:
    with pytest.raises(UploadError, match="larger than 0 MB"):
        check_upload(_png(), 100, tmp_path)
```

- [ ] **Step 2: Run them to see them fail**

Run: `uv run pytest tests/unit/web/test_uploads.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'claimlens.web.uploads'`.

- [ ] **Step 3: Implement uploads**

`src/claimlens/web/uploads.py`:

```python
"""Photo uploads: checked by content, size-limited, saved under their content hash."""

from __future__ import annotations

import hashlib
import io
from pathlib import Path

from PIL import Image, UnidentifiedImageError

_FORMATS = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp"}


class UploadError(ValueError):
    """A plain message for the customer."""


def check_upload(data: bytes, max_bytes: int, folder: Path) -> Path:
    if len(data) > max_bytes:
        raise UploadError(
            f"That photo is larger than {max_bytes // 1_000_000} MB. Please send a smaller one."
        )
    try:
        with Image.open(io.BytesIO(data)) as image:
            image.verify()
            kind = image.format or ""
    except (UnidentifiedImageError, OSError, SyntaxError):
        kind = ""
    if kind not in _FORMATS:
        raise UploadError("Please send a photo as JPEG, PNG or WebP.")
    folder.mkdir(parents=True, exist_ok=True)
    kept = folder / f"{hashlib.sha256(data).hexdigest()}{_FORMATS[kind]}"
    if not kept.exists():
        kept.write_bytes(data)
    return kept
```

Run: `uv run pytest tests/unit/web/test_uploads.py -v` → PASS.

- [ ] **Step 4: Write the failing intake API tests**

`tests/unit/web/test_intake_api.py`:

```python
import io
import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.sqlite import SqliteSaver
from PIL import Image

from claimlens.agent.chat_model import GatewayChatModel
from claimlens.events.store import SQLiteEventStore
from claimlens.intake_agent.graph import IntakeState, build_intake_graph
from claimlens.intake_agent.session import IntakeSessions, file_intake_claim
from claimlens.llm.budget import Budget
from claimlens.llm.cache import ResponseCache
from claimlens.llm.gateway import Gateway
from claimlens.llm.provider import FakeProvider, ProviderReply
from claimlens.llm.types import LLMError
from claimlens.web.app import create_app
from claimlens.web.services import WebServices
from tests.unit.test_intake_graph import CONFIG, LLM, TODAY, lookup
from tests.unit.test_intake_session import SCRIPT


def _png(seed: int) -> bytes:
    buffer = io.BytesIO()
    Image.effect_noise((640, 480), 60 + seed).convert("RGB").save(buffer, "PNG")
    return buffer.getvalue()


def _chat(
    services: WebServices, tmp_path: Path, script: list[ProviderReply | Exception]
) -> TestClient:
    def build() -> IntakeSessions:
        gateway = Gateway(
            LLM,
            FakeProvider(script),
            Budget(tmp_path / "b.sqlite", LLM.limits, today=lambda: TODAY),
            ResponseCache(tmp_path / "c.sqlite"),
            lambda c: None,
            sleep=lambda s: None,
        )
        model = GatewayChatModel(gateway=gateway, tier="fast", max_tokens=400)

        def submit(state: IntakeState) -> str:
            store = SQLiteEventStore(services.db_path)
            try:
                return str(file_intake_claim(store, services.blobs, state, CONFIG))
            finally:
                store.close()

        conn = sqlite3.connect(str(tmp_path / "intake.sqlite"), check_same_thread=False)
        graph = build_intake_graph(
            model, lookup, CONFIG, submit, lambda: TODAY, checkpointer=SqliteSaver(conn)
        )
        return IntakeSessions(graph, system="system prompt")

    services.intake = build
    return TestClient(create_app(services))


def _photo(client: TestClient, session: str, seed: int) -> dict:  # type: ignore[type-arg]
    files = {"photo": (f"p{seed}.png", _png(seed), "image/png")}
    return client.post(f"/api/intake/{session}/reply", data={"text": ""}, files=files).json()


def test_a_chat_files_a_claim_and_starts_the_pipeline(
    services: WebServices, tmp_path: Path
) -> None:
    client = _chat(services, tmp_path, list(SCRIPT))
    start = client.post("/api/intake")
    assert start.status_code == 201
    session = start.json()["session_id"]
    turn = client.post(
        f"/api/intake/{session}/reply", data={"text": "P-1001, bollard yesterday"}
    ).json()
    assert turn["photo_kind"] == "overview"
    for seed in (1, 2):
        turn = _photo(client, session, seed)
    turn = _photo(client, session, 3)
    assert turn["claim_id"] is not None
    detail = client.get(f"/api/claims/{turn['claim_id']}").json()
    assert detail["status"] == "decided"  # the inline runner processed it
    assert detail["stages"][0]["state"] == "done"  # intake chat


def test_the_customer_never_sees_a_route_or_price(services: WebServices, tmp_path: Path) -> None:
    client = _chat(services, tmp_path, list(SCRIPT))
    session = client.post("/api/intake").json()["session_id"]
    client.post(f"/api/intake/{session}/reply", data={"text": "P-1001"})
    turns = [_photo(client, session, seed) for seed in (1, 2, 3)]
    assert set(turns[-1]) == {"session_id", "message", "photo_kind", "claim_id"}
    for word in ("FAST_TRACK", "ADJUSTER_REVIEW", "FRAUD_REVIEW", "$"):
        assert word not in turns[-1]["message"]


def test_a_reloaded_chat_continues(services: WebServices, tmp_path: Path) -> None:
    client = _chat(services, tmp_path, list(SCRIPT))
    session = client.post("/api/intake").json()["session_id"]
    client.post(f"/api/intake/{session}/reply", data={"text": "P-1001"})
    again = client.get(f"/api/intake/{session}").json()
    assert again["photo_kind"] == "overview"


def test_an_unknown_session_is_404(services: WebServices, tmp_path: Path) -> None:
    client = _chat(services, tmp_path, list(SCRIPT))
    assert client.get("/api/intake/nope").status_code == 404
    assert client.post("/api/intake/nope/reply", data={"text": "hi"}).status_code == 404


def test_a_bad_photo_is_422_and_the_chat_waits(services: WebServices, tmp_path: Path) -> None:
    client = _chat(services, tmp_path, list(SCRIPT))
    session = client.post("/api/intake").json()["session_id"]
    client.post(f"/api/intake/{session}/reply", data={"text": "P-1001"})
    files = {"photo": ("car.jpg", b"not an image", "image/jpeg")}
    response = client.post(f"/api/intake/{session}/reply", data={"text": ""}, files=files)
    assert response.status_code == 422
    assert "JPEG, PNG or WebP" in response.json()["detail"]
    assert client.get(f"/api/intake/{session}").json()["photo_kind"] == "overview"


def test_an_llm_outage_is_503_and_the_answers_are_kept(
    services: WebServices, tmp_path: Path
) -> None:
    script: list[ProviderReply | Exception] = [SCRIPT[0], LLMError("overloaded")]
    client = _chat(services, tmp_path, script)
    session = client.post("/api/intake").json()["session_id"]
    response = client.post(f"/api/intake/{session}/reply", data={"text": "P-1001"})
    assert response.status_code == 503
    assert "answers are saved" in response.json()["detail"]


def test_no_intake_configured_is_503(client: TestClient) -> None:
    assert client.post("/api/intake").status_code == 503
```

Before running, check that `FakeProvider` raises an `Exception` item from its script (as the M6a stalled-session tests do); if the gateway retries a failed call, add enough `LLMError` items for every retry and the fallback model.

- [ ] **Step 5: Run them to see them fail**

Run: `uv run pytest tests/unit/web/test_intake_api.py -v`
Expected: FAIL with 404 on `/api/intake`.

- [ ] **Step 6: Implement the intake router**

`src/claimlens/web/routers/intake.py`:

```python
"""The customer's chat with the intake agent. Every call runs on the intake worker thread."""

from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from claimlens.intake_agent.session import AgentTurn
from claimlens.llm.types import LLMError
from claimlens.web.app import get_services
from claimlens.web.schemas import ChatTurn
from claimlens.web.services import WebServices
from claimlens.web.uploads import UploadError, check_upload

router = APIRouter(prefix="/api/intake", tags=["intake"])
Services = Annotated[WebServices, Depends(get_services)]
SAVED = "Sorry, something went wrong on our side. Your answers are saved; please try again shortly."


def _chat_turn(turn: AgentTurn) -> ChatTurn:
    return ChatTurn(
        session_id=turn.session_id,
        message=turn.message,
        photo_kind=turn.photo_kind,
        claim_id=turn.claim_id,
    )


def _available(services: WebServices) -> None:
    if services.intake is None:
        raise HTTPException(503, "The intake chat is not available.")


@router.post("", response_model=ChatTurn, status_code=201, summary="Start a claim chat")
def start(services: Services) -> ChatTurn:
    _available(services)
    try:
        turn = services.intake_worker.call(lambda: services.sessions().start())
    except LLMError:
        raise HTTPException(503, SAVED) from None
    return _chat_turn(turn)


@router.get("/{session_id}", response_model=ChatTurn, summary="The question a chat is waiting on")
def show(session_id: str, services: Services) -> ChatTurn:
    _available(services)
    try:
        turn = services.intake_worker.call(lambda: services.sessions().resume(session_id))
    except LLMError:
        raise HTTPException(503, SAVED) from None
    if turn is None:
        raise HTTPException(404, "No open chat with that reference.")
    return _chat_turn(turn)


@router.post(
    "/{session_id}/reply", response_model=ChatTurn, summary="Answer, with an optional photo"
)
def reply(
    session_id: str,
    services: Services,
    text: Annotated[str, Form(max_length=4000)] = "",
    photo: Annotated[UploadFile | None, File()] = None,
) -> ChatTurn:
    _available(services)
    kept = None
    if photo is not None:
        limit = services.settings.max_upload_mb * 1_000_000
        try:
            kept = check_upload(photo.file.read(limit + 1), limit, services.upload_dir)
        except UploadError as exc:
            raise HTTPException(422, str(exc)) from None

    def answer() -> AgentTurn:
        sessions = services.sessions()
        if sessions.pending(session_id) is None:
            raise LookupError(session_id)
        return sessions.reply(session_id, text, kept)

    try:
        turn = services.intake_worker.call(answer)
    except LookupError:
        raise HTTPException(404, "No open chat with that reference.") from None
    except LLMError:
        raise HTTPException(503, SAVED) from None
    if turn.claim_id is not None:
        services.runner.start(UUID(turn.claim_id))
    return _chat_turn(turn)
```

In `create_app`, import `intake` too and `app.include_router(intake.router)`.

- [ ] **Step 7: Run the tests**

Run: `uv run pytest tests/unit/web -v && uv run mypy`
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add src/claimlens/web/uploads.py src/claimlens/web/routers/intake.py src/claimlens/web/app.py tests/unit/web/test_uploads.py tests/unit/web/test_intake_api.py
git commit -m "feat: intake chat API with checked photo uploads"
```

---

### Task 7: The pages

**Files:**
- Create: `src/claimlens/web/routers/pages.py`, `src/claimlens/web/templates/{base,chat,claims,claim,not_found,log_failed}.html`, `src/claimlens/web/static/{chat.js,claim.js}`
- Modify: `src/claimlens/web/static/app.css` (full styles), `src/claimlens/web/app.py` (templates, pages router, HTML 404)
- Test: `tests/unit/web/test_pages.py`

**Interfaces:**
- Consumes: `get_claim` data via `claim_detail`, `list_claims` logic, `load_claim`, `LOG_FAILED`.
- Produces: `GET /` (chat), `GET /claims` (list, `?filter=review|fraud`), `GET /claims/{claim_id}` (detail). `templates = Jinja2Templates(directory=HERE / "templates")` in `app.py`.

- [ ] **Step 1: Write the failing tests**

`tests/unit/web/test_pages.py`:

```python
import sqlite3
from pathlib import Path

from fastapi.testclient import TestClient

from claimlens.web.services import WebServices
from tests.unit.web.helpers import decided_claim


def _decided(services: WebServices, tmp_path: Path) -> str:
    with services.open_store() as store:
        return str(decided_claim(tmp_path, store))


def test_the_chat_page_renders(client: TestClient) -> None:
    page = client.get("/")
    assert page.status_code == 200
    assert "File a claim" in page.text
    assert "/static/chat.js" in page.text


def test_the_claims_page_lists_and_filters(
    client: TestClient, services: WebServices, tmp_path: Path
) -> None:
    claim = _decided(services, tmp_path)
    assert claim in client.get("/claims").text
    assert claim not in client.get("/claims?filter=fraud").text


def test_the_claim_page_shows_the_evidence_and_escapes_customer_text(
    client: TestClient, services: WebServices, tmp_path: Path
) -> None:
    claim = _decided(services, tmp_path)
    page = client.get(f"/claims/{claim}").text
    assert "Log verified" in page
    assert "dent · back_bumper · 0.90" in page
    assert "&lt;b&gt;slowly&lt;/b&gt;" in page
    assert "<b>slowly</b>" not in page
    assert 'id="review-form"' in page


def test_a_tampered_claim_page_hides_the_review_form(
    client: TestClient, services: WebServices, tmp_path: Path
) -> None:
    claim = _decided(services, tmp_path)
    with sqlite3.connect(services.db_path) as conn:
        conn.execute("UPDATE events SET data = replace(data, 'P-1001', 'P-1002') WHERE seq = 1")
    page = client.get(f"/claims/{claim}")
    assert page.status_code == 409
    assert "Log failed verification" in page.text
    assert 'id="review-form"' not in page.text


def test_an_unknown_claim_page_is_a_404_page(client: TestClient) -> None:
    page = client.get("/claims/not-a-claim")
    assert page.status_code == 404
    assert "not found" in page.text.lower()


def test_the_chat_page_never_mentions_routes(client: TestClient) -> None:
    page = client.get("/").text
    for word in ("FAST_TRACK", "ADJUSTER_REVIEW", "FRAUD_REVIEW", "deductible"):
        assert word not in page
```

- [ ] **Step 2: Run them to see them fail**

Run: `uv run pytest tests/unit/web/test_pages.py -v`
Expected: FAIL (404 on `/`).

- [ ] **Step 3: Implement the pages router and templates wiring**

In `app.py` add `from fastapi.templating import Jinja2Templates`, a module-level `templates = Jinja2Templates(directory=HERE / "templates")` (Jinja2 autoescapes `.html` by default), import `pages` with the other routers and `app.include_router(pages.router)`.

`src/claimlens/web/routers/pages.py`:

```python
"""Server-rendered pages. All customer text goes through Jinja2 autoescaping."""

from __future__ import annotations

from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse

from claimlens.events.projection import fold
from claimlens.web.app import get_services, templates
from claimlens.web.routers.claims import list_claims, load_claim, parse_claim_id
from claimlens.web.services import WebServices
from claimlens.web.views import claim_detail

router = APIRouter(include_in_schema=False)
Services = Annotated[WebServices, Depends(get_services)]
_REVIEW = {"ADJUSTER_REVIEW", "FRAUD_REVIEW"}


@router.get("/", response_class=HTMLResponse)
def chat_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "chat.html", {"title": "File a claim"})


@router.get("/claims", response_class=HTMLResponse)
def claims_page(
    request: Request, services: Services, filter: Literal["all", "review", "fraud"] = "all"
) -> HTMLResponse:
    rows = list_claims(services)
    if filter == "review":
        rows = [r for r in rows if r.route in _REVIEW and r.status == "decided"]
    elif filter == "fraud":
        rows = [r for r in rows if r.route == "FRAUD_REVIEW"]
    return templates.TemplateResponse(
        request, "claims.html", {"title": "Claims", "rows": rows, "filter": filter}
    )


@router.get("/claims/{claim_id}", response_class=HTMLResponse)
def claim_page(request: Request, claim_id: str, services: Services) -> HTMLResponse:
    try:
        cid = parse_claim_id(claim_id)
        with services.open_store() as store:
            events = load_claim(store, cid)
    except HTTPException as exc:
        name = "log_failed.html" if exc.status_code == 409 else "not_found.html"
        return templates.TemplateResponse(
            request, name, {"title": "Claim", "claim_id": claim_id}, status_code=exc.status_code
        )
    state = fold(events)
    detail = claim_detail(
        state,
        events,
        running=services.runner.running(cid),
        error=services.runner.error(cid),
        similar=services.similar(state, events),
    )
    return templates.TemplateResponse(
        request,
        "claim.html",
        {"title": f"Claim {claim_id[:8]}", "c": detail, "poll": services.settings.poll_seconds},
    )
```

- [ ] **Step 4: Write the templates**

`templates/base.html`:

```html
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{{ title }} · ClaimLens</title>
  <link rel="stylesheet" href="/static/app.css">
</head>
<body>
  <header class="top">
    <a class="brand" href="/">ClaimLens</a>
    <nav aria-label="Main">
      <a href="/">File a claim</a>
      <a href="/claims">Claims</a>
      <a href="/docs">API</a>
    </nav>
    <span class="local">Local prototype</span>
  </header>
  <main class="page">{% block main %}{% endblock %}</main>
  {% block scripts %}{% endblock %}
</body>
</html>
```

`templates/chat.html`:

```html
{% extends "base.html" %}
{% block main %}
<section class="chat" aria-labelledby="chat-title">
  <h1 id="chat-title">File a claim</h1>
  <p class="muted">Tell us what happened. We will ask for a few details and three photos.</p>
  <ol id="messages" class="messages" aria-live="polite"></ol>
  <form id="chat-form" class="composer" hidden>
    <label for="text" class="sr">Your answer</label>
    <input id="text" name="text" autocomplete="off" maxlength="4000" placeholder="Type your answer">
    <label id="photo-label" for="photo" class="photo-pick" hidden>Add the photo</label>
    <input id="photo" name="photo" type="file" accept="image/jpeg,image/png,image/webp" hidden>
    <button id="send" type="submit">Send</button>
  </form>
  <button id="start" type="button" class="primary">Start a claim</button>
  <p id="chat-error" class="error" role="alert" hidden></p>
  <p id="done" class="done" hidden></p>
</section>
{% endblock %}
{% block scripts %}<script src="/static/chat.js"></script>{% endblock %}
```

`templates/claims.html`:

```html
{% extends "base.html" %}
{% block main %}
<h1>Claims</h1>
<nav class="filters" aria-label="Filter claims">
  <a href="/claims" {% if filter == "all" %}aria-current="page"{% endif %}>All</a>
  <a href="/claims?filter=review" {% if filter == "review" %}aria-current="page"{% endif %}>Needs review</a>
  <a href="/claims?filter=fraud" {% if filter == "fraud" %}aria-current="page"{% endif %}>Fraud review</a>
</nav>
<div class="table-wrap">
<table>
  <thead><tr><th>Claim</th><th>Policy</th><th>Filed</th><th>Status</th><th>Route</th><th>Rule</th><th>Review</th></tr></thead>
  <tbody>
  {% for r in rows %}
    <tr>
      <td><a class="mono" href="/claims/{{ r.claim_id }}">{{ r.claim_id }}</a></td>
      <td>{{ r.policy_id }}</td>
      <td class="mono">{{ r.filed_at }}</td>
      <td><span class="chip s-{{ r.status }}">{{ r.status.replace("_", " ") }}</span></td>
      <td>{% if r.route %}<span class="chip r-{{ r.route }}">{{ r.route }}</span>{% endif %}</td>
      <td class="mono">{{ r.rule_id or "" }}</td>
      <td>{{ r.review_action or "" }}</td>
    </tr>
  {% else %}
    <tr><td colspan="7" class="muted">No claims yet. File one from the chat, or start the app with --seed.</td></tr>
  {% endfor %}
  </tbody>
</table>
</div>
{% endblock %}
```

`templates/claim.html`:

```html
{% extends "base.html" %}
{% block main %}
<div class="claim-head">
  <div>
    <p class="eyebrow">Claim · policy {{ c.policy_id }}</p>
    <h1 class="mono">{{ c.claim_id }}</h1>
  </div>
  <div class="verdict">
    {% if c.route %}<span class="chip big r-{{ c.route }}">{{ c.route }}</span>
      <span class="mono">{{ c.rule_id }}</span>{% endif %}
    <span class="chip s-{{ c.status }}">{{ c.status }}</span>
    <span class="verified" title="Every event's hash matches the chain">Log verified · {{ c.event_count }} events</span>
  </div>
</div>
{% if c.decision_reason %}<p class="reason">{{ c.decision_reason }}</p>{% endif %}
{% if c.status == "stopped" %}
  <div class="alert" role="alert">
    <p>Processing stopped{% if c.error %}: {{ c.error }}{% endif %}.</p>
    <button id="resume" type="button" class="primary" data-claim="{{ c.claim_id }}">Resume</button>
  </div>
{% endif %}

<section aria-labelledby="t-stages">
  <h2 id="t-stages">Stages</h2>
  <ol class="stages">
  {% for s in c.stages %}
    <li class="stage st-{{ s.state }}"><span class="dot" aria-hidden="true"></span>
      <span class="label">{{ s.label }}</span>
      <span class="state">{{ s.state }}</span>
      {% if s.detail %}<span class="detail">{{ s.detail }}</span>{% endif %}</li>
  {% endfor %}
  </ol>
</section>

<div class="grid">
<section aria-labelledby="t-photos">
  <h2 id="t-photos">Photos</h2>
  <div class="photos">
  {% for p in c.photos %}
    <figure class="photo">
      <div class="frame">
        <img src="/api/claims/{{ c.claim_id }}/photos/{{ p.photo_id }}" alt="Claim photo {{ p.photo_id }}{% if p.kind %} ({{ p.kind }}){% endif %}">
        {% for b in p.boxes %}
          <span class="box" style="left: {{ b.left }}%; top: {{ b.top }}%; width: {{ b.width }}%; height: {{ b.height }}%"><span class="tag">{{ b.label }}</span></span>
        {% endfor %}
      </div>
      <figcaption>{{ p.kind or p.photo_id }} · {{ p.status }}{% if p.reject_reason %}: {{ p.reject_reason }}{% endif %}</figcaption>
    </figure>
  {% else %}
    <p class="muted">No photos.</p>
  {% endfor %}
  </div>
</section>

<section aria-labelledby="t-evidence">
  <h2 id="t-evidence">Evidence</h2>
  <dl class="facts">
    <dt>Story</dt><dd>{{ c.description }}</dd>
    {% for k, v in c.facts.items() %}<dt>{{ k.replace("_", " ") }}</dt><dd>{{ v }}</dd>{% endfor %}
    <dt>Damage</dt><dd>{{ c.damage | join("; ") or "none found" }}</dd>
    <dt>Estimate</dt><dd>{% if c.cost_low is not none %}${{ c.cost_low }}–${{ c.cost_high }}{% else %}not priced{% endif %}</dd>
    <dt>Coverage</dt><dd>{{ c.coverage or "not checked" }}</dd>
    <dt>Fraud signals</dt><dd>{{ c.fraud_signals | join("; ") or "none" }}</dd>
  </dl>
</section>
</div>

{% if c.agent %}
<section aria-labelledby="t-agent" class="agent">
  <h2 id="t-agent">Triage agent</h2>
  <p><span class="chip r-{{ c.agent.route_suggestion }}">{{ c.agent.route_suggestion }}</span>
     confidence {{ c.agent.confidence }} · {{ c.agent.llm_calls }} model call(s) · ${{ "%.4f"|format(c.agent.llm_cost_usd) }}</p>
  <p>{{ c.agent.rationale }}</p>
  <dl class="facts">
    <dt>Clauses cited</dt><dd class="mono">{{ c.agent.policy_citations | join(", ") or "none" }}</dd>
    <dt>Evidence cited</dt><dd class="mono">{{ c.agent.citations | join(", ") or "none" }}</dd>
    <dt>Skills used</dt><dd class="mono">{{ c.agent.skills_used | join(", ") or "none" }}</dd>
    <dt>Tools used</dt><dd class="mono">{{ c.agent.tools_used | join(", ") or "none" }}</dd>
    <dt>Open questions</dt><dd>{{ c.agent.open_questions | join(" · ") or "none" }}</dd>
  </dl>
</section>
{% endif %}

<section aria-labelledby="t-similar">
  <h2 id="t-similar">Similar claims in memory</h2>
  {% if c.similar %}
  <ul class="similar">{% for s in c.similar %}
    <li><a class="mono" href="/claims/{{ s.claim_id }}">{{ s.claim_id[:8] }}</a> · {{ s.reason }}{% if s.distance is not none %} ({{ s.distance }} bits){% endif %} · {{ s.route }} · {{ s.damage }}</li>
  {% endfor %}</ul>
  {% else %}<p class="muted">None found (or memory is off).</p>{% endif %}
</section>

{% if c.route and c.status != "running" %}
<section aria-labelledby="t-review" class="review">
  <h2 id="t-review">Review</h2>
  <form id="review-form" data-claim="{{ c.claim_id }}">
    <label for="reviewer">Your name</label>
    <input id="reviewer" name="reviewer" required maxlength="80">
    <label for="action">Decision</label>
    <select id="action" name="action">
      <option value="approve">Approve</option>
      <option value="request_info">Ask for information</option>
      <option value="override">Change the route</option>
      <option value="deny">Deny</option>
    </select>
    <label for="final_route">New route (for a change)</label>
    <select id="final_route" name="final_route">
      <option value="">—</option>
      <option>FAST_TRACK</option><option>ADJUSTER_REVIEW</option><option>FRAUD_REVIEW</option>
    </select>
    <label for="note">Note</label>
    <textarea id="note" name="note" maxlength="2000" rows="3"></textarea>
    <button type="submit" class="primary">Record decision</button>
    <p id="review-error" class="error" role="alert" hidden></p>
  </form>
</section>
{% endif %}

<details class="audit">
  <summary>Audit trail ({{ c.event_count }} events)</summary>
  <ol class="events">{% for e in c.events %}
    <li class="mono">{{ e.seq }} · {{ e.type }} · {{ e.actor }} · {{ e.occurred_at }}</li>
  {% endfor %}</ol>
</details>
{% endblock %}
{% block scripts %}
<script id="claim-data" type="application/json">{{ {"id": c.claim_id, "status": c.status, "poll": poll} | tojson }}</script>
<script src="/static/claim.js"></script>
{% endblock %}
```

`templates/not_found.html`:

```html
{% extends "base.html" %}
{% block main %}
<h1>Claim not found</h1>
<p>There is no claim <span class="mono">{{ claim_id }}</span>. <a href="/claims">Back to all claims</a>.</p>
{% endblock %}
```

`templates/log_failed.html`:

```html
{% extends "base.html" %}
{% block main %}
<div class="alert danger" role="alert">
  <h1>Log failed verification</h1>
  <p>The audit log of claim <span class="mono">{{ claim_id }}</span> has been changed since it was written.
     Nobody can act on this claim until it is investigated.</p>
</div>
{% endblock %}
```

- [ ] **Step 5: Write the page scripts**

`static/chat.js`:

```javascript
// The customer chat. Messages are inserted with textContent only.
(() => {
  const list = document.getElementById("messages");
  const form = document.getElementById("chat-form");
  const text = document.getElementById("text");
  const photo = document.getElementById("photo");
  const photoLabel = document.getElementById("photo-label");
  const start = document.getElementById("start");
  const error = document.getElementById("chat-error");
  const done = document.getElementById("done");
  let session = new URLSearchParams(location.search).get("session");

  function say(who, words) {
    const item = document.createElement("li");
    item.className = who;
    item.textContent = words;
    list.appendChild(item);
    item.scrollIntoView({ block: "end" });
  }

  function show(turn) {
    session = turn.session_id;
    history.replaceState(null, "", `/?session=${encodeURIComponent(session)}`);
    say("agent", turn.message);
    error.hidden = true;
    start.hidden = true;
    if (turn.claim_id) {
      form.hidden = true;
      done.hidden = false;
      done.textContent = `Claim received. Your reference is ${turn.claim_id}.`;
      const link = document.createElement("a");
      link.href = `/claims/${turn.claim_id}`;
      link.textContent = " Open it as the adjuster";
      done.appendChild(link);
      return;
    }
    form.hidden = false;
    photoLabel.hidden = photo.hidden = !turn.photo_kind;
    photoLabel.textContent = turn.photo_kind ? `Add the ${turn.photo_kind.replace("_", " ")} photo` : "";
    (turn.photo_kind ? photo : text).focus();
  }

  async function call(url, options) {
    const response = await fetch(url, options);
    const body = await response.json();
    if (!response.ok) throw new Error(body.detail || "Something went wrong.");
    return body;
  }

  function fail(err) {
    error.textContent = err.message;
    error.hidden = false;
  }

  start.addEventListener("click", () => call("/api/intake", { method: "POST" }).then(show, fail));

  form.addEventListener("submit", (event) => {
    event.preventDefault();
    const data = new FormData();
    data.append("text", text.value);
    if (photo.files.length) data.append("photo", photo.files[0]);
    if (!text.value && !photo.files.length) return;
    say("customer", photo.files.length ? `[photo] ${photo.files[0].name}` : text.value);
    form.querySelector("button").disabled = true;
    call(`/api/intake/${encodeURIComponent(session)}/reply`, { method: "POST", body: data })
      .then(show, fail)
      .finally(() => {
        form.querySelector("button").disabled = false;
        text.value = "";
        photo.value = "";
      });
  });

  if (session) {
    call(`/api/intake/${encodeURIComponent(session)}`).then(show, () => { session = null; });
  }
})();
```

`static/claim.js`:

```javascript
// The adjuster's claim page: follow a running claim, resume a stopped one, record a review.
(() => {
  const data = JSON.parse(document.getElementById("claim-data").textContent);

  if (data.status === "running") {
    let last = null;
    const tick = async () => {
      const response = await fetch(`/api/claims/${data.id}`);
      if (!response.ok) return;
      const claim = await response.json();
      const now = JSON.stringify([claim.status, claim.stages]);
      if (last !== null && now !== last) location.reload();
      last = now;
      if (claim.status === "running") setTimeout(tick, data.poll * 1000);
      else location.reload();
    };
    setTimeout(tick, data.poll * 1000);
  }

  const resume = document.getElementById("resume");
  if (resume) {
    resume.addEventListener("click", async () => {
      resume.disabled = true;
      await fetch(`/api/claims/${data.id}/resume`, { method: "POST" });
      location.reload();
    });
  }

  const form = document.getElementById("review-form");
  if (form) {
    const error = document.getElementById("review-error");
    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      const values = Object.fromEntries(new FormData(form));
      if (!values.final_route) delete values.final_route;
      const response = await fetch(`/api/claims/${data.id}/review`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(values),
      });
      if (response.ok) { location.reload(); return; }
      const body = await response.json();
      error.textContent = body.detail || "The decision was not recorded.";
      error.hidden = false;
    });
  }
})();
```

- [ ] **Step 6: Write the stylesheet**

`static/app.css`:

```css
:root {
  --bg: #f4f6f8; --surface: #ffffff; --ink: #14202b; --muted: #5b6875; --rule: #d5dde4;
  --accent: #2c5b8a; --ok: #2a7a69; --warn: #a8701a; --risk: #b03838;
  --body: "Segoe UI", system-ui, -apple-system, Helvetica, Arial, sans-serif;
  --mono: ui-monospace, "Cascadia Mono", Consolas, monospace;
  color-scheme: light;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #0f151b; --surface: #161f28; --ink: #e2e8ed; --muted: #94a2ae; --rule: #2b3642;
    --accent: #78a9da; --ok: #5fbfa8; --warn: #e0a64a; --risk: #e57373; color-scheme: dark;
  }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--ink); font: 15px/1.55 var(--body); }
a { color: var(--accent); }
:focus-visible { outline: 3px solid var(--accent); outline-offset: 2px; }
.top { display: flex; flex-wrap: wrap; gap: 8px 24px; align-items: center; padding: 12px 20px;
  background: var(--surface); border-bottom: 1px solid var(--rule); }
.brand { font-weight: 700; text-decoration: none; color: var(--ink); letter-spacing: .02em; }
.top nav { display: flex; gap: 16px; }
.local { margin-left: auto; font: 12px var(--mono); color: var(--muted); }
.page { max-width: 1100px; margin: 0 auto; padding: 24px 16px 64px; display: grid; gap: 24px; }
h1 { font-size: 26px; margin: 0; text-wrap: balance; }
h2 { font-size: 17px; margin: 0 0 10px; }
.muted { color: var(--muted); }
.mono { font-family: var(--mono); font-size: 13px; word-break: break-all; }
.eyebrow { margin: 0; font: 12px var(--mono); text-transform: uppercase; letter-spacing: .1em; color: var(--muted); }
.sr { position: absolute; width: 1px; height: 1px; overflow: hidden; clip: rect(0 0 0 0); }
button { font: inherit; padding: 8px 14px; border: 1px solid var(--rule); background: var(--surface);
  color: var(--ink); border-radius: 6px; cursor: pointer; }
button.primary { background: var(--accent); border-color: var(--accent); color: var(--surface); }
button:disabled { opacity: .6; cursor: wait; }
input, select, textarea { font: inherit; padding: 8px 10px; border: 1px solid var(--rule);
  border-radius: 6px; background: var(--surface); color: var(--ink); width: 100%; }
.error { color: var(--risk); }
.chip { display: inline-block; font: 12px/1 var(--mono); padding: 4px 8px; border: 1px solid; border-radius: 4px; }
.chip.big { font-size: 14px; padding: 6px 10px; }
.r-FAST_TRACK, .s-reviewed { color: var(--ok); }
.r-ADJUSTER_REVIEW, .s-running { color: var(--warn); }
.r-FRAUD_REVIEW, .s-stopped, .s-log_failed { color: var(--risk); }
.s-decided { color: var(--accent); }
/* chat */
.chat { max-width: 720px; display: grid; gap: 14px; }
.messages { list-style: none; margin: 0; padding: 0; display: grid; gap: 10px; }
.messages li { max-width: 85%; padding: 10px 14px; border-radius: 12px; background: var(--surface);
  border: 1px solid var(--rule); white-space: pre-wrap; }
.messages li.customer { justify-self: end; background: color-mix(in srgb, var(--accent) 14%, var(--surface)); }
.composer { display: grid; grid-template-columns: 1fr auto; gap: 8px; }
.composer .photo-pick, .composer #photo { grid-column: 1 / -1; }
.done { padding: 12px 14px; border-left: 4px solid var(--ok); background: var(--surface); }
/* claims list */
.filters { display: flex; gap: 14px; }
.filters a[aria-current] { font-weight: 700; text-decoration: none; color: var(--ink); }
.table-wrap { overflow-x: auto; background: var(--surface); border: 1px solid var(--rule); border-radius: 8px; }
table { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }
th, td { text-align: left; padding: 10px 12px; border-bottom: 1px solid var(--rule); vertical-align: top; }
th { font-size: 12px; text-transform: uppercase; letter-spacing: .06em; color: var(--muted); }
/* claim */
.claim-head { display: flex; flex-wrap: wrap; justify-content: space-between; gap: 12px; align-items: end; }
.verdict { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
.verified { font: 12px var(--mono); color: var(--ok); }
.reason { margin: 0; padding: 10px 14px; background: var(--surface); border-left: 4px solid var(--accent); }
.alert { padding: 12px 16px; border-left: 4px solid var(--warn); background: var(--surface); }
.alert.danger { border-left-color: var(--risk); }
.stages { list-style: none; margin: 0; padding: 0; display: grid;
  grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap: 8px; }
.stage { background: var(--surface); border: 1px solid var(--rule); border-radius: 8px; padding: 10px;
  display: grid; gap: 2px; min-width: 0; }
.stage .dot { width: 10px; height: 10px; border-radius: 50%; background: var(--rule); }
.stage .state { font: 12px var(--mono); color: var(--muted); }
.stage .detail { font-size: 13px; color: var(--muted); overflow-wrap: anywhere; }
.st-done .dot { background: var(--ok); } .st-running .dot { background: var(--warn); }
.st-failed .dot { background: var(--risk); } .st-failed .state { color: var(--risk); }
.grid { display: grid; grid-template-columns: minmax(0, 1.2fr) minmax(0, 1fr); gap: 24px; }
@media (max-width: 800px) { .grid { grid-template-columns: minmax(0, 1fr); } }
.photos { display: grid; gap: 14px; }
.photo { margin: 0; }
.frame { position: relative; line-height: 0; }
.frame img { width: 100%; height: auto; border-radius: 6px; }
.box { position: absolute; border: 2px solid var(--risk); border-radius: 2px; }
.box .tag { position: absolute; left: -2px; top: -22px; white-space: nowrap; line-height: 1.4;
  font: 12px var(--mono); padding: 1px 6px; background: var(--risk); color: #fff; }
figcaption { font-size: 13px; color: var(--muted); padding-top: 6px; }
.facts { display: grid; grid-template-columns: max-content minmax(0, 1fr); gap: 6px 14px; margin: 0; }
.facts dt { color: var(--muted); text-transform: capitalize; }
.facts dd { margin: 0; overflow-wrap: anywhere; }
.agent, .review { background: var(--surface); border: 1px solid var(--rule); border-radius: 8px; padding: 16px; }
#review-form { display: grid; gap: 8px; max-width: 520px; }
.similar { margin: 0; padding-left: 18px; }
.audit summary { cursor: pointer; }
.events { font-size: 13px; }
@media (prefers-reduced-motion: reduce) { * { scroll-behavior: auto !important; } }
```

(The box label keeps `#fff` text on the risk red in both themes on purpose: it sits on a solid red fill, which reads in light and dark.)

- [ ] **Step 7: Run the tests**

Run: `uv run pytest tests/unit/web -v && uv run mypy && uv run ruff check . && uv run ruff format --check .`
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add src/claimlens/web tests/unit/web/test_pages.py
git commit -m "feat: chat, claims and claim pages"
```

---

### Task 8: `claimlens serve`, seed data and the real wiring

**Files:**
- Create: `src/claimlens/web/serve.py`
- Modify: `src/claimlens/cli.py` (sub-command and dispatch)
- Test: `tests/unit/web/test_serve.py`

**Interfaces:**
- Consumes: `make_deps`, `_make_detector`, `_make_agent`, `_open_memory`, `MEMORY_PATH`, `build_intake`, `file_intake_claim`, `load_intake_config`.
- Produces: `build_services(args, detector_factory) -> WebServices`; `seed_claims(services, photos: Sequence[Path]) -> list[UUID]`; `run_serve(args, detector_factory, run=uvicorn.run) -> int`. CLI: `claimlens serve [--host H] [--port N] [--seed] [--stub-agent] [--no-memory]`.

- [ ] **Step 1: Write the failing tests**

`tests/unit/web/test_serve.py`:

```python
from pathlib import Path
from typing import Any

import pytest

from claimlens.cli import main
from tests.fakes import CONFIG_DIR, FakeDetector
from tests.unit.web.helpers import image


def _base(tmp_path: Path) -> list[str]:
    return [
        "--db",
        str(tmp_path / "c.db"),
        "--blobs",
        str(tmp_path / "b"),
        "--config",
        str(CONFIG_DIR),
    ]


def test_serve_starts_uvicorn_locally(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    started: dict[str, Any] = {}
    monkeypatch.setattr("uvicorn.run", lambda app, **kw: started.update(kw, app=app))
    monkeypatch.chdir(tmp_path)
    code = main(
        [*_base(tmp_path), "serve", "--stub-agent", "--no-memory", "--port", "8123"],
        detector_factory=lambda *_: FakeDetector(),
    )
    assert code == 0
    assert started["host"] == "127.0.0.1"
    assert started["port"] == 8123


def test_serve_refuses_a_public_host(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = main(
        [*_base(tmp_path), "serve", "--host", "0.0.0.0"], detector_factory=lambda *_: FakeDetector()
    )
    assert code == 2
    assert "local only" in capsys.readouterr().err


def test_seed_files_three_claims_and_the_reused_photo_is_flagged(tmp_path: Path) -> None:
    import argparse

    from claimlens.events.projection import fold
    from claimlens.events.store import SQLiteEventStore
    from claimlens.web.serve import build_services, seed_claims

    args = argparse.Namespace(
        db=tmp_path / "c.db",
        blobs=tmp_path / "b",
        config=CONFIG_DIR,
        detector=None,
        weights=None,
        agent="stub",
        memory=False,
        embedder_factory=None,
        agent_factory=None,
        llm_daily_cap=None,
    )
    services = build_services(args, lambda *_: FakeDetector(), inline=True)
    photos = [image(tmp_path / "s1.png", (200, 30, 30)), image(tmp_path / "s2.png", (30, 30, 200))]
    claims = seed_claims(services, photos)
    assert len(claims) == 3
    store = SQLiteEventStore(tmp_path / "c.db")
    routes = [fold(store.load(c)).decision.route.value for c in claims]  # type: ignore[union-attr]
    store.close()
    assert routes[2] == "FRAUD_REVIEW"  # the third reuses the first photo
```

- [ ] **Step 2: Run them to see them fail**

Run: `uv run pytest tests/unit/web/test_serve.py -v`
Expected: FAIL (`invalid choice: 'serve'`).

- [ ] **Step 3: Implement `web/serve.py`**

```python
"""`claimlens serve`: the real wiring for the web prototype."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any
from uuid import UUID

from claimlens.blobs import BlobStore
from claimlens.events.store import SQLiteEventStore
from claimlens.intake import submit_claim
from claimlens.web.runner import ClaimRunner
from claimlens.web.services import WebServices, pipeline_processor
from claimlens.web.settings import load_web_settings
from claimlens.web.workers import Worker

LOCAL_HOSTS = {"127.0.0.1", "localhost"}
SEED_STORIES = (
    "Reversed into a bollard in a car park.",
    "Another car clipped my rear bumper at low speed.",
    "Same dent as before, I think it got worse.",
)


def build_services(
    args: argparse.Namespace, detector_factory: Callable[..., Any], *, inline: bool = False
) -> WebServices:
    from claimlens.cli import MEMORY_PATH, _make_agent, _make_detector, _open_memory, make_deps

    settings = load_web_settings(args.config / "web.toml")
    blobs = BlobStore(args.blobs)
    memory = _open_memory(args, MEMORY_PATH)
    detectors: list[Any] = []

    def deps_for(store: SQLiteEventStore) -> Any:
        if not detectors:  # built on the claims worker, on first use
            detectors.append(_make_detector(args, detector_factory))
        return make_deps(
            store, blobs, args.config, detectors[0], _make_agent(args, args.db), memory
        )

    services = WebServices(
        db_path=args.db,
        blobs=blobs,
        settings=settings,
        runner=ClaimRunner(Worker("claims", inline=inline), pipeline_processor(args.db, deps_for)),
        intake_worker=Worker("intake", inline=inline),
        intake=None,
        upload_dir=Path("var/web-uploads"),
        memory=memory,
    )
    services.intake = _intake_factory(args, services)
    return services


def _intake_factory(args: argparse.Namespace, services: WebServices) -> Callable[[], Any]:
    def build() -> Any:
        from claimlens.intake_agent.config import load_intake_config
        from claimlens.intake_agent.session import build_intake, file_intake_claim

        config = load_intake_config(args.config / "intake.toml")

        def submit(state: Any) -> str:  # files the claim; the runner processes it
            store = SQLiteEventStore(services.db_path)
            try:
                return str(file_intake_claim(store, services.blobs, state, config))
            finally:
                store.close()

        def unused() -> Any:
            raise RuntimeError("the web app files claims through its own submit")

        return build_intake(
            args.config, Path.cwd(), Path("var/intake.sqlite"), unused, submit=submit
        )

    return build


def seed_claims(services: WebServices, photos: Sequence[Path]) -> list[UUID]:
    """Three form-filed claims so the adjuster screens are not empty; the third reuses the
    first photo, which shows the reused-photo check."""
    chosen = [photos[0], photos[1 % len(photos)], photos[0]]
    claims = []
    with services.open_store() as store:
        for story, photo in zip(SEED_STORIES, chosen, strict=True):
            claims.append(
                submit_claim(
                    store,
                    services.blobs,
                    policy_id="P-1001",
                    description=story,
                    photo_paths=[photo],
                )
            )
    for claim in claims:
        services.runner.start(claim)
    return claims


def run_serve(
    args: argparse.Namespace,
    detector_factory: Callable[..., Any],
    run: Callable[..., None] | None = None,
) -> int:
    settings = load_web_settings(args.config / "web.toml")
    host = args.host or settings.host
    if host not in LOCAL_HOSTS:
        print(
            "error: the prototype is local only (127.0.0.1) until sign-in exists", file=sys.stderr
        )
        return 2
    args.agent = "stub" if args.stub_agent else "llm"
    args.memory = not args.no_memory
    try:
        import uvicorn

        from claimlens.web.app import create_app
    except ImportError:
        print("error: the web app needs: uv sync --group web --group agent", file=sys.stderr)
        return 2
    services = build_services(args, detector_factory)
    if args.seed:
        seed_claims(services, sorted(Path("tests/fixtures/images").glob("*.jpg")))
    (run or uvicorn.run)(create_app(services), host=host, port=args.port or settings.port)
    return 0
```

The `run` parameter defaults to `uvicorn.run`, looked up at call time, so the test's `monkeypatch.setattr("uvicorn.run", ...)` takes effect.

- [ ] **Step 4: Wire the CLI**

In `build_parser()` in `src/claimlens/cli.py`, after the `review-claim` parser:

```python
    serve = sub.add_parser("serve", help="run the local web prototype on http://127.0.0.1:8000")
    serve.add_argument("--host", default=None, help="local only: 127.0.0.1 (default) or localhost")
    serve.add_argument("--port", type=int, default=None)
    serve.add_argument("--seed", action="store_true", help="file three sample claims at start")
    serve.add_argument("--stub-agent", action="store_true", help="use the rule-based stub agent")
    serve.add_argument("--no-memory", action="store_true", help="do not use the claim memory")
```

In `main()`, next to the `intake` branch (after `args.embedder_factory = embedder_factory`):

```python
        if args.command == "serve":
            from claimlens.web.serve import run_serve

            return run_serve(args, detector_factory)
```

- [ ] **Step 5: Run the tests and the full suite**

Run: `uv run pytest tests/unit/web -v && uv run pytest && uv run mypy && uv run ruff check . && uv run ruff format --check .`
Expected: all pass, coverage at or above 85%.

- [ ] **Step 6: Smoke-run locally (free)**

Run: `uv run claimlens serve --stub-agent --no-memory --seed`, open `http://127.0.0.1:8000/claims`, check three claims appear and the third shows `FRAUD_REVIEW`; stop with Ctrl+C.

- [ ] **Step 7: Commit**

```bash
git add src/claimlens/web/serve.py src/claimlens/cli.py tests/unit/web/test_serve.py
git commit -m "feat: claimlens serve, with seed claims and local-only binding"
```

---

### Task 9: The browser test (Playwright, local only)

**Files:**
- Create: `tests/browser/__init__.py`, `tests/browser/test_web_flow.py`

**Interfaces:**
- Consumes: `create_app`, the `services` wiring from `tests/unit/web/test_intake_api.py::_chat` (fake LLM), uvicorn in a thread.

- [ ] **Step 1: Install the browser (local only)**

Run: `uv sync --group web --group agent --group browser && uv run playwright install chromium`

- [ ] **Step 2: Write the test**

`tests/browser/test_web_flow.py`:

```python
"""File a claim through the chat with photos, see it decided, approve it. Local only."""

import threading
import time
from pathlib import Path

import pytest

pytest.importorskip("playwright.sync_api")
import uvicorn  # noqa: E402
from playwright.sync_api import Error as PlaywrightError  # noqa: E402
from playwright.sync_api import sync_playwright  # noqa: E402

from claimlens.web.app import create_app  # noqa: E402
from tests.unit.test_intake_session import SCRIPT  # noqa: E402
from tests.unit.web.conftest import services as _services  # noqa: E402,F401
from tests.unit.web.test_intake_api import _chat, _png  # noqa: E402

pytestmark = pytest.mark.browser


def test_file_follow_and_review_a_claim(tmp_path: Path, services) -> None:  # type: ignore[no-untyped-def]
    _chat(services, tmp_path, list(SCRIPT))  # sets services.intake with the fake LLM
    server = uvicorn.Server(
        uvicorn.Config(create_app(services), host="127.0.0.1", port=8765, log_level="warning")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    while not server.started:
        time.sleep(0.05)
    photos = []
    for seed in (1, 2, 3):
        path = tmp_path / f"p{seed}.png"
        path.write_bytes(_png(seed))
        photos.append(path)
    try:
        with sync_playwright() as p:
            try:
                browser = p.chromium.launch()
            except PlaywrightError:
                pytest.skip("no browser installed: uv run playwright install chromium")
            page = browser.new_page()
            page.goto("http://127.0.0.1:8765/")
            page.click("#start")
            page.fill("#text", "P-1001, I reversed into a bollard yesterday")
            page.click("#send")
            for path in photos:
                page.wait_for_selector("#photo:not([hidden])")
                page.set_input_files("#photo", str(path))
                page.click("#send")
            page.wait_for_selector("#done:not([hidden])")
            page.click("#done a")
            page.wait_for_selector(".chip.big")
            page.fill("#reviewer", "Sam")
            page.select_option("#action", "approve")
            page.click("#review-form button[type=submit]")
            page.wait_for_selector(".s-reviewed")
            browser.close()
    finally:
        server.should_exit = True
        thread.join(timeout=5)
```

The intake test's `services` fixture uses inline workers, which is right here: the claim is decided before the chat reply returns.

- [ ] **Step 3: Run it**

Run: `uv run pytest tests/browser -m browser -v --no-cov`
Expected: PASS locally. In CI (no `browser` group) the module is skipped by `importorskip`.

- [ ] **Step 4: Commit**

```bash
git add tests/browser pyproject.toml uv.lock
git commit -m "test: browser test of the whole web flow (local only)"
```

---

### Task 10: Docs, the live check and the screenshot

**Files:**
- Create: `docs/adr/0018-web-prototype.md`, `docs/images/claim-page.png` (screenshot from a publishable photo only)
- Modify: `README.md` (section "Run the web prototype", status line), `docs/roadmap.md` (M8a items)

- [ ] **Step 1: Write ADR 0018**

`docs/adr/0018-web-prototype.md` records: context (a prototype to see and show the flow, before the trimmed M7); decision (FastAPI + Jinja2 + plain JS over Gradio, owner's choice 2026-10-03; no business logic in the web layer; single worker thread per area because SQLite connections are thread-bound; per-request stores opened inside endpoint bodies; local-only binding until sign-in; uploads checked by content; publishable photos only); consequences (the API at `/docs` is the base for M8b; one-second polling; no accounts; Docker in M8b).

- [ ] **Step 2: Live check (about $0.10, owner-approved budget)**

Run: `uv run claimlens serve`, file one claim through the chat with three photos from a source that may be published (Roboflow `car-seg`, CC BY 4.0, or the owner's photo), watch it decided, approve it. Then `uv run claimlens verify <claim-id>` → `Chain OK`.

- [ ] **Step 3: Screenshot**

Save the claim page as `docs/images/claim-page.png`. Check by eye that it contains no CarDD or unknown-source photo. Add the credit line for a `car-seg` image under the screenshot in the README.

- [ ] **Step 4: README and roadmap**

README: a "Run the web prototype" section (`uv sync --group web --group agent --group knowledge`, `uv run claimlens serve [--seed] [--stub-agent] [--no-memory]`, the four screens, the screenshot, local-only note); status line "M0–M6 and M8a complete; next: the trimmed M7, then M8b". Roadmap: tick M8a items, add the trimmed M7 list and the plan change.

- [ ] **Step 5: Full checks and commit**

Run: `uv run ruff check . && uv run ruff format --check . && uv run mypy && uv run pytest && uv run claimlens eval-gate`
Expected: all pass.

```bash
git add docs/adr/0018-web-prototype.md docs/images/claim-page.png README.md docs/roadmap.md
git commit -m "docs: ADR 0018, README section and screenshot for the web prototype"
```
