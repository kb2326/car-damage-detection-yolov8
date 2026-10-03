# M8a: The Local Web Prototype (design)

- **Status:** Draft for owner review
- **Date:** 2026-10-03
- **Parent design:** [`2026-10-01-claimlens-design.md`](2026-10-01-claimlens-design.md) (section 18, M8)
- **Builds on:** ADR 0003 (event log), ADR 0013 (triage agent), ADR 0015 (human review),
  ADR 0016 (intake agent), ADR 0017 (memory and skills)

## 1. Purpose

Show the whole ClaimLens flow working in a browser, end to end, so it is easy to understand, demo
and record. One person plays both roles on their own laptop:

| Role | What they do |
|---|---|
| **Customer** | Files a claim by chatting with the intake agent and uploading photos |
| **Adjuster** | Watches the claim go through each stage, reads the evidence and the agent's reasoning, and approves, denies, asks for information or overrides the route |

The web layer adds **no business logic**. It calls what already exists: `IntakeSessions`,
`submit_claim`, `process_claim`, the review queue (`pending_reviews`, `record_review`), `fold` and
the hash-chain check. The principles hold: models measure, the agent reasons, rules decide; only a
person can deny.

## 2. Decisions taken with the owner (2026-10-03)

- **Plan change:** M8a (this prototype, local only) comes before a trimmed M7 (red-team suite,
  near-copy fraud rule, system card). **Nothing goes public until that M7 is done.**
- **Own web pages, not Gradio:** FastAPI with server-rendered Jinja2 templates and a little plain
  JavaScript. No front-end framework and no build step. The same FastAPI app becomes the public API
  in M8b.
- **Standard practice throughout** (owner's request): the conventional FastAPI layout, typed
  request and response models, OpenAPI docs at `/docs`, dependency injection, `TestClient` tests,
  and standard web safety (section 6).
- **Real LLM with the existing caps:** about $0.05 per intake chat (Haiku) and $0.01 per triage
  (Sonnet). Spend for M8a: under $2 (development smoke checks plus one live check before merge).

## 3. Architecture

```
browser ──HTTP──▶ FastAPI app (127.0.0.1:8000)
                    ├── routers/pages.py    HTML pages (Jinja2)
                    ├── routers/intake.py   /api/intake/...   → IntakeSessions
                    ├── routers/claims.py   /api/claims/...   → fold, events, chain check, photos
                    └── routers/review.py   /api/claims/{id}/review → record_review
                  ClaimRunner (background thread) → process_claim
```

New package `src/claimlens/web/`:

| Module | Purpose |
|---|---|
| `app.py` | `create_app(settings, services)`: builds the app; tests pass fake services |
| `settings.py` | `WebSettings` from `config/web.toml` (host, port, upload limit, poll interval) |
| `services.py` | `WebServices`: how to open an event store, the blob store, `IntakeSessions`, pipeline deps, memory; built once at start-up |
| `runner.py` | `ClaimRunner`: runs `process_claim` in a background worker thread and remembers which claims are running |
| `schemas.py` | Pydantic response models: `ChatTurn`, `ClaimSummary`, `ClaimDetail`, `StageStatus`, `EventItem`, `ReviewRequest`, `ApiError` |
| `views.py` | Pure functions that turn a `ClaimState` and its events into the view models (stage list, boxes to draw, agent section) |
| `routers/*.py` | One router per area; thin: validate, call services, return schemas |
| `templates/`, `static/` | Jinja2 templates, one stylesheet, one small script per page |

- **One event store per request and per worker.** The SQLite store is opened through a FastAPI
  dependency that yields a new `SQLiteEventStore` and closes it afterwards; the worker opens its
  own. No connection is shared across threads.
- **`claimlens serve`** starts uvicorn on `127.0.0.1` (the port from `--port`, default 8000). It
  uses the LLM triage agent and the claim memory by default; `--stub-agent` and `--no-memory` turn
  them off (for free local runs).
- **New dependencies:** `jinja2`, `uvicorn`, `python-multipart` (uploads); `playwright` as a dev
  dependency for the one browser test.

## 4. Screens

### 4.1 File a claim (customer), `/`

- A chat with the intake agent. "Start a claim" calls `POST /api/intake`; each message calls
  `POST /api/intake/{session}/reply` (text, and an optional photo as a multipart upload).
- When the agent asks for a photo, the chat shows an upload control for that photo kind; coaching
  messages ("too dark, please retake") appear as agent messages.
- At hand-over the chat shows "Claim received. Your reference is …" and a link the *adjuster* uses.
  The customer sees nothing about cover, price or outcome (the same rule as the terminal chat).
- The hand-over starts the pipeline through `ClaimRunner`.
- A reloaded page continues the session (the session id is kept in the page's URL, `/?session=…`).

### 4.2 Claims list (adjuster), `/claims`

A table of all claims, newest first: reference, policy, filed at, status (running, decided,
reviewed, stopped), route, rule, and the reviewer's action. Filters: all, needs review, fraud
review.

### 4.3 Claim detail (adjuster), `/claims/{id}`

The centrepiece, built from the claim's events only:

1. **Stage timeline:** intake, quality, damage and parts, integrity, pricing, coverage, agent,
   decision, memory, review. Each stage is done, running, failed (with the reason) or not reached.
   While the claim runs, the page polls `GET /api/claims/{id}` every second and updates.
2. **Photos with labelled boxes:** each accepted photo, with a box for each finding and a label such
   as `dent · back_bumper · 0.77`. The boxes are the ones recorded in `DamageDetected`; models are
   never re-run. Rejected photos show their rejection reason.
3. **Facts and evidence:** intake facts, damage list, cost range, coverage, integrity signals.
4. **The agent:** route recommended, reasons, cited clause ids, skills used (`name@vN`), similar past
   claims and why they matched, and open questions.
5. **Decision:** the route and the rule that fired, with the rule's plain description.
6. **Audit:** a "Log verified · N events" badge, and the full event list (type, actor, time) on
   demand.
7. **Review** (section 4.4).

### 4.4 Review

On a decided claim, a form with reviewer name, action (approve, override, deny, request
information), a final route for an override, and a note. `POST /api/claims/{id}/review` calls
`record_review` (with memory, as the CLI does). Only this form, filled in by a person, can deny.

## 5. Errors

| Situation | What the user sees | Behind it |
|---|---|---|
| LLM down or a cap hit during intake | "Your answers are saved; please try again shortly" | Existing stalled-session handling; the session resumes |
| LLM fails during triage | The claim is decided by rule R2 (to a person) | Unchanged pipeline |
| Pipeline stops midway (crash, restart) | "Stopped at stage X" and a **Resume** button | `POST /api/claims/{id}/resume` → `ClaimRunner` → `process_claim` |
| Hash chain fails verification | Red banner "Log failed verification"; review form hidden | `ChainIntegrityError` → `409` on review |
| Upload of the wrong type or too large | Plain message in the chat, asks again | `422` |
| Unknown claim or session | "Not found" page | `404` |
| Review of an undecided claim, or a second resume while running | Plain message | `409` |

API errors are JSON `{"detail": "<plain message>"}` with the standard status code. No stack trace,
path or secret ever reaches the browser.

## 6. Standard web safety (local, but done properly)

- Binds to `127.0.0.1` only. A different host is refused at start-up until M7 and M8b add sign-in.
- Jinja2 autoescaping is on, so customer text cannot inject HTML; page scripts read data from JSON
  responses and insert it as text, never as HTML.
- Uploads: JPEG, PNG or WebP only, checked by content (Pillow) as well as by type; at most 10 MB
  (`config/web.toml`); stored through the existing content-addressed blob store. No path ever comes
  from the request.
- Photos are served only by blob id through `/api/claims/{id}/photos/{photo_id}`, and only for a
  photo that belongs to that claim.
- Forms post JSON or multipart through `fetch`; state-changing requests are `POST` only.
- Accessibility basics: labels on every control, keyboard navigation, visible focus, readable
  contrast in light and dark mode, alt text on photos.

## 7. Seed data

`claimlens serve --seed` files three sample claims from local photos (through the form path, with no
intake chat) so the adjuster screens are not empty on first open. Two use different photos; the
third reuses one, which shows the reused-photo check sending a claim to fraud review. They are
processed with the configured agent, so with `--agent llm` the seed costs about $0.03.

**Photos that may be published.** The two test photos in `tests/fixtures/images` came with the
original course project and their source is not recorded, so they are treated like CarDD
(non-commercial, never published). Anything published (the README screenshot, later the public
demo) uses only a photo with a clear licence: a Roboflow `car-seg` image (CC BY 4.0, credited) or
the owner's own photo. **No CarDD or unknown-source photo ever appears in a published file.**

## 8. Testing

- **API tests** with FastAPI's `TestClient`, a fake detector and `FakeProvider` (no paid calls, no
  weights): start a chat, reply, upload a photo (good, wrong type, too large), hand over, run the
  pipeline (the runner runs inline in tests), read the claim detail, review it, and every error in
  section 5.
- **View tests:** `views.py` turns known event logs into the expected stage list, boxes and agent
  section (pure functions, no web).
- **Page tests:** each page renders; customer text is escaped; the review form is hidden when the
  chain fails; the customer page never shows a route or price.
- **Browser test** (Playwright, marked `browser`, skipped when no browser is installed, so CI stays
  as it is): file a claim through the chat with photos, see it decided, approve it.
- **Live check before merge** (about $0.10): one real chat claim through to review. The README
  screenshot uses a photo that may be published (section 7), never a CarDD or unknown-source one.
- Coverage stays at or above 85%; ruff, mypy (strict) and the eval gate as usual.

## 9. Done means

- `claimlens serve` opens a working app on `http://127.0.0.1:8000` with the four screens.
- A claim filed by chat is decided and reviewed entirely in the browser, and `claimlens verify`
  passes on it.
- `/docs` lists the API with typed models.
- All tests pass in CI; the browser test passes locally.
- README section "Run the web prototype", with the screenshot; ADR 0018 records the web layer.

## 10. Out of scope

- Sign-in, user accounts and roles (needed before going public; M7/M8b).
- Docker, deployment, monitoring (M8b).
- Blurring faces and plates (only if the public demo shows photos).
- The trimmed M7 items.
- Real-time push (WebSockets); one-second polling is enough locally.
