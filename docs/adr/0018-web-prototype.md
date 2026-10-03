# ADR 0018: A local web prototype on FastAPI, before the public demo

- **Status:** Accepted
- **Date:** 2026-10-03

## Context

By M6 the whole claim flow worked: chat intake, vision models, the triage agent, rules, memory
and human review. It was only reachable from the terminal. The owner wanted to see and show the
flow working before more trust work, so the order changed:

1. M8a: a local web prototype (this ADR);
2. a trimmed M7 (a red-team suite, near-copy photos as a fraud rule, a system card) **before
   anything goes public**;
3. M8b: the public demo, Docker and monitoring.

## Decision

- **FastAPI with server-rendered pages**, not Gradio (owner's choice, 2026-10-03):
  - Jinja2 templates, one stylesheet and two small scripts; no front-end framework, no build step;
  - a typed JSON API (`/api/...`), documented at `/docs`, which becomes the public API in M8b;
  - Gradio would have been faster to start but could not lay out the claim page the way an
    adjuster's screen needs, and M8 needed FastAPI anyway.
- **No business logic in the web layer.** It calls `IntakeSessions`, `submit_claim`,
  `process_claim`, `record_review`, `fold` and the store's chain check. Routes still come only from
  the rules, and only the review form (a person) can deny.
- **Four screens:** the customer chat (`/`), the claims list (`/claims`), the claim page
  (`/claims/{id}`: stage timeline, photos with the recorded damage boxes, evidence, the agent's
  reasoning, similar claims in memory, the route and rule, "log verified", the audit trail) and the
  review form on the claim page.
- **Threads:** the event store and the LLM gateway (budget and cache) hold SQLite connections that
  only work on the thread that opened them. So:
  - the intake chat runs on one dedicated worker thread, and the pipeline on another
    (`ClaimRunner`, one claim at a time, a double click cannot start a claim twice);
  - each request opens and closes its own event store inside the endpoint body, never in a FastAPI
    dependency, which may run on a different thread.
- **The chat files the claim; the runner processes it.** `file_intake_claim` was split out of the
  intake hand-over, so the customer gets their reference at once and the claim page shows the
  stages as they finish (polled every second).
- **Safety, even locally:**
  - binds to `127.0.0.1`; any other host is refused until sign-in exists;
  - answers only to the host names `127.0.0.1` and `localhost` (a DNS-rebinding page is refused),
    and refuses a state-changing request from another site's page (Origin check);
  - templates autoescape and the scripts insert text only;
  - uploads must be a real JPEG, PNG or WebP (checked by content), at most 10 MB, and are stored
    under their content hash; a photo is served only for the claim it belongs to;
  - a claim whose log fails verification shows a red banner and cannot be reviewed;
  - errors are plain JSON messages with standard status codes.
- **`claimlens serve`** uses the LLM agent and memory by default (`--stub-agent` and `--no-memory`
  for free runs); `--seed` files three sample claims in the background, the third reusing the
  first photo, which rule R1 sends to fraud review.
- **Published images use only photos with a clear licence** (Roboflow `car-seg`, CC BY 4.0,
  credited). CarDD photos and the two unknown-source test photos are never published.

## Consequences

- 64 tests for the web layer (10 of them from the final review's findings), and one Playwright browser test that files a claim by
  chat, follows it and approves it (local only; CI skips it).
- **Live check (2026-10-03), Haiku chat and Sonnet agent:**
  - The first attempt hit the $1 daily LLM cap, already used by the day's M6 evaluations. The chat
    handed the claim over after one photo and the agent's budget refusal sent it to a person (R2),
    with the reason on the claim page: the safeguards worked as designed.
  - With the owner's approval, `serve --llm-daily-cap` raised the cap for the run. A full chat then
    collected every fact and three CC BY 4.0 `car-seg` photos in 12 turns; the claim was decided,
    written to memory and reviewed through the review API, and `claimlens verify` reported `Chain OK`
    (23 events). The agent cost $0.018.
  - Rule R1 sent it to fraud review because the aborted first attempt had used the same photo, and
    the agent's reasons named the reuse and the story's mismatch with the damage.
  - On these undamaged cars the damage model still reported low-confidence findings (0.08 to
    0.39), which inflate the cost estimate; rule R6 (0.65) keeps such claims away from a fast-track.
- **The final review** found an LLM outage could leave a filed claim unprocessed, that a reviewer's
  route change vanished from the review lists, and no Host/Origin checks; all were fixed test-first,
  with phone JPEGs (MPO), reloads after the hand-over, error text and reviews during processing.
- **Limits:** no sign-in or roles; one-second polling instead of push; one worker per area, so two
  chats wait for each other; no Docker yet (M8b).
