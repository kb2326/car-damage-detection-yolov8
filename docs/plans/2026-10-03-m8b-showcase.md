# M8b: The Public Showcase Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Publish a read-only showcase of ClaimLens on Hugging Face Spaces, then finish the project.

**Architecture:** A `--showcase` mode of the existing FastAPI app: one middleware refuses every write, the event store opens read-only, the pages hide input and show a recorded chat. Sample claims are built once with the real models and agents from openly licensed photos and committed under `showcase/`. A slim Docker image (no PyTorch, no weights, no key) runs it; a GitHub Action deploys it to the Space.

**Tech Stack:** FastAPI, SQLite (`mode=ro`), Docker, Hugging Face Spaces (Docker SDK), GitHub Actions.

**Spec:** `docs/specs/2026-10-03-m8b-showcase-design.md`

## Global Constraints

- No model weights, CarDD or unknown-source photos, or API key in the image or the Space.
- Published photos: Wikimedia Commons CC BY / CC BY-SA only, credited with author, licence and link; no readable number plate.
- Showcase: every non-GET/HEAD/OPTIONS request is 403; the store is read-only.
- `0.0.0.0` is allowed only with `--showcase`; the Host check uses `CLAIMLENS_ALLOWED_HOSTS`.
- Spend: about $0.15 for the sample build (owner-approved plan).

## Review Focus

1. A write endpoint added later must still be refused in showcase mode (middleware, not per-route; test over every POST route).
2. The read-only store must refuse an append (test).
3. A public host without `--showcase` must still be refused (test).
4. Every committed sample photo is credited and verifies (test).
5. Pages in showcase mode show no review form, no resume button and no chat input (test).

### Task 1: Read-only event store

- [ ] Test: `SQLiteEventStore(path, read_only=True)` loads events and verifies the chain; `append` raises `sqlite3.OperationalError`; a missing file raises `FileNotFoundError`.
- [ ] Implement: `read_only` keyword; connect with `file:{path}?mode=ro` (`uri=True`) and skip the schema statement.

### Task 2: Showcase mode in the web app

- [ ] Tests (`tests/unit/web/test_showcase.py`): every POST route → 403 with the showcase message; GET pages 200; banner and links on every page; the chat page shows the recorded transcript and has no form; the claim page has no review form or Resume button; security headers on every response (`X-Content-Type-Options`, `Content-Security-Policy`, `Referrer-Policy`); `CLAIMLENS_ALLOWED_HOSTS` sets the allowed hosts; `serve --host 0.0.0.0` refused without `--showcase`, accepted with it.
- [ ] Implement: `WebSettings.showcase`, `WebServices.read_only` and `transcript`; middleware; template flags; `serve --showcase --data DIR`; environment overrides.

### Task 3: Sample claims

- [ ] `scripts/build_showcase.py`: downloads the five Commons photos (URL, author, licence recorded), builds the five claims with the real fused detector and LLM agent (memory on, so claim 4's re-save is a near-copy), records the chat for claim 5 with the scripted customer, records the reviews, writes `showcase/` and `CREDITS.md`.
- [ ] Test (`tests/unit/test_showcase_data.py`): every claim in `showcase/claims.db` verifies; every blob is listed in `CREDITS.md` with a licence and a source URL; the transcript has customer and agent turns only.
- [ ] Run it (about $0.15); screenshot the showcase locally and check it.

### Task 4: Container and deployment

- [ ] `Dockerfile`, `.dockerignore`, `hf-space/README.md`, `.github/workflows/deploy-space.yml` (manual dispatch and releases; needs `HF_TOKEN`).
- [ ] Build and run the image locally; `/claims` returns 200 and lists five claims.

### Task 5: Finish

- [ ] ADR 0020, README (showcase link), system card (showcase), roadmap (M8 done), retro for M7 and M8, a LinkedIn post draft for the owner.
- [ ] Owner: Hugging Face account and `HF_TOKEN`; first deploy; check the live Space.
