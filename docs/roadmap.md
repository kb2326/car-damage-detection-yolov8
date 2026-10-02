# ClaimLens Roadmap & To-Do

Plain-language guide to the whole project. Tick items as they are done.
Design detail lives in [`specs/2026-10-01-claimlens-design.md`](specs/2026-10-01-claimlens-design.md).

## The project in one story

Sam scrapes a car against a pole and opens the ClaimLens app.

1. **A chat assistant (intake agent)** asks what happened and guides Sam to take the right photos.
2. **Photo checks** reject blurry or dark photos and blur faces and licence plates.
3. **Our own trained vision models** find *which part* is damaged (rear bumper) and *what kind* of
   damage it is (dent, 8% of the bumper).
4. **Fraud checks** look for reused photos, edited or AI-generated photos, or a story that doesn't
   match the damage.
5. **Pricing** turns "dent on rear bumper, moderate" into a cost range, such as $600–900.
6. **A reasoning assistant (triage agent)** reads all the evidence, checks Sam's policy, and writes
   a recommendation with sources.
7. **Fixed business rules** make the final call: *fast-track*, *send to an adjuster*, or *send to
   the fraud team*. The AI can never deny a claim. Only a person can.
8. **Every step is written to a tamper-proof log**, so anyone can later see exactly why a decision
   was made.
9. **Adjuster corrections feed back** into testing and retraining, so the system improves.

## How each phase is run

At the start of every phase, Claude explains:

1. **What** we are building, in plain words
2. **Why** we need it, and what breaks without it
3. **Where** it sits in the story above
4. **To-do checklist** for the phase
5. **Done looks like**: the demo you will be able to show
6. **New terms**: a short glossary

At the end of every phase: a recap, what you learned, three quick questions to check your
understanding, and a short retrospective in `docs/retros/`.

## LinkedIn: one post at the end

LinkedIn only, as a single post after M8. Material is collected here as each milestone finishes:

| Milestone | Story material |
|---|---|
| M0 | Audit of the course project: every label the app showed was wrong (7-class model, 17-class list); val mAP50 0.137 |
| M1 | Working end-to-end skeleton on day one; baseline route accuracy 0.78, escalation recall 1.00, 0 of 11 correct fast-tracks ("safe but not yet useful") |
| M2 | Roboflow v6 turned out to be CarDD re-uploaded; dedupe found the same dent photo in train and valid under two ids (4 CarDD and 29 parts images moved); foundation-model part labels: 133 of 455 proposals passed review, wheels and lights yes, doors and bumpers no; the CPU run overheated the laptop, so it moved to a free Kaggle GPU (1 minute for 100 images) |

---

## M0 · Foundations (Oct 1–7)

*Set up the workshop before building: a clean, standard project layout, automatic checks, and a
written plan.*

- [x] Audit the original course project
- [x] Restructure the repo (`src/`, `tests/`, `docs/`, `data/`, `models/`, `legacy/`)
- [x] Tooling: uv, ruff, mypy, pytest, pre-commit, GitHub Actions CI
- [x] ADR 0001 (decision records) and ADR 0002 (repo structure)
- [x] Design spec and visual blueprint
- [ ] **You:** review the spec and blueprint
- [ ] **You:** back up `data/raw/` and `models/legacy/` (they exist only on this laptop)
- [x] Commit, open a pull request, CI passes, merge to `main` (PR #1)
- [x] PR/FAQ: the launch press release, written first (`docs/prfaq.md`)
- [ ] Implementation plan for M1
- [x] Retro

## M1 · Walking skeleton (Oct 8–18)

*Build the thinnest possible version of the whole story, end to end, so every later phase improves
something that already works.*

- [x] Data types for claims, photos, findings and decisions (Pydantic)
- [x] Tamper-proof event log with hash chain
- [x] Rebuild claim state from events (fold) and resume a claim
- [x] Pipeline stages wired up, using the old model with its labels fixed
- [x] Placeholder triage agent (no LLM yet)
- [x] Decision rules (§11 of the spec)
- [x] 50 golden test claims, plus a script that scores the pipeline on them
- [ ] **You:** review 10 golden claims (g001, g004, ... g028)
- [x] Command-line demo: one claim in, one decision and an audit trail out
- [x] Baseline: route accuracy 0.78, escalation recall 1.00, 0 of 11 correct fast-tracks
- [x] Retro (`docs/retros/m1-walking-skeleton.md`)

## M2 · Data engine (Oct 19 – Nov 1)

*Better data matters more than a bigger model.*

**M2a: pipeline (done)**

- [x] Found that Roboflow v6 is CarDD; terms recorded in ADR 0004
- [ ] **You:** submit the CarDD licence form (https://cardd-ustc.github.io)
- [x] DVC data versioning (ADR 0005), local remote
- [x] Taxonomy by name, data contract, converters for COCO and YOLO
- [x] Exact and perceptual dedupe; whole clusters per split; golden photos protected
- [x] Car-part dataset chosen (Roboflow `car-seg`, CC BY 4.0)
- [x] `damage-v1` (4,000 images) and `parts-v1` (3,833 images) built; data card

**M2b: review and labelling (done)**

- [x] FiftyOne review UI and `claimlens review launch | export | apply`
- [x] Grounding DINO + SAM 2 part masks on 100 test images, run on a Kaggle GPU (ADR 0006)
- [x] `fusion-eval-v1`: 133 of 455 part masks approved, 77 images (AI-reviewed, not human-verified)
- [ ] **You:** spot-check `fusion-eval-v1` in FiftyOne (`uv run claimlens review launch parts`
  opens with my decisions pre-tagged; swap the tag on any you disagree with, then `review export parts`)
- [x] Golden claims v1: 97 cases, duplicates removed, 50 CarDD test cases added (30 fast-tracks)
- [ ] **You:** review the golden claims (`uv run claimlens review launch golden`)
- [x] Baseline v1: route accuracy 0.73, escalation recall 1.00, 4 of 30 correct fast-tracks
- [x] Retro (`docs/retros/m2-data-engine.md`)
- [ ] Optional: move the DVC remote to DagsHub

## M3 · Vision models (Nov 2–15)

*Train the "eyes" of the system and prove how good they are.*

- [ ] Damage segmentation model (YOLO11-seg) trained on Colab
- [ ] Part segmentation model
- [ ] Fusion: which damage is on which part, and how big it is
- [ ] Calibrate confidence scores; choose threshold τ
- [ ] MLflow experiment tracking and model registry; ONNX export
- [ ] Model card (target: mask mAP50 ≥ 0.50)
- [ ] Stretch: RT-DETR comparison

## M4 · Tools & integration (Nov 16–26)

*Give the agent safe, standard ways to use our models and (mock) company systems.*

- [ ] MCP servers: vision, policy-admin, claims-system, payments
- [ ] Per-agent permissions (scopes), with tests
- [ ] LLM gateway: model routing, retries, fallback, caching, cost caps
- [ ] Fictional policy documents plus hybrid search with citations

## M5 · Triage agent (Nov 27 – Dec 10)

*Add the reasoning assistant, then measure it like a product.*

- [ ] Agent loop with step, cost and time limits
- [ ] Bake-off: our own loop vs LangGraph behind the same interface, on the golden set (ADR 0007)
- [ ] Human review queue and approval tokens
- [ ] 150 golden claims
- [ ] LLM judge, checked against 50 of your own labels
- [ ] CI blocks changes that lower eval scores
- [ ] Tracing in Phoenix (OpenTelemetry)

## M6 · Intake agent & memory (Dec 11–24)

*Add the conversation with the customer, and give the system a governed memory.*

- [ ] Multi-turn intake agent with guided photo capture
- [ ] Episodic memory (claim history, image similarity) and semantic memory (policies, similar claims)
- [ ] Three Agent Skills (`SKILL.md`), each approved by you
- [ ] Stretch: simulated claimant tests with pass^k

## M7 · Trust & governance (Dec 25 – Jan 4)

*Try to break the system before others do.*

- [ ] Fraud checks: photo reuse, metadata, AI-generated image score, story vs photo
- [ ] Blur faces and licence plates
- [ ] Red-team test suite mapped to the OWASP agentic top 10
- [ ] AIS Program document and system card

## M8 · Ship (Jan 5–15)

*Put it in front of people.*

- [ ] Docker image and FastAPI service
- [ ] Public Gradio demo on Hugging Face Spaces
- [ ] Drift monitoring (Evidently)
- [ ] Final write-up, demo video, the single LinkedIn post
- [ ] Stretch: A2A repair-shop partner agent, adjuster copilot
