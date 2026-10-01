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
understanding, a short retrospective in `docs/retros/`, and the next part of the LinkedIn series.

## LinkedIn series: "Building an AI claims adjuster in public"

LinkedIn only, posted as one numbered series. One part per milestone, each posted after that
milestone's demo works. No standalone posts in between.

Every part uses the same format: a one-line hook, what was built, one visual, one lesson learned,
a link to the repo, and a teaser for the next part.

| Part | After | Working title |
|---|---|---|
| 1/9 | M0 | I audited my own deep learning class project. Every label it showed was wrong. |
| 2/9 | M1 | An ugly end-to-end system on day 10, and why that beats a perfect model |
| 3/9 | M2 | Labelling thousands of images with foundation models |
| 4/9 | M3 | From 0.137 mAP to a model I trust: what actually helped |
| 5/9 | M4 | Turning my own models into MCP tools for an agent |
| 6/9 | M5 | Why most of my AI claims adjuster is deliberately not an agent |
| 7/9 | M6 | Agent memory an auditor can trust |
| 8/9 | M7 | I tried to commit insurance fraud against my own AI |
| 9/9 | M8 | Live demo, numbers, and everything I learned |

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
- [ ] Retro, and LinkedIn series part 1/9

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
- [ ] Retro, and LinkedIn series part 2/9

## M2 · Data engine (Oct 19 – Nov 1)

*Better data matters more than a bigger model.*

- [ ] Confirm CarDD license terms
- [ ] DVC data versioning
- [ ] Remove duplicates; split by vehicle so test photos are never seen in training
- [ ] Auto-labelling with Grounding DINO + SAM 2, then human correction in FiftyOne
- [ ] Choose a car-part dataset
- [ ] Data card

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
- [ ] Final write-up, demo video, LinkedIn series part 9/9
- [ ] Stretch: A2A repair-shop partner agent, adjuster copilot
