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
| M3a | Trained two YOLO11-seg damage models on a free Kaggle GPU (test mask mAP50 0.733, up from the course model's 0.137); golden fast-tracks 4 to 11 of 30; the safety gate caught my own design mistake (outline vs box sizing) before it shipped |
| M3b | Part model plus fusion: "dent on rear bumper, 6% of the part"; calibration showed my model was slightly under-confident (T 0.80); ONNX turned out slower than PyTorch on a laptop CPU; the safety gate caught the same sizing mistake a second time in a new code path |
| M4a | Our models and mock company systems as MCP tools; Claude Code can use them, and the server refuses anything outside the profile's job, including a direct call to an unadvertised tool; payments need a human-signed token |
| M4b | One doorway to the LLM with hard spend caps ($0.03 per claim): two real Claude calls cost $0.0006 and the repeat was free from cache; policy search that returns the exact clause to quote (LanceDB hybrid search), so the agent can never cite a clause that does not exist |

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

**M3a: damage model (done)**

- [x] Damage segmentation model (YOLO11n/s-seg) trained on a free Kaggle GPU (ADR 0008)
- [x] Champion `damage-yolo11s-v1` chosen on validation: test mask mAP50 0.733 (target 0.50)
- [x] MLflow experiment tracking and model registry (local SQLite logbook)
- [x] Golden v1 with our model: route accuracy 0.80, escalation recall 1.00, 11 of 30 correct fast-tracks
- [x] Gate caught a design mistake: sizing damage by its outline broke safety (0.97); pricing by box, as the rate card defines, restored 1.00

**M3b: parts, fusion, calibration (done)**

- [x] Part segmentation model `parts-yolo11n-v1`: test mask mAP50 0.746 (22 classes)
- [x] Fusion: which damage is on which part, and how big it is (88% of golden findings get a part)
- [x] Part-ratio severity (rate card v1), box fallback (ADR 0009)
- [x] Calibrate confidence: T = 0.80, ECE 0.059 to 0.040; recommended R6 threshold 0.65
- [x] R6 threshold set to 0.65 (decision-policy-v1, owner-approved): golden 0.78 / 1.00 / 9 of 30
- [x] ONNX export (slower than `.pt` on this CPU, so `.pt` stays the default)
- [x] Golden v1 with the fused detector: 0.80 / 1.00 / 11 of 30
- [x] Retro (`docs/retros/m3-vision-models.md`)
- [x] Model card (`docs/model-card.md`; target mask mAP50 ≥ 0.50 met: damage 0.733, parts 0.746)
- [ ] Stretch: RT-DETR comparison (dropped for now)

## M4 · Tools & integration (Nov 16–26)

*Give the agent safe, standard ways to use our models and (mock) company systems.*

**M4a: MCP tools (done)**

- [x] MCP servers: vision, policy-admin, claims-system, payments (ADR 0010)
- [x] Per-agent permissions (scopes), enforced at listing and at call, with a scope-matrix test
- [x] Payments: human `operator` profile plus signed approval token (`claimlens approve-payment`)
- [x] Idempotent writes; every call audited (claim log or `var/mcp-audit.jsonl`)
- [x] Red-team starter tests: ASI02 tool misuse, ASI03 privilege abuse, injected note as data
- [x] Claude Code demo (`.mcp.json`, read-only `demo` profile); real stdio test with the fused model
- [ ] **You:** try the Claude Code demo (README, "Use ClaimLens from Claude Code")

**M4b: LLM gateway and policy search (done)**

- [x] LLM gateway: tiers (Sonnet 5.5 / Haiku 4.5), retries, Claude-only fallback, response cache (ADR 0011)
- [x] Hard cost caps: $0.03 per claim, $1.00 per day, checked before each call
- [x] Structured outputs validated with one repair; call log with prompt hashes, never the key
- [x] Live smoke test on the real API: about $0.0006 for two calls; a cache hit costs $0
- [x] Fictional policy wordings (56 clauses) plus LanceDB hybrid search (BM25 + vectors, RRF) (ADR 0012)
- [x] Citation check (`verify_citations`) and MCP tool `search_policy_clauses`
- [x] Policy search quality: recall@5 1.00 on 10 questions (recall@1 0.60)

## M5 · Triage agent (Nov 27 – Dec 10)

*Add the reasoning assistant, then measure it like a product.*

- [x] Agent loop with step, cost and time limits (LangGraph on our gateway and scoped MCP tools; M5a)
- [x] LangGraph chosen by the owner without a bake-off (ADR 0013, supersedes ADR 0007)
- [x] Citations checked in code: every clause exists and belongs to the claimant's wording
- [x] Golden v1 with the LLM agent: escalation recall 1.00, $0.009 per claim; fast-tracks 1 of 30 (stub 9): the agent holds back claims whose vision output is impossible (fusion fix needed)
- [x] Fusion fix: damage only on plausible parts, part ratio capped at 1 (ADR 0009 update)
- [x] Human review queue: `claimlens queue`, `claimlens review-claim`; only a person can deny; payments wait for a review (ADR 0015)
- [x] 150 golden claims (v2): 53 narrative cases; the agent catches 43 of 43 with the right clause, fast-tracks 3 of 10 harmless stories (too cautious)
- [x] LLM judge built (pass rate 0.77); checking it against your 50 labels was **skipped by the owner**, so it is not validated and stays out of the gate (open item)
- [x] CI blocks changes that lower eval scores or skip re-evaluation (`claimlens eval-gate`, ADR 0014)
- [x] Tracing in Phoenix (OpenTelemetry), opt-in with `CLAIMLENS_TRACING=1`

## M6 · Intake agent & memory (Dec 11–24)

*Add the conversation with the customer, and give the system a governed memory.*

- [x] Multi-turn intake agent with guided photo capture: `claimlens intake`, pause and resume, photo coaching with retakes (M6a, ADR 0016); 3 live scripted customers complete intake, $0.18
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
