# ClaimLens: System Design Specification

- **Status:** Draft for review
- **Date:** 2026-10-01
- **Owner:** kb2326
- **Schedule:** ~15.5 weeks at ~10 h/week (2026-10-01 → 2027-01-15)

---

## 1. Summary

ClaimLens is an AI-assisted **vehicle damage claims triage** system. A claimant reports a loss and
uploads photos. Custom-trained computer-vision models measure the damage (which part, which damage
type, how much area). A deterministic workflow gathers evidence (integrity checks, cost estimate,
policy lookup). A supervised LLM agent reasons over the evidence and recommends a route.
Deterministic business rules make the final routing decision. Humans handle everything that is
high-value, uncertain, suspicious, or a denial.

It is built as a portfolio project that follows enterprise and insurtech practice end to end:
product definition, data engine, model training, agent engineering, evaluation, security,
observability, governance, and deployment.

## 2. Problem and goals

**Problem.** Claims adjusters spend most of their time on simple claims. Inspection, coverage
checks, estimating and routing take days, and outcomes vary between adjusters.

**Goals**

| ID | Goal |
|---|---|
| G1 | Fast-track simple, covered, low-value claims with an explanation an auditor can verify. |
| G2 | Escalate everything else to a human with evidence already assembled. |
| G3 | Make every decision reproducible and auditable from an immutable event log. |
| G4 | Demonstrate the 2026 agentic AI reference stack in a credible, measured way. |

**Non-goals (will not do)**

- Automatic claim denial or automatic payment without human approval.
- Multi-agent swarms or more agents than needed (three, with clear boundaries).
- Agent-authored memory without human review.
- Fine-tuning the LLM (context engineering plus evals instead).
- GraphRAG, Kubernetes, Kafka, microservices, a feature store.
- Voice intake (stretch at most). Real insurer data or real insurer branding.

## 3. Users and success metrics

| Persona | Need |
|---|---|
| **Claimant** | Report a loss quickly, get guided photo capture, get a fast decision. |
| **Adjuster** | Receive escalated claims with evidence, rationale and citations; override easily. |
| **Fraud investigator (SIU)** | Receive flagged claims with the specific integrity signals. |
| **Auditor / regulator** | Reconstruct why any decision was made. |

**Targets** (hypotheses, measured on held-out data):

| Area | Metric | Target |
|---|---|---|
| Vision | Damage mask mAP50 (test split) | ≥ 0.50 (legacy box mAP50: 0.137) |
| Vision | Expected calibration error after calibration | ≤ 0.05 |
| Triage | Escalation recall (claims needing a human that are escalated) | ≥ 0.95 |
| Triage | Route accuracy on golden claims | ≥ 0.85 |
| Intake agent | pass^4 on simulated-claimant tasks | ≥ 0.70 |
| Safety | Red-team suite: successful hijacks of a routing decision | 0 |
| Ops | p95 end-to-end latency per claim (CPU, excluding human wait) | ≤ 15 s |
| Ops | LLM cost per claim | ≤ $0.05 |
| Audit | Decisions fully reconstructable from the event log | 100% |

## 4. Design principles

1. **Models measure, the agent reasons, rules decide.**
2. **The system never auto-denies.** Outcomes are `FAST_TRACK`, `ADJUSTER_REVIEW`, or
   `FRAUD_REVIEW`. Only humans deny.
3. **Workflows where possible, agents where necessary** (Anthropic, *Building Effective Agents*).
4. **Own the control flow** and treat the agent as a stateless reducer over persisted state
   (12-Factor Agents).
5. **Everything is an event.** One append-only, hash-chained log provides durability, memory,
   audit, and replay.
6. **Least privilege.** Each agent has its own identity and tool scopes.
7. **Evals before features.** No component merges without a measurable evaluation.
8. **Fail safe.** Any failure or uncertainty routes to a human.

## 5. Architecture

### 5.1 Claim flow

```
Claimant ──▶ INTAKE AGENT (multi-turn: facts + guided photo capture)
                 │ ClaimReported, PhotoUploaded events
                 ▼
 ┌─────────────── DETERMINISTIC WORKFLOW ───────────────────────────────┐
 │ 1 Intake gate   quality check (blur/dark/not-a-car), PII redaction   │
 │ 2 Perception    damage seg + part seg → fusion → DamageFinding[]     │
 │ 3 Integrity     pHash reuse, EXIF, synthetic-image score,            │
 │                 narrative-vs-photo consistency → FraudSignal[]       │
 │ 4 Pricing       rate card (part × damage × severity) → CostEstimate  │
 └───────────────────────────────┬──────────────────────────────────────┘
                                 ▼
                 TRIAGE AGENT (tools: policy search, coverage, claim
                 history, similar claims, skills) → AgentRecommendation
                                 ▼
                 DECISION POLICY (deterministic rules, override agent)
                                 ▼
          FAST_TRACK │ ADJUSTER_REVIEW │ FRAUD_REVIEW  ──▶ human queue
                                 ▼
          Adjuster decision/corrections ──▶ events ──▶ evals + retraining
```

Every arrow writes events to the claim event log (§6).

### 5.2 Layers (2026 reference stack mapping)

| # | Layer | ClaimLens implementation |
|---|---|---|
| 1 | Specialised models | Damage instance segmentation, part segmentation, fusion |
| 2 | Control flow | Hand-written workflow + agent loop (no framework owns the loop) |
| 3 | Durable execution | Event-sourced claim state; resume any claim from its log |
| 4 | Model gateway | `claimlens.llm`: routing, fallback, retries, caching, cost caps |
| 5 | Context engineering | Just-in-time retrieval, compaction, notes outside context, progressive disclosure |
| 6 | Knowledge | Hybrid BM25 + embedding RAG over fictional policy documents, with citations |
| 7 | Memory | Working / episodic / semantic / procedural (§7) |
| 8 | Tools | MCP servers (§9) |
| 9 | Agent-to-agent | A2A repair-shop partner agent (stretch) |
| 10 | Human-in-the-loop | `request_human_review` tool, approval tokens, review queue |
| 11 | Guardrails | Decision policy rules, schema validation, input/output filters |
| 12 | Security & identity | OWASP ASI01–10 threat model (§14), per-agent scopes |
| 13 | Evaluation | §13 |
| 14 | Observability | OpenTelemetry GenAI spans → Arize Phoenix; cost/latency dashboards |
| 15 | LLMOps | Versioned prompts, eval-gated prompt/model changes |
| 16 | Governance | AIS Program doc (NAIC), system card, model inventory, audit log |

## 6. Event-sourced claim state

**Event envelope**

```
ClaimEvent {
  event_id: UUID, claim_id: UUID, seq: int,           # seq is gapless per claim
  type: str, payload: dict, actor: Actor,             # actor = system | agent:<name> | human:<id>
  occurred_at: datetime, schema_version: int,
  prev_hash: str, hash: str                           # sha256(prev_hash + canonical(event))
}
```

**Core event types:** `ClaimReported`, `PhotoUploaded`, `PhotoRejected`, `PhotoRedacted`,
`DamageDetected`, `FraudSignalRaised`, `CostEstimated`, `PolicyRetrieved`, `AgentStepCompleted`,
`AgentRecommended`, `HumanReviewRequested`, `RouteDecided`, `HumanDecided`, `HumanOverrode`,
`MemoryWritten`, `ClaimClosed`.

**Rules**

- Events are append-only. Corrections are new events (e.g. `HumanOverrode`), never edits.
- `ClaimState` is a **projection**: `fold(events) -> ClaimState`. The agent is a pure function
  `step(ClaimState) -> list[ClaimEvent]`.
- The hash chain is verified on read; a broken chain stops the claim and raises an alert.
- Storage: SQLite (`events` table, unique `(claim_id, seq)`). Photos are stored by content hash
  in a local blob store; events reference hashes, not bytes.
- Resume = load events, fold, continue. This is how a claim waits days for a human or photos.

## 7. Memory

| Type | Content | Storage | Write policy |
|---|---|---|---|
| Working | Current `ClaimState` + agent scratch notes | Event log projection | Automatic (it is the event log) |
| Episodic | Prior claims for the policy, vehicle, claimant; image embeddings | SQLite + LanceDB (DINOv2 embeddings) | Written by the workflow, not the agent |
| Semantic | Policy documents; resolved similar claims with adjuster outcomes | LanceDB hybrid index | Ingestion pipeline only |
| Procedural | Adjuster procedures as **Agent Skills** (`SKILL.md`) | `skills/` directory, versioned in git | Human-authored or human-approved PR only |

**Memory governance:** every memory read is logged as part of an `AgentStepCompleted` event;
every write is a `MemoryWritten` event with provenance. PII retention is configurable, and deleting
a claimant replaces PII in projections and the blob store with a tombstone while the event chain
keeps hashes only.

## 8. Agents

| Agent | Purpose | Tools (scope) | Model tier |
|---|---|---|---|
| **Intake** | Multi-turn first notice of loss: extract structured facts, guide photo capture | `record_fact`, `request_photo(angle)`, `check_photo_quality`, `lookup_policy_summary` (read) | Fast/cheap |
| **Triage** | Reason over evidence, check coverage, recommend route with citations | `policy_search`, `get_coverage`, `get_claim_history`, `find_similar_claims`, `load_skill`, `request_human_review` (read + escalate) | Strong |
| **Copilot** (stretch) | Answer adjuster questions over the claim file; draft letters | read-only claim file + `draft_letter` | Strong |

**Agent loop** (owned by us): build context → call LLM with tools and output schema → validate →
execute permitted tools → append events → repeat until a terminal output or limits are reached.
Limits per run: max 12 steps, max $0.03 LLM spend, 60 s wall clock. Exceeding a limit emits
`HumanReviewRequested(reason="agent_budget_exceeded")`.

**Output contract:** `AgentRecommendation { route_suggestion, confidence: low|medium|high,
rationale, citations: list[EvidenceRef], open_questions }`. Every citation must reference an event
or a policy clause that exists, and this is checked in code.

## 9. Tools and MCP servers

| MCP server | Tools | Side-effect class |
|---|---|---|
| `vision` | `segment_damage`, `segment_parts`, `assess_quality` | read |
| `policy-admin` (mock) | `get_policy`, `get_coverage`, `search_policy_clauses` | read |
| `claims-system` (mock) | `get_claim_history`, `find_similar_claims`, `add_note`, `assign_queue` | read / write |
| `payments` (mock) | `issue_payment` | **consequential**: requires a human approval token |
| `repair-network` (mock, stretch) | `find_shops`, `request_estimate` (via A2A partner agent) | write |

Each server runs as its own process with its own credentials; each agent receives only the scopes
listed in §8. Tool inputs and outputs are Pydantic-validated. Write tools are idempotent (they take
an idempotency key).

## 10. LLM gateway (`claimlens.llm`)

- One interface: `LLM.generate(messages, tools, output_schema, tier) -> LLMResponse`.
- Providers: Claude (default) and Ollama (local fallback). Tier → model mapping lives in config.
- Retries with backoff, provider fallback, prompt caching for policy documents and system prompts,
  per-claim cost accounting and hard cap, structured outputs validated against Pydantic schemas.
- Prompts are versioned files (`prompts/<agent>/<version>.md`); the active version is in config and
  recorded in each `AgentStepCompleted` event.
- Tests use a recorded/fake provider so CI needs no API key.

## 11. Decision policy

Evaluated in order; the first matching rule wins. All thresholds live in versioned config.

| Rule | Route |
|---|---|
| Any fraud signal score ≥ 0.5, or a reused image is detected | `FRAUD_REVIEW` |
| Photo set incomplete or quality gate failed after 2 retries | `ADJUSTER_REVIEW` |
| Coverage not confirmed by `get_coverage` | `ADJUSTER_REVIEW` |
| Cost estimate high bound > $3,000 | `ADJUSTER_REVIEW` |
| Any damage finding below the calibrated confidence threshold τ (chosen in M3 from the PR curve) | `ADJUSTER_REVIEW` |
| Agent confidence `low`, or open questions not empty | `ADJUSTER_REVIEW` |
| Agent suggests anything other than fast-track | `ADJUSTER_REVIEW` |
| Otherwise | `FAST_TRACK` (payment still requires a human approval token) |

## 12. Data and models

**Datasets**

- Roboflow car-damage v6 (CC BY 4.0): 339 images, 7 classes. Baseline and test fixture only.
- CarDD (Wang et al., IEEE T-ITS 2023): about 4,000 images with instance masks for dent, scratch,
  crack, glass shatter, lamp broken, tire flat. **License to be confirmed in M2 before any public
  release of derived weights.**
- Car-part segmentation dataset (to be selected in M2 from public CC-licensed sources).

**Data engine (M2):** DVC-tracked `raw → interim → processed`; perceptual-hash deduplication across
splits; splits grouped by source vehicle; FiftyOne review of label errors; Grounding DINO + SAM 2
pre-labelling of unlabelled or box-only images, followed by human correction; a data card.

**Models (M3):** YOLO11-seg (primary) and RT-DETR (comparison) for damage; a YOLO11-seg part model.
Fusion assigns each damage mask to the part with the highest intersection-over-damage-area and
computes `area_ratio = damage_area / part_area`. Severity bands (minor, moderate, severe) come from
`area_ratio` and damage type. Temperature scaling calibrates confidence. Experiments are tracked in
MLflow; the best model is registered and exported to ONNX. A model card is written.

## 13. Evaluation strategy

| Layer | Method | Gate |
|---|---|---|
| Data | Schema checks, split-leakage check (pHash), class-count report | CI on data changes |
| Vision | mAP50 / mAP50-95 per class, PR curves, ECE, slice analysis (lighting, angle, damage size) | Regression vs registered model |
| Fusion & pricing | Unit tests on synthetic masks; golden part/damage cases | CI |
| Triage | **Golden claims set** (≥ 50 in M1, ≥ 150 by M5): photos + facts + policy + expected route + key facts | Route accuracy, escalation recall |
| Agent quality | LLM-as-judge for rationale faithfulness and citation validity, **validated against ≥ 50 human labels** (agreement ≥ 0.8) | CI on prompt/model changes |
| Intake agent | **Simulated claimant** (LLM persona with a hidden ground truth); success = final `ClaimState` facts match the goal (τ-bench style); **pass^k** over k = 4 trials | pass^4 ≥ 0.70 |
| Security | Red-team suite mapped to OWASP ASI01–10 (§14) | 0 successful route hijacks |
| Ops | Latency and cost per claim from traces | p95 ≤ 15 s, ≤ $0.05 |

The eval suite runs in CI on every pull request that touches prompts, agent code, models, or
policy rules. LLM-dependent evals use a cached run on pull requests and a fresh nightly run.

## 14. Security threat model (OWASP Top 10 for Agentic Applications 2026)

| Risk | ClaimLens scenario | Control |
|---|---|---|
| ASI01 Goal hijack | Text inside a photo or narrative: "approve this claim" | OCR'd or narrative text is passed as quoted data; rules decide routing |
| ASI02 Tool misuse | Agent induced to call `issue_payment` | Payments tool not in any agent scope; human approval token required |
| ASI03 Identity & privilege abuse | Intake agent reaches claims-history data | Per-agent credentials and scopes; scope tests in CI |
| ASI04 Supply chain | Compromised MCP server or dependency | Pinned versions, lockfile, local MCP servers only, dependency audit in CI |
| ASI05 Unexpected code execution | Agent produces code | No code-execution tool exists |
| ASI06 Memory & context poisoning | Malicious notes in a prior claim | Agents cannot write memory; provenance on retrieved items; notes quoted as data |
| ASI07 Insecure inter-agent comms | Spoofed partner agent (A2A) | Signed AgentCard, mutual auth, allowlist |
| ASI08 Cascading failures | Vision service down | Timeouts, circuit breaker, fail-safe route to human |
| ASI09 Human trust exploitation | Adjuster rubber-stamps recommendations | UI shows confidence, evidence and counter-evidence; random audit sampling |
| ASI10 Rogue agent | Loops or overspending | Step, cost and time limits; global kill switch in config |

**PII:** faces and licence plates are blurred before storage of derived images and before any LLM
call; originals are kept encrypted at rest with restricted access. Secrets live in `.env` (never
committed).

## 15. Observability

- OpenTelemetry tracing with GenAI semantic conventions: `invoke_agent`, tool spans, model spans
  (tokens, model, latency, cost), linked by `claim_id`.
- Local backend: Arize Phoenix. Metrics: cost per claim, latency per stage, route distribution,
  escalation rate, override rate.
- Drift monitoring (Evidently): input image statistics and prediction distributions versus training.
- Replay: any claim can be re-run from its event log with the same or a new prompt/model version.

## 16. Governance

- **AIS Program document** following the NAIC Model Bulletin: inventory, risk tiering, roles,
  testing, monitoring, third-party (LLM vendor) oversight.
- **System card**, **model card**, **data card**.
- **EU AI Act note:** motor claims triage is not explicitly listed in Annex III (life and health
  insurance pricing and risk assessment are). ClaimLens nevertheless meets high-risk-style controls
  voluntarily: risk management, data governance, logging, human oversight, accuracy reporting.
- **Audit:** tamper-evident event log; every route decision cites the rule that fired.

## 17. Error handling

| Failure | Behaviour |
|---|---|
| Invalid upload (type, size > 15 MB, unreadable) | Reject with a 4xx error and a `PhotoRejected` event |
| Quality gate failure | Intake agent requests a retake (max 2), then `ADJUSTER_REVIEW` |
| Model or tool timeout or error | Retry once, then circuit-break and route to `ADJUSTER_REVIEW` with reason |
| LLM provider error | Retry with backoff, fall back to the secondary provider, then `ADJUSTER_REVIEW` |
| Output fails schema or citation validation | One repair attempt with the validation error in context, then `ADJUSTER_REVIEW` |
| Budget or step limit exceeded | `ADJUSTER_REVIEW` (reason recorded) |
| Event hash-chain mismatch | Halt the claim, alert, no further processing |

## 18. Testing strategy

- **Unit:** schemas, event folding, hash chain, fusion geometry, pricing, decision rules,
  gateway (fake provider). Property-based tests (Hypothesis) for fold/replay and hash chain.
- **Contract:** MCP tool schemas and per-agent scope enforcement.
- **Integration:** full claim through the workflow with a fake LLM and small ONNX models
  (marked `integration`).
- **Evals:** §13. **Coverage target:** ≥ 85% line coverage on `src/claimlens` (excluding UI).

## 19. Technology stack

Python 3.12, uv, ruff, mypy, pytest, Hypothesis, pre-commit, GitHub Actions · Pydantic v2 ·
SQLite, LanceDB · Ultralytics (YOLO11-seg, RT-DETR), ONNX Runtime, Grounding DINO, SAM 2, DINOv2 ·
DVC, FiftyOne, MLflow · Anthropic SDK, Ollama · MCP Python SDK · FastAPI · Gradio (Hugging Face
Spaces) · OpenTelemetry, Arize Phoenix, Evidently · Docker.

Each significant choice receives an ADR at the milestone where it is first used.

## 20. Milestones

| Milestone | Dates | Exit criteria (must) | Should / Could |
|---|---|---|---|
| **M0 Foundations** | Oct 1–7 | Repo restructured; CI green; ADRs 0001–0002; legacy audit; AGENTS.md; this spec approved; PR/FAQ | — |
| **M1 Walking skeleton** | Oct 8–18 | Event log + fold + hash chain; workflow stages wired with the legacy model (labels fixed) and stub agent; decision policy; golden claims v0 (50); CLI demo | Gradio demo |
| **M2 Data engine** | Oct 19 – Nov 1 | CarDD license confirmed; DVC; dedup + grouped splits; auto-labelling pipeline; FiftyOne review; data card | Part dataset |
| **M3 Vision models** | Nov 2–15 | YOLO11-seg damage + part models; fusion; calibration; MLflow registry; ONNX; model card; damage mAP50 ≥ 0.50 | RT-DETR comparison |
| **M4 Tools & integration** | Nov 16–26 | MCP servers (vision, policy-admin, claims-system); LLM gateway; policy RAG with citations; scope tests | Reranker |
| **M5 Triage agent** | Nov 27 – Dec 10 | Agent loop + limits; human review queue; golden claims v1 (150); validated LLM judge; CI eval gates; OTel → Phoenix | Prompt registry UI |
| **M6 Intake agent & memory** | Dec 11–24 | Intake agent; episodic + semantic memory; Agent Skills (≥ 3); memory governance | Simulated claimant + pass^k |
| **M7 Trust & governance** | Dec 25 – Jan 4 | Integrity checks; PII redaction; OWASP red-team suite passing | AIS Program doc, system card |
| **M8 Ship** | Jan 5–15 | Docker; FastAPI; public Gradio demo on HF Spaces; Evidently monitoring; final write-up | A2A partner agent; copilot |

Each milestone ends with a demo, a short retrospective in `docs/retros/`, and an optional
build-in-public post.

## 21. Risks and open questions

| Risk | Mitigation |
|---|---|
| CarDD license disallows public release of derived weights | Confirm in M2; fall back to CC-licensed data for public weights and keep CarDD-trained weights private |
| No local GPU | Train on Colab/Kaggle; inference via ONNX on CPU |
| LLM cost overrun | Gateway cost caps, caching, cheap tier for intake, cached evals in CI |
| Golden claims are synthetic | Document limitations; vary personas and adversarial cases; keep human-labelled subset |
| Scope creep | MoSCoW per milestone; stretch items can be dropped without breaking the system |
| Holiday period (M6–M7) | Two-week buffer absorbed by stretch items |

**Open questions** (resolved at the named milestone):

- Car-part segmentation dataset choice (M2).
- Embedding model for the policy RAG (M4).
- Whether a framework (LangGraph or Temporal) is needed beyond the owned loop, decided by an ADR
  in M5 based on real pain points.
