# M6b: Claim Memory, Agent Skills and Simulated Customers Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A workflow-written claim memory in LanceDB with near-copy photo search; three
owner-approved Agent Skills the triage agent can load; and a simulated-customer evaluation of the
intake agent with pass^k.

**Architecture:**
- `claimlens.memory`:
  - `records.py` builds a `MemoryRecord` from a decided claim (perceptual hashes, damage and
    summary; no customer text);
  - `index.py` holds `ClaimMemory` on LanceDB (upsert, forget, rebuild, similar).
- The workflow writes memory after `RouteDecided`, and `record_review` updates it, each with a
  `MemoryWritten` event.
- `find_similar_claims` uses the memory when one is configured.
- `claimlens.skills` parses `skills/*/SKILL.md` and loads only approved skills. The triage agent
  gets a `load_skill` tool and the `triage/v2` prompt.
- `claimlens.evals.intake_sim` runs LLM-played customers against the real intake sessions and
  scores pass^k.

**Tech Stack:** LanceDB 0.39, `imagehash` (pHash, already a dependency), the existing embedders
(`FastEmbedder`, `FakeEmbedder`), LangChain `StructuredTool`, the gateway on the `fast` tier.

**Spec:** [`docs/specs/2026-10-03-m6b-memory-skills-design.md`](../specs/2026-10-03-m6b-memory-skills-design.md)

---

## Plain-language briefing

**What we're building:**
1. **A memory of past claims** that the system keeps, never the agents. The triage agent can ask
   "seen anything like this before?" and learn about a re-used photo (even cropped or
   re-saved), earlier claims on the same policy, and how similar claims ended.
2. **Three written procedures** (glass, flat tyres, exclusions) that the triage agent can read
   when a claim needs one. You approve each before it can be used.
3. **Pretend customers** played by an AI, to measure how reliably the intake chat gets the facts
   right: each one 4 times, and it only counts if all 4 are right.

**What you do:** approve the three skills (about 5 minutes), and approve the spend (about $6,
already agreed in the spec).

**To-do:**
1. Memory records and events (Task 1)
2. The memory index: upsert, forget, rebuild, similar (Task 2)
3. Writing memory from the workflow and reviews; `claimlens memory` (Task 3)
4. `find_similar_claims` on memory (Task 4)
5. Skills: format, loader, approval (Task 5)
6. The triage agent loads skills; prompt v2 (Task 6)
7. The simulated customers and pass^k (Task 7)
8. Owner approval of the skills, live runs, ADR 0017, docs (Task 8)

## Global Constraints

- Package code lives in `src/claimlens`; no imports from `legacy/`. `uv run …`; ruff (line 100),
  mypy strict, pytest; coverage ≥ 95%.
- Never commit `data/`, `models/`, `var/`, `.env`; never print the API key.
- **Agents never write memory or skills**; the workflow writes memory, people write or approve
  skills.
- **No decision rule changes**; memory is evidence only.
- A memory record holds no customer free text (no description, no transcript).
- Live runs only in Task 8 (golden v2 re-run about $1.50, eval-intake about $4), each with
  `--llm-daily-cap`.
- Edit Python with exact-string tools; avoid shell heredocs containing `\n` or non-ASCII.
- Sync with `--inexact --no-install-project … --reinstall-package opencv-python-headless`.

## Review Focus

1. **A memory write fails after the decision** (LanceDB error). The decision must stand, and the
   failure must be recorded, not raised (Task 3).
2. **A claim is decided twice** (`process_claim` called again). The memory upsert must not
   duplicate the record (Task 2).
3. **A record is built from a claim with no accepted photos or no estimate.** Empty fields, not
   a crash (Task 1).
4. **A skill file has bad front matter, a missing name, or `approved_by` blank.** Refused with a
   clear reason; never loaded (Task 5).
5. **The simulated customer leaks its instructions or answers for the agent.** The harness stops
   at a turn limit and scores honestly (Task 7).

---

### Task 1: Memory records and events

**Files:** Create `src/claimlens/memory/__init__.py`, `src/claimlens/memory/records.py`; Modify
`src/claimlens/events/payloads.py`, `src/claimlens/events/projection.py`; Test
`tests/unit/test_memory_records.py`

**Interfaces:**
- `MemoryRecord(claim_id, policy_id, decided_on: str, route, rule_id, review_action: str = "",
  final_route: str = "", damage: str, cost_low: int, cost_high: int,
  photo_phashes: tuple[str, ...], summary: str, source_seq: tuple[int, ...])`
- `phash_of(path: Path) -> str`: a 16-hex-character perceptual hash.
- `build_record(state: ClaimState, events: Sequence[ClaimEvent], photo_path: Callable[[str],
  Path]) -> MemoryRecord`:
  - photo hashes for accepted photos;
  - damage from findings;
  - summary from damage, cost, intake yes/no facts and the outcome;
  - `source_seq` from `DamageDetected`, `CostEstimated`, `RouteDecided` and `HumanReviewed`.
- Events:
  - `MemoryWritten(record_id: str, fields: tuple[str, ...], source_seq: tuple[int, ...])`;
  - `MemoryForgotten(reason: str)`.
  The fold ignores both.

Tests:
- a decided claim produces the expected record;
- no photos or no estimate gives empty values;
- a review updates `review_action` and `final_route`;
- the description text never appears in any field;
- the events fold.

### Task 2: The memory index

**Files:** Create `src/claimlens/memory/index.py`; Test `tests/unit/test_memory_index.py`

**Interfaces:**
- `ClaimMemory(path: Path, embedder: Embedder)`:
  - `upsert(record)`: delete then add, so the claim id stays unique;
  - `forget(claim_id) -> bool`;
  - `get(claim_id) -> MemoryRecord | None`;
  - `all() -> list[MemoryRecord]`;
  - `similar(record, limit=5) -> list[SimilarMemory]`.
- `SimilarMemory(claim_id, reason, distance: int | None, route, review_action, damage,
  cost_low, cost_high)`.
- Search order:
  1. near-copy photos (Hamming ≤ 6, `PHASH_MAX_DISTANCE`);
  2. the same policy;
  3. similar damage (vector search on the summary, keeping only records that share a damage
     type);
  4. the record itself excluded, duplicates removed, cut to `limit`.
- `rebuild(memory, store, photo_path) -> int`: rebuilds from every decided claim in the event
  store.
- Claim ids are validated as UUIDs before use in filters.

Tests:
- upsert twice keeps one row;
- forget;
- a near-copy (re-encoded at JPEG quality 60, cropped by 5%) is found with its distance;
- an unrelated photo is not;
- same policy;
- similar damage;
- the limit is applied;
- rebuild from a store with three decided claims;
- a bad id is refused.

### Task 3: Writing memory

**Files:** Modify `src/claimlens/workflow.py` (`PipelineDeps.memory: ClaimMemory | None = None`;
after `RouteDecided`), `src/claimlens/review_queue.py` (`record_review(…, memory=None)`),
`src/claimlens/cli.py` (`claimlens memory rebuild | forget <claim> | show <claim>`, plus memory
wiring for `run` and `review-claim` when the `knowledge` group is installed); Test
`tests/unit/test_memory_workflow.py`

- After `RouteDecided`, the workflow builds the record, upserts it and appends `MemoryWritten`.
  On any exception it appends `StageFailed(stage="memory", …)` and returns the decision
  unchanged.
- `record_review` updates the record and appends `MemoryWritten` when memory is configured.
- `memory forget` appends `MemoryForgotten`.
- The memory lives in `var/memory`; the eval harness uses one memory per case directory.

Tests:
- a processed claim is in memory with a `MemoryWritten` event;
- processing again changes nothing (no second event);
- a failing memory gives a `StageFailed(memory)` and the same route;
- a review updates memory;
- the CLI's rebuild, forget and show.

### Task 4: `find_similar_claims` on memory

**Files:** Modify `src/claimlens/mcp/claims_system.py` (`build_claims_system(…, memory=None)`),
`src/claimlens/mcp/schemas.py` (`SimilarClaim` gains `route`, `review_action`, `damage`,
`cost_low`, `cost_high`, `distance`), `src/claimlens/mcp/serve.py`,
`src/claimlens/agent/factory.py`; Test `tests/unit/test_mcp_claims_system.py` (append)

- With memory: build the current claim's record from its log (it may not be decided yet, so a
  provisional record) and return `memory.similar(...)`.
- Without memory: the old behaviour (existing tests unchanged).
- The triage agent's factory passes the memory.

Tests:
- an earlier claim with a near-copy photo is returned as `near-copy photo` with its outcome;
- no customer text appears in the result;
- the intake profile still cannot call the tool.

### Task 5: Skills

**Files:** Create `src/claimlens/skills.py`, `skills/glass-claims/SKILL.md`,
`skills/flat-tyre-claims/SKILL.md`, `skills/exclusion-review/SKILL.md`; Test
`tests/unit/test_skills.py`

**Interfaces:**
- `Skill(name, description, version: int, approved_by: str, approved_on: str, body: str)`
- `parse_skill(path) -> Skill`: front matter between `---` lines, `key: value` pairs with quotes
  optional (no YAML dependency). A missing `name` or `description`, a name that differs from the
  folder, or a body under 50 characters → `ValueError`.
- `load_skills(root) -> dict[str, Skill]`: **approved skills only**. `approved_by` must be
  non-blank and `approved_on` an ISO date. `skipped` reasons are returned through
  `load_skills_report(root) -> tuple[dict, list[str]]`.
- The three skills are written from the policy wordings (10 to 20 lines each), with
  `approved_by: ""` until the owner approves (Task 8).

Tests:
- parse each repo skill;
- unapproved skills are not loaded;
- bad front matter;
- a name mismatch;
- the approval date format.

### Task 6: The triage agent loads skills

**Files:** Create `prompts/triage/v2.md`; Modify `config/agent.toml` (`prompt = "triage/v2"`,
`skills_dir = "skills"`), `src/claimlens/agent/config.py`, `src/claimlens/agent/langgraph_agent.py`,
`src/claimlens/agent/factory.py`, `src/claimlens/domain.py` (`AgentRecommendation.skills_used:
tuple[str, ...] = ()`); Test `tests/unit/test_agent_skills.py`

- `prompts/triage/v2.md` = v1 plus a "Procedures" section: load a listed skill with
  `load_skill(name)` when the claim matches its description; a skill is procedure, not evidence.
- The agent appends "Approved procedures:" plus `- name: description` lines to the system text
  (only approved skills; none means no section).
- `load_skill` is a `StructuredTool` that returns the body wrapped as
  `<procedure name="…" version="…">…</procedure>`; an unknown name is a `ToolException`. Each
  load is recorded and ends up in `skills_used` as `name@vN`.

Tests:
- the system text lists approved skills only;
- `load_skill` returns the body;
- `skills_used` on the recommendation;
- an unknown skill gives an error result;
- no skills means no tool and no section;
- the existing triage tests pass with `triage/v2`.

### Task 7: Simulated customers

**Files:** Create `src/claimlens/evals/intake_sim.py`, `prompts/customer/v1.md`,
`evals/intake/personas.jsonl`; Modify `src/claimlens/cli.py` (`eval-intake`); Test
`tests/unit/test_intake_sim.py`

**Interfaces:**
- `Persona(persona_id, style, truth: dict[str, str], story: str, photo_trouble: bool)`,
  validated against the intake facts.
- `run_trial(persona, sessions, customer: Callable[[list[dict]], str], photos: PhotoFixtures,
  max_turns=40) -> Trial(persona_id, passed: bool, failures: list[str], turns: int,
  handover: str)`:
  - the harness answers photo requests itself (a dark photo first when `photo_trouble`);
  - otherwise it asks the customer model.
- `customer_model(gateway, prompt, persona) -> Callable` on the `fast` tier: plain text and no
  tools; the persona's truth and style go in the system text.
- `score(trial facts, persona) -> list[str]` (the failed fields, per spec 5.3).
- `pass_at_k(trials_by_persona) -> tuple[float, float]` gives pass@1 and pass^k.
- `render_intake_report(...)`.
- CLI: `claimlens eval-intake --personas FILE -k 4 --report FILE --llm-daily-cap USD`.

Tests:
- a scripted customer and a scripted agent produce a passing trial;
- a wrong date is a failure;
- a story-changer's corrected date passes;
- the pass^k arithmetic (one failed trial fails the persona);
- the persona file schema;
- the turn limit ends a stuck trial as failed.

### Task 8: Approval, live runs and docs

- **Owner approval of the three skills** (sets `approved_by` and `approved_on`). Stop here until
  approved; do the docs meanwhile.
- **Golden v2 re-run** with `triage/v2` and memory (`--scorecard`):
  - escalation recall and citation validity must be 1.00; otherwise stop and report;
  - copy the scorecard to the baseline.
- **`eval-intake`:** 10 personas, k = 4, about $4. Report pass@1 and pass^4 honestly.
- **ADR 0017** (memory, skills, simulated customers), roadmap and README.
