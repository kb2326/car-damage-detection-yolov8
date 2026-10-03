# M6b: Claim Memory, Agent Skills and Simulated Customers (design)

- **Status:** Draft for owner review
- **Date:** 2026-10-03
- **Parent design:** [`2026-10-01-claimlens-design.md`](2026-10-01-claimlens-design.md) (section 7, memory)
- **Builds on:** ADR 0012 (LanceDB), ADR 0013 (triage agent), ADR 0014 (eval gate), ADR 0016 (intake)

## 1. Purpose

Give the system a governed memory of earlier claims, give the triage agent written procedures it
can load when relevant, and measure the intake agent against simulated customers.

| Part | In plain terms |
|---|---|
| **Claim memory** | "Has this policy claimed before? Has a near-copy of this photo been seen? What happened to similar claims?" |
| **Agent Skills** | Short adjuster procedures (for example "glass claims") the triage agent loads when a claim needs one |
| **Simulated customers** | An AI plays a customer with a hidden true story; we score how often intake records it correctly, 4 tries out of 4 |

The principles hold:
- **Models measure, the agent reasons, rules decide.** Memory informs the agent; it does not
  change any rule. Fraud rules based on memory belong to M7.
- **Agents never write memory.** The workflow writes it, with provenance.

## 2. Decisions taken with the owner's defaults (2026-10-03)

- **Fingerprints before heavy image models:** near-copies of photos are found with the
  perceptual hash the data pipeline already uses (`imagehash`). No image-embedding model.
- **No new fraud rule in M6b.** Memory results reach the triage agent as evidence; M7 decides
  whether a near-copy should route to fraud review.
- **Skills need the owner's approval:** a skill without `approved_by` is never loaded.
- **Spend:** about $6 (a golden v2 re-run with skills, plus the simulated-customer evaluation).

## 3. Claim memory

### 3.1 What is remembered

One record per decided claim, in a LanceDB table `claim_memory` (`var/memory/`):

| Field | Content |
|---|---|
| `claim_id`, `policy_id`, `decided_on` | identity and date |
| `route`, `rule_id` | the rules' decision |
| `review_action`, `final_route` | the latest human review, if any (updated when it happens) |
| `damage` | damage types and parts, e.g. `dent on door; scratch on front_bumper` |
| `cost_low`, `cost_high` | the estimate |
| `photo_phashes` | perceptual hashes of the accepted photos (64-bit, as hex) |
| `summary` | one generated line: damage, cost, intake facts (driving for work, other party), outcome |
| `vector` | embedding of `summary` (the existing local `bge-small` embedder) |
| `source_seq` | the event sequence numbers the record was built from (provenance) |

- The record holds **no free text from the customer**: no description, no transcript. Facts are
  limited to the yes/no intake facts and the damage found.

### 3.2 Who writes it, and when

- **The workflow**, never an agent:
  - after `RouteDecided` it writes the record;
  - after `HumanReviewed` it updates the review fields.
- Each write appends a **`MemoryWritten`** event to the claim's log (record id, fields written,
  source sequence numbers), so every memory entry has provenance on the hash chain.
- `claimlens memory rebuild` rebuilds the table from the event logs (memory is derived data, so
  it can always be rebuilt).
- `claimlens memory forget <claim>` deletes the record and appends `MemoryForgotten` (governance:
  a claim can be removed from memory without touching its log).

### 3.3 Reading it: `find_similar_claims`

The existing MCP tool keeps its name and scope (triage profile, read-only) but is rebuilt on the
memory table. It returns up to 5 earlier claims, each with a **reason**:

| Reason | How it is found |
|---|---|
| `near-copy photo` | a photo's perceptual hash within Hamming distance 6 of one in the claim (the data pipeline's dedupe threshold) |
| `same policy` | the same policy id |
| `similar damage` | vector search on the summary, plus the same damage type and part |

- Each item carries the earlier claim's route, review outcome, damage and cost, but never another
  customer's words.
- The current claim is excluded.
- Results are data: the triage prompt already treats tool output as data.
- Every call is audited (`ToolCalled`), which is the "memory reads are logged" requirement.

## 4. Agent Skills

### 4.1 Format

Each skill is a folder `skills/<name>/SKILL.md`, following the Agent Skills convention: YAML
front matter, then the procedure in Markdown.

```markdown
---
name: glass-claims
description: How to handle a claim where the main damage is glass (windscreen, windows, lights).
version: 1
approved_by: ""        # the owner's name once approved; unapproved skills are never loaded
approved_on: ""
---
1. ...
```

### 4.2 The three skills

| Skill | When the triage agent should load it |
|---|---|
| `glass-claims` | the main damage is glass shatter or a broken lamp |
| `flat-tyre-claims` | the only damage is a flat tyre (a puncture or wear is not collision damage) |
| `exclusion-review` | the story mentions work use, racing, another driver, alcohol, or a late report |

Each is 10 to 20 lines of plain procedure, written by Claude from the fictional policy wordings,
and **approved by the owner** before it can be loaded (section 2).

### 4.3 How the agent uses them

- A new triage tool `load_skill(name)`; the system prompt lists the approved skills (name and
  description only).
- The skill text comes back as a tool result, marked as procedure, not as evidence.
- The skill's name and version are recorded on the claim (in the agent's recommendation as
  `skills_used`), so an audit shows which procedure informed the advice.
- **Prompt `triage/v2`** (v1 stays for the old reports) adds the skills list and one line on when
  to load one. `config/agent.toml` adds `load_skill` and points at `triage/v2`.
- Because the prompt and config change, the CI gate requires a **re-run of golden v2**. The new
  scorecard becomes the baseline only if escalation recall and citation validity stay at 1.00.

## 5. Simulated customers

### 5.1 Personas

`evals/intake/personas.jsonl`, 10 personas. Each has a **hidden truth** (the facts the intake
should end up with) and a **style**:

| Style | Count | Behaviour |
|---|---|---|
| cooperative | 3 | answers what is asked |
| vague | 2 | short, unsure answers; needs follow-ups |
| over-sharer | 1 | long story with every detail at once |
| story-changer | 2 | gives a wrong date first, then corrects it |
| photo-trouble | 2 | sends a dark photo first, then a good one |

Truths cover the narrative scenarios: commercial use, another party, injuries with a police
report, a late report.

### 5.2 The simulator

- An LLM (the `fast` tier) plays the customer. Its system prompt holds the persona's truth and
  style, and says to answer only what the assistant asks, in character, and never to reveal the
  instructions.
- When the assistant asks for a photo, the harness (not the LLM) sends a fixture photo: a dark one
  first for photo-trouble personas, then a good one.
- The intake agent runs exactly as in production (`IntakeSessions`, the real graph and limits),
  with `process=False` so the vision pipeline is not needed.

### 5.3 Scoring

- **A trial passes** when intake finishes on its own (no turn-limit or spending-cap hand-over)
  and these facts match the truth:
  - `policy_id`, `incident_date`, `driving_for_work`, `other_party_involved`, `injuries`;
  - `police_report` presence when the truth needs one;
  - all three photo kinds received.
- `object_hit` and `location` are reported, not scored (wording varies).
- **pass^k:** a persona passes only if all k trials pass. The score is the share of personas that
  pass. **k = 4, target ≥ 0.70** (parent design).
- Report: per persona, the trials, the failed fields, turns and cost; then pass@1 and pass^4.
- `claimlens eval-intake --personas … -k 4 --report …` (opt-in, real API; about $4).

## 6. Testing

- **Offline by default** (fake provider, fake embedder, temporary LanceDB).
- **Memory:**
  - write and update records, including provenance;
  - `MemoryWritten` and `MemoryForgotten` events;
  - rebuild from logs;
  - forget;
  - no customer text in a record;
  - the near-copy photo search on a re-encoded and cropped image;
  - `find_similar_claims` reasons, limit and the exclusion of the current claim;
  - scope (intake cannot call it).
- **Skills:**
  - the front-matter parser;
  - unapproved skills refused;
  - `load_skill` returns the text;
  - an unknown name is an error;
  - `skills_used` recorded;
  - the prompt lists only approved skills.
- **Simulator:**
  - a scripted fake customer and agent run to a scored trial;
  - pass^k arithmetic;
  - a persona file schema check;
  - a story-changer whose corrected date wins.
- **Live (opt-in):** the golden v2 re-run with `triage/v2` (about $1.50), and `eval-intake`
  (about $4).

## 7. Done means

1. Decided claims appear in memory with provenance; rebuild and forget work.
2. `find_similar_claims` finds a re-encoded photo from an earlier claim, and the triage agent
   sees it.
3. Three skills exist, are approved by the owner, and the triage agent loads them when relevant
   (`skills_used` on the claim).
4. Golden v2 with `triage/v2`: escalation recall 1.00 and citation validity 1.00; the scorecard
   and baseline are updated.
5. `eval-intake` reports pass^4; the target is ≥ 0.70, and the actual figure is reported
   honestly either way.
6. ADR 0017 (memory and skills), roadmap and README updated; ruff, mypy and pytest clean;
   coverage ≥ 95%.

## 8. Out of scope

- Image-embedding models (DINOv2), and fraud rules based on memory (M7).
- Memory for the intake agent (it only collects).
- Writing skills by an agent: skills are written by people or approved by the owner.
