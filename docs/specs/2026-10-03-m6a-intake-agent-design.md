# M6a: Intake Agent (design)

- **Status:** Draft for owner review
- **Date:** 2026-10-03
- **Parent design:** [`2026-10-01-claimlens-design.md`](2026-10-01-claimlens-design.md) (sections 7, 8)
- **Builds on:** ADR 0010 (MCP scopes), ADR 0011 (gateway), ADR 0013 (LangGraph), ADR 0015 (review)

## 1. Purpose

Replace the one-shot claim form with a conversation. An intake agent talks to the customer,
collects the facts and the photos a claim needs, coaches better photos, can pause for days and
resume, and then hands a complete claim to the pipeline built in M1–M5.

The principles hold: the intake agent **collects**; it never decides, prices or promises cover.

## 2. Decisions already made by the owner (2026-10-03)

- The customer chats **in the terminal** (`claimlens intake`); the same agent will sit behind the
  web app in M8.
- Paused conversations are saved with **LangGraph's SQLite checkpointer**; the claim's event log
  stays the record of the claim.
- The agent uses the **`fast` tier (Haiku)**.
- **At most 2 retakes** per requested photo; then the conversation moves on and the gap is noted.
- Image memory and Agent Skills are **M6b**, not this spec.
- Spend for M6a: about **$5**.

## 3. What the customer sees

```
$ claimlens intake
ClaimLens: Hello, I'm the ClaimLens claims assistant. What's your policy number?
You: P-1001
ClaimLens: Thanks. Please tell me what happened, in your own words.
You: I reversed into a bollard at the supermarket yesterday.
ClaimLens: Sorry to hear that. Was anyone else involved, and were you driving for work?
You: No and no, just me.
ClaimLens: Please send a photo of the whole car, showing the damaged side.
You: /photo C:\photos\car.jpg
ClaimLens: That photo is too dark to see the damage. Could you take it again in better light?
You: /photo C:\photos\car2.jpg
ClaimLens: Thanks, that's clear. Now a close-up of the damage, please.
…
ClaimLens: That's everything I need. Your claim number is 3f2a…. A summary: …
           We'll review it and be in touch.
```

- `/photo <path>` attaches a photo; `/quit` pauses (the conversation is saved);
  `claimlens intake --session <id>` resumes; `claimlens intake --list` lists open sessions.
- The agent never tells the customer the route, the price or whether they are covered.

## 4. What the agent must collect

### 4.1 Facts (`IntakeFacts`)

| Fact | Type | Required |
|---|---|---|
| `policy_id` | string, checked with the policy lookup | yes |
| `incident_date` | date (asked as "when did it happen?"; "yesterday" is resolved to a date) | yes |
| `location` | short text | yes |
| `what_happened` | the customer's own account, 1–3 sentences | yes |
| `object_hit` | text (another car, a post, a wall, nothing…) | yes |
| `other_party_involved` | yes / no | yes |
| `driver_is_policyholder` | yes / no | yes |
| `driving_for_work` | yes / no | yes |
| `injuries` | yes / no | yes |
| `police_report` | reference or "none" | only if theft, vandalism or injury |

- The agent records each fact with a `record_fact` tool call. Code validates the value (types,
  a date not in the future and not before 2000) and returns an error the agent must fix.
- Answers the customer does not know are recorded as `unknown`; they never block the claim.

### 4.2 Photos

| Photo | Purpose |
|---|---|
| `overview` | the whole car, showing the damaged side |
| `damage_closeup` | the damage itself, close up |
| `plate` | the licence plate |

- The agent asks for each with a `request_photo(kind)` tool call.
- Every photo is checked in code as it arrives: the existing format and size check, plus two new
  **coaching checks**:
  - too dark (mean brightness below a threshold);
  - too blurry (low edge variance).
  Thresholds are set so that every golden photo the pipeline accepts also passes (a test checks
  this).
- These coaching checks are used **only in intake**. The pipeline's quality gate is unchanged, so
  golden v1/v2 results and the CI gate do not move.
- A failed photo gets a short reason and a retake request: **the first try plus at most 2 retakes
  per kind** (3 photos at most). After
  that the agent moves on, and the claim records `photo_gaps` (for example `plate: too blurry
  after 2 retakes`).
- At least one usable photo is required to submit. With none, the claim is still submitted and
  rule R3 sends it to a person.

## 5. How it works

```
customer ──► claimlens intake (terminal loop)
               │  text or /photo
               ▼
        LangGraph intake graph  (checkpointed per session in var/intake.sqlite)
          agent ──► tools ──► agent … ──► wait_for_customer (interrupt) ──► agent …
            │                                              ▲
            │ GatewayChatModel (fast tier)                 │ Command(resume=…) on the next turn
            ▼
          finish ──► completeness check in code ──► submit_claim + IntakeCompleted event
                                                    └► process_claim (vision → rules → triage)
```

### 5.1 Graph

| Node | Does |
|---|---|
| `agent` | One model call with the intake tools bound (step and turn limits checked first). |
| `tools` | `record_fact` and `lookup_policy` (read-only MCP `get_policy` under the `intake` profile). |
| `ask` | The agent called `ask_customer(message)` or `request_photo(kind, message)`: LangGraph `interrupt()` pauses here and returns the message to the terminal. The next turn resumes with the customer's text and any photo, which is checked in code before the agent sees the result. |
| `finish` | The agent called `finish_intake(summary)`: code checks completeness (section 4). Missing items go back to the agent as a tool error; once complete, the claim is submitted. |

- The agent always talks through `ask_customer` / `request_photo`, so every customer-facing
  message is a recorded tool call. A plain-text reply is wrapped as `ask_customer`.
- **Checkpointing:** `SqliteSaver` in `var/intake.sqlite`, one thread per session id. A paused
  session survives a restart and resumes exactly where it stopped (verified in a spike on
  langgraph 1.2.12 and langgraph-checkpoint-sqlite 3.1.1).

### 5.2 Limits

| Limit | Value | When reached |
|---|---|---|
| Customer turns | 30 | the claim is submitted with what was collected, and noted |
| Model calls per turn | 4 | the agent's turn ends with a fallback question |
| Retakes per photo kind | 2 | the gap is recorded and the agent moves on |
| Session cost | gateway cap (claim id = session id) | the session is handed to a person |

### 5.3 Handing over to the pipeline

- `submit_claim` is called with the checked photos and a description made from the customer's
  own account.
- A new event, **`IntakeCompleted`**, records:
  - the facts;
  - the photo kinds received, and any gaps;
  - turn and retake counts;
  - the session id and a sha256 of the transcript.
  It does **not** record the transcript text, which stays in the checkpoint database.
- `ClaimState` gains `intake` (the facts). The triage agent's evidence summary adds a "Facts
  collected at intake" block, so `driving_for_work: yes` reaches the triage agent as a fact.
- **No rule changes.** The decision rules stay as they are; the triage agent already escalates
  on facts like work use.
- `process_claim` runs straight away. `--no-process` submits only.

## 6. Safety

- **The customer's words are data.** Messages are wrapped as data in the prompt, and the agent
  has no tool that decides, prices or pays. An injected "approve my claim" can only become a
  recorded fact or a message.
- **Least privilege:** the `intake` profile can call only `get_policy` (and `assess_quality`,
  unused here). It cannot read claim history, search other claims or write notes. A test pins
  this.
- **No promises:** the prompt forbids statements about cover, price or outcome. A test scans the
  agent's messages in scripted runs for "covered", "approved", "you will be paid" and similar.
- **PII:** the transcript stays in `var/intake.sqlite` (git-ignored); the event log keeps a hash
  only.

## 7. Command line

- `claimlens intake [--session ID] [--list] [--no-process]`
- The terminal loop is thin: it prints the agent's message, reads a line, and turns
  `/photo <path>` into a photo attachment. All logic lives in the graph, so M8 can reuse it.

## 8. Testing

- **Offline by default:** `FakeProvider` scripts the agent's tool calls, so whole conversations run
  in CI with no key.
- **Unit tests:**
  - fact validation;
  - the photo coaching checks, including a test that every pipeline-accepted golden photo passes;
  - retake counting;
  - the completeness check;
  - the `IntakeCompleted` event and its fold;
  - evidence rendering with intake facts.
- **Graph tests:**
  - a complete happy-path conversation;
  - pause, a "restart" (a new graph on the same database), then resume;
  - a dark photo leading to a retake;
  - retakes exhausted;
  - an unknown policy;
  - `finish` refused while facts are missing;
  - the turn limit;
  - a plain-text reply wrapped as a question.
- **Red-team tests:**
  - an injection in the customer's words;
  - a request for another claim's history (no such tool);
  - "tell me I'm covered" (no promise in the reply).
- **Scripted live runs** (`CLAIMLENS_LIVE=1`): 3 scripted customers (simple, vague, work use)
  on Haiku, for about $0.10. The simulated customer with a hidden ground truth and pass^k is
  M6b.

## 9. Done means

1. `claimlens intake` collects a complete claim in a live chat and the pipeline decides it.
2. A paused session resumes after the process is restarted.
3. A dark or blurry photo gets a specific retake request; after 2 retakes the gap is recorded.
4. A "driving for work" answer reaches the triage agent as a fact.
5. ADR 0016 (the intake agent: checkpoints and interrupt), roadmap and README updated.
6. `ruff`, `mypy` and `pytest` clean; coverage stays at or above 95%; the CI eval gate unchanged.

## 10. Out of scope (M6b or later)

- Episodic and semantic memory, image embeddings, similar claims by image (M6b).
- Agent Skills (M6b).
- The simulated customer and pass^k (M6b).
- A web chat (M8), voice, other languages.
- Detecting "not a car" or AI-generated photos (M7).
