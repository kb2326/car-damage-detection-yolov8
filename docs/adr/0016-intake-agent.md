# ADR 0016: A checkpointed intake agent that collects, never decides

- **Status:** Accepted
- **Date:** 2026-10-03

## Context

Until M6 a claim started as a one-shot form: a policy number, a description and photos. Nothing
asked for what was missing, and nothing told the customer a photo was too dark. A real first
notice of loss is a conversation, and it can stop for hours or days (the customer is at the
roadside, or has to find the licence plate).

## Decision

- **A LangGraph intake graph** (`claimlens.intake_agent`):
  - `agent` makes one model call through `GatewayChatModel` on the `fast` tier (Haiku);
  - `act` records facts and looks up the policy;
  - `ask` pauses with `interrupt()` whenever the agent speaks to the customer;
  - `finish` checks completeness in code and hands the claim over.
- **Pausing is LangGraph's checkpointer** (`SqliteSaver` on `var/intake.sqlite`, one thread per
  session). A paused session resumes exactly where it stopped after a restart (tested). This is
  the M6 use that ADR 0013 chose LangGraph for. The claim's event log stays the record of the
  claim; the checkpoint holds the conversation.
- **Facts are data, validated in code:**
  - ten facts (policy, date, place, what happened, what was hit, other party, driver, work use,
    injuries, and a police report when relevant);
  - "yesterday" and "3 days ago" become dates;
  - yes/no answers are normalised, and "unknown" is allowed except for the policy and the story;
  - the policy is checked through the `intake` MCP profile (`get_policy` only).
- **Photo coaching, intake only:** each photo gets the pipeline's format and size check plus "too
  dark" (mean brightness < 40) and "too blurry" (edge variance < 100).
  - The thresholds sit well below every golden photo the pipeline accepts (darkest 72, least
    sharp 181), and a test checks all 94.
  - Each photo gets the first try plus two retakes; then the gap is recorded and the chat moves
    on.
  - The pipeline's own quality gate is unchanged.
- **Hand-over:**
  - `submit_claim` files the claim. A claim with no usable photo is still filed, and rule R3
    sends it to a person.
  - An `IntakeCompleted` event records the facts, photo kinds, gaps, turn and retake counts, and
    a sha256 of the transcript. The transcript text stays out of the log.
  - The triage agent's evidence summary shows the facts as data.
  - No rule changed.
- **No promises:** the prompt forbids statements about cover, cost or outcome, and code replaces
  any outgoing message that matches "you're covered", "approved", "will be paid", "we'll pay" or
  "guarantee" with a neutral line.
- **Limits:** 30 customer turns (then the claim is filed with what was collected), 4 model calls
  per turn (then a fallback question), and a per-session spending cap (claim id
  `intake-<session>`).
- **Interface:** `claimlens intake` in the terminal: `/photo <path>`, `/quit` to pause,
  `--session` to resume, `--list` to see paused sessions. `IntakeSessions` (start, reply, pending,
  open sessions) is the reusable front end for the M8 web app.

## Consequences

- **Live check (Haiku, 2026-10-03):** three scripted customers (simple, vague, driving for work)
  each completed intake in 9 to 13 turns, with every required fact and all three photos, and
  `driving_for_work` correct. Cost: $0.18 for 61 calls.
- The triage agent now sees facts such as "driving for work: yes" directly, instead of having to
  find them in free text.
- **Limits:**
  - photo coaching does not detect "not a car" or AI-generated images (M7);
  - the scripted customers are keyword responders. The simulated customer with a hidden truth,
    and pass^k, are M6b.
