# ADR 0003: Event-sourced claim state

- **Status:** Accepted
- **Date:** 2026-10-08

## Context

Claims are long-running: a claim may wait days for photos or an adjuster. Every routing decision
must be explainable to an auditor, and agent behaviour must be replayable for debugging and
evaluation.

## Decision

Store each claim as an append-only sequence of typed events in SQLite. Each event records the
SHA-256 of the previous event, forming a hash chain that is verified on every read. Claim state is
never stored; it is rebuilt with `fold(events)`. Workflow and agent steps read state and only
append new events. Corrections are new events, never edits.

## Consequences

- One mechanism gives durability (resume after a crash), working memory, an audit trail and replay.
- Every new piece of state needs an event type and a fold rule.
- Folding the full log on every step is O(n) per step; acceptable at tens of events per claim.
- The hash chain detects tampering but does not prevent someone with database access from
  rewriting the whole chain; anchoring chain heads externally is out of scope for now.
