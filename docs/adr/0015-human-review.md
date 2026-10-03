# ADR 0015: Human review is an event, and only a person can deny

- **Status:** Accepted
- **Date:** 2026-10-03

## Context

Rules route a claim to `FAST_TRACK`, `ADJUSTER_REVIEW` or `FRAUD_REVIEW`. Until M5b nothing
recorded what the adjuster then did, so the audit trail stopped at the rule, and a payment could
be approved for a claim no person had looked at.

## Decision

- **A new event, `HumanReviewed`:** reviewer name, action, a final route for an override, and a
  note. It is appended to the claim's hash-chained log with a `human` actor.
- **Four actions:** `approve`, `override` (needs a route and a note), `deny` (needs a note) and
  `request_info` (keeps the claim in the queue).
- **Deny exists only here.** No model, agent or rule can produce it, so "the system never
  auto-denies" is now enforced by the data model, not only by convention.
- **One payment rule, `payable(state)`,** used by both `claimlens approve-payment` and the MCP
  `issue_payment` tool:
  - a fast-tracked claim can be paid unless a reviewer denied it;
  - a claim on `ADJUSTER_REVIEW` waits for an `approve` or `override`;
  - a claim on `FRAUD_REVIEW` is never paid through this path, whatever the review says;
  - a denied claim is never paid.
- **Command line:** `claimlens queue [--route R]` lists waiting claims, oldest first, with the
  rule, the agent's rationale, its citations and its open questions.
  `claimlens review-claim <id> --approve | --override R | --deny | --request-info --reviewer NAME
  [--note TEXT]` records the decision. (`claimlens review` was already the FiftyOne labelling
  command.)
- **An override does not rewrite `RouteDecided`.** The rule's route and the person's route are both
  kept on the log, which is what an auditor needs.

## Why not LangGraph `interrupt()`

`interrupt()` pauses a running graph until a person answers. The triage agent's run is short and
finishes before any person looks at the claim; the review happens later, on the claim, through the
event log. `interrupt()` fits the M6 intake conversation, where the graph itself must wait.

## Consequences

- Every decision about money now ends with a named person on the audit trail.
- A web screen for the queue comes with the M8 demo; the command line is enough for now.
