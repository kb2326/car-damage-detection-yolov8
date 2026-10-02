# ADR 0007: Choose the agent framework by a bake-off

- **Status:** Proposed (decided in M5)
- **Date:** 2026-10-02

## Context

Frameworks such as LangGraph and Google ADK provide agent loops, state, checkpoints and tracing.
Our own loop is small and fully auditable. Which is better for a regulated claims workflow is
an empirical question, not one to settle up front.

## Decision

- In M5, build the triage agent twice behind the existing `TriageAgent` interface: our own loop and
  a LangGraph implementation of the same agent.
- Run both on the same golden set and compare route accuracy, escalation recall, cost per claim,
  latency, and how easy each trace is to audit.
- Build the M6 intake agent (conversational, long-running) with the winner, likely LangGraph.
- ADK stays optional, for the A2A repair-shop partner agent in M8.

## Consequences

- The decision rules stay outside any framework, so "rules decide" holds whichever agent wins.
- The comparison itself becomes a write-up for the final LinkedIn post.
