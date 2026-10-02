# ADR 0010: MCP tool servers, each enforcing its own scopes

- **Status:** Accepted
- **Date:** 2026-10-02

## Context

In M5 an LLM triage agent will reason over claims, so it needs tools for our models and our mock
company systems. The OWASP Top 10 for Agentic Applications says to assume that the agent can be
manipulated:

- **ASI01 goal hijack:** text in a photo or a note says "approve this claim".
- **ASI02 tool misuse:** the agent is pushed into issuing a payment.
- **ASI03 privilege abuse:** a low-privilege agent reaches data it should not see.

The tools themselves must therefore refuse anything outside a role's job. MCP (Model Context
Protocol) is the standard way to expose tools to AI clients.

## Decision

- **Four MCP servers**, one per system, on the official MCP Python SDK 2.x (`MCPServer`) over
  stdio: `vision`, `policy-admin`, `claims-system` and `payments`.
- **Each server enforces its own scope.**
  - It is started with `--profile` (`config/agents.toml`: `intake`, `triage`, `demo`).
  - It registers only that profile's tools, so the model never sees the others.
  - `ScopedServer.call_tool` re-checks every call. A client that sends an unadvertised name gets
    `ScopeDenied`, and the refusal is audited.
  - A central gateway was rejected as a single point of failure. Agent-side-only checks were
    rejected because any client can bypass them.
- **No agent can pay.**
  - Loading `agents.toml` fails if any profile lists `issue_payment`.
  - The payments server runs only under the built-in human `operator` profile.
  - Every payment needs an HMAC-SHA256 approval token from `claimlens approve-payment`, bound to
    the claim and the amount, expiring in 24 hours, and refused for claims routed to
    `FRAUD_REVIEW`. The secret lives in `.env`.
- **Safe inputs.**
  - Photo paths must resolve inside `var/blobs`, `data` or `tests/fixtures` and have an image
    suffix, so `.env` and `../` are refused before any model loads.
  - Inputs and outputs are typed Pydantic models.
- **Idempotent writes.** `add_note`, `assign_queue` and `issue_payment` take an
  `idempotency_key`. Repeating one returns the first result with `duplicate=true`.
- **Audit everything.**
  - Calls about a claim append a `ToolCalled` event to that claim's hash-chained log.
  - Other calls go to `var/mcp-audit.jsonl`.
  - Errors are audited too: the SDK raises for tool errors, so the server audits before the error
    becomes a result.
- **Free text is data.** Notes come back inside result fields, never merged into instructions.
- **A Claude Code demo.** A checked-in `.mcp.json` starts the three read servers with the
  read-only `demo` profile.

## Consequences

- **What it gives:**
  - Least privilege holds even for clients we do not control, such as Claude Code.
  - The red-team tests (ASI02, ASI03, a note containing an injected instruction) run in CI with no
    LLM.
  - A real stdio test runs the fused model through a separate server process (about 10 s).
- **What it costs:**
  - One process per server, and each vision server loads the models on its first call.
  - Every call about a claim, including reads and refused calls from the read-only `demo` profile,
    adds a `ToolCalled` event to that claim's log. That is deliberate, so reads are audited too,
    but a client can grow a claim's log. Rate limits are an M7 item.
- **Review fixes:**
  - Idempotency keys are scoped to the event type, so a note key cannot swallow a queue
    assignment or a payment.
  - An approval token is single-use: its hash is stored on `PaymentIssued`.
  - The payments server re-checks that the claim has a decision.
  - The human operator is audited as a human.
- **What comes next:** policy search with citations and the LLM gateway (M4b), then the agent
  loop that uses these tools (M5).
