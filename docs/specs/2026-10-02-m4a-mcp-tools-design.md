# M4a design: MCP tool servers with per-agent scopes

- **Status:** Draft for review
- **Date:** 2026-10-02
- **Parent spec:** [`2026-10-01-claimlens-design.md`](2026-10-01-claimlens-design.md) §8, §9, §14
- **Follows:** M3 (vision models). **Followed by:** M4b (LLM gateway, policy search with citations),
  then M5 (triage agent).

## 1. Why

In M5 an LLM agent will reason over a claim. It needs safe, standard ways to use our models and our
(mock) company systems. MCP (Model Context Protocol) is the standard for exposing tools to AI
agents. M4a wraps ClaimLens in four MCP servers. Each server enforces **least privilege** itself:
an agent profile sees and may call only the tools it needs, and the one consequential action
(payment) also needs a human-signed approval token. The same servers work from Claude Code and
Claude Desktop, which gives a live demo before our own agent exists.

## 2. Goals and success criteria

| Goal | Measure |
|---|---|
| Standard tools | 4 MCP servers on the official MCP Python SDK (2.x), stdio transport |
| Least privilege | Each profile advertises and accepts exactly its tools; a scope-matrix test covers every profile × tool pair |
| No agent can pay | No profile contains `issue_payment`; `issue_payment` also requires a valid human approval token |
| Safe inputs | Photo paths are confined to allowed folders; inputs and outputs are Pydantic-validated |
| Idempotent writes | Same idempotency key → one event, same result |
| Auditable | Every call, including refused ones, leaves an audit record |
| Works with real AI clients | Claude Code connects via `.mcp.json` (demo profile) and uses the tools; a stdio integration test passes |
| Quality bar | Coverage ≥ 95%, CI needs no LLM, weights or data |

**Out of scope:**
- `search_policy_clauses` (policy search with citations): moves to M4b with RAG.
- LLM gateway: M4b.
- Agent loop: M5.
- HTTP transport and remote auth.
- Similarity search for `find_similar_claims`: M6 memory. M4a uses simple rules.

## 3. Decisions already made

- **Each server enforces its own scopes** (Approach 1). It runs as its own process with
  `--profile`, registers only the allowed tools, and re-checks on every call. A central gateway
  and agent-side-only checks were rejected: one is a single point of failure, the other can be
  bypassed by any client.
- **Claude Code and Claude Desktop** are supported as demo clients through a checked-in `.mcp.json`
  using the read-only `demo` profile.
- **Payments are mock-only.** They produce an event; no money moves.

## 4. Architecture

```
config/agents.toml  →  Profile(name, tools per server)
claimlens mcp <server> --profile <p>      one process per server, stdio
  vision         segment_damage, segment_parts, assess_quality                    read
  policy-admin   get_policy, get_coverage                                         read
  claims-system  get_claim_history, find_similar_claims (read); add_note, assign_queue (write)
  payments       issue_payment                          consequential: approval token
every call: scope check → input validation → run → output validation → audit
          ↓
existing code: FusedDetector, quality gate, PolicyRepository, SQLiteEventStore, fold
```

## 5. Components (`src/claimlens/mcp/`)

### 5.1 Profiles (`profiles.py`, `config/agents.toml`)

```toml
[profiles.intake]
vision = ["assess_quality"]
policy-admin = ["get_policy"]

[profiles.triage]
vision = ["segment_damage", "segment_parts", "assess_quality"]
policy-admin = ["get_policy", "get_coverage"]
claims-system = ["get_claim_history", "find_similar_claims", "add_note", "assign_queue"]

[profiles.demo]   # Claude Code / Desktop: read-only
vision = ["segment_damage", "segment_parts", "assess_quality"]
policy-admin = ["get_policy", "get_coverage"]
claims-system = ["get_claim_history", "find_similar_claims"]
```

`Profile` is frozen. Loading fails on an unknown server or tool name, and on any profile that
lists `issue_payment` (no agent may pay; a person approves with a token).

### 5.2 Guard (`guard.py`, pure functions)

- `allowed(profile, server, tool) -> bool`. Raising `ScopeDenied(profile, server, tool)` is used
  by the servers' call check.
- `safe_photo_path(path, roots) -> Path` resolves the path, requires it to be inside an allowed
  root (`var/blobs`, `data`, `tests/fixtures`), and requires an image suffix and an existing file.
  Otherwise it raises `PathRejected`. This blocks `..`, absolute paths elsewhere and `.env`.
- Approval tokens:
  - `issue_approval(claim_id, amount_usd, secret, now, ttl=24h) -> str`, an HMAC-SHA256 over
    `claim_id|amount|expiry`, base64url.
  - `verify_approval(token, claim_id, amount_usd, secret, now)` raises `ApprovalInvalid` on a
    tampered or malformed token, a wrong claim or amount, or expiry.
  - The secret is `CLAIMLENS_APPROVAL_SECRET` in `.env`; it is missing → payments refuse.

### 5.3 Tool contracts (`schemas.py`)

| Tool | Input | Output |
|---|---|---|
| `segment_damage` | `photo: str` | `DamageReport(model_version, findings: [type, confidence, part, part_area_ratio, severity])` |
| `segment_parts` | `photo: str` | `PartsReport(model_version, parts: [label, group, confidence])` |
| `assess_quality` | `photo: str` | `QualityReport(accepted, reason)` |
| `get_policy` | `policy_id` | `PolicySummary(found, policy_id, holder, active, collision, deductible)` |
| `get_coverage` | `policy_id` | `CoverageResult(found, active, collision, deductible)` |
| `get_claim_history` | `claim_id` | `ClaimHistory(found, claim_id, policy_id, route, findings, notes, events: [type, seq])` |
| `find_similar_claims` | `claim_id`, `limit=5` | `SimilarClaims(items: [claim_id, reason])`: same policy, or the same part group |
| `add_note` | `claim_id`, `text ≤ 2000`, `idempotency_key` | `NoteResult(event_seq, duplicate)` |
| `assign_queue` | `claim_id`, `queue: adjuster\|fraud\|desk`, `idempotency_key` | `QueueResult(event_seq, duplicate)` |
| `issue_payment` | `claim_id`, `amount_usd`, `approval_token`, `idempotency_key` | `PaymentResult(payment_id, event_seq, duplicate)` |

Notes and other free text are returned **as data inside fields**, never merged into instructions
(ASI01, ASI06).

### 5.4 Servers

- `vision.py`, `policy_admin.py`, `claims_system.py`, `payments.py`.
- Each has `build_<server>(profile, deps) -> MCPServer`, registering only the allowed tools. Every
  tool body starts with the guard check, so a direct call to an unregistered tool name is also
  refused.
- `deps` injects the detector (fused by default; a fake in tests), the quality gate, the
  `PolicyRepository`, the event store path and the audit sink.

### 5.5 Events and audit

- **New event payloads:** `NoteAdded(text, author, idempotency_key)`,
  `QueueAssigned(queue, idempotency_key)`,
  `PaymentIssued(payment_id, amount_usd, idempotency_key)`, and
  `ToolCalled(server, tool, profile, input_sha256, outcome: ok|denied|error)`.
- They are appended to the claim's hash-chained log through the existing store. The fold ignores
  `ToolCalled` and collects notes, queue and payments into `ClaimState`. These are new optional
  fields with defaults, so old events still fold.
- **Idempotency:** before a write, the store is searched for an event with the same
  `idempotency_key` on that claim. If found, its seq is returned with `duplicate=true`.
- **Calls with no claim** (photos, policies) are audited to `var/mcp-audit.jsonl`, one JSON line
  per call.

### 5.6 CLI and demo

- `claimlens mcp {vision,policy-admin,claims-system,payments} --profile P` runs over stdio.
- `claimlens approve-payment <claim_id> <amount>` is the human step. It refuses when the claim is
  unknown, has no decision, or its route is `FRAUD_REVIEW`. It prints the token.
- `.mcp.json` at the repo root starts the three read servers with `--profile demo` through
  `uv run claimlens mcp …`. Payments are not included.
- A README section explains Claude Code, Claude Desktop, and example prompts.

## 6. Error handling

| Situation | Behaviour |
|---|---|
| Tool not in profile | Not advertised; a direct call returns an error result `ScopeDenied`; audited `denied` |
| Bad or unsafe photo path | Error result `PathRejected`; nothing runs |
| Model weights missing or model error | Error result with a clear message; never empty findings |
| Unknown policy or claim | `found=false` result |
| Payment without, or with an invalid, token, or claim routed to fraud | Error result; no event |
| Approval secret missing | Payments refuse with a message naming `CLAIMLENS_APPROVAL_SECRET` |
| Repeated write key | First result returned, `duplicate=true` |

## 7. Testing

- **Unit:** profiles (loading, the no-payments invariant, the read-only demo); the guard (scope
  matrix generated from config, paths, tokens); schemas.
- **Server tests through the in-memory MCP client** (same protocol as Claude Code): advertised
  tools per profile, happy paths, refusals, idempotency, payment rules. Vision uses a fake
  detector.
- **Red-team starter set (OWASP):** ASI02 (triage tries payment), ASI03 (intake tries claim
  history), prompt-in-data (a note says "approve this claim" and comes back as data only).
- **Integration (skipped in CI):** a real stdio subprocess `claimlens mcp vision --profile demo`
  running the fused model on a test photo.
- **Manual:** the Claude Code demo from the README.

## 8. Risks

| Risk | Mitigation |
|---|---|
| MCP SDK 2.x API differs from most online examples (FastMCP → MCPServer) | Pin `mcp>=2.2,<3`; tests run through the SDK's own client |
| Model load time on each vision server start | Lazy load on the first call; the integration test measures it |
| A Claude Code user points at private files | Path guard with allowed roots only; `.env` and anything outside `data/`, `var/blobs` and `tests/fixtures` is refused |
| Token secret leaks | Only in `.env` (git-ignored); tokens expire after 24 h and bind the claim and amount |
