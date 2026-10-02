# M5a: LLM Triage Agent on LangGraph (design)

- **Status:** Draft for owner review
- **Date:** 2026-10-02
- **Parent design:** [`2026-10-01-claimlens-design.md`](2026-10-01-claimlens-design.md)
- **Builds on:** ADR 0010 (MCP tools and scopes), ADR 0011 (LLM gateway), ADR 0012 (policy search)

## 1. Purpose

Replace the placeholder triage agent with an LLM agent that reads the evidence on a claim, looks
things up with the M4 tools, and writes a recommendation with reasons and citations.

The principle does not change: **models measure, the agent reasons, rules decide.** The agent
advises. Rules R1–R9 decide. Nothing in M5a can deny or pay a claim.

## 2. Decisions already made by the owner

- **Framework: LangGraph.** The bake-off planned in ADR 0007 is dropped. The M6 intake agent
  (multi-turn, pauses for days) needs LangGraph's pause-and-resume, so we use one framework for
  both agents. ADR 0013 records this and supersedes ADR 0007.
- **Split:** M5a builds the agent. M5b measures it (150 golden claims, LLM judge, CI gates,
  tracing, human review queue).
- **Spending while developing:** a fixed 20-claim subset; full runs only at the end.

## 3. What the agent adds, and what it cannot change

The stub agent says "fast-track" whenever any damage was found. Rules R1–R6 run before the
agent's advice is read, so the agent's advice matters only for claims that already passed them.

- The LLM agent can make the outcome **more cautious** (send a claim to a person, with a reason
  and a cited clause).
- It cannot fast-track a claim the rules would stop.
- So escalation recall cannot fall below the stub's. The gate stays: **escalation recall = 1.00
  on golden v1.**
- Its value is in what rules cannot see: a story that does not match the damage, or a story that
  triggers a policy exclusion (racing, commercial use, a late report). Golden v1 has few such
  cases; M5b adds them.

## 4. Architecture

```
ClaimState ──► LangGraphTriageAgent.recommend(state)
                 │
                 ├─ render_evidence(state)  → one user message (evidence as data)
                 │
                 └─ LangGraph graph
                      agent ──► tools ──► agent ──► … ──► finalize ──► END
                        │                                   │
                        │  GatewayChatModel                 │  checks schema, clause ids,
                        ▼                                   │  evidence ids; one repair
                      Gateway (caps, cache, log)            ▼
                        ▼                              AgentRecommendation
                      Anthropic API
```

### 4.1 Components

| Unit | File | Purpose |
|---|---|---|
| Tool use in the gateway | `llm/types.py`, `llm/gateway.py`, `llm/provider.py`, `llm/wire.py`, `llm/anthropic_provider.py` | Requests carry tool definitions; replies carry tool calls. Caps, cache and log apply as before. |
| `GatewayChatModel` | `agent/chat_model.py` | A LangChain `BaseChatModel` whose `_generate` calls our gateway. LangGraph never talks to Anthropic directly. |
| MCP tool adapter | `agent/tools.py` | Turns the tools of our `ScopedServer`s into LangChain tools through the MCP in-memory client. |
| Evidence renderer | `agent/evidence.py` | Deterministic text summary of `ClaimState` with event ids. |
| Graph | `agent/graph.py` | Nodes `agent`, `tools`, `nudge`, `finalize`; limits; citation checks. |
| Agent | `agent/langgraph_agent.py` | Implements the existing `TriageAgent` protocol. |
| Config | `config/agent.toml`, `agent/config.py` | Tier, prompt id, tool allowlist, limits. |
| Prompt | `prompts/triage/v1.md` | Versioned system prompt. |

### 4.2 Why these choices follow LangGraph's own standards

- **`StateGraph` with `MessagesState`,** `ToolNode` for tool execution, and conditional edges.
  This is the documented shape of a tool-calling agent. We write the graph ourselves instead of
  using the prebuilt agent so the `finalize` checks and limits are explicit and testable.
- **Custom chat model:** LangChain's documented extension point is subclassing `BaseChatModel`
  (`_generate`, `_llm_type`, `bind_tools`). It reports `usage_metadata` on each `AIMessage`.
- **Step limit:** our own counter in graph state, with LangGraph's `recursion_limit` as a
  backstop (`GraphRecursionError` is caught).
- **No checkpointer in M5a.** A triage run is short and has no human pause. Our event log stays
  the only record of a claim. LangGraph's checkpointer and `interrupt()` arrive in M6, where the
  conversation must pause and resume.
- **Verified versions (2026-10-02):** langgraph 1.2.12, langchain-core 1.6.6,
  langgraph-prebuilt 1.1.0.

### 4.3 A finding that shapes the design

`langchain-mcp-adapters` 0.3.1 imports `mcp.server.fastmcp`, which no longer exists in MCP SDK
2.x (we use 2.2). So we write our own adapter, about 50 lines, on `mcp.client.Client`. It keeps
scopes and the audit trail in force, because every call still goes through `ScopedServer`.

## 5. Tool use in the gateway

- `ToolSpec(name, description, input_schema)`, `ToolCall(id, name, arguments)`,
  `ToolResult(tool_call_id, content, is_error)`.
- `Message` gains `tool_calls` (assistant) and `tool_results` (user). `content` may be empty when
  either is present.
- `LLMRequest` gains `tools` and `tool_choice` (a tool name that the model must call).
- `LLMResponse` and the cached reply gain `tool_calls`.
- `tools` and `output_schema` cannot be combined: with tools, the structured answer is itself a
  tool call (`submit_recommendation`).
- The cache key, the prompt hash and the worst-case cost estimate include the tool definitions
  and the tool calls and results in the messages.
- `llm/wire.py` holds the pure mapping to and from Anthropic's block format, so it is unit-tested
  without the SDK.

## 6. The agent's tools

Read-only, through the `triage` profile, limited further by an allowlist in `config/agent.toml`:

| Tool | Server | Use |
|---|---|---|
| `get_policy`, `get_coverage` | policy-admin | Confirm cover and deductible |
| `search_policy_clauses` | policy-admin | Find the wording to cite |
| `get_claim_history` | claims-system | Earlier events on this claim |
| `find_similar_claims` | claims-system | Earlier claims with similar photos |
| `submit_recommendation` | (the graph itself) | The final structured answer |

- The agent gets **no write tools** (`add_note`, `assign_queue`) and no vision tools. The
  workflow already ran the models; an agent that re-runs them would cost time and add nothing.
- The MCP servers are built in-process against the claim's own store, so tool calls are audited
  on the claim's log (`ToolCalled`).
- **Bound arguments:** the adapter fills in `claim_id` and `policy_id` itself and hides them from
  the model. The agent can therefore read only its own claim and its own claimant's policy, and
  cannot be talked into looking up another one.

## 7. The graph

**State:** messages, `steps` (model calls so far), `repairs`, `recommendation`.

| Node | Does |
|---|---|
| `agent` | Checks the step and time limits, then calls the model with the tools bound. On the last allowed step it forces `submit_recommendation` with `tool_choice`. |
| `tools` | LangGraph `ToolNode`. A tool error goes back to the model as an error message, not an exception. |
| `nudge` | The model replied with plain text and no tool call: remind it to call `submit_recommendation`. Counts as a repair. |
| `finalize` | Validates the submitted recommendation. Valid → END. Invalid and a repair is left → send the problems back as the tool result. Otherwise → fail. |

**`finalize` checks:**
1. The arguments match the `Recommendation` schema.
2. Every cited clause id exists **and belongs to the claimant's policy wording**.
3. Every cited evidence id is one of the labels in the evidence message (`E1`, `E2`, ...). The
   agent sees short labels, not raw event ids; the code maps them back.
4. A recommendation other than `FAST_TRACK` gives at least one reason.

**Limits** (`config/agent.toml`):

| Limit | Value | When reached |
|---|---|---|
| `max_steps` (model calls) | 6 | `AgentFailed` |
| `max_seconds` (wall clock) | 90 | `AgentFailed` |
| `max_repairs` | 1 | `AgentFailed` |
| `max_tokens` per call | 700 | (clamped by the gateway) |
| Cost per claim | gateway cap | `BudgetExceeded` |

## 8. Failure handling: always to a person

`recommend()` raises on any failure: `AgentFailed`, `BudgetExceeded`, `LLMUnavailable`,
`InvalidModelOutput`, or a tool-server error. The workflow already turns an exception in the
agent stage into `StageFailed("agent")`, and rule R2 then routes the claim to
`ADJUSTER_REVIEW`. No new routing code is needed, and no failure can lead to a fast-track.

## 9. The recommendation

```python
class Recommendation(BaseModel):  # what the model submits
    route_suggestion: Route  # FAST_TRACK, ADJUSTER_REVIEW or FRAUD_REVIEW
    confidence: Confidence  # low, medium, high
    rationale: str  # 1 to 4 sentences
    evidence_ids: list[str]  # labels from the evidence message, e.g. E1
    policy_citations: list[str]  # clause ids, e.g. STD-8.5
    open_questions: list[str]
```

- `AgentRecommendation` gains `policy_citations: tuple[str, ...] = ()`. The default keeps every
  earlier event readable.
- `citations` continues to hold evidence event ids.

## 10. Untrusted text

The claimant's description, tool results and policy text are data, never instructions.

- The evidence message wraps the description in `<claimant_description>` tags, and the system
  prompt says that text inside tags is data.
- The agent has no write tools, so an injected instruction has nothing to act with.
- Even a fully hijacked agent can only return a recommendation; rules R1–R6 still apply.
- Red-team tests cover: an instruction in the description, an instruction in a tool result, a
  made-up clause id, and a clause from another wording.

## 11. Cost (needs the owner's approval)

A tool-using agent makes about 3 to 5 model calls per claim, each re-sending the growing
conversation. The current cap of $0.03 per claim allows about two calls.

- **Proposed:** raise `per_claim_usd` to **$0.10**. The daily cap stays at $1.00 for normal use.
- `eval-triage` gets `--llm-daily-cap` so an evaluation run can raise the daily cap for that run
  only (default: the config value).
- **Expected spend for M5a:** 20-claim subset, at most $2.00 a run (a few runs); one full run of
  97 claims, at most $9.70. Expected total under $15, likely about half. Re-runs are not free: tool
  results contain run-specific data, so the response cache rarely hits across runs.

## 12. Command line

- `claimlens run --agent stub|llm` and `claimlens eval-triage --agent stub|llm`. The default
  stays `stub`, so nothing needs a key unless asked.
- `claimlens eval-triage --cases FILE` runs only the listed case ids
  (`evals/golden/v1/dev20.txt`, a fixed stratified subset).
- The report gains an agent section: model calls per claim, tool calls per claim, cost per claim,
  failures by reason, and how often each rule decided.

## 13. Testing

- **Offline and free by default.** `FakeProvider` scripts tool calls and answers, so the whole
  graph runs in CI with no key. The MCP servers run in-process on a temporary store, with the
  fake embedder for policy search.
- **Unit tests:** tool mapping (`wire.py`), gateway tool use (cache, budget, log), the chat model
  (message conversion, usage, `bind_tools`), the tool adapter (scopes still enforced, errors
  returned as text), the evidence renderer, each `finalize` check, each limit.
- **Graph tests:** happy path; tool error then recovery; plain-text reply then nudge; invalid
  citation then repair; second invalid answer → `AgentFailed`; step limit; time limit; budget.
- **Workflow test:** an agent failure becomes R2 `ADJUSTER_REVIEW`.
- **Red-team tests:** section 10.
- **Live tests** (`CLAIMLENS_LIVE=1`): one real claim; then the 20-claim subset; then golden v1.
- **Dependencies:** group `agent` (`langgraph`). CI installs it.

## 14. Done means

1. `claimlens run --agent llm` produces a cited recommendation on a real claim.
2. Golden v1 with the LLM agent: **escalation recall = 1.00**; route accuracy and fast-tracks
   reported next to the stub's (0.78, 9 of 30).
3. Every cited clause exists and belongs to the right wording (checked in code, 100%).
4. Cost per claim and model calls per claim are reported.
5. ADR 0013 (LangGraph; supersedes ADR 0007), roadmap and README updated.
6. `ruff`, `mypy`, `pytest` clean; coverage stays at or above 95%.

## 15. Out of scope (M5b or later)

- 150 golden claims, the LLM judge, CI eval gates, tracing, the human review queue (M5b).
- Checkpointing, `interrupt()` and multi-turn conversation (M6).
- Agent write actions (`add_note`, `assign_queue`): after M5b, once quality is measured.
