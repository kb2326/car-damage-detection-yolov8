# ADR 0013: LangGraph for the agents, on our own gateway and tools

- **Status:** Accepted
- **Date:** 2026-10-03
- **Supersedes:** ADR 0007 (agent framework bake-off)

## Context

ADR 0007 planned to build the triage agent twice, once as our own loop and once on LangGraph,
and keep the better one. Before M5 the owner chose LangGraph directly. The M6 intake agent is a
conversation that pauses for days and resumes, which is what LangGraph's checkpoints and
`interrupt()` are for, and one framework for both agents is simpler than two.

What had to stay true whatever the framework:

1. Every model call goes through our gateway (ADR 0011), or the cost caps and the call log stop
   meaning anything.
2. Tools come from our MCP servers with the agent's profile (ADR 0010), so scopes and the audit
   trail stay in force.
3. The claim's hash-chained event log stays the only record of a claim.

## Decision

- **LangGraph `StateGraph`** with `MessagesState`, LangGraph's `ToolNode` and conditional edges.
  We write the graph ourselves instead of using the prebuilt agent, so the checks and limits are
  explicit nodes that tests can reach:
  - `agent` (one model call, with the step and time limits; the last step forces
    `submit_recommendation`),
  - `tools` (`ToolNode`; a tool error goes back to the model as an error result),
  - `nudge` (a plain-text reply gets one reminder),
  - `finalize` (validates the answer, the clause ids and the evidence labels; one repair).
- **The model is always `GatewayChatModel`,** a LangChain `BaseChatModel` whose `_generate` calls
  `Gateway.generate`. The gateway learned tool use for this (`ToolSpec`, `ToolCall`,
  `ToolResult`, `tool_choice`). No LangChain provider package is installed.
- **Tools come through our own MCP adapter** (`agent/tools.py`, about 50 lines on
  `mcp.client.Client`). `langchain-mcp-adapters` 0.3.1 imports `mcp.server.fastmcp`, which MCP SDK
  2.x removed. The adapter binds `claim_id` and `policy_id` and hides them from the model, so the
  agent can read only its own claim and its claimant's policy.
- **Read-only tools:** policy lookup, coverage, policy search, claim history, similar claims.
  `config/agent.toml` refuses write tools at load time.
- **Citations are checked in code:** every clause must exist **and** belong to the claimant's
  wording; evidence is cited by labels (`E1`) that code maps back to event ids.
- **Limits:** 6 model calls, 90 seconds, 1 repair, 700 output tokens per call, and the gateway's
  caps ($0.10 per claim, raised from $0.03 by the owner for a tool-using agent).
- **A failed tool blocks a fast-track:** if any tool call failed during the review (for example
  the policy index is missing), `finalize` refuses `FAST_TRACK`, because something went
  unchecked. The agent also opens the policy index before the first model call, so a missing
  index fails the claim to a person at once.
- **Any failure raises,** the workflow records a `StageFailed`, and rule R2 sends the claim to a
  person. No new routing code, and no failure can fast-track a claim.
- **No checkpointer in M5a.** A triage run is short and has no human pause. Checkpoints and
  `interrupt()` arrive with the M6 intake agent.
- Only `src/claimlens/agent/` imports `langgraph` and `langchain_core` (group `agent`).

## Consequences

- **What we gave up:** a measured comparison with a plain loop. The ADR 0007 write-up for the
  final post becomes "why we chose" instead of "what the numbers said".
- **Results** (fused detector, decision policy v1, R6 = 0.65):

  | Run | Route accuracy | Escalation recall | Correct fast-tracks | Cost |
  |---|---|---|---|---|
  | Stub agent, golden v1 (97) | 0.78 | 1.00 | 9 of 30 | $0 |
  | LLM agent, dev subset (20) | 0.70 | 1.00 | 0 of 6 | $0.20 |
  | LLM agent, golden v1 (97) | 0.70 | 1.00 | 1 of 30 | $0.91 |

- **The agent is much more cautious than the stub.** On golden v1 it held back 8 claims the
  stub fast-tracked (all through R7: low confidence or open questions). In 6 of them it found
  **impossible vision output**: a flat tyre "on the front bumper" or "on the trunk", glass
  "on the hood" or "on the front bumper", and damage covering 108% to 298% of a part. The other
  two had a rejected photo, a wide estimate, or no damage found. These are real defects in the
  M3 fusion step (a damage type matched to a part it cannot be on, and part ratios above 1)
  that the stub and the rules could not see.
- **So the 0.78 → 0.70 drop is mostly the agent being right.** Golden v1 labels come from rules
  applied to the true annotations, so they count these catches as errors. Fixing fusion (part
  ratio capped and checked, damage types restricted to plausible parts) is the way to win the
  fast-tracks back, not a less careful prompt. M5b's narrative cases and the LLM judge measure
  what the agent adds.
- **Cost is well under the cap:** $0.009 per claim on average ($0.013 at most), 2.0 model calls
  and 0.7 tool calls per claim, no agent failures. Prompt caching cuts the second call's input
  cost.
- **The workflow retries a failed stage once,** so a failing agent can run twice. Spend stays
  bounded by the claim cap, and the response cache answers identical first calls.
- **Tools run with `anyio.run` inside LangGraph's sync nodes.** That works for today's sync
  callers. If M6 runs the graph asynchronously, the adapter must switch to async tools.
- **Replacing the framework** means rewriting `agent/graph.py` and `agent/chat_model.py`; the
  gateway, tools adapter, evidence, checks and workflow do not depend on LangGraph.
