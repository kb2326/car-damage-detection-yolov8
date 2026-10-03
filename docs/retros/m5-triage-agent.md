# M5 retrospective: the LLM triage agent (M5a, fusion fix, M5b)

- **Dates:** 2026-10-02 to 2026-10-03
- **Plans:** [`m5a-triage-agent`](../plans/2026-10-02-m5a-triage-agent.md),
  [`m5b-agent-evaluation`](../plans/2026-10-02-m5b-agent-evaluation.md)
- **Decisions:** ADR 0013 (LangGraph), ADR 0009 update (fusion), ADR 0014 (eval gate and judge),
  ADR 0015 (human review)

## What we shipped

- **M5a: the agent** (PR #13).
  - A LangGraph agent reads a claim's evidence, looks things up with read-only MCP tools and
    recommends a route with reasons and cited clauses.
  - Every model call goes through our gateway, which learned tool use.
  - Code checks that every clause exists and is in the customer's own wording.
  - Any failure sends the claim to a person (rule R2).
  - Default is still the rule-based stub (`--agent llm` turns the agent on).
- **The fusion fix** (PR #14). Part ratio is now overlap ÷ part area (never above 100%), and a damage
  type can only be on a plausible part (a flat tyre on a wheel, a broken lamp on a light).
- **M5b: measuring it.**
  - Golden v2: 150 cases, including 53 story cases.
  - Agent quality scored by code, and an LLM judge with a labelling page and an agreement score.
  - A CI gate on a committed scorecard, and opt-in tracing to Phoenix.
  - A human review queue in which only a person can deny.
- 587 tests, 97% line coverage, strict typing.

## Numbers

| Run | Route accuracy | Escalation recall | Fast-tracks / harmless stories | Cost per claim |
|---|---|---|---|---|
| Stub, golden v1 (97) | 0.78 | 1.00 | 9 of 30 | $0 |
| LLM agent, golden v1 | 0.70 | 1.00 | 1 of 30 | $0.009 |
| LLM agent, golden v2 (150) | 0.76 | 1.00 | 3 of 10 harmless stories | $0.010 (list price) |

On golden v2 the agent catches **43 of 43** stories that must go to a person, always citing the
right clause, with no failures. The LLM judge passes 0.77 of recommendations (not validated: the
owner skipped labelling). API spend for all of M5: about $5.

## What went well

- **The agent found a real model defect.** Its reasons for holding claims back pointed at
  impossible vision output (a flat tyre "on the front bumper", damage covering 298% of a part).
  Reading the agent's reasons, not just its score, turned a "worse" number into a fix.
- **The safety gate held everywhere.** Escalation recall stayed 1.00 on every run: the stub, the
  agent, both golden sets, before and after the fusion fix.
- **Independent reviews caught money bugs before merge:**
  - a missing policy index could still end in a fast-track;
  - a reviewer's override to fraud would have unlocked payment;
  - the gate accepted partial runs;
  - cached answers made a run look cheaper than it was.
- **Cheap runs.** A full 150-claim evaluation costs about $1, and the response cache makes repeats
  nearly free.

## What did not

- **The agent is too cautious with harmless stories** (3 of 10 fast-tracked). Golden v1's
  generic descriptions ("Damage reported after a low-speed collision") make it ask questions
  instead of passing claims. This is the main number to raise next, by prompt changes measured
  by the gate.
- **Only 9 golden photos pass rules R1–R6,** so the 53 story cases reuse them heavily (36 use the
  same dent photo).
- **The plan carried a bug** (overrides and payments) that the implementation copied faithfully.
  The review caught it; plans need the same scrutiny as code.
- **Shell editing slips:** several Python edits through shell heredocs turned `\n` into real line
  breaks. The commit hooks caught each one, but it cost time. Exact-string edits are safer.
- **Tracing was not checked against a live Phoenix**, only with an in-memory exporter.

## Open items

- Validate the judge: label 50 items (`claimlens judge export`, then `judge agreement`).
- Spot-check the 53 story cases (written by Claude, not reviewed by a person).
- Raise the harmless-story pass rate with a new prompt version (`prompts/triage/v2.md`); the gate
  will hold recall and the catch rate while it changes.
- Try a traced run in Phoenix (`uvx arize-phoenix serve`, then `CLAIMLENS_TRACING=1`).
- Update the ClaimLens Explained page for M5.

## Next: M6

The intake agent (a conversation that pauses and resumes with LangGraph checkpoints and
`interrupt()`), and memory: claim history and similar claims in LanceDB.
