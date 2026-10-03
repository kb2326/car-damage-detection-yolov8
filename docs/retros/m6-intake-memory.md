# M6 retrospective: the intake agent, memory and skills (M6a + M6b)

- **Dates:** 2026-10-03
- **Plans:** [`m6a-intake-agent`](../plans/2026-10-03-m6a-intake-agent.md),
  [`m6b-memory-skills`](../plans/2026-10-03-m6b-memory-skills.md)
- **Decisions:** ADR 0016 (intake agent), ADR 0017 (memory, skills, simulated customers)

## What we shipped

- **M6a, the intake agent** (PR #16).
  - `claimlens intake` is a chat that collects 10 validated facts and 3 photos, and coaches
    retakes.
  - It pauses and resumes with LangGraph checkpoints and `interrupt()`, which is what LangGraph
    was chosen for (ADR 0013).
  - It hands a complete claim to the pipeline.
  - It never promises, denies or prices; such messages are not sent.
- **M6b, memory and skills.**
  - A claim memory written only by the workflow, with provenance on the hash chain.
  - Near-copy photos are found by perceptual hash.
  - The triage agent's `find_similar_claims` answers from memory.
  - Three owner-approved procedures the triage agent loads when relevant.
  - Simulated customers measured with pass^k.
- 724 tests, 96% coverage, strict typing.

## Numbers

| Measure | Result |
|---|---|
| Intake: scripted live customers | 3 of 3 complete with correct facts (11 to 17 turns) |
| Intake: simulated customers, pass@1 / pass^4 | **0.85 / 0.80** (target 0.70) |
| Golden v2, escalation recall | **1.00** |
| Golden v2, harmless stories fast-tracked | **5 of 10** (was 3 of 10) |
| Golden v2, stories caught / citation validity | 43 of 43 / 1.00 |
| Skills loaded | 86 of 150 claims |
| API spend for M6 | about $7 |

## What went well

- **Measuring before deciding.** The near-copy threshold came from golden photos, not a guess:
  re-saves moved the hash at most 2 bits, different photos were never closer than 18. Choosing
  10 instead of the pipeline's 6 is backed by that data.
- **Skills helped where the agent was weakest.** With procedures for glass, flat tyres and
  exclusions, the agent fast-tracked more harmless stories without losing any risky ones.
- **The reviews again found real problems before merge:**
  - a session lost after an error;
  - a possible double claim;
  - gaps in the promise filter;
  - the spending cap crashing instead of handing over.
- **Independent trials.** Giving each simulated trial its own response cache made pass^4
  meaningful; with a shared cache all four trials would have been replays.

## What did not

- **Corrected answers.** Customers who correct their date often end up with the first date
  recorded. The intake prompt needs to say "record a corrected answer again".
- **Strictness costs turns.** Blocking promise-like messages makes the agent rephrase, which
  costs a turn or two (11 to 17 turns instead of 9 to 13).
- **Long conversations can hit the per-session cap**, which hands the claim to a person. That
  is safe, but a sign that prompts should be shorter.
- **The plan's code was not written test-first once** (the intake graph). It was set aside, the
  tests written and seen failing, then restored. It is recorded in the ledger.

## Open items

- Intake prompt v2: re-record corrected facts; measure with `eval-intake`.
- Validate the LLM judge (50 owner labels, from M5).
- Near-copy detection for heavier crops (M7).
- Use memory in the MCP demo server.

## Next: M7

Trust and governance:
- fraud checks: photo reuse as a rule (using the memory), metadata, AI-generated images, story
  versus photo;
- face and licence-plate blurring;
- a red-team suite mapped to the OWASP agentic top 10;
- the AIS programme document and a system card.
