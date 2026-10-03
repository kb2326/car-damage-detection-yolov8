# M7 and M8 retrospective: trust, the web app and the showcase

- **Dates:** 2026-10-03
- **Plans:** [`m8a-web-prototype`](../plans/2026-10-03-m8a-web-prototype.md),
  [`m7-trust`](../plans/2026-10-03-m7-trust.md), [`m8b-showcase`](../plans/2026-10-03-m8b-showcase.md)
- **Decisions:** ADR 0018 (web prototype), ADR 0019 (red team, near-copy fraud, system card),
  ADR 0020 (read-only showcase)

## What we shipped

- **M8a, the web app** (PR #18): chat intake with photo upload, claims list, claim page (stages,
  damage boxes, evidence, the agent's reasoning, similar claims, verified log), review form. Local
  only, Host and Origin checks, a Playwright browser test, a live check.
- **M7, trust** (PR #19): 34 red-team attacks mapped to the OWASP agentic top 10, run in
  "hijacked" mode (the fake model obeys the attacker), a live injection run, a dependency audit in
  CI, near-copy photos as a fraud signal, and the system card.
- **M8b, the showcase:** a read-only mode of the same app, five sample claims recorded with the real
  models and agents from openly licensed photos, a slim Docker image and a Hugging Face deploy
  workflow.

## Numbers

| Measure | Result |
|---|---|
| Red team, hijacked | 33 of 33 run held (+ dependency audit in CI) |
| Red team, live (Sonnet, 15 injected stories) | 15 of 15 to a person; agent fooled 0 times; $0.17 |
| Web live check | 12-turn chat → decided → reviewed, `Chain OK` (23 events) |
| Showcase | 5 claims, 1.8 MB, image 1.15 GB, no weights or keys |
| Tests | about 850, 96% coverage |

## What went well

- **The order change paid off.** Building the web app before the trust work gave the red team a
  real surface (Host, Origin, uploads) and made the system card concrete.
- **Hijacked mode** is a stronger claim than "the model resisted": it tests the controls. A
  mutation check (rules trusting a confident agent) proved the tests guard them.
- **Reviews kept catching real problems:** an LLM outage stranding a claim, a reviewer's override
  vanishing from the queues, no Host check; a memory failure hiding an exact-copy fraud signal;
  skipped tests counting as held; and an overclaim in the docs about story-only exclusions.
- **Safeguards showed up in practice:** the daily cap stopped the first live check and the first
  showcase chat, each time handing the claim to a person with the reason shown.

## What did not

- **The first showcase chats looped** because the scripted customer was too crude, and because the
  intake agent keeps asking for a photo the customer cannot send. The second is a real product gap.
- **Open-licence damage photos were hard to find.** `car-seg` shows undamaged cars; the
  showcase uses Wikimedia Commons instead.
- **Shell edits through heredocs** broke Python files several times again (quotes, `\n`); writing
  files with the editor or a script file is reliable.

## Open items

- Intake: record a photo gap when the customer cannot send it; re-record corrected answers.
- A code check for exclusions found only in the story (today the agent is the only control).
- Validate the LLM judge against 50 owner labels; heavier-crop near-copies; a lighter image.
- The ClaimLens Explained page (M8a, M7, M8b) when the owner is in that account; the LinkedIn post.
