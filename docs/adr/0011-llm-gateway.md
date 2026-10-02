# ADR 0011: One LLM gateway with cost caps

- **Status:** Accepted
- **Date:** 2026-10-02

## Context

In M5 an LLM triage agent starts reasoning over claims. An agent that can call a model freely has
three problems. A bug can spend without limit. A provider outage becomes a claim failure. And
nobody can later say what was asked, of which model, at what cost.

## Decision

- **One gateway** (`claimlens.llm.Gateway`): every model call goes through it. Only
  `llm/anthropic_provider.py` imports the Anthropic SDK.
- **Our own thin layer on the official SDK,** not LiteLLM: a small package we can read and test,
  with providers behind a small `Provider` protocol (`AnthropicProvider`, and `FakeProvider` for
  tests).
- **Tiers in config** (`config/llm.toml`): `strong` → `claude-sonnet-5-5` and `fast` →
  `claude-haiku-4-5`, with prices per million tokens ($2 / $10 and $1 / $5).
- **Order of work for each call:**
  1. cache lookup (a hit is free);
  2. budget check;
  3. provider call, with up to 3 attempts and exponential backoff on transient errors;
  4. fallback to the next model;
  5. schema validation, with one repair request;
  6. charge, cache and log.
- **Caps checked before a call:** $0.03 per claim and $1.00 per day, stored in SQLite so a
  restart cannot reset them. Over the cap → `BudgetExceeded`, and M5 routes the claim to a person.
  - A call is refused when a cap is already reached, **or when its worst case could pass it**.
    The worst case is the prompt (estimated at 3 characters per token) plus a full-length reply.
  - `max_tokens` is clamped to a ceiling (4,096).
  - The input estimate is approximate, so a small overshoot is still possible, but not the
    10-times overshoot the first version allowed.
- **Everything spent is charged and recorded,** whatever the outcome: `ok`, `invalid_output`,
  `unavailable` (all models down) or `rejected` (the provider refused the request). A failure
  during the repair call still charges the first call.
- **Prompt-cache tokens count:** cache writes are billed at 1.25 times and cache reads at 0.1
  times the input price, and both are added to the spend.
- **Claude-only fallback** (Sonnet → Haiku), on the owner's call: no local model to install or
  heat the laptop.
  - Transient errors (timeout, connection, 429, overload, 5xx) retry and then fall back.
  - Fatal errors (401, 403, 400, 404, and any other unclassified API error) stop at once, because
    another model would fail the same way.
- **Structured output:** the Pydantic model's JSON Schema goes into the system prompt. The reply
  is parsed (fenced or surrounded JSON is accepted) and validated. Invalid twice →
  `InvalidModelOutput`, never a half-parsed answer.
- **Call records** (`LLMCall`): model, tokens, cost, latency, cache hit, attempts, outcome,
  prompt id and a **sha256 of the prompt**. They never hold the prompt text or the API key. They
  go to `var/llm-calls.jsonl`, plus an `LLMCalled` event on the claim's log when the call belongs
  to a claim.
- **Versioned prompts:** `prompts/<name>/<version>.md`; the id and hash are recorded per call.

## Consequences

- **Verified live (2026-10-02):**

  | Call | Tokens in / out | Cost | Time |
  |---|---|---|---|
  | Haiku 4.5 | 38 / 5 | $0.000063 | 1.0 s |
  | Sonnet 5.5, structured answer | 156 / 18 | $0.000492 | 1.5 s |
  | The same request again (cache) | – | $0 | 3 ms |

- A typical triage call (about 3,000 tokens in, 500 out) costs about $0.011 on Sonnet. With the
  worst-case check, the $0.03 cap allows two such calls per claim.
- The real gateway is built by `claimlens.llm.factory.build_gateway`, which refuses to start
  without `ANTHROPIC_API_KEY` and logs claim calls as `LLMCalled` events.
- CI needs no key: `FakeProvider` scripts replies and failures. The live test runs only with
  `CLAIMLENS_LIVE=1`.
- A cache hit is not refused at the cap, because it costs nothing.
- Adding a provider means implementing `Provider.complete` and adding its models and prices to
  the config.
