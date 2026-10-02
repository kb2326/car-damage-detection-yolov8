# M4b design: LLM gateway and policy search with citations

- **Status:** Draft for review
- **Date:** 2026-10-02
- **Parent spec:** [`2026-10-01-claimlens-design.md`](2026-10-01-claimlens-design.md) §7, §10, §12
- **Follows:** [M4a](2026-10-02-m4a-mcp-tools-design.md) (MCP tools). **Followed by:** M5 (triage
  agent).

## 1. Why

The M5 triage agent needs two things before it can reason safely:

1. **One controlled way to call the LLM.** That means fixed models per role, retries, a fallback,
   caching, hard cost caps, validated structured output and a log of every call.
2. **Policy knowledge with citations.** The agent must quote the exact clause behind any coverage
   statement, and code must be able to check that the clause exists.

## 2. Goals and success criteria

| Goal | Measure | Target |
|---|---|---|
| One gateway | Every LLM call goes through `claimlens.llm.Gateway` | 100% (nothing else imports `anthropic`) |
| Spend can't run away | Calls refused once a claim's or the day's cap is reached | Per claim $0.03, per day $1.00 |
| Resilience | Retries on timeout and overload, then a fallback model | Sonnet 5.5 → Haiku 4.5 |
| Valid outputs | Structured replies are validated, with one repair attempt | Invalid replies never returned |
| Search quality | recall@5 on 10 hand-labelled questions | ≥ 0.9 |
| Citations checkable | `verify_citations` rejects unknown clause ids | Tested |
| Real API path works | Opt-in live smoke test (Haiku + Sonnet, structured) | Passes; cost < $0.01 |
| Free CI | Tests use a fake provider and a fake embedder | No key, no model download |

**Out of scope:**
- A local or second-vendor provider. The fallback is Claude-only, on the owner's call; the
  `Provider` protocol allows adding one later.
- The agent loop (M5).
- Claim-history memory (M6).

## 3. Decisions already made

- **Gateway on the official Anthropic SDK, with our own thin layer.** LiteLLM was rejected as a
  large dependency that hides the mechanics.
- **Tiers:** `strong` → `claude-sonnet-5-5` ($2 / $10 per million input / output tokens) and
  `fast` → `claude-haiku-4-5` ($1 / $5). Prices live in config.
- **Policy search on LanceDB** (embedded, files in `var/lancedb/`). It has native BM25 full-text
  search plus vector search, fused by reciprocal rank fusion (RRF). Chosen over hand-rolled
  SQLite + NumPy, Chroma and sqlite-vec, and reused for memory in M6.
- **Local embeddings:** `BAAI/bge-small-en-v1.5` via `fastembed` (ONNX, about 130 MB, CPU). It
  sits behind an `Embedder` protocol, with a fake in tests.

## 4. Components

### 4.1 `claimlens.llm` (gateway)

- `config/llm.toml`: the tier → model map; per-model prices (USD per million input and output
  tokens); retries (3, with exponential backoff from 1 s); the fallback order; the caps
  (`per_claim_usd = 0.03`, `per_day_usd = 1.00`); and the default `max_tokens`.
- `LLMRequest(messages, system, tier, max_tokens, output_schema: type[BaseModel] | None,
  claim_id: str | None, prompt_id: str | None)`.
- `LLMResponse(text, parsed: BaseModel | None, model, input_tokens, output_tokens, cost_usd,
  cached, latency_ms, attempts)`.
- `Provider` protocol: `complete(model, system, messages, max_tokens) -> ProviderReply(text,
  input_tokens, output_tokens)`. It raises `ProviderTransientError` (timeout, overload, rate limit)
  or `ProviderFatalError` (authentication, bad request).
  - `AnthropicProvider` reads `ANTHROPIC_API_KEY` through `read_secret` and sets
    `cache_control` on the system prompt.
  - `FakeProvider` replays scripted replies and errors.
- `Budget`: spend per claim and per day, kept in `var/llm-budget.sqlite`. `check(claim_id)`
  happens before the call and `charge(claim_id, cost)` after it.
- `ResponseCache`: SQLite keyed by `sha256(model, system, messages, schema name, max_tokens)`. A
  hit costs $0 and is marked `cached`.
- `Gateway.generate(request)`:
  1. budget check;
  2. cache lookup;
  3. for each model in `[tier model, *fallback]`, up to 3 attempts on transient errors;
  4. on a fatal error, stop;
  5. validate the schema, with one repair request if invalid;
  6. charge, cache and log.
  - Errors: `BudgetExceeded`, `LLMUnavailable`, `InvalidModelOutput`.
- **Structured output:** the schema's JSON Schema is put into the system prompt with "reply with
  only JSON", then parsed with `model_validate_json`. The repair prompt quotes the validation
  error.
- `LLMCall` log records: request id, claim id, prompt id/version, model, tokens, cost, latency,
  cached, attempts, outcome. When a claim id is set they are appended as an `LLMCalled` event
  (new payload), otherwise to `var/llm-calls.jsonl`. They never contain the API key or full
  prompts; the prompt is stored as a sha256.
- **Prompts:** `prompts/<name>/<version>.md`, loaded by `load_prompt(name, version) ->
  Prompt(id, version, text, sha256)`. M4b ships `prompts/smoke/v1.md` for the live test.

### 4.2 `claimlens.knowledge` (policy search)

- `knowledge/policies/{basic,standard,premium}.md`: fictional policy wordings. Clauses are written
  as `### STD-4.2 Glass damage` followed by text; ids are `BAS-`, `STD-` and `PRM-` plus a section
  number.
- `config/policies.toml` gains a `wording` per policy (`basic` | `standard` | `premium`), linking
  policy ids to documents.
- `parse_policy(path) -> list[Clause(clause_id, wording, title, text)]`. Duplicate or malformed
  ids are errors.
- `Embedder` protocol: `embed(texts) -> list[list[float]]`, plus `dim` and `name`.
  `FastEmbedder` (bge-small, 384 dimensions) and `FakeEmbedder` (deterministic hashed
  bag-of-words, for tests).
- `build_index(clauses, embedder, path)` writes the LanceDB table `policy_clauses` (clause_id,
  wording, title, text, vector) with a full-text index on `text` and `title`, plus an
  `index.json` manifest (embedder name, number of clauses, documents' sha256).
- `search_clauses(index, query, *, wording=None, k=5) -> list[ClauseHit(clause_id, wording,
  title, text, score)]`: LanceDB hybrid query with the RRF reranker, filtered by wording when
  given.
- `verify_citations(index, clause_ids) -> list[str]` returns the unknown ids. An empty list means
  every citation is real.
- CLI:
  - `claimlens knowledge build` (re)builds the index.
  - `claimlens knowledge search "<question>" [--policy P-1001]` prints hits.
  - `claimlens knowledge eval` computes recall@5 on `evals/knowledge/questions.jsonl` (10 labelled
    questions) and writes `evals/reports/<date>-policy-search-v1.md`.

### 4.3 MCP

`policy-admin` gains `search_policy_clauses(query, policy_id=None, k=5) -> ClauseResults`. The
policy id is mapped to its wording through `config/policies.toml`. The tool is added to
`SERVERS` and to the `triage` and `demo` profiles, but not to `intake`.

## 5. Error handling

| Situation | Behaviour |
|---|---|
| No `ANTHROPIC_API_KEY` | `LLMUnavailable("set ANTHROPIC_API_KEY in .env")` before any network call |
| Claim or day cap reached | `BudgetExceeded`; no call made |
| Timeout, overload or 429 | Retry with backoff, then the fallback model, then `LLMUnavailable` |
| 401 / 400 | `ProviderFatalError`, which becomes `LLMUnavailable` with the reason; no fallback (it would fail the same way) |
| Invalid JSON or schema | One repair request; then `InvalidModelOutput` (costs are still charged) |
| Index missing | `search_clauses` raises "run `claimlens knowledge build`"; the MCP tool returns an error result |
| Embedding model not downloaded | `knowledge build` downloads it once (about 130 MB); search reuses the cached model |

## 6. Testing

- **Gateway unit tests (fake provider):**
  - tier mapping; retries and backoff (with a fake sleep);
  - fallback after transient errors; no fallback on a fatal error;
  - both caps (refused before calling); cache hit is free;
  - schema success, repair success and repair failure; cost arithmetic;
  - `LLMCalled` event or JSONL record with no secrets.
- **Knowledge unit tests (tiny LanceDB in tmp_path, fake embedder):**
  - parser (ids, duplicates);
  - keyword hit; wording filter; fused ranking;
  - `verify_citations`; missing index error.
- **MCP:** profile visibility for `search_policy_clauses`, the policy-id → wording filter, and the
  intake refusal.
- **Quality:** `knowledge eval` with the real embedder, run locally; recall@5 ≥ 0.9 is reported,
  not run in CI.
- **Live smoke** (`CLAIMLENS_LIVE=1`, skipped otherwise): Haiku plain text, and Sonnet with a small
  schema. Asserts the parsed output, a cost under $0.01, and the logged record.

## 7. Risks

| Risk | Mitigation |
|---|---|
| Price or model changes | Prices and model ids in config; the live test checks the model ids are accepted |
| LanceDB API differences on Windows or between versions | Pin the version; the tests exercise the hybrid query |
| fastembed download in CI | CI uses the fake embedder; the real model is only used locally |
| The fake embedder flatters search quality | Quality is measured with the real embedder (`knowledge eval`) |
| A prompt contains secrets | Logs store only prompt hashes; the key is never placed in a request body or log |
