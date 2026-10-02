# ADR 0012: Policy search with citable clauses on LanceDB

- **Status:** Accepted
- **Date:** 2026-10-02

## Context

A coverage statement from an AI agent is only trustworthy if it points at the policy wording
behind it. The structured lookup (`get_coverage`) knows that a policy is active and has collision
cover. It does not know that a rental car is covered for 10 days or that racing is excluded.

## Decision

- **Fictional policy wordings as documents** (`knowledge/policies/{basic,standard,premium}.md`).
  They hold 56 clauses, each with a stable id in its heading (`### STD-6.1 Rental vehicle after a
  covered loss`). `config/policies.toml` links each policy to a wording.
- **LanceDB, embedded.** One table `policy_clauses` in `var/lancedb/` (files, no server), with
  native BM25 full-text search and vector search, fused by reciprocal rank fusion (RRF).
  - Chosen over hand-rolled SQLite + NumPy, which would re-implement what LanceDB provides.
  - Chosen over Chroma (weaker keyword search) and sqlite-vec (fusion by hand).
  - The same store is planned for M6 memory.
- **One combined `content` field** (title + text) carries the full-text index, because LanceDB
  0.39's native full-text search indexes a single field.
- **Local embeddings:** `BAAI/bge-small-en-v1.5` through `fastembed` (ONNX on CPU, about 130 MB,
  no PyTorch). It sits behind an `Embedder` protocol; tests use a deterministic hashed
  bag-of-words fake.
- **Search is scoped to the claimant's policy:** the policy id maps to a wording, and the query is
  pre-filtered to it. Wordings are checked against a fixed allowlist, so the filter cannot be
  injected.
- **Citations are checkable:** `verify_citations(ids)` returns any id that does not exist; M5 will
  reject recommendations that cite unknown clauses.
- **Exposed as an MCP tool:** `search_policy_clauses` on `policy-admin`, for the `triage` and
  `demo` profiles (not `intake`).

## Consequences

- **Quality** (`evals/reports/2026-10-02-policy-search-v1.md`): recall@5 = 1.00 on 10
  everyday-language questions; the right clause is first for 6 and second for 4 (recall@1 0.60).
  With 17 to 20 clauses per wording, recall@5 is a generous bar; the set must grow before these
  numbers mean much.
- `claimlens knowledge build` must run once per machine (it downloads the embedding model on
  first use, then takes seconds). The index is not committed.
- Clause text is data written by us; the agent quotes it, and rules still decide the route.
