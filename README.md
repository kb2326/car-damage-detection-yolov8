# ClaimLens

**AI-assisted vehicle damage claims triage.** Custom-trained computer-vision models act as tools
for an auditable adjuster agent that runs inside a deterministic workflow, with humans in control
of every consequential decision.

> Status: **M0 – Foundations** (in progress). This repository started as a STAT 5350 course
> project (a YOLOv8 car-damage detector). See [`legacy/README.md`](legacy/README.md) for the
> original code and the audit that motivated the rebuild.

## Why

Filing a car-damage claim means an adjuster must inspect photos, identify damaged parts, check the
policy, estimate cost, watch for fraud, and route the claim. ClaimLens aims to **fast-track simple
claims in minutes with an explanation an auditor can verify, and escalate everything else to a
human with the evidence already assembled.**

## Design principles

1. **Models measure, the agent reasons, rules decide.** CV models produce measurable evidence;
   the LLM agent interprets it; deterministic business rules make the final routing decision.
2. **The system never auto-denies.** It can fast-track or escalate. Only humans deny claims.
3. **Everything is an event.** Claim state is an append-only, hash-chained event log that provides
   durability, memory, audit, and replay.
4. **Evals before features.** Every component ships with a measurable evaluation.

## Repository layout

```
src/claimlens/      Python package (domain code)
tests/              unit / integration tests and fixtures
training/           data and training pipelines (from M2)
evals/              golden claims, eval harness, reports (from M1)
docs/               PR/FAQ, ADRs, specs, cards, learning notes
data/               datasets — versioned with DVC, not git (see data/README.md)
models/             model weights — not in git (see models/README.md)
legacy/             the original course project, frozen for comparison
```

## Getting started

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync                      # create .venv with Python 3.12 and install dev tools
uv run pre-commit install    # run lint/format checks on every commit
uv run pytest                # tests
uv run ruff check . && uv run mypy
```

## Roadmap

| Milestone | Focus |
|---|---|
| M0 | Foundations: repo, tooling, CI, ADRs, legacy audit |
| M1 | Walking skeleton: event-sourced claim state, thin end-to-end pipeline, golden claims |
| M2 | Data engine: CarDD + foundation-model auto-labelling, DVC, FiftyOne |
| M3 | Vision models: damage + part instance segmentation, calibration, model card |
| M4 | Tools & integration: MCP servers, LLM gateway, policy RAG |
| M5 | Triage agent: human-in-the-loop, LLM evals, CI gates, tracing |
| M6 | Intake agent & memory: multi-turn intake, Agent Skills, user-simulator evals |
| M7 | Trust & governance: OWASP agentic threat model, red-team, fraud, PII, AIS program |
| M8 | Ship: ONNX, Docker, public demo, monitoring |

Full design: [`docs/specs/2026-10-01-claimlens-design.md`](docs/specs/2026-10-01-claimlens-design.md).

## License

Code: MIT (see [`LICENSE`](LICENSE)). Datasets keep their own licenses — see
[`data/README.md`](data/README.md).
