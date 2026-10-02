# M4b: LLM Gateway and Policy Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** One cost-capped, logged, validated way to call Claude, and hybrid search over fictional
policy wordings that returns citable clauses, exposed as an MCP tool.

**Architecture:**
- `claimlens.llm.Gateway` runs every call through: budget check → cache → provider with retries →
  fallback model → schema validation (one repair) → charge, cache and log. Providers sit behind a
  protocol: `AnthropicProvider` (official SDK) and `FakeProvider`.
- `claimlens.knowledge` parses `knowledge/policies/*.md` into clauses, embeds them (fastembed
  bge-small, or a fake in tests), and stores them in LanceDB with a full-text index on `content`.
  Search is hybrid (BM25 + vector, RRF).
- `policy-admin` gains `search_policy_clauses`.

**Tech Stack:** anthropic 1.11 SDK, lancedb 0.39 (native FTS on a single field, RRFReranker),
fastembed 0.8 (`BAAI/bge-small-en-v1.5`, 384-dim), pyarrow, SQLite, Pydantic 2, MCP SDK 2.2.

**Spec:** [`docs/specs/2026-10-02-m4b-llm-knowledge-design.md`](../specs/2026-10-02-m4b-llm-knowledge-design.md)

---

## Plain-language briefing

**What we're building:**
1. A **gateway**: one doorway to Claude. It picks the model for each role, retries on hiccups,
   falls back from Sonnet to Haiku, remembers answers it has already paid for, **refuses to spend
   more than $0.03 per claim or $1 per day**, checks that structured answers are valid, and writes
   down every call.
2. **Policy search**: fictional policy documents split into numbered clauses, so the agent can
   look up and **quote** the wording ("STD-6.1 Rental vehicle: up to 10 days…"). Code can check
   that any quoted clause really exists.

**Why:** an AI agent with unlimited spend and no sources is neither safe nor auditable.

**Where it sits:** under the M5 triage agent. The agent will think through the gateway and look
things up through the search tool.

**To-do:**
1. Gateway config and types (Task 1)
2. Providers: fake and Anthropic (Task 2)
3. Budget and cache (Task 3)
4. The gateway itself, plus call logging and prompts (Task 4)
5. Policy documents and the clause parser (Task 5)
6. Embedders and the LanceDB index with hybrid search (Task 6)
7. `claimlens knowledge build/search/eval` and the quality report (Task 7)
8. The MCP tool `search_policy_clauses` (Task 8)
9. The live smoke test, ADRs 0011 and 0012, the roadmap (Task 9)

**Done looks like:**
- `uv run claimlens knowledge search "is a rental car covered after a collision?" --policy P-1001`
  prints `STD-6.1 Rental vehicle …`.
- The live test makes two real, tiny Claude calls for under a cent, each logged with its cost.

**New terms:**
- **Gateway:** the single code path to the LLM.
- **Tier:** a role-based model choice (fast or strong).
- **Fallback:** a second model tried when the first is unavailable.
- **Cost cap:** a hard spending limit.
- **RAG (retrieval-augmented generation):** looking things up and giving the model the sources.
- **BM25:** classic keyword ranking.
- **Embedding:** numbers capturing a text's meaning.
- **Hybrid search:** keyword and meaning search combined.
- **RRF (reciprocal rank fusion):** a way to merge two ranked lists.
- **recall@5:** how often the right answer is in the top 5.

---

## Global Constraints

- Python 3.12; `uv run …`; ruff 100, mypy strict, coverage ≥ 95%. Provider and embedder adapters
  that need the network or models (`llm/anthropic_provider.py`, `knowledge/fastembedder.py`) are
  excluded from coverage.
- Nothing outside `src/claimlens/llm/anthropic_provider.py` imports `anthropic`.
- The API key is read only with `read_secret("ANTHROPIC_API_KEY")` and never logged, cached or put
  in an exception message.
- Caps from `config/llm.toml`: `per_claim_usd = 0.03` and `per_day_usd = 1.00`, checked **before**
  each provider call.
- Fallback: transient errors (timeout, connection, 429, overload, 5xx) → retry, then the next
  model. Fatal errors (401, 403, 400, 404) → stop.
- CI uses `FakeProvider` and `FakeEmbedder` only: no key, no model download, no network.
- New dependencies: `anthropic>=1.11,<2` (main); the `knowledge` group = `lancedb>=0.39,<0.40`,
  `fastembed>=0.8,<0.9`. CI installs `--group knowledge`.

## Review Focus

- **A provider that hangs or overloads on every model:** `LLMUnavailable` after the configured
  attempts, never an endless loop. Pinned in Task 4 (`test_all_models_down_is_unavailable`).
- **A claim exactly at its cap:** the next call is refused before any provider call. Pinned in
  Task 3/4 (`test_claim_at_cap_is_refused_before_calling`).
- **A model reply wrapped in ```json fences, or with prose around it:** parsed when the JSON is
  valid, otherwise one repair. Pinned in Task 4 (`test_fenced_json_is_parsed`).
- **A query for a policy whose wording has no clauses, or no index yet:** a clear error, not
  empty silence. Pinned in Task 6 (`test_missing_index_is_a_clear_error`).
- **The API key appearing in logs:** never. Pinned in Task 4 (`test_logs_hold_no_secrets`).

---

### Task 1: Gateway config and types

**Files:** Create `config/llm.toml`, `src/claimlens/llm/__init__.py`, `src/claimlens/llm/config.py`, `src/claimlens/llm/types.py`; modify `pyproject.toml`; test `tests/unit/test_llm_config.py`.

- [ ] **Step 1: Dependencies.**
  - Add `"anthropic>=1.11,<2",` to `dependencies`.
  - Add the group `knowledge = ["fastembed>=0.8,<0.9", "lancedb>=0.39,<0.40"]`.
  - Add `"lancedb", "lancedb.*", "fastembed", "fastembed.*", "pyarrow", "pyarrow.*"` to the mypy
    override with `ultralytics`.
  - Add `"*/claimlens/llm/anthropic_provider.py", "*/claimlens/knowledge/fastembedder.py",` to the
    coverage omit list.
  - Change the CI install to `uv sync --locked --group training --group knowledge`.
  - Then run `uv lock && uv sync --group training --group review --group vision --group knowledge --reinstall-package opencv-python-headless`.

- [ ] **Step 2: Config**

`config/llm.toml`:

```toml
# LLM gateway (spec docs/specs/2026-10-02-m4b-llm-knowledge-design.md). Prices: USD per million tokens.

[tiers]
strong = "claude-sonnet-5-5"
fast = "claude-haiku-4-5"

[fallback]
# Tried in order after the tier's model fails with transient errors (Claude-only, owner's call).
models = ["claude-haiku-4-5"]

[prices]
"claude-sonnet-5-5" = { input = 2.0, output = 10.0 }
"claude-haiku-4-5" = { input = 1.0, output = 5.0 }

[limits]
per_claim_usd = 0.03
per_day_usd = 1.00
max_tokens = 1024
attempts_per_model = 3
backoff_seconds = 1.0
```

- [ ] **Step 3: Failing tests**

`tests/unit/test_llm_config.py`:

```python
from pathlib import Path

import pytest
from pydantic import ValidationError

from claimlens.llm.config import LLMConfig, load_llm_config
from claimlens.llm.types import LLMRequest, Message

ROOT = Path(__file__).resolve().parents[2]


def test_repo_config_loads() -> None:
    config = load_llm_config(ROOT / "config" / "llm.toml")
    assert config.tiers == {"strong": "claude-sonnet-5-5", "fast": "claude-haiku-4-5"}
    assert config.models_for("strong") == ["claude-sonnet-5-5", "claude-haiku-4-5"]
    assert config.models_for("fast") == ["claude-haiku-4-5"]
    assert config.limits.per_claim_usd == 0.03


def test_cost_uses_per_million_prices() -> None:
    config = load_llm_config(ROOT / "config" / "llm.toml")
    assert config.cost("claude-sonnet-5-5", 3000, 500) == pytest.approx(0.011)
    assert config.cost("claude-haiku-4-5", 1_000_000, 0) == pytest.approx(1.0)


def test_every_model_needs_a_price(tmp_path: Path) -> None:
    path = tmp_path / "llm.toml"
    path.write_text(
        '[tiers]\nstrong = "m1"\n[fallback]\nmodels = []\n[prices]\n'
        "[limits]\nper_claim_usd = 1\nper_day_usd = 1\nmax_tokens = 10\n"
        "attempts_per_model = 1\nbackoff_seconds = 0\n",
        encoding="utf-8",
    )
    with pytest.raises(ValidationError, match="no price"):
        load_llm_config(path)


def test_unknown_tier_is_an_error() -> None:
    config = load_llm_config(ROOT / "config" / "llm.toml")
    with pytest.raises(ValueError, match="unknown tier"):
        config.models_for("genius")


def test_request_needs_a_message() -> None:
    with pytest.raises(ValidationError):
        LLMRequest(messages=[], tier="fast")
    assert LLMRequest(messages=[Message(role="user", content="hi")], tier="fast").max_tokens is None
    assert isinstance(load_llm_config(ROOT / "config" / "llm.toml"), LLMConfig)
```

- [ ] **Step 4: Run; fails** (`ModuleNotFoundError: claimlens.llm`).

- [ ] **Step 5: Implement**

`src/claimlens/llm/__init__.py`:

```python
"""LLM gateway: the only way ClaimLens calls a language model (spec 2026-10-02-m4b)."""
```

`src/claimlens/llm/config.py`:

```python
"""Gateway configuration: tiers, fallback, prices and spend limits."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Self

from pydantic import Field, model_validator

from claimlens.domain import Frozen


class Price(Frozen):
    input: float = Field(ge=0.0)
    output: float = Field(ge=0.0)


class Limits(Frozen):
    per_claim_usd: float = Field(gt=0.0)
    per_day_usd: float = Field(gt=0.0)
    max_tokens: int = Field(gt=0)
    attempts_per_model: int = Field(ge=1)
    backoff_seconds: float = Field(ge=0.0)


class LLMConfig(Frozen):
    tiers: dict[str, str]
    fallback: tuple[str, ...]
    prices: dict[str, Price]
    limits: Limits

    @model_validator(mode="after")
    def _priced(self) -> Self:
        for model in {*self.tiers.values(), *self.fallback}:
            if model not in self.prices:
                raise ValueError(f"model {model!r} has no price in [prices]")
        return self

    def models_for(self, tier: str) -> list[str]:
        if tier not in self.tiers:
            raise ValueError(f"unknown tier {tier!r}; known: {sorted(self.tiers)}")
        primary = self.tiers[tier]
        return [primary, *(m for m in self.fallback if m != primary)]

    def cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        price = self.prices[model]
        return (input_tokens * price.input + output_tokens * price.output) / 1_000_000


def load_llm_config(path: Path) -> LLMConfig:
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    return LLMConfig.model_validate(
        {
            "tiers": data["tiers"],
            "fallback": tuple(data.get("fallback", {}).get("models", [])),
            "prices": data["prices"],
            "limits": data["limits"],
        }
    )
```

`src/claimlens/llm/types.py`:

```python
"""Requests, responses and errors of the LLM gateway."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from claimlens.domain import Frozen


class Message(Frozen):
    role: Literal["user", "assistant"]
    content: str


class LLMRequest(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    messages: list[Message] = Field(min_length=1)
    tier: str
    system: str = ""
    max_tokens: int | None = None
    output_schema: type[BaseModel] | None = None
    claim_id: str | None = None
    prompt_id: str | None = None


class LLMResponse(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    text: str
    parsed: BaseModel | None
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    cached: bool
    latency_ms: int
    attempts: int


class LLMError(Exception):
    """Base class for gateway errors; messages never contain secrets."""


class BudgetExceeded(LLMError):  # noqa: N818 - spec name
    pass


class LLMUnavailable(LLMError):  # noqa: N818 - spec name
    pass


class InvalidModelOutput(LLMError):
    pass
```

- [ ] **Step 6: Pass; commit** `feat: add LLM gateway config and types`.

---

### Task 2: Providers

**Files:** Create `src/claimlens/llm/provider.py`, `src/claimlens/llm/anthropic_provider.py`; test `tests/unit/test_llm_provider.py`.

**Interfaces:**
- `ProviderReply(text, input_tokens, output_tokens)`.
- `ProviderTransientError`, `ProviderFatalError`.
- The `Provider` protocol: `complete(model, system, messages, max_tokens) -> ProviderReply`.
- `FakeProvider(script: list[ProviderReply | Exception])`, with `calls: list[dict]`.
- `AnthropicProvider(api_key: str, timeout_s: float = 60.0)`.

- [ ] **Step 1: Failing tests**

`tests/unit/test_llm_provider.py`:

```python
import pytest

from claimlens.llm.provider import FakeProvider, ProviderReply, ProviderTransientError
from claimlens.llm.types import Message


def test_fake_provider_replays_its_script_and_records_calls() -> None:
    fake = FakeProvider([ProviderTransientError("overloaded"), ProviderReply("hi", 10, 2)])
    with pytest.raises(ProviderTransientError):
        fake.complete("m", "", [Message(role="user", content="x")], 5)
    assert fake.complete("m", "sys", [Message(role="user", content="x")], 5).text == "hi"
    assert [c["model"] for c in fake.calls] == ["m", "m"]
    assert fake.calls[1]["system"] == "sys"


def test_fake_provider_runs_out_loudly() -> None:
    with pytest.raises(AssertionError, match="script"):
        FakeProvider([]).complete("m", "", [Message(role="user", content="x")], 5)
```

- [ ] **Step 2: Run; fails.**

- [ ] **Step 3: Implement**

`src/claimlens/llm/provider.py`:

```python
"""Provider protocol and a scripted fake for tests (no network, no key)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from claimlens.llm.types import Message


@dataclass(frozen=True)
class ProviderReply:
    text: str
    input_tokens: int
    output_tokens: int


class ProviderTransientError(Exception):
    """Worth retrying: timeout, connection, rate limit, overload, server error."""


class ProviderFatalError(Exception):
    """Not worth retrying or falling back: authentication, permission, bad request."""


class Provider(Protocol):
    def complete(
        self, model: str, system: str, messages: Sequence[Message], max_tokens: int
    ) -> ProviderReply: ...


@dataclass
class FakeProvider:
    script: list[ProviderReply | Exception]
    calls: list[dict[str, Any]] = field(default_factory=list)

    def complete(
        self, model: str, system: str, messages: Sequence[Message], max_tokens: int
    ) -> ProviderReply:
        self.calls.append(
            {"model": model, "system": system, "messages": list(messages), "max_tokens": max_tokens}
        )
        assert self.script, "FakeProvider script ran out"
        step = self.script.pop(0)
        if isinstance(step, Exception):
            raise step
        return step
```

`src/claimlens/llm/anthropic_provider.py` (coverage-excluded):

```python
"""Claude through the official Anthropic SDK. The only module that imports `anthropic`."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import anthropic

from claimlens.llm.provider import ProviderFatalError, ProviderReply, ProviderTransientError
from claimlens.llm.types import Message

_TRANSIENT: tuple[type[Exception], ...] = (
    anthropic.APITimeoutError,
    anthropic.APIConnectionError,
    anthropic.RateLimitError,
    anthropic.OverloadedError,
    anthropic.InternalServerError,
    anthropic.ServiceUnavailableError,
)
_FATAL: tuple[type[Exception], ...] = (
    anthropic.AuthenticationError,
    anthropic.PermissionDeniedError,
    anthropic.BadRequestError,
    anthropic.NotFoundError,
)


class AnthropicProvider:
    def __init__(self, api_key: str, timeout_s: float = 60.0) -> None:
        # The gateway owns retries, so the SDK must not retry on its own.
        self._client: Any = anthropic.Anthropic(api_key=api_key, timeout=timeout_s, max_retries=0)

    def complete(
        self, model: str, system: str, messages: Sequence[Message], max_tokens: int
    ) -> ProviderReply:
        system_blocks = (
            [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
            if system
            else []
        )
        try:
            response = self._client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=system_blocks,
                messages=[{"role": m.role, "content": m.content} for m in messages],
            )
        except _TRANSIENT as exc:
            raise ProviderTransientError(f"{type(exc).__name__}") from None
        except _FATAL as exc:
            raise ProviderFatalError(
                f"{type(exc).__name__}: {getattr(exc, 'status_code', '')}"
            ) from None
        text = "".join(block.text for block in response.content if block.type == "text")
        return ProviderReply(
            text=text,
            input_tokens=int(response.usage.input_tokens),
            output_tokens=int(response.usage.output_tokens),
        )
```

(Exception messages carry only the class name and status, never the request or the key.)

- [ ] **Step 4: Pass; commit** `feat: add LLM providers (fake and Anthropic)`.

---

### Task 3: Budget and cache

**Files:** Create `src/claimlens/llm/budget.py`, `src/claimlens/llm/cache.py`; test `tests/unit/test_llm_budget_cache.py`.

**Interfaces:**
- `Budget(path, limits, *, today=…)`, with `check(claim_id)` (raises `BudgetExceeded`),
  `charge(claim_id, cost)`, `spent_claim(claim_id)` and `spent_today()`.
- `ResponseCache(path)`, with `key(model, system, messages, schema_name, max_tokens) -> str`,
  `get(key) -> CachedReply | None` and `put(key, CachedReply)`.
- `CachedReply(text, model, input_tokens, output_tokens)`.

- [ ] **Step 1: Failing tests**

`tests/unit/test_llm_budget_cache.py`:

```python
from datetime import date
from pathlib import Path

import pytest

from claimlens.llm.budget import Budget
from claimlens.llm.cache import CachedReply, ResponseCache
from claimlens.llm.config import Limits
from claimlens.llm.types import BudgetExceeded, Message

LIMITS = Limits(
    per_claim_usd=0.03, per_day_usd=0.05, max_tokens=10, attempts_per_model=1, backoff_seconds=0
)


def test_claim_at_cap_is_refused_before_calling(tmp_path: Path) -> None:
    budget = Budget(tmp_path / "b.sqlite", LIMITS, today=lambda: date(2026, 10, 2))
    budget.check("c1")
    budget.charge("c1", 0.03)
    with pytest.raises(BudgetExceeded, match="claim c1"):
        budget.check("c1")
    budget.check("c2")


def test_daily_cap_covers_all_claims_and_resets_next_day(tmp_path: Path) -> None:
    day = [date(2026, 10, 2)]
    budget = Budget(tmp_path / "b.sqlite", LIMITS, today=lambda: day[0])
    budget.charge("c1", 0.02)
    budget.charge("c2", 0.02)
    budget.charge(None, 0.01)
    with pytest.raises(BudgetExceeded, match="daily"):
        budget.check("c3")
    day[0] = date(2026, 10, 3)
    budget.check("c3")
    assert budget.spent_claim("c1") == pytest.approx(0.02)


def test_spend_survives_a_restart(tmp_path: Path) -> None:
    Budget(tmp_path / "b.sqlite", LIMITS, today=lambda: date(2026, 10, 2)).charge("c1", 0.03)
    with pytest.raises(BudgetExceeded):
        Budget(tmp_path / "b.sqlite", LIMITS, today=lambda: date(2026, 10, 2)).check("c1")


def test_cache_round_trip_and_keys_differ_by_input(tmp_path: Path) -> None:
    cache = ResponseCache(tmp_path / "c.sqlite")
    msgs = [Message(role="user", content="hi")]
    key = cache.key("m", "sys", msgs, None, 10)
    assert cache.get(key) is None
    cache.put(key, CachedReply(text="yo", model="m", input_tokens=3, output_tokens=1))
    assert cache.get(key) == CachedReply(text="yo", model="m", input_tokens=3, output_tokens=1)
    assert cache.key("m", "sys2", msgs, None, 10) != key
    assert cache.key("m", "sys", msgs, "Schema", 10) != key
```

- [ ] **Step 2: Run; fails.**

- [ ] **Step 3: Implement**

`src/claimlens/llm/budget.py`:

```python
"""Hard spend caps per claim and per day, persisted so a restart cannot reset them."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from datetime import date
from pathlib import Path

from claimlens.llm.config import Limits
from claimlens.llm.types import BudgetExceeded

_SCHEMA = "CREATE TABLE IF NOT EXISTS spend (day TEXT NOT NULL, claim_id TEXT, usd REAL NOT NULL)"


class Budget:
    def __init__(
        self, path: Path, limits: Limits, *, today: Callable[[], date] = date.today
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), isolation_level=None)
        self._conn.execute(_SCHEMA)
        self._limits = limits
        self._today = today

    def spent_claim(self, claim_id: str) -> float:
        row = self._conn.execute(
            "SELECT COALESCE(SUM(usd), 0) FROM spend WHERE claim_id = ?", (claim_id,)
        ).fetchone()
        return float(row[0])

    def spent_today(self) -> float:
        row = self._conn.execute(
            "SELECT COALESCE(SUM(usd), 0) FROM spend WHERE day = ?", (self._today().isoformat(),)
        ).fetchone()
        return float(row[0])

    def check(self, claim_id: str | None) -> None:
        if self.spent_today() >= self._limits.per_day_usd:
            raise BudgetExceeded(f"daily LLM cap of ${self._limits.per_day_usd:.2f} reached")
        if claim_id is not None and self.spent_claim(claim_id) >= self._limits.per_claim_usd:
            raise BudgetExceeded(
                f"LLM cap of ${self._limits.per_claim_usd:.2f} reached for claim {claim_id}"
            )

    def charge(self, claim_id: str | None, usd: float) -> None:
        self._conn.execute(
            "INSERT INTO spend (day, claim_id, usd) VALUES (?, ?, ?)",
            (self._today().isoformat(), claim_id, usd),
        )
```

`src/claimlens/llm/cache.py`:

```python
"""Response cache: an identical request is answered from disk, for free and reproducibly."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Sequence
from pathlib import Path

from claimlens.domain import Frozen
from claimlens.llm.types import Message

_SCHEMA = "CREATE TABLE IF NOT EXISTS replies (key TEXT PRIMARY KEY, body TEXT NOT NULL)"


class CachedReply(Frozen):
    text: str
    model: str
    input_tokens: int
    output_tokens: int


class ResponseCache:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), isolation_level=None)
        self._conn.execute(_SCHEMA)

    @staticmethod
    def key(
        model: str,
        system: str,
        messages: Sequence[Message],
        schema_name: str | None,
        max_tokens: int,
    ) -> str:
        payload = json.dumps(
            [model, system, [m.model_dump() for m in messages], schema_name, max_tokens],
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    def get(self, key: str) -> CachedReply | None:
        row = self._conn.execute("SELECT body FROM replies WHERE key = ?", (key,)).fetchone()
        return None if row is None else CachedReply.model_validate_json(row[0])

    def put(self, key: str, reply: CachedReply) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO replies (key, body) VALUES (?, ?)",
            (key, reply.model_dump_json()),
        )
```

- [ ] **Step 4: Pass; commit** `feat: add LLM spend caps and response cache`.

---

### Task 4: Gateway, call log and prompts

**Files:** Create `src/claimlens/llm/gateway.py`, `src/claimlens/llm/log.py`, `src/claimlens/llm/prompts.py`, `prompts/smoke/v1.md`; modify `src/claimlens/events/payloads.py` and `projection.py` (an `LLMCalled` event, ignored by the fold); test `tests/unit/test_llm_gateway.py`.

**Interfaces:**
- `LLMCall` record (`request_id, claim_id, prompt_id, model, input_tokens, output_tokens,
  cost_usd, latency_ms, cached, attempts, outcome, prompt_sha256`).
- `CallLog = Callable[[LLMCall], None]`, `jsonl_call_log(path)` and
  `event_call_log(store_path, fallback)`.
- `Gateway(config, provider, budget, cache, log, *, sleep=time.sleep, clock=time.perf_counter)`
  with `.generate(request) -> LLMResponse`.
- `Prompt(name, version, text, sha256)` and `load_prompt(root, name, version)`.

- [ ] **Step 1: Failing tests**

`tests/unit/test_llm_gateway.py`:

```python
from datetime import date
from pathlib import Path

import pytest
from pydantic import BaseModel

from claimlens.llm.budget import Budget
from claimlens.llm.cache import ResponseCache
from claimlens.llm.config import load_llm_config
from claimlens.llm.gateway import Gateway
from claimlens.llm.log import LLMCall, jsonl_call_log
from claimlens.llm.prompts import load_prompt
from claimlens.llm.provider import (
    FakeProvider,
    ProviderFatalError,
    ProviderReply,
    ProviderTransientError,
)
from claimlens.llm.types import (
    BudgetExceeded,
    InvalidModelOutput,
    LLMRequest,
    LLMUnavailable,
    Message,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG = load_llm_config(ROOT / "config" / "llm.toml")


class Verdict(BaseModel):
    route: str
    confidence: float


def _gateway(
    tmp_path: Path, script: list[ProviderReply | Exception]
) -> tuple[Gateway, FakeProvider, list[LLMCall]]:
    fake = FakeProvider(script)
    calls: list[LLMCall] = []
    gateway = Gateway(
        CONFIG,
        fake,
        Budget(tmp_path / "b.sqlite", CONFIG.limits, today=lambda: date(2026, 10, 2)),
        ResponseCache(tmp_path / "c.sqlite"),
        calls.append,
        sleep=lambda s: None,
    )
    return gateway, fake, calls


def _ask(text: str = "hello", **kw: object) -> LLMRequest:
    return LLMRequest(messages=[Message(role="user", content=text)], **kw)  # type: ignore[arg-type]


def test_strong_tier_uses_sonnet_and_charges(tmp_path: Path) -> None:
    gateway, fake, calls = _gateway(tmp_path, [ProviderReply("hi", 3000, 500)])
    response = gateway.generate(_ask(tier="strong", claim_id="c1"))
    assert response.model == "claude-sonnet-5-5"
    assert response.cost_usd == pytest.approx(0.011)
    assert fake.calls[0]["max_tokens"] == CONFIG.limits.max_tokens
    assert calls[0].outcome == "ok"
    assert calls[0].claim_id == "c1"


def test_transient_errors_retry_then_fall_back(tmp_path: Path) -> None:
    script: list[ProviderReply | Exception] = [ProviderTransientError("overloaded")] * 3 + [
        ProviderReply("ok", 10, 2)
    ]
    gateway, fake, _ = _gateway(tmp_path, script)
    response = gateway.generate(_ask(tier="strong"))
    assert [c["model"] for c in fake.calls] == ["claude-sonnet-5-5"] * 3 + ["claude-haiku-4-5"]
    assert response.model == "claude-haiku-4-5"
    assert response.attempts == 4


def test_all_models_down_is_unavailable(tmp_path: Path) -> None:
    gateway, fake, calls = _gateway(tmp_path, [ProviderTransientError("down")] * 6)
    with pytest.raises(LLMUnavailable):
        gateway.generate(_ask(tier="strong"))
    assert len(fake.calls) == 6
    assert calls[-1].outcome == "unavailable"


def test_fatal_error_does_not_fall_back(tmp_path: Path) -> None:
    gateway, fake, _ = _gateway(tmp_path, [ProviderFatalError("AuthenticationError: 401")])
    with pytest.raises(LLMUnavailable, match="AuthenticationError"):
        gateway.generate(_ask(tier="strong"))
    assert len(fake.calls) == 1


def test_claim_at_cap_is_refused_before_calling(tmp_path: Path) -> None:
    gateway, fake, _ = _gateway(tmp_path, [ProviderReply("x", 15000, 0)] * 2)
    gateway.generate(_ask(tier="strong", claim_id="c1"))  # $0.03 spent
    with pytest.raises(BudgetExceeded):
        gateway.generate(_ask("again", tier="strong", claim_id="c1"))
    assert len(fake.calls) == 1


def test_cache_hit_is_free(tmp_path: Path) -> None:
    gateway, fake, calls = _gateway(tmp_path, [ProviderReply("hi", 100, 10)])
    first = gateway.generate(_ask(tier="fast"))
    second = gateway.generate(_ask(tier="fast"))
    assert len(fake.calls) == 1
    assert second.cached
    assert second.cost_usd == 0.0
    assert second.text == first.text
    assert calls[-1].cached


def test_structured_output_is_validated(tmp_path: Path) -> None:
    gateway, fake, _ = _gateway(
        tmp_path, [ProviderReply('{"route": "FAST_TRACK", "confidence": 0.8}', 50, 10)]
    )
    response = gateway.generate(_ask(tier="fast", output_schema=Verdict))
    assert response.parsed == Verdict(route="FAST_TRACK", confidence=0.8)
    assert "JSON Schema" in fake.calls[0]["system"]


def test_fenced_json_is_parsed(tmp_path: Path) -> None:
    reply = 'Here you go:\n```json\n{"route": "ADJUSTER_REVIEW", "confidence": 0.4}\n```'
    gateway, _, _ = _gateway(tmp_path, [ProviderReply(reply, 50, 10)])
    response = gateway.generate(_ask(tier="fast", output_schema=Verdict))
    assert response.parsed == Verdict(route="ADJUSTER_REVIEW", confidence=0.4)


def test_invalid_output_gets_one_repair(tmp_path: Path) -> None:
    gateway, fake, _ = _gateway(
        tmp_path,
        [
            ProviderReply('{"route": 3}', 50, 10),
            ProviderReply('{"route": "FAST_TRACK", "confidence": 1}', 60, 10),
        ],
    )
    response = gateway.generate(_ask(tier="fast", output_schema=Verdict))
    assert response.parsed == Verdict(route="FAST_TRACK", confidence=1)
    assert "not valid" in fake.calls[1]["messages"][-1].content
    assert response.input_tokens == 110


def test_repair_failure_raises_and_still_charges(tmp_path: Path) -> None:
    gateway, _, calls = _gateway(tmp_path, [ProviderReply("nope", 50, 10)] * 2)
    with pytest.raises(InvalidModelOutput):
        gateway.generate(_ask(tier="fast", output_schema=Verdict, claim_id="c9"))
    assert calls[-1].outcome == "invalid_output"
    assert calls[-1].cost_usd > 0


def test_logs_hold_no_secrets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-should-never-appear")
    log_path = tmp_path / "calls.jsonl"
    gateway = Gateway(
        CONFIG,
        FakeProvider([ProviderReply("hi", 1, 1)]),
        Budget(tmp_path / "b.sqlite", CONFIG.limits),
        ResponseCache(tmp_path / "c.sqlite"),
        jsonl_call_log(log_path),
        sleep=lambda s: None,
    )
    gateway.generate(_ask("my secret question", tier="fast"))
    text = log_path.read_text(encoding="utf-8")
    assert "sk-ant" not in text
    assert "my secret question" not in text
    assert '"prompt_sha256"' in text


def test_prompts_are_versioned_files() -> None:
    prompt = load_prompt(ROOT / "prompts", "smoke", "v1")
    assert prompt.name == "smoke"
    assert prompt.version == "v1"
    assert len(prompt.sha256) == 64
    with pytest.raises(FileNotFoundError):
        load_prompt(ROOT / "prompts", "smoke", "v99")
```

- [ ] **Step 2: Run; fails.**

- [ ] **Step 3: Implement**

`prompts/smoke/v1.md`:

```markdown
You are a test assistant for the ClaimLens LLM gateway. Answer briefly and exactly as asked.
```

In `payloads.py`:
- add `LLM_CALLED = "LLMCalled"` to `EventType`;
- add the class below and register it in `PAYLOAD_TYPES`.

```python
class LLMCalled(Payload):
    event_type: ClassVar[EventType] = EventType.LLM_CALLED
    request_id: str
    model: str
    prompt_id: str | None
    input_tokens: int
    output_tokens: int
    cost_usd: float
    cached: bool
    outcome: str
```

In `projection.py`, import `LLMCalled` and add a fold case `case LLMCalled(): pass`.

`src/claimlens/llm/log.py`:

```python
"""Call records: what each LLM call cost and how it went. Never the key or the prompt text."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from uuid import UUID

from claimlens.domain import Frozen
from claimlens.events.envelope import Actor, ActorKind
from claimlens.events.payloads import LLMCalled
from claimlens.events.store import SQLiteEventStore


class LLMCall(Frozen):
    request_id: str
    claim_id: str | None
    prompt_id: str | None
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: int
    cached: bool
    attempts: int
    outcome: str
    prompt_sha256: str


CallLog = Callable[[LLMCall], None]


def jsonl_call_log(path: Path) -> CallLog:
    def log(call: LLMCall) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(call.model_dump_json() + "\n")

    return log


def event_call_log(store_path: Path, fallback: CallLog) -> CallLog:
    """Append to the claim's event log when the call belongs to a known claim."""

    def log(call: LLMCall) -> None:
        fallback(call)
        if call.claim_id is None:
            return
        try:
            uuid = UUID(call.claim_id)
        except ValueError:
            return
        store = SQLiteEventStore(store_path)
        try:
            if uuid in store.claim_ids():
                store.append(
                    uuid,
                    LLMCalled(
                        request_id=call.request_id,
                        model=call.model,
                        prompt_id=call.prompt_id,
                        input_tokens=call.input_tokens,
                        output_tokens=call.output_tokens,
                        cost_usd=call.cost_usd,
                        cached=call.cached,
                        outcome=call.outcome,
                    ),
                    Actor(kind=ActorKind.SYSTEM, name="llm-gateway"),
                )
        finally:
            store.close()

    return log
```

`src/claimlens/llm/prompts.py`:

```python
"""Versioned prompt files: prompts/<name>/<version>.md."""

from __future__ import annotations

import hashlib
from pathlib import Path

from claimlens.domain import Frozen


class Prompt(Frozen):
    name: str
    version: str
    text: str
    sha256: str

    @property
    def id(self) -> str:
        return f"{self.name}/{self.version}"


def load_prompt(root: Path, name: str, version: str) -> Prompt:
    path = root / name / f"{version}.md"
    if not path.is_file():
        raise FileNotFoundError(f"prompt not found: {path}")
    text = path.read_text(encoding="utf-8").strip()
    return Prompt(
        name=name, version=version, text=text, sha256=hashlib.sha256(text.encode()).hexdigest()
    )
```

`src/claimlens/llm/gateway.py`:

```python
"""The gateway: budget → cache → provider with retries → fallback → validation → charge and log."""

from __future__ import annotations

import hashlib
import json
import re
import time
import uuid
from collections.abc import Callable

from pydantic import BaseModel, ValidationError

from claimlens.llm.budget import Budget
from claimlens.llm.cache import CachedReply, ResponseCache
from claimlens.llm.config import LLMConfig
from claimlens.llm.log import CallLog, LLMCall
from claimlens.llm.provider import (
    Provider,
    ProviderFatalError,
    ProviderReply,
    ProviderTransientError,
)
from claimlens.llm.types import InvalidModelOutput, LLMRequest, LLMResponse, LLMUnavailable, Message

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def _extract_json(text: str) -> str:
    fenced = _FENCE.search(text)
    if fenced:
        return fenced.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    return text[start : end + 1] if start != -1 and end > start else text


def _schema_system(system: str, schema: type[BaseModel]) -> str:
    return (
        f"{system}\n\nReply with only a JSON object (no prose) that matches this JSON Schema:\n"
        f"{json.dumps(schema.model_json_schema(), sort_keys=True)}"
    ).strip()


class Gateway:
    def __init__(
        self,
        config: LLMConfig,
        provider: Provider,
        budget: Budget,
        cache: ResponseCache,
        log: CallLog,
        *,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._config = config
        self._provider = provider
        self._budget = budget
        self._cache = cache
        self._log = log
        self._sleep = sleep
        self._clock = clock

    def _call(
        self, models: list[str], system: str, messages: list[Message], max_tokens: int
    ) -> tuple[str, ProviderReply, int]:
        attempts = 0
        for model in models:
            for attempt in range(self._config.limits.attempts_per_model):
                attempts += 1
                try:
                    return (
                        model,
                        self._provider.complete(model, system, messages, max_tokens),
                        attempts,
                    )
                except ProviderTransientError:
                    if attempt + 1 < self._config.limits.attempts_per_model:
                        self._sleep(self._config.limits.backoff_seconds * 2**attempt)
                except ProviderFatalError as exc:
                    raise LLMUnavailable(f"LLM request rejected: {exc}") from None
        raise _AllDown(attempts)

    def generate(self, request: LLMRequest) -> LLMResponse:
        started = self._clock()
        request_id = uuid.uuid4().hex
        schema = request.output_schema
        system = _schema_system(request.system, schema) if schema else request.system
        max_tokens = request.max_tokens or self._config.limits.max_tokens
        models = self._config.models_for(request.tier)
        prompt_sha = hashlib.sha256(
            json.dumps([system, [m.model_dump() for m in request.messages]]).encode()
        ).hexdigest()

        def record(
            outcome: str, model: str, tin: int, tout: int, cost: float, cached: bool, attempts: int
        ) -> None:
            self._log(
                LLMCall(
                    request_id=request_id,
                    claim_id=request.claim_id,
                    prompt_id=request.prompt_id,
                    model=model,
                    input_tokens=tin,
                    output_tokens=tout,
                    cost_usd=round(cost, 6),
                    latency_ms=int((self._clock() - started) * 1000),
                    cached=cached,
                    attempts=attempts,
                    outcome=outcome,
                    prompt_sha256=prompt_sha,
                )
            )

        key = self._cache.key(
            models[0], system, request.messages, schema.__name__ if schema else None, max_tokens
        )
        hit = self._cache.get(key)
        if hit is not None:
            parsed = schema.model_validate_json(_extract_json(hit.text)) if schema else None
            record("ok", hit.model, hit.input_tokens, hit.output_tokens, 0.0, True, 0)
            return LLMResponse(
                text=hit.text,
                parsed=parsed,
                model=hit.model,
                input_tokens=hit.input_tokens,
                output_tokens=hit.output_tokens,
                cost_usd=0.0,
                cached=True,
                latency_ms=int((self._clock() - started) * 1000),
                attempts=0,
            )

        self._budget.check(request.claim_id)
        messages = list(request.messages)
        try:
            model, reply, attempts = self._call(models, system, messages, max_tokens)
        except _AllDown as down:
            record("unavailable", models[-1], 0, 0, 0.0, False, down.attempts)
            raise LLMUnavailable("all LLM models are unavailable; route to a person") from None
        tin, tout = reply.input_tokens, reply.output_tokens
        cost = self._config.cost(model, tin, tout)
        text = reply.text
        parsed: BaseModel | None = None
        if schema is not None:
            try:
                parsed = schema.model_validate_json(_extract_json(text))
            except ValidationError as first_error:
                repair = [
                    *messages,
                    Message(role="assistant", content=text),
                    Message(
                        role="user",
                        content=(
                            f"Your reply was not valid: {first_error.errors()[:3]}. "
                            "Reply again with only the JSON object."
                        ),
                    ),
                ]
                try:
                    model, reply, more = self._call([model], system, repair, max_tokens)
                except _AllDown as down:
                    self._budget.charge(request.claim_id, cost)
                    record("unavailable", model, tin, tout, cost, False, attempts + down.attempts)
                    raise LLMUnavailable(
                        "LLM unavailable during repair; route to a person"
                    ) from None
                attempts += more
                tin, tout = tin + reply.input_tokens, tout + reply.output_tokens
                cost += self._config.cost(model, reply.input_tokens, reply.output_tokens)
                text = reply.text
                try:
                    parsed = schema.model_validate_json(_extract_json(text))
                except ValidationError:
                    self._budget.charge(request.claim_id, cost)
                    record("invalid_output", model, tin, tout, cost, False, attempts)
                    raise InvalidModelOutput("model output failed validation twice") from None
        self._budget.charge(request.claim_id, cost)
        self._cache.put(
            key, CachedReply(text=text, model=model, input_tokens=tin, output_tokens=tout)
        )
        record("ok", model, tin, tout, cost, False, attempts)
        return LLMResponse(
            text=text,
            parsed=parsed,
            model=model,
            input_tokens=tin,
            output_tokens=tout,
            cost_usd=cost,
            cached=False,
            latency_ms=int((self._clock() - started) * 1000),
            attempts=attempts,
        )


class _AllDown(Exception):
    def __init__(self, attempts: int) -> None:
        super().__init__(attempts)
        self.attempts = attempts
```

The `invalid_output` test uses scripted replies `"nope"` twice: the first and the repair both
fail. Ruff format will re-wrap the long lines; split `record`'s signature if needed.

- [ ] **Step 4: Pass, full suite; commit** `feat: add the LLM gateway with retries, fallback, caps, cache and call log`.

---

### Task 5: Policy wordings and the clause parser

**Files:**
- Create `knowledge/policies/basic.md`, `knowledge/policies/standard.md`,
  `knowledge/policies/premium.md`, `src/claimlens/knowledge/__init__.py` and
  `src/claimlens/knowledge/clauses.py`.
- Modify `config/policies.toml` (a `wording` field on each policy) and `src/claimlens/policy.py`
  (`PolicyRecord.wording`).
- Test `tests/unit/test_knowledge_clauses.py`.

**Interfaces:**
- `Clause(clause_id, wording, title, text)`.
- `parse_policy(path) -> list[Clause]`.
- `load_wordings(root) -> list[Clause]`.
- `PolicyRecord.wording: Literal["basic", "standard", "premium"] = "standard"`.

- [ ] **Step 1: Write the three wordings.** These are fictional documents. Each clause heading is
  `### <ID> <Title>`, where the ID prefix matches the wording (`BAS`, `STD`, `PRM`), followed by
  plain sentences. Each wording must cover at least these sections, with **consistent but
  different terms**:
  - 1 Definitions
  - 2 Collision damage to your vehicle (basic: only when shown on the schedule)
  - 3 Deductible (with the amounts on the schedule; glass is different)
  - 4 Glass (basic: deductible applies; standard: no deductible for repair; premium: no
    deductible for repair or replacement)
  - 5 Theft and vandalism
  - 6 Rental vehicle (basic: not covered; standard: up to 10 days at $40/day; premium: up to
    30 days at $60/day)
  - 7 Towing (basic: none; standard: up to $100; premium: unlimited within 100 miles)
  - 8 Exclusions (racing, intentional damage, wear and tear, unlicensed driver, commercial use)
  - 9 Making a claim (report within 30 days, photos, police report for theft)
  - 10 Fraud (a false claim voids the policy and is referred to the fraud team)

  Write them as normal prose, 2–4 sentences per clause.

- [ ] **Step 2: Failing tests**

`tests/unit/test_knowledge_clauses.py`:

```python
from pathlib import Path

import pytest

from claimlens.knowledge.clauses import load_wordings, parse_policy
from claimlens.policy import load_policies

ROOT = Path(__file__).resolve().parents[2]


def test_repo_wordings_parse_with_unique_ids() -> None:
    clauses = load_wordings(ROOT / "knowledge" / "policies")
    ids = [c.clause_id for c in clauses]
    assert len(ids) == len(set(ids))
    assert {c.wording for c in clauses} == {"basic", "standard", "premium"}
    assert all(c.text for c in clauses)
    rental = next(c for c in clauses if c.clause_id == "STD-6.1")
    assert "10 days" in rental.text


def test_clause_id_prefix_must_match_the_wording(tmp_path: Path) -> None:
    path = tmp_path / "basic.md"
    path.write_text("# Basic\n\n### STD-1.1 Wrong prefix\nText.\n", encoding="utf-8")
    with pytest.raises(ValueError, match="prefix"):
        parse_policy(path)


def test_duplicate_ids_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "basic.md"
    path.write_text("### BAS-1.1 A\nx\n### BAS-1.1 B\ny\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        parse_policy(path)


def test_every_policy_has_a_known_wording() -> None:
    wordings = {c.wording for c in load_wordings(ROOT / "knowledge" / "policies")}
    policies = load_policies(ROOT / "config" / "policies.toml")
    for policy_id in ("P-1001", "P-1003", "P-1004", "P-2002"):
        record = policies.get_record(policy_id)
        assert record is not None
        assert record.wording in wordings
```

- [ ] **Step 3: Run; fails.**

- [ ] **Step 4: Implement**

`src/claimlens/knowledge/__init__.py`:

```python
"""Policy knowledge: fictional policy wordings, hybrid search, and citation checks."""
```

`src/claimlens/knowledge/clauses.py`:

```python
"""Split a policy wording into citable clauses (`### STD-6.1 Rental vehicle`)."""

from __future__ import annotations

import re
from pathlib import Path

from claimlens.domain import Frozen

PREFIXES = {"basic": "BAS", "standard": "STD", "premium": "PRM"}
_HEADING = re.compile(r"^###\s+([A-Z]{3}-\d+(?:\.\d+)?)\s+(.+?)\s*$")


class Clause(Frozen):
    clause_id: str
    wording: str
    title: str
    text: str


def parse_policy(path: Path) -> list[Clause]:
    wording = path.stem
    if wording not in PREFIXES:
        raise ValueError(f"unknown wording file {path.name}; expected one of {sorted(PREFIXES)}")
    clauses: list[Clause] = []
    current: tuple[str, str] | None = None
    body: list[str] = []

    def flush() -> None:
        if current is not None:
            clauses.append(
                Clause(
                    clause_id=current[0],
                    wording=wording,
                    title=current[1],
                    text=" ".join(body).strip(),
                )
            )

    for line in path.read_text(encoding="utf-8").splitlines():
        match = _HEADING.match(line)
        if match:
            flush()
            clause_id, title = match.groups()
            if not clause_id.startswith(PREFIXES[wording] + "-"):
                raise ValueError(
                    f"{path.name}: {clause_id} does not use the {PREFIXES[wording]} prefix"
                )
            current, body = (clause_id, title), []
        elif current is not None and line.strip() and not line.startswith("#"):
            body.append(line.strip())
    flush()
    ids = [c.clause_id for c in clauses]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise ValueError(f"{path.name}: duplicate clause ids {duplicates}")
    return clauses


def load_wordings(root: Path) -> list[Clause]:
    return [clause for path in sorted(root.glob("*.md")) for clause in parse_policy(path)]
```

In `policy.py`, add `wording: Literal["basic", "standard", "premium"] = "standard"` to
`PolicyRecord` (with `from typing import Literal`). In `config/policies.toml`, add a `wording`
line to each policy:
- P-1001 standard, P-1002 standard, P-1003 premium, P-1004 basic, P-1005 standard,
  P-1006 premium, P-2001 basic, P-2002 basic.
- Then edit the header comment: "wording selects the policy document in knowledge/policies/."

- [ ] **Step 5: Pass; commit** `feat: add fictional policy wordings and the clause parser`.

---

### Task 6: Embedders and the LanceDB hybrid index

**Files:** Create `src/claimlens/knowledge/embed.py`, `src/claimlens/knowledge/fastembedder.py`, `src/claimlens/knowledge/index.py`; test `tests/unit/test_knowledge_index.py`.

**Interfaces:**
- The `Embedder` protocol (`name`, `dim`, `embed(texts) -> list[list[float]]`) and
  `FakeEmbedder(dim=64)`.
- `FastEmbedder(model="BAAI/bge-small-en-v1.5")`.
- `IndexMissingError(Exception)`.
- `build_index(clauses, embedder, path) -> int`.
- `PolicyIndex.open(path, embedder)`, with `.search(query, *, wording=None, k=5) -> list[ClauseHit]`,
  `.verify_citations(ids) -> list[str]` and `.get(clause_id) -> Clause | None`.
- `ClauseHit(clause_id, wording, title, text, score)`.

- [ ] **Step 1: Failing tests**

`tests/unit/test_knowledge_index.py`:

```python
from pathlib import Path

import pytest

from claimlens.knowledge.clauses import Clause
from claimlens.knowledge.embed import FakeEmbedder
from claimlens.knowledge.index import IndexMissingError, PolicyIndex, build_index

pytest.importorskip("lancedb")

CLAUSES = [
    Clause(
        clause_id="STD-4.1",
        wording="standard",
        title="Glass damage",
        text="Windscreen repair has no deductible.",
    ),
    Clause(
        clause_id="STD-6.1",
        wording="standard",
        title="Rental vehicle",
        text="A rental car is covered for up to 10 days after a collision.",
    ),
    Clause(
        clause_id="BAS-6.1",
        wording="basic",
        title="Rental vehicle",
        text="Rental cars are not covered.",
    ),
    Clause(
        clause_id="STD-8.1",
        wording="standard",
        title="Exclusions",
        text="Damage while racing is excluded.",
    ),
]


def _index(tmp_path: Path) -> PolicyIndex:
    embedder = FakeEmbedder()
    assert build_index(CLAUSES, embedder, tmp_path / "lancedb") == 4
    return PolicyIndex.open(tmp_path / "lancedb", embedder)


def test_keyword_query_finds_the_clause(tmp_path: Path) -> None:
    hits = _index(tmp_path).search("racing exclusion")
    assert hits[0].clause_id == "STD-8.1"


def test_wording_filter_limits_results(tmp_path: Path) -> None:
    hits = _index(tmp_path).search("rental car", wording="basic")
    assert [h.clause_id for h in hits] == ["BAS-6.1"]


def test_hybrid_ranks_the_best_match_first(tmp_path: Path) -> None:
    hits = _index(tmp_path).search(
        "is a rental car covered after a collision", wording="standard", k=3
    )
    assert hits[0].clause_id == "STD-6.1"
    assert hits[0].score >= hits[-1].score


def test_citations_are_verified(tmp_path: Path) -> None:
    index = _index(tmp_path)
    assert index.verify_citations(["STD-6.1", "BAS-6.1"]) == []
    assert index.verify_citations(["STD-6.1", "STD-99.9"]) == ["STD-99.9"]
    clause = index.get("STD-4.1")
    assert clause is not None
    assert clause.title == "Glass damage"


def test_missing_index_is_a_clear_error(tmp_path: Path) -> None:
    with pytest.raises(IndexMissingError, match="knowledge build"):
        PolicyIndex.open(tmp_path / "nowhere", FakeEmbedder())


def test_fake_embedder_is_deterministic_and_normalised() -> None:
    a, b = FakeEmbedder().embed(["rental car", "rental car"])
    assert a == b
    assert abs(sum(x * x for x in a) - 1.0) < 1e-6
```

- [ ] **Step 2: Run; fails.**

- [ ] **Step 3: Implement**

`src/claimlens/knowledge/embed.py`:

```python
"""Text embedders: the real one (fastembed, local ONNX) and a deterministic fake for tests."""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence
from typing import Protocol

_WORD = re.compile(r"[a-z0-9]+")


class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class FakeEmbedder:
    """Hashed bag of words, L2-normalised: similar words → similar vectors. Tests only."""

    name = "fake-hashed-bow"

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            vector = [0.0] * self.dim
            for word in _WORD.findall(text.lower()):
                vector[int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dim] += 1.0
            norm = math.sqrt(sum(v * v for v in vector)) or 1.0
            vectors.append([v / norm for v in vector])
        return vectors
```

`src/claimlens/knowledge/fastembedder.py` (coverage-excluded):

```python
"""bge-small-en-v1.5 through fastembed (ONNX on CPU; downloads ~130 MB once)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


class FastEmbedder:
    def __init__(self, model: str = "BAAI/bge-small-en-v1.5") -> None:
        from fastembed import TextEmbedding

        self._model: Any = TextEmbedding(model_name=model)
        self.name = model
        self.dim = 384

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [[float(x) for x in vector] for vector in self._model.embed(list(texts))]
```

`src/claimlens/knowledge/index.py`:

```python
"""LanceDB table of policy clauses with hybrid (BM25 + vector, RRF) search."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from claimlens.domain import Frozen
from claimlens.knowledge.clauses import Clause
from claimlens.knowledge.embed import Embedder

TABLE = "policy_clauses"


class IndexMissingError(Exception):
    pass


class ClauseHit(Frozen):
    clause_id: str
    wording: str
    title: str
    text: str
    score: float


def _lancedb() -> Any:
    import lancedb

    return lancedb


def build_index(clauses: Sequence[Clause], embedder: Embedder, path: Path) -> int:
    vectors = embedder.embed([f"{c.title}. {c.text}" for c in clauses])
    rows = [
        {
            "clause_id": c.clause_id,
            "wording": c.wording,
            "title": c.title,
            "text": c.text,
            "content": f"{c.title}. {c.text}",
            "vector": vector,
        }
        for c, vector in zip(clauses, vectors, strict=True)
    ]
    db = _lancedb().connect(str(path))
    table = db.create_table(TABLE, data=rows, mode="overwrite")
    table.create_fts_index("content", replace=True)
    return len(rows)


class PolicyIndex:
    def __init__(self, table: Any, embedder: Embedder) -> None:
        self._table = table
        self._embedder = embedder

    @classmethod
    def open(cls, path: Path, embedder: Embedder) -> PolicyIndex:
        if not path.exists():
            raise IndexMissingError(f"no policy index at {path}: run `claimlens knowledge build`")
        db = _lancedb().connect(str(path))
        try:
            table = db.open_table(TABLE)
        except (FileNotFoundError, ValueError) as exc:
            raise IndexMissingError(
                f"no policy index at {path}: run `claimlens knowledge build`"
            ) from exc
        return cls(table, embedder)

    def search(self, query: str, *, wording: str | None = None, k: int = 5) -> list[ClauseHit]:
        from lancedb.rerankers import RRFReranker

        vector = self._embedder.embed([query])[0]
        builder = self._table.search(query_type="hybrid").vector(vector).text(query)
        if wording is not None:
            builder = builder.where(f"wording = '{wording}'", prefilter=True)
        rows = builder.rerank(RRFReranker()).limit(k).to_list()
        return [
            ClauseHit(
                clause_id=r["clause_id"],
                wording=r["wording"],
                title=r["title"],
                text=r["text"],
                score=float(r.get("_relevance_score", 0.0)),
            )
            for r in rows
        ]

    def _all(self) -> dict[str, Clause]:
        rows = self._table.to_arrow().select(["clause_id", "wording", "title", "text"]).to_pylist()
        return {r["clause_id"]: Clause.model_validate(r) for r in rows}

    def get(self, clause_id: str) -> Clause | None:
        return self._all().get(clause_id)

    def verify_citations(self, clause_ids: Sequence[str]) -> list[str]:
        known = self._all()
        return [c for c in clause_ids if c not in known]
```

`wording` values come only from the `PolicyRecord` Literal (`basic`, `standard`, `premium`), never
from free text, so the `where` string cannot be injected. Add an explicit check
`if wording not in {"basic", "standard", "premium"}: raise ValueError` at the top of `search`.

- [ ] **Step 4: Pass; commit** `feat: add the LanceDB policy index with hybrid search and citation checks`.

---

### Task 7: `claimlens knowledge build/search/eval` and the quality report

**Files:** Create `src/claimlens/knowledge/commands.py`, `evals/knowledge/questions.jsonl` (10 questions); modify `src/claimlens/cli.py`; test `tests/unit/test_knowledge_cli.py`.

**Interfaces:**
- `add_knowledge_parser(sub)` and `run_knowledge_command(args, *, embedder_factory) -> int`.
- `EmbedderFactory = Callable[[], Embedder]`.
- `recall_at_k(index, questions, k) -> tuple[float, list[tuple[str, str, list[str]]]]`.
- `main(..., embedder_factory=_fastembedder)`.

- [ ] **Step 1: Questions.** `evals/knowledge/questions.jsonl` holds one JSON object per line,
  `{"question": …, "policy_id": …, "expected": "<clause id>"}`, ten in all. They cover:
  - rental for P-1001 → STD-6.1;
  - rental for P-1004 → BAS-6.1;
  - a windscreen chip for P-1003 → PRM-4.x;
  - towing for P-1001 → STD-7.x;
  - racing → an exclusion clause;
  - the theft police report → section 9;
  - the deductible for glass on basic → BAS-4.x or BAS-3.x;
  - a false claim → section 10;
  - collision on basic → BAS-2.x;
  - the reporting deadline → section 9.

  Use the exact ids from the wordings written in Task 5.

- [ ] **Step 2: Failing tests**

`tests/unit/test_knowledge_cli.py`:

```python
import json
from pathlib import Path

import pytest

from claimlens.cli import main
from claimlens.knowledge.embed import FakeEmbedder
from tests.fakes import CONFIG_DIR

pytest.importorskip("lancedb")
ROOT = Path(__file__).resolve().parents[2]


def _run(tmp_path: Path, *args: str) -> int:
    return main(
        [
            "--config",
            str(CONFIG_DIR),
            "knowledge",
            *args,
            "--root",
            str(ROOT),
            "--index",
            str(tmp_path / "idx"),
        ],
        embedder_factory=FakeEmbedder,
    )


def test_build_then_search_with_a_policy(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _run(tmp_path, "build") == 0
    assert "clauses" in capsys.readouterr().out
    assert _run(tmp_path, "search", "rental car after a collision", "--policy", "P-1004") == 0
    out = capsys.readouterr().out
    assert "BAS-6.1" in out


def test_eval_writes_a_report(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert _run(tmp_path, "build") == 0
    report = tmp_path / "report.md"
    assert _run(tmp_path, "eval", "--out", str(report)) == 0
    text = report.read_text(encoding="utf-8")
    assert "recall@5" in text
    assert (
        len(
            json.loads(
                "["
                + ",".join(
                    (ROOT / "evals/knowledge/questions.jsonl")
                    .read_text(encoding="utf-8")
                    .split("\n")[:-1]
                )
                + "]"
            )
        )
        == 10
    )


def test_search_without_an_index_is_a_clear_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _run(tmp_path, "search", "anything") == 1
    assert "knowledge build" in capsys.readouterr().err
```

- [ ] **Step 3: Run; fails.**

- [ ] **Step 4: Implement**

`src/claimlens/knowledge/commands.py`:

```python
"""`claimlens knowledge build | search | eval`."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from datetime import date
from pathlib import Path

from claimlens.knowledge.clauses import load_wordings
from claimlens.knowledge.embed import Embedder
from claimlens.knowledge.index import IndexMissingError, PolicyIndex, build_index
from claimlens.policy import load_policies

EmbedderFactory = Callable[[], Embedder]


def add_knowledge_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    knowledge = sub.add_parser("knowledge", help="policy wordings: build, search, evaluate")
    ksub = knowledge.add_subparsers(dest="knowledge_command", required=True)
    for name in ("build", "search", "eval"):
        p = ksub.add_parser(name)
        p.add_argument("--root", type=Path, default=Path("."))
        p.add_argument("--index", type=Path, default=Path("var/lancedb"))
        if name == "search":
            p.add_argument("query")
            p.add_argument("--policy", default=None, help="policy id, e.g. P-1001")
            p.add_argument("-k", type=int, default=5)
        if name == "eval":
            p.add_argument("--out", type=Path, default=None)


def recall_at_k(
    index: PolicyIndex,
    questions: Sequence[dict[str, str]],
    wording_of: Callable[[str], str],
    k: int,
) -> tuple[float, list[tuple[str, str, list[str]]]]:
    rows: list[tuple[str, str, list[str]]] = []
    for q in questions:
        hits = index.search(q["question"], wording=wording_of(q["policy_id"]), k=k)
        rows.append((q["question"], q["expected"], [h.clause_id for h in hits]))
    found = sum(expected in ids for _, expected, ids in rows)
    return found / max(len(rows), 1), rows


def run_knowledge_command(args: argparse.Namespace, *, embedder_factory: EmbedderFactory) -> int:
    policies = load_policies(args.config / "policies.toml")

    def wording_of(policy_id: str) -> str:
        record = policies.get_record(policy_id)
        if record is None:
            raise ValueError(f"unknown policy {policy_id}")
        return record.wording

    try:
        if args.knowledge_command == "build":
            clauses = load_wordings(args.root / "knowledge" / "policies")
            count = build_index(clauses, embedder_factory(), args.index)
            print(f"Indexed {count} clauses into {args.index}")
            return 0
        index = PolicyIndex.open(args.index, embedder_factory())
        if args.knowledge_command == "search":
            wording = wording_of(args.policy) if args.policy else None
            for hit in index.search(args.query, wording=wording, k=args.k):
                print(f"{hit.clause_id}  {hit.title}  ({hit.score:.3f})\n    {hit.text}")
            return 0
        lines = (args.root / "evals" / "knowledge" / "questions.jsonl").read_text(encoding="utf-8")
        questions = [json.loads(line) for line in lines.splitlines() if line.strip()]
        recall, rows = recall_at_k(index, questions, wording_of, 5)
        report = [
            "# Policy search v1",
            "",
            f"- Date: {date.today().isoformat()}",
            f"- Questions: {len(rows)}",
            f"- **recall@5: {recall:.2f}** (target ≥ 0.90)",
            "",
            "| Question | Expected | Top 5 | Found |",
            "|---|---|---|---|",
            *[
                f"| {q} | {e} | {', '.join(ids)} | {'yes' if e in ids else 'no'} |"
                for q, e, ids in rows
            ],
        ]
        out = (
            args.out
            or args.root / "evals" / "reports" / f"{date.today().isoformat()}-policy-search-v1.md"
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(report) + "\n", encoding="utf-8")
        print(f"recall@5 = {recall:.2f}; report written to {out}")
        return 0
    except (IndexMissingError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
```

In `cli.py`:
- Import `add_knowledge_parser`, `run_knowledge_command` and `EmbedderFactory`.
- Add the default factory:

```python
def _fastembedder() -> Embedder:
    from claimlens.knowledge.fastembedder import FastEmbedder

    return FastEmbedder()
```

- Call `add_knowledge_parser(sub)` in `build_parser`.
- Add `embedder_factory: EmbedderFactory = _fastembedder` to `main`, and pass it through
  `_dispatch`. Route `if args.command == "knowledge": return run_knowledge_command(args, embedder_factory=embedder_factory)`.

- [ ] **Step 5: Pass with the fake embedder.** Then run it **locally with the real embedder** and
  check the target:

```bash
uv run claimlens knowledge build
uv run claimlens knowledge search "is a rental car covered after a collision?" --policy P-1001
uv run claimlens knowledge eval
```

Expected: `recall@5 >= 0.90`. If it is lower, look at the misses before changing anything.
Usually the question wording or the clause title is the problem; do not tune the index to the
test.

- [ ] **Step 6: Commit** `feat: add claimlens knowledge build, search and eval`, including the report.

---

### Task 8: MCP `search_policy_clauses`

**Files:** Modify `src/claimlens/mcp/profiles.py` (`SERVERS`), `config/agents.toml`, `src/claimlens/mcp/schemas.py`, `src/claimlens/mcp/policy_admin.py`, `src/claimlens/mcp/serve.py`; test `tests/unit/test_mcp_vision_policy.py` (append).

**Interfaces:**
- `ClauseOut(clause_id, title, text, score)` and `ClauseResults(wording: str | None, clauses: list[ClauseOut])`.
- `build_policy_admin(profile, policies, audit, index: Callable[[], PolicyIndex] | None = None)`.

- [ ] **Step 1: Failing test.** Append:

```python
def test_search_policy_clauses_uses_the_policy_wording(tmp_path: Path) -> None:
    pytest.importorskip("lancedb")
    from claimlens.knowledge.clauses import load_wordings
    from claimlens.knowledge.embed import FakeEmbedder
    from claimlens.knowledge.index import PolicyIndex, build_index

    build_index(load_wordings(ROOT / "knowledge" / "policies"), FakeEmbedder(), tmp_path / "idx")
    policies = load_policies(ROOT / "config" / "policies.toml")

    def index() -> PolicyIndex:
        return PolicyIndex.open(tmp_path / "idx", FakeEmbedder())

    server = build_policy_admin(PROFILES["demo"], policies, MemoryAudit(), index)
    result = call(server, "search_policy_clauses", {"query": "rental car", "policy_id": "P-1004"})
    content = result.structured_content
    assert content is not None
    assert content["wording"] == "basic"
    assert all(c["clause_id"].startswith("BAS-") for c in content["clauses"])
    intake = build_policy_admin(PROFILES["intake"], policies, MemoryAudit(), index)
    assert "search_policy_clauses" not in tool_names(intake)
```

(Add `import pytest` to that file if missing.)

- [ ] **Step 2: Run; fails.**

- [ ] **Step 3: Implement.**
  - In `profiles.py`, set `SERVERS["policy-admin"] = ("get_policy", "get_coverage", "search_policy_clauses")`.
  - In `agents.toml`, add `"search_policy_clauses"` to the `triage` and `demo` policy-admin lists.
  - Add the schemas, and in `policy_admin.py`:

```python
def search_policy_clauses(
    query: Annotated[str, Field(min_length=2, max_length=500)],
    policy_id: str | None = None,
    k: Annotated[int, Field(ge=1, le=10)] = 5,
) -> ClauseResults:
    with tool_errors():
        if index is None:
            raise ValueError("policy search is not configured on this server")
        wording = None
        if policy_id is not None:
            record = policies.get_record(policy_id)
            if record is None:
                raise ValueError(f"unknown policy {policy_id}")
            wording = record.wording
        hits = index().search(query, wording=wording, k=k)
        return ClauseResults(
            wording=wording,
            clauses=[
                ClauseOut(
                    clause_id=h.clause_id, title=h.title, text=h.text, score=round(h.score, 4)
                )
                for h in hits
            ],
        )


server.register(
    search_policy_clauses,
    "search_policy_clauses",
    "Search policy wording; returns citable clauses.",
)
```

  - Make `tool_errors` in `base.py` also catch `IndexMissingError`, imported from
    `claimlens.knowledge.index`.
  - In `serve.py`, pass `index=lambda: PolicyIndex.open(Path("var/lancedb"), FastEmbedder())`,
    imported lazily and cached in a one-element list.

- [ ] **Step 4: Pass; full suite; commit** `feat: add search_policy_clauses to the policy-admin MCP server`.

---

### Task 9: Live smoke test, ADRs and roadmap

**Files:** Create `tests/integration/test_llm_live.py`, `docs/adr/0011-llm-gateway.md`, `docs/adr/0012-policy-search.md`; modify `docs/roadmap.md`, `README.md` (an example prompt for `search_policy_clauses`).

- [ ] **Step 1: The live test**

`tests/integration/test_llm_live.py`:

```python
import os
from pathlib import Path

import pytest
from pydantic import BaseModel

from claimlens.data.config import read_secret

ROOT = Path(__file__).resolve().parents[2]
pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        os.environ.get("CLAIMLENS_LIVE") != "1" or not read_secret("ANTHROPIC_API_KEY"),
        reason="live LLM test: set CLAIMLENS_LIVE=1 and ANTHROPIC_API_KEY",
    ),
]


class Answer(BaseModel):
    city: str
    country: str


def test_real_calls_through_the_gateway(tmp_path: Path) -> None:
    from claimlens.llm.anthropic_provider import AnthropicProvider
    from claimlens.llm.budget import Budget
    from claimlens.llm.cache import ResponseCache
    from claimlens.llm.config import load_llm_config
    from claimlens.llm.gateway import Gateway
    from claimlens.llm.log import LLMCall
    from claimlens.llm.prompts import load_prompt
    from claimlens.llm.types import LLMRequest, Message

    config = load_llm_config(ROOT / "config" / "llm.toml")
    key = read_secret("ANTHROPIC_API_KEY")
    assert key
    calls: list[LLMCall] = []
    gateway = Gateway(
        config,
        AnthropicProvider(key),
        Budget(tmp_path / "b.sqlite", config.limits),
        ResponseCache(tmp_path / "c.sqlite"),
        calls.append,
    )
    prompt = load_prompt(ROOT / "prompts", "smoke", "v1")
    fast = gateway.generate(
        LLMRequest(
            messages=[Message(role="user", content="Reply with exactly the word: pong")],
            tier="fast",
            system=prompt.text,
            max_tokens=10,
            prompt_id=prompt.id,
        )
    )
    assert "pong" in fast.text.lower()
    assert fast.model == "claude-haiku-4-5"
    strong = gateway.generate(
        LLMRequest(
            messages=[Message(role="user", content="Where is the Eiffel Tower?")],
            tier="strong",
            system=prompt.text,
            max_tokens=100,
            output_schema=Answer,
            prompt_id=prompt.id,
        )
    )
    assert isinstance(strong.parsed, Answer)
    assert strong.parsed.city.lower() == "paris"
    assert strong.model == "claude-sonnet-5-5"
    total = fast.cost_usd + strong.cost_usd
    assert 0 < total < 0.01
    assert [c.outcome for c in calls] == ["ok", "ok"]
```

Run: `CLAIMLENS_LIVE=1 uv run pytest tests/integration/test_llm_live.py -q -s`. Expected: it
passes, with a total cost under $0.01. Record the actual cost and token counts in the ledger.

- [ ] **Step 2: ADR 0011, the LLM gateway.**
  - **Context:** the agent needs a model; cost, safety and audit.
  - **Decision:**
    - our own thin gateway on the Anthropic SDK (not LiteLLM);
    - tiers in config;
    - Claude-only fallback;
    - per-claim and per-day caps checked before calls;
    - SQLite cache;
    - JSON-schema output with one repair;
    - call records with prompt hashes only;
    - versioned prompt files.
  - **Consequences:** the live test result; cost per call; how to add a provider.

- [ ] **Step 3: ADR 0012, policy search.**
  - **Context:** citable policy knowledge.
  - **Decision:**
    - LanceDB embedded (it has BM25 and vectors with RRF built in), chosen over hand-rolled
      SQLite + NumPy, Chroma and sqlite-vec;
    - bge-small via fastembed (ONNX, no PyTorch);
    - clause ids;
    - citation verification;
    - one combined `content` field, because native full-text search indexes one field.
  - **Consequences:** recall@5 from the report; LanceDB reused for M6 memory.

- [ ] **Step 4: Roadmap and README.**
  - Tick the M4b items in the roadmap and add LinkedIn material.
  - Add the README example prompt "What does policy P-1001 say about a rental car after a
    collision? Quote the clause."
  - Final checks; commit `docs: add ADRs 0011 and 0012, the M4b roadmap and live test`.

---

## Self-review notes

- **Spec coverage:** §4.1 → Tasks 1–4; §4.2 → Tasks 5–7; §4.3 → Task 8; §5 errors → Tasks 3, 4,
  6 and 7; §6 testing → every task plus Task 9.
- **Rulings already in the plan:**
  - A single full-text field `content`, because LanceDB 0.39 native FTS indexes one field (found
    while probing the API).
  - `PolicyRecord.wording` defaults to `standard`.
  - The budget is checked only on cache misses: a cache hit costs nothing, so it isn't refused at
    the cap.
