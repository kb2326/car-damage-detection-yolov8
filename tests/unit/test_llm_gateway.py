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
    return LLMRequest(messages=[Message(role="user", content=text)], **kw)


def test_strong_tier_uses_sonnet_and_charges(tmp_path: Path) -> None:
    gateway, fake, calls = _gateway(tmp_path, [ProviderReply("hi", 3000, 500)])
    response = gateway.generate(_ask(tier="strong", claim_id="c1"))
    assert response.model == "claude-sonnet-5-5"
    assert response.cost_usd == pytest.approx(0.011)
    assert fake.calls[0]["max_tokens"] == CONFIG.limits.max_tokens
    assert calls[0].outcome == "ok"
    assert calls[0].claim_id == "c1"


def test_transient_errors_retry_then_fall_back(tmp_path: Path) -> None:
    script: list[ProviderReply | Exception] = [
        ProviderTransientError("overloaded"),
        ProviderTransientError("overloaded"),
        ProviderTransientError("overloaded"),
        ProviderReply("ok", 10, 2),
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
