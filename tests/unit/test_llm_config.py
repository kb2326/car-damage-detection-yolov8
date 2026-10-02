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
