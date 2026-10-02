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
