"""Build the real gateway from repo config. The API key is read here and nowhere else."""

from __future__ import annotations

from pathlib import Path

from claimlens.data.config import read_secret
from claimlens.llm.budget import Budget
from claimlens.llm.cache import ResponseCache
from claimlens.llm.config import load_llm_config
from claimlens.llm.gateway import Gateway
from claimlens.llm.log import event_call_log, jsonl_call_log
from claimlens.llm.types import LLMUnavailable


def build_gateway(
    config_dir: Path,
    repo_root: Path,
    store_path: Path | None = None,
    *,
    per_day_usd: float | None = None,
    cache_path: Path | None = None,
) -> Gateway:
    """Gateway on the Anthropic API. Calls with a claim id are also logged on that claim."""
    key = read_secret("ANTHROPIC_API_KEY", repo_root / ".env")
    if not key:
        raise LLMUnavailable("set ANTHROPIC_API_KEY in .env to use the LLM gateway")
    from claimlens.llm.anthropic_provider import AnthropicProvider

    config = load_llm_config(config_dir / "llm.toml")
    if per_day_usd is not None:  # an evaluation run may raise the daily cap for itself only
        limits = config.limits.model_copy(update={"per_day_usd": per_day_usd})
        config = config.model_copy(update={"limits": limits})
    var = repo_root / "var"
    log = jsonl_call_log(var / "llm-calls.jsonl")
    if store_path is not None:
        log = event_call_log(store_path, log)
    return Gateway(
        config,
        AnthropicProvider(key),
        Budget(var / "llm-budget.sqlite", config.limits),
        # An evaluation that repeats trials gives each its own cache, so repeats are real.
        ResponseCache(cache_path or var / "llm-cache.sqlite"),
        log,
    )
