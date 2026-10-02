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
    max_tokens_ceiling: int = Field(default=4096, gt=0)
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
