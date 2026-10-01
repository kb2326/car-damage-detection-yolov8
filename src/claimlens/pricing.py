"""Repair cost range from damage findings and a versioned rate card."""

from __future__ import annotations

import tomllib
from collections.abc import Sequence
from pathlib import Path
from typing import Self

from pydantic import Field, model_validator

from claimlens.domain import CostEstimate, DamageFinding, DamageType, Frozen, Severity

_SEVERITY_ORDER = {Severity.MINOR: 0, Severity.MODERATE: 1, Severity.SEVERE: 2}


class RateCard(Frozen):
    version: str
    minor_max_fraction: float = Field(gt=0.0, lt=1.0)
    moderate_max_fraction: float = Field(gt=0.0, lt=1.0)
    rates: dict[DamageType, dict[Severity, tuple[int, int]]]

    @model_validator(mode="after")
    def _complete(self) -> Self:
        if self.minor_max_fraction >= self.moderate_max_fraction:
            raise ValueError("minor_max_fraction must be below moderate_max_fraction")
        for damage_type in DamageType:
            bands = self.rates.get(damage_type)
            if bands is None or set(bands) != set(Severity):
                raise ValueError(f"rate card is missing rates for {damage_type.value}")
            for low, high in bands.values():
                if not 0 <= low <= high:
                    raise ValueError(f"invalid range for {damage_type.value}: {low}-{high}")
        return self


def load_rate_card(path: Path) -> RateCard:
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    return RateCard.model_validate(
        {"version": data["version"], **data["severity"], "rates": data["rates"]}
    )


def severity_for(fraction: float, card: RateCard) -> Severity:
    if fraction < card.minor_max_fraction:
        return Severity.MINOR
    if fraction < card.moderate_max_fraction:
        return Severity.MODERATE
    return Severity.SEVERE


def estimate_cost(findings: Sequence[DamageFinding], card: RateCard) -> CostEstimate:
    worst: dict[DamageType, Severity] = {}
    for finding in findings:
        severity = severity_for(finding.image_area_fraction, card)
        current = worst.get(finding.damage_type)
        if current is None or _SEVERITY_ORDER[severity] > _SEVERITY_ORDER[current]:
            worst[finding.damage_type] = severity
    if not worst:
        return CostEstimate(low=0, high=0, basis=f"{card.version}: no damage found")
    low = sum(card.rates[t][s][0] for t, s in worst.items())
    high = sum(card.rates[t][s][1] for t, s in worst.items())
    parts = ", ".join(f"{t.value}/{s.value}" for t, s in sorted(worst.items()))
    return CostEstimate(low=low, high=high, basis=f"{card.version}: {parts}")
