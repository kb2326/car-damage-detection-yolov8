"""Core domain types shared across ClaimLens."""

from __future__ import annotations

from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Frozen(BaseModel):
    """Immutable model that rejects unknown fields."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class DamageType(StrEnum):
    CRACK = "crack"
    DENT = "dent"
    GLASS_SHATTER = "glass_shatter"
    LAMP_BROKEN = "lamp_broken"
    SCRATCH = "scratch"
    TIRE_FLAT = "tire_flat"
    SMASH = "smash"


class Severity(StrEnum):
    MINOR = "minor"
    MODERATE = "moderate"
    SEVERE = "severe"


class Route(StrEnum):
    """Possible outcomes. There is deliberately no route that denies a claim."""

    FAST_TRACK = "FAST_TRACK"
    ADJUSTER_REVIEW = "ADJUSTER_REVIEW"
    FRAUD_REVIEW = "FRAUD_REVIEW"


class Confidence(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class BoundingBox(Frozen):
    """Axis-aligned box in pixel coordinates of the original image."""

    x1: float
    y1: float
    x2: float
    y2: float

    @model_validator(mode="after")
    def _ordered(self) -> Self:
        if self.x2 < self.x1 or self.y2 < self.y1:
            raise ValueError("box corners must satisfy x1 <= x2 and y1 <= y2")
        return self


class DamageFinding(Frozen):
    photo_id: str
    damage_type: DamageType
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: BoundingBox
    image_area_fraction: float = Field(ge=0.0, le=1.0)
    part: str | None = None
    part_area_ratio: float | None = Field(default=None, ge=0.0)


class FraudSignal(Frozen):
    kind: str
    score: float = Field(ge=0.0, le=1.0)
    detail: str


class CostEstimate(Frozen):
    low: int = Field(ge=0)
    high: int = Field(ge=0)
    currency: str = "USD"
    basis: str

    @model_validator(mode="after")
    def _ordered(self) -> Self:
        if self.low > self.high:
            raise ValueError("low must not exceed high")
        return self


class Coverage(Frozen):
    policy_id: str
    found: bool
    active: bool
    collision: bool
    deductible: int = Field(ge=0)

    @property
    def confirmed(self) -> bool:
        return self.found and self.active and self.collision


class AgentRecommendation(Frozen):
    route_suggestion: Route
    confidence: Confidence
    rationale: str
    citations: tuple[str, ...]
    open_questions: tuple[str, ...] = ()


class Decision(Frozen):
    route: Route
    rule_id: str
    reason: str
    policy_version: str
