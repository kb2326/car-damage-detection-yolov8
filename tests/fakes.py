"""Test doubles shared across test modules."""

from __future__ import annotations

from pathlib import Path
from uuid import UUID

from claimlens.agent import TriageAgent
from claimlens.agent.stub import StubTriageAgent
from claimlens.blobs import BlobStore
from claimlens.decision import load_decision_config
from claimlens.domain import (
    AgentRecommendation,
    BoundingBox,
    Coverage,
    DamageFinding,
    DamageType,
    Route,
)
from claimlens.events.projection import ClaimState
from claimlens.events.store import SQLiteEventStore
from claimlens.policy import PolicyRepository, load_policies
from claimlens.pricing import load_rate_card
from claimlens.workflow import PipelineDeps

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "config"


class SimulatedCrashError(BaseException):
    """Stands in for the process dying mid-pipeline. Stage error handling does not catch it."""


def dent(photo_id: str = "p1", confidence: float = 0.9) -> DamageFinding:
    return DamageFinding(
        photo_id=photo_id,
        damage_type=DamageType.DENT,
        confidence=confidence,
        bbox=BoundingBox(x1=0, y1=0, x2=64, y2=64),
        image_area_fraction=0.01,
    )


class FakeDetector:
    model_version = "fake-detector-v1"

    def __init__(
        self,
        *,
        findings: dict[str, list[DamageFinding]] | None = None,
        fail_photos: frozenset[str] | None = None,
        crash_on: str | None = None,
    ) -> None:
        self._findings = findings
        self._fail_photos = fail_photos or frozenset()
        self._crash_on = crash_on
        self.calls: list[str] = []

    def detect(self, image_path: Path, photo_id: str) -> list[DamageFinding]:
        self.calls.append(photo_id)
        if photo_id == self._crash_on:
            raise SimulatedCrashError(photo_id)
        if photo_id in self._fail_photos:
            raise RuntimeError("model server unavailable")
        if self._findings is None:
            return [dent(photo_id)]
        return list(self._findings.get(photo_id, []))


class FailingAgent:
    agent_version = "failing-agent-v1"

    def recommend(self, state: ClaimState) -> AgentRecommendation:
        raise RuntimeError("LLM provider timeout")


class FailingPolicies(PolicyRepository):
    """Policy system that is down, like a remote service timing out."""

    def __init__(self) -> None:
        super().__init__([])

    def get_coverage(self, policy_id: str) -> Coverage:
        raise ConnectionError("policy system unavailable")


def make_test_deps(
    workdir: Path,
    store: SQLiteEventStore,
    detector: FakeDetector,
    agent: TriageAgent | None = None,
    policies: PolicyRepository | None = None,
) -> PipelineDeps:
    return PipelineDeps(
        store=store,
        blobs=BlobStore(workdir / "blobs"),
        detector=detector,
        policies=policies or load_policies(CONFIG_DIR / "policies.toml"),
        rate_card=load_rate_card(CONFIG_DIR / "rate_card.toml"),
        decision_config=load_decision_config(CONFIG_DIR / "decision_policy.toml"),
        agent=agent or StubTriageAgent(),
    )


def decided_claim(tmp_path: Path, route: Route) -> tuple[SQLiteEventStore, UUID]:
    """A store with one claim decided on `route` (for review-queue and payment tests)."""
    from uuid import uuid4

    from claimlens.domain import Decision
    from claimlens.events.envelope import Actor, ActorKind
    from claimlens.events.payloads import ClaimReported, RouteDecided

    store = SQLiteEventStore(tmp_path / "claims.db")
    claim = uuid4()
    system = Actor(kind=ActorKind.SYSTEM, name="test")
    store.append(claim, ClaimReported(policy_id="P-1001", description="scrape"), system)
    decision = Decision(route=route, rule_id="R7", reason="test", policy_version="v")
    store.append(claim, RouteDecided(decision=decision), system)
    return store, claim


def undecided_claim(tmp_path: Path) -> tuple[SQLiteEventStore, UUID]:
    from uuid import uuid4

    from claimlens.events.envelope import Actor, ActorKind
    from claimlens.events.payloads import ClaimReported

    store = SQLiteEventStore(tmp_path / "claims.db")
    claim = uuid4()
    store.append(
        claim,
        ClaimReported(policy_id="P-1001", description="scrape"),
        Actor(kind=ActorKind.SYSTEM, name="test"),
    )
    return store, claim
