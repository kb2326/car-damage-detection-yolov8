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
