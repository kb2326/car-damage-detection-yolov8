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
