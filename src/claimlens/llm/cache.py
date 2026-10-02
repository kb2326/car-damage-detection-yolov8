"""Response cache: an identical request is answered from disk, for free and reproducibly."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Sequence
from pathlib import Path

from claimlens.domain import Frozen
from claimlens.llm.types import Message

_SCHEMA = "CREATE TABLE IF NOT EXISTS replies (key TEXT PRIMARY KEY, body TEXT NOT NULL)"


class CachedReply(Frozen):
    text: str
    model: str
    input_tokens: int
    output_tokens: int


class ResponseCache:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), isolation_level=None)
        self._conn.execute(_SCHEMA)

    @staticmethod
    def key(
        model: str,
        system: str,
        messages: Sequence[Message],
        schema_name: str | None,
        max_tokens: int,
    ) -> str:
        payload = json.dumps(
            [model, system, [m.model_dump() for m in messages], schema_name, max_tokens],
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    def get(self, key: str) -> CachedReply | None:
        row = self._conn.execute("SELECT body FROM replies WHERE key = ?", (key,)).fetchone()
        return None if row is None else CachedReply.model_validate_json(row[0])

    def put(self, key: str, reply: CachedReply) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO replies (key, body) VALUES (?, ?)",
            (key, reply.model_dump_json()),
        )
