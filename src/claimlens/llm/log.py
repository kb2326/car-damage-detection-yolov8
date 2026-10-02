"""Call records: what each LLM call cost and how it went. Never the key or the prompt text."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from uuid import UUID

from claimlens.domain import Frozen
from claimlens.events.envelope import Actor, ActorKind
from claimlens.events.payloads import LLMCalled
from claimlens.events.store import SQLiteEventStore


class LLMCall(Frozen):
    request_id: str
    claim_id: str | None
    prompt_id: str | None
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: int
    cached: bool
    attempts: int
    outcome: str
    prompt_sha256: str


CallLog = Callable[[LLMCall], None]


def jsonl_call_log(path: Path) -> CallLog:
    def log(call: LLMCall) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(call.model_dump_json() + "\n")

    return log


def event_call_log(store_path: Path, fallback: CallLog) -> CallLog:
    """Append to the claim's event log when the call belongs to a known claim."""

    def log(call: LLMCall) -> None:
        fallback(call)
        if call.claim_id is None:
            return
        try:
            uuid = UUID(call.claim_id)
        except ValueError:
            return
        store = SQLiteEventStore(store_path)
        try:
            if uuid in store.claim_ids():
                store.append(
                    uuid,
                    LLMCalled(
                        request_id=call.request_id,
                        model=call.model,
                        prompt_id=call.prompt_id,
                        input_tokens=call.input_tokens,
                        output_tokens=call.output_tokens,
                        cost_usd=call.cost_usd,
                        cached=call.cached,
                        outcome=call.outcome,
                    ),
                    Actor(kind=ActorKind.SYSTEM, name="llm-gateway"),
                )
        finally:
            store.close()

    return log
