"""`claimlens memory rebuild | show <claim> | forget <claim>`."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path
from uuid import UUID

from claimlens.blobs import BlobStore
from claimlens.events.envelope import Actor, ActorKind
from claimlens.events.payloads import MemoryForgotten
from claimlens.events.store import SQLiteEventStore
from claimlens.knowledge.embed import Embedder
from claimlens.memory.index import ClaimMemory, rebuild

MEMORY_PATH = Path("var/memory")


def add_memory_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    memory = sub.add_parser("memory", help="the claim memory: rebuild, show, forget")
    msub = memory.add_subparsers(dest="memory_command", required=True)
    msub.add_parser("rebuild", help="rebuild memory from every decided claim's log")
    for name in ("show", "forget"):
        p = msub.add_parser(name)
        p.add_argument("claim_id", type=UUID)
    memory.add_argument("--memory-dir", type=Path, default=MEMORY_PATH)


def run_memory_command(args: argparse.Namespace, embedder_factory: Callable[[], Embedder]) -> int:
    memory = ClaimMemory(args.memory_dir, embedder_factory())
    store = SQLiteEventStore(args.db)
    try:
        if args.memory_command == "rebuild":
            result = rebuild(memory, store, BlobStore(args.blobs).path)
            print(f"Remembered {result.remembered} claim(s) in {args.memory_dir}")
            for line in result.skipped:
                print(f"Skipped {line}")
            return 1 if result.skipped else 0
        if args.memory_command == "show":
            record = memory.get(str(args.claim_id))
            if record is None:
                print(f"Claim {args.claim_id} is not in memory.")
                return 1
            print(record.model_dump_json(indent=1))
            return 0
        if args.claim_id not in set(store.claim_ids()):
            print(f"There is no claim {args.claim_id}.")
            return 1
        # The event comes first, so a rebuild or a later review never brings the claim back,
        # even when it was never in memory or the delete below fails.
        store.append(
            args.claim_id,
            MemoryForgotten(reason="removed with claimlens memory forget"),
            Actor(kind=ActorKind.HUMAN, name="operator"),
        )
        removed = memory.forget(str(args.claim_id))
        print(
            f"Forgot claim {args.claim_id}"
            + ("" if removed else " (it was not in memory)")
            + "; its log is kept."
        )
        return 0
    finally:
        store.close()
