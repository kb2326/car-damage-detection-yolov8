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
            count = rebuild(memory, store, BlobStore(args.blobs).path)
            print(f"Remembered {count} claim(s) in {args.memory_dir}")
            return 0
        if args.memory_command == "show":
            record = memory.get(str(args.claim_id))
            if record is None:
                print(f"Claim {args.claim_id} is not in memory.")
                return 1
            print(record.model_dump_json(indent=1))
            return 0
        if not memory.forget(str(args.claim_id)):
            print(f"Claim {args.claim_id} is not in memory.")
            return 1
        store.append(
            args.claim_id,
            MemoryForgotten(reason="removed with claimlens memory forget"),
            Actor(kind=ActorKind.HUMAN, name="operator"),
        )
        print(f"Forgot claim {args.claim_id}; its log is kept.")
        return 0
    finally:
        store.close()
