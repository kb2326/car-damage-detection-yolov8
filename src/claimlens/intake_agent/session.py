"""Intake sessions: start, reply, pause and resume, and the hand-over to the claims pipeline."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any
from uuid import UUID

from langgraph.types import Command

from claimlens.events.envelope import Actor, ActorKind
from claimlens.events.payloads import IntakeCompleted
from claimlens.intake import submit_claim
from claimlens.intake_agent.config import IntakeConfig
from claimlens.intake_agent.graph import IntakeState, initial_state
from claimlens.workflow import PipelineDeps, process_claim

KICKOFF = "A customer has opened a new claim chat. Greet them and ask for their policy number."


@dataclass(frozen=True)
class AgentTurn:
    session_id: str
    message: str  # what the assistant says to the customer
    photo_kind: str | None  # the photo the assistant is waiting for, if any
    claim_id: str | None  # set once the claim has been submitted


class IntakeSessions:
    """A thin front end over the checkpointed intake graph (the CLI now, a web app later)."""

    def __init__(self, graph: Any, system: str, kickoff: str = KICKOFF) -> None:
        self._graph = graph
        self._system = system
        self._kickoff = kickoff

    @staticmethod
    def _config(session_id: str) -> dict[str, Any]:
        return {"configurable": {"thread_id": session_id}}

    def _turn(self, session_id: str, out: dict[str, Any]) -> AgentTurn:
        if out.get("__interrupt__"):
            value = out["__interrupt__"][0].value
            return AgentTurn(session_id, value["message"], value["photo_kind"], None)
        return AgentTurn(session_id, out["outgoing"], None, out["claim_id"])

    def start(self) -> AgentTurn:
        session_id = uuid.uuid4().hex[:12]
        out = self._graph.invoke(
            initial_state(self._system, self._kickoff, session_id), self._config(session_id)
        )
        return self._turn(session_id, out)

    def pending(self, session_id: str) -> AgentTurn | None:
        """The message a paused session is waiting on, or None if it is finished or unknown."""
        snapshot = self._graph.get_state(self._config(session_id))
        for task in snapshot.tasks:
            for paused in task.interrupts:
                value = paused.value
                return AgentTurn(session_id, value["message"], value["photo_kind"], None)
        return None

    def reply(self, session_id: str, text: str, photo: Path | None = None) -> AgentTurn:
        if self.pending(session_id) is None:
            raise ValueError(f"no open intake session {session_id!r}")
        resume = {"text": text, "photo": str(photo) if photo else None}
        out = self._graph.invoke(Command(resume=resume), self._config(session_id))
        return self._turn(session_id, out)

    def open_sessions(self) -> list[str]:
        saver = self._graph.checkpointer
        ids = {c.config["configurable"]["thread_id"] for c in saver.list(None)}
        return sorted(i for i in ids if self.pending(i) is not None)


def transcript_sha256(transcript: Sequence[dict[str, str]]) -> str:
    """A fingerprint of the conversation for the claim's log; the text itself stays private."""
    return hashlib.sha256(json.dumps(list(transcript), sort_keys=True).encode()).hexdigest()


def pipeline_submitter(
    deps_factory: Callable[[], PipelineDeps], config: IntakeConfig, *, process: bool
) -> Callable[[IntakeState], str]:
    """File the collected claim, record IntakeCompleted, and (optionally) run the pipeline."""

    def submit(state: IntakeState) -> str:
        deps = deps_factory()
        photos = state["photos"]
        kinds = [k for k in config.photo_kinds if k in photos]
        kinds += sorted(k for k in photos if k not in config.photo_kinds)
        facts = state["facts"]
        claim_id: UUID = submit_claim(
            deps.store,
            deps.blobs,
            policy_id=facts.get("policy_id", "unknown"),
            description=facts.get("what_happened", ""),
            photo_paths=[Path(photos[k]) for k in kinds],
            allow_no_photos=True,
        )
        deps.store.append(
            claim_id,
            IntakeCompleted(
                session_id=state["session_id"],
                facts=dict(facts),
                photo_kinds={kind: f"p{n}" for n, kind in enumerate(kinds, start=1)},
                photo_gaps=dict(state["gaps"]),
                turns=state["turns"],
                retakes=sum(state["retakes"].values()),
                transcript_sha256=transcript_sha256(state["transcript"]),
            ),
            Actor(kind=ActorKind.AGENT, name=config.version),
        )
        if process:
            process_claim(claim_id, deps)
        return str(claim_id)

    return submit


def build_intake(
    config_dir: Path,
    repo_root: Path,
    checkpoint_path: Path,
    deps_factory: Callable[[], PipelineDeps],
    *,
    gateway: Any = None,
    process: bool = True,
    today: Callable[[], date] = date.today,
) -> IntakeSessions:
    """The real wiring: gateway on the fast tier, the intake profile's policy lookup, and a
    SQLite checkpointer so sessions survive restarts."""
    from langgraph.checkpoint.sqlite import SqliteSaver

    from claimlens.agent.chat_model import GatewayChatModel
    from claimlens.agent.tools import load_tools
    from claimlens.intake_agent.config import load_intake_config
    from claimlens.intake_agent.graph import build_intake_graph
    from claimlens.llm.factory import build_gateway
    from claimlens.llm.prompts import load_prompt
    from claimlens.mcp.base import jsonl_audit
    from claimlens.mcp.policy_admin import build_policy_admin
    from claimlens.mcp.profiles import load_profiles
    from claimlens.policy import load_policies

    config = load_intake_config(config_dir / "intake.toml")
    prompt = load_prompt(repo_root / "prompts", config.prompt_name, config.prompt_version)
    profile = load_profiles(config_dir / "agents.toml")[config.profile]
    server = build_policy_admin(
        profile,
        load_policies(config_dir / "policies.toml"),
        jsonl_audit(repo_root / "var" / "mcp-audit.jsonl"),
    )
    (get_policy,) = load_tools([server], allow=["get_policy"], bound={})

    def lookup(policy_id: str) -> str:
        return str(get_policy.invoke({"policy_id": policy_id}))

    model = GatewayChatModel(
        gateway=gateway or build_gateway(config_dir, repo_root),
        tier=config.tier,
        prompt_id=prompt.id,
        max_tokens=config.max_tokens,
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(checkpoint_path), check_same_thread=False)
    graph = build_intake_graph(
        model,
        lookup,
        config,
        pipeline_submitter(deps_factory, config, process=process),
        today,
        checkpointer=SqliteSaver(conn),
    )
    return IntakeSessions(graph, system=prompt.text)
