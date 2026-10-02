"""Agent profiles: which MCP tools each role may see and call."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

from claimlens.domain import Frozen

SERVERS: dict[str, tuple[str, ...]] = {
    "vision": ("segment_damage", "segment_parts", "assess_quality"),
    "policy-admin": ("get_policy", "get_coverage"),
    "claims-system": ("get_claim_history", "find_similar_claims", "add_note", "assign_queue"),
    "payments": ("issue_payment",),
}
WRITE_TOOLS = frozenset({"add_note", "assign_queue", "issue_payment"})


class Profile(Frozen):
    name: str
    tools: dict[str, tuple[str, ...]]

    def allows(self, server: str, tool: str) -> bool:
        return tool in self.tools.get(server, ())


# A person operating payments; never loadable from agents.toml.
OPERATOR = Profile(name="operator", tools={"payments": ("issue_payment",)})


def load_profiles(path: Path) -> dict[str, Profile]:
    data: dict[str, Any] = tomllib.loads(path.read_text(encoding="utf-8"))
    profiles: dict[str, Profile] = {}
    for name, servers in data.get("profiles", {}).items():
        if name == OPERATOR.name:
            raise ValueError(f"profile name {name!r} is reserved for the human payments operator")
        tools: dict[str, tuple[str, ...]] = {}
        for server, names in servers.items():
            if server not in SERVERS:
                raise ValueError(f"profile {name}: unknown server {server!r}")
            for tool in names:
                if tool not in SERVERS[server]:
                    raise ValueError(f"profile {name}: unknown tool {server}.{tool}")
                if tool == "issue_payment":
                    raise ValueError(
                        f"profile {name}: no agent profile may list issue_payment; "
                        "payments need a human operator and an approval token"
                    )
            tools[server] = tuple(names)
        profiles[name] = Profile(name=name, tools=tools)
    return profiles
