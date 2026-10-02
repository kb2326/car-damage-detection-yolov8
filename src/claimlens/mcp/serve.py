"""Assemble one MCP server for a profile from repo config (used by `claimlens mcp`)."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

from claimlens.data.config import read_secret
from claimlens.data.taxonomy import load_part_groups
from claimlens.mcp.base import ScopedServer, jsonl_audit
from claimlens.mcp.claims_system import build_claims_system, event_audit
from claimlens.mcp.payments import build_payments
from claimlens.mcp.policy_admin import build_policy_admin
from claimlens.mcp.profiles import OPERATOR, Profile, load_profiles
from claimlens.mcp.vision import VisionDeps, build_vision
from claimlens.policy import load_policies
from claimlens.pricing import load_rate_card
from claimlens.quality import QualityConfig

PHOTO_ROOTS = (Path("var/blobs"), Path("data"), Path("tests/fixtures"))


def _profile(name: str, config_dir: Path) -> Profile:
    if name == OPERATOR.name:
        return OPERATOR
    profiles = load_profiles(config_dir / "agents.toml")
    if name not in profiles:
        raise ValueError(f"unknown profile {name!r}; known: {sorted(profiles)} or 'operator'")
    return profiles[name]


def build_server(
    name: str,
    profile_name: str,
    config_dir: Path,
    db_path: Path,
    *,
    damage_factory: Callable[[], Any],
    parts_factory: Callable[[], Any],
) -> ScopedServer:
    profile = _profile(profile_name, config_dir)
    fallback = jsonl_audit(Path("var/mcp-audit.jsonl"))
    if name == "vision":
        deps = VisionDeps(
            damage=damage_factory,
            parts=parts_factory,
            groups=load_part_groups(config_dir / "taxonomy.toml"),
            rate_card=load_rate_card(config_dir / "rate_card.toml"),
            quality=QualityConfig(),
            roots=PHOTO_ROOTS,
        )
        return build_vision(profile, deps, fallback)
    if name == "policy-admin":
        return build_policy_admin(profile, load_policies(config_dir / "policies.toml"), fallback)
    if name == "claims-system":
        return build_claims_system(profile, db_path, event_audit(db_path, fallback))
    if name == "payments":
        secret = read_secret("CLAIMLENS_APPROVAL_SECRET") or ""
        return build_payments(profile, db_path, secret, event_audit(db_path, fallback))
    raise ValueError(f"unknown MCP server {name!r}")
