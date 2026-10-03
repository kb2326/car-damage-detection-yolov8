"""Build the real LLM triage agent from repo config."""

from __future__ import annotations

from pathlib import Path

from claimlens.agent.config import load_agent_config
from claimlens.agent.langgraph_agent import LangGraphTriageAgent
from claimlens.knowledge.index import PolicyIndex
from claimlens.llm.factory import build_gateway
from claimlens.llm.gateway import Gateway
from claimlens.llm.prompts import load_prompt
from claimlens.mcp.base import jsonl_audit
from claimlens.mcp.claims_system import build_claims_system, event_audit
from claimlens.mcp.policy_admin import build_policy_admin
from claimlens.mcp.profiles import load_profiles
from claimlens.policy import load_policies


def build_llm_agent(
    config_dir: Path,
    repo_root: Path,
    store_path: Path,
    *,
    gateway: Gateway | None = None,
    index: PolicyIndex | None = None,
    per_day_usd: float | None = None,
) -> LangGraphTriageAgent:
    config = load_agent_config(config_dir / "agent.toml")
    profile = load_profiles(config_dir / "agents.toml")[config.profile]
    policies = load_policies(config_dir / "policies.toml")
    audit = event_audit(store_path, jsonl_audit(repo_root / "var" / "mcp-audit.jsonl"))
    cache: list[PolicyIndex] = [] if index is None else [index]

    def get_index() -> PolicyIndex:
        if not cache:  # the embedding model loads on first use
            from claimlens.knowledge.fastembedder import FastEmbedder

            cache.append(PolicyIndex.open(repo_root / "var" / "lancedb", FastEmbedder()))
        return cache[0]

    servers = [
        build_policy_admin(profile, policies, audit, get_index),
        build_claims_system(profile, store_path, audit),
    ]
    return LangGraphTriageAgent(
        gateway or build_gateway(config_dir, repo_root, store_path, per_day_usd=per_day_usd),
        config,
        load_prompt(repo_root / "prompts", config.prompt_name, config.prompt_version),
        servers,
        policies,
        get_index,
    )
