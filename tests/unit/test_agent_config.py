from pathlib import Path

import pytest

from claimlens.agent.config import load_agent_config
from claimlens.mcp.profiles import load_profiles

ROOT = Path(__file__).resolve().parents[2]


def test_repo_agent_config_loads() -> None:
    config = load_agent_config(ROOT / "config" / "agent.toml")
    assert config.version == "triage-agent-v1"
    assert config.tier == "strong"
    assert (config.prompt_name, config.prompt_version) == ("triage", "v2")
    assert config.max_steps == 6


def test_agent_tools_are_read_only_and_inside_the_profile() -> None:
    config = load_agent_config(ROOT / "config" / "agent.toml")
    profile = load_profiles(ROOT / "config" / "agents.toml")[config.profile]
    allowed = {tool for tools in profile.tools.values() for tool in tools}
    assert set(config.tools) <= allowed
    assert not set(config.tools) & {"add_note", "assign_queue", "issue_payment"}


def test_write_tools_are_refused(tmp_path: Path) -> None:
    text = (ROOT / "config" / "agent.toml").read_text(encoding="utf-8")
    bad = tmp_path / "agent.toml"
    bad.write_text(text.replace('"get_policy"', '"add_note"'), encoding="utf-8")
    with pytest.raises(ValueError, match="read-only"):
        load_agent_config(bad)
