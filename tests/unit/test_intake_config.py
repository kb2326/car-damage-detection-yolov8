from pathlib import Path

from claimlens.intake_agent.config import load_intake_config

ROOT = Path(__file__).resolve().parents[2]


def test_repo_intake_config_loads() -> None:
    config = load_intake_config(ROOT / "config" / "intake.toml")
    assert config.tier == "fast"
    assert (config.prompt_name, config.prompt_version) == ("intake", "v1")
    assert config.photo_kinds == ("overview", "damage_closeup", "plate")
    assert config.max_retakes == 2
    assert config.min_brightness == 40
