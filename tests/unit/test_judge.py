import json
from datetime import date
from pathlib import Path

import pytest

from claimlens.domain import AgentRecommendation, Confidence, Route
from claimlens.evals.judge import JudgeItem, judge_item, render_judge_input
from claimlens.llm.budget import Budget
from claimlens.llm.cache import ResponseCache
from claimlens.llm.config import load_llm_config
from claimlens.llm.gateway import Gateway
from claimlens.llm.prompts import load_prompt
from claimlens.llm.provider import FakeProvider, ProviderReply
from claimlens.llm.types import InvalidModelOutput

ROOT = Path(__file__).resolve().parents[2]
LLM = load_llm_config(ROOT / "config" / "llm.toml")
PROMPT = load_prompt(ROOT / "prompts", "judge", "v1")
ITEM = JudgeItem(
    case_id="n001",
    evidence=(
        "<claimant_description>Delivering pizza.</claimant_description>\n- [E1] dent on front_door"
    ),
    recommendation=AgentRecommendation(
        route_suggestion=Route.ADJUSTER_REVIEW,
        confidence=Confidence.HIGH,
        rationale="The car was used for deliveries, which STD-8.5 excludes.",
        citations=("evt-1",),
        policy_citations=("STD-8.5",),
    ),
    clauses={"STD-8.5": "We do not cover damage while your vehicle is used for hire, delivery ..."},
)


def _reply(**answers: object) -> ProviderReply:
    base = {
        "grounded": True,
        "grounded_reason": "ok",
        "citation_relevant": True,
        "citation_relevant_reason": "ok",
        "actionable": True,
        "actionable_reason": "ok",
    }
    return ProviderReply(json.dumps({**base, **answers}), 900, 80)


def _gateway(
    tmp_path: Path, script: list[ProviderReply | Exception]
) -> tuple[Gateway, FakeProvider]:
    fake = FakeProvider(script)
    return (
        Gateway(
            LLM,
            fake,
            Budget(tmp_path / "b.sqlite", LLM.limits, today=lambda: date(2026, 10, 2)),
            ResponseCache(tmp_path / "c.sqlite"),
            lambda c: None,
            sleep=lambda s: None,
        ),
        fake,
    )


def test_all_yes_is_a_pass(tmp_path: Path) -> None:
    gateway, fake = _gateway(tmp_path, [_reply()])
    judgement = judge_item(gateway, PROMPT, ITEM)
    assert judgement.verdict == "pass"
    assert judgement.judge_version == "judge/v1"
    assert fake.calls[0]["system"].startswith(PROMPT.text)


def test_any_no_is_a_fail(tmp_path: Path) -> None:
    gateway, _ = _gateway(
        tmp_path, [_reply(grounded=False, grounded_reason="No delivery in evidence.")]
    )
    judgement = judge_item(gateway, PROMPT, ITEM)
    assert judgement.verdict == "fail"
    assert judgement.answer.grounded_reason == "No delivery in evidence."


def test_the_verdict_is_computed_not_read_from_the_model(tmp_path: Path) -> None:
    sneaky = ProviderReply(
        json.dumps(
            {
                "grounded": False,
                "grounded_reason": "x",
                "citation_relevant": True,
                "citation_relevant_reason": "x",
                "actionable": True,
                "actionable_reason": "x",
                "verdict": "pass",
            }
        ),
        900,
        80,
    )
    gateway, _ = _gateway(tmp_path, [sneaky, _reply(grounded=False)])
    assert judge_item(gateway, PROMPT, ITEM).verdict == "fail"


def test_the_judge_input_has_the_clause_text_and_not_the_expected_route() -> None:
    text = render_judge_input(ITEM)
    assert "STD-8.5: We do not cover damage" in text
    assert "The car was used for deliveries" in text
    assert "expected" not in text.lower()


def test_invalid_judge_output_twice_is_an_error(tmp_path: Path) -> None:
    gateway, _ = _gateway(
        tmp_path, [ProviderReply("no json", 10, 5), ProviderReply("still none", 10, 5)]
    )
    with pytest.raises(InvalidModelOutput):
        judge_item(gateway, PROMPT, ITEM)
