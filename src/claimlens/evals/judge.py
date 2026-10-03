"""LLM judge for agent recommendations. It scores the reasoning; it never sees the answer key."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from claimlens.domain import AgentRecommendation, Frozen
from claimlens.llm.gateway import Gateway
from claimlens.llm.prompts import Prompt
from claimlens.llm.types import InvalidModelOutput, LLMRequest, Message


class JudgeAnswer(BaseModel):
    model_config = ConfigDict(extra="ignore")  # anything else the model adds is dropped

    grounded: bool
    grounded_reason: str
    citation_relevant: bool
    citation_relevant_reason: str
    actionable: bool
    actionable_reason: str


class JudgeItem(Frozen):
    case_id: str
    evidence: str
    recommendation: AgentRecommendation
    clauses: dict[str, str]


class Judgement(Frozen):
    case_id: str
    verdict: Literal["pass", "fail"]
    answer: JudgeAnswer
    judge_version: str
    cost_usd: float


def render_judge_input(item: JudgeItem) -> str:
    rec = item.recommendation
    clauses = "\n".join(f"{cid}: {text}" for cid, text in sorted(item.clauses.items())) or "(none)"
    questions = "\n".join(f"- {q}" for q in rec.open_questions) or "(none)"
    return (
        f"## Evidence\n{item.evidence}\n\n"
        f"## Cited policy clauses\n{clauses}\n\n"
        f"## Recommendation to assess\n"
        f"Suggested route: {rec.route_suggestion.value}\n"
        f"Confidence: {rec.confidence.value}\n"
        f"Rationale: {rec.rationale}\n"
        f"Open questions:\n{questions}"
    )


def judge_item(gateway: Gateway, prompt: Prompt, item: JudgeItem) -> Judgement:
    response = gateway.generate(
        LLMRequest(
            messages=[Message(role="user", content=render_judge_input(item))],
            tier="strong",
            system=prompt.text,
            output_schema=JudgeAnswer,
            prompt_id=prompt.id,
        )
    )
    answer = response.parsed
    if not isinstance(answer, JudgeAnswer):
        raise InvalidModelOutput("the judge returned no structured answer")
    ok = answer.grounded and answer.citation_relevant and answer.actionable
    return Judgement(
        case_id=item.case_id,
        verdict="pass" if ok else "fail",
        answer=answer,
        judge_version=prompt.id,
        cost_usd=response.cost_usd,
    )
