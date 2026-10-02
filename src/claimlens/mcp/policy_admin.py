"""MCP `policy-admin` server (mock policy system): policy lookups and citable policy wording."""

from collections.abc import Callable
from typing import Annotated

from pydantic import Field

from claimlens.knowledge.index import PolicyIndex
from claimlens.mcp.base import AuditSink, ScopedServer, tool_errors
from claimlens.mcp.profiles import Profile
from claimlens.mcp.schemas import (
    ClauseOut,
    ClauseResults,
    CoverageResult,
    PolicySummary,
)
from claimlens.policy import PolicyRepository


def build_policy_admin(
    profile: Profile,
    policies: PolicyRepository,
    audit: AuditSink,
    index: Callable[[], PolicyIndex] | None = None,
) -> ScopedServer:
    server = ScopedServer("policy-admin", profile, audit)

    def get_policy(policy_id: str) -> PolicySummary:
        record = policies.get_record(policy_id)
        if record is None:
            return PolicySummary(found=False, policy_id=policy_id)
        return PolicySummary(
            found=True,
            policy_id=policy_id,
            status=record.status.value,
            collision=record.collision,
            deductible=record.deductible,
        )

    def get_coverage(policy_id: str) -> CoverageResult:
        c = policies.get_coverage(policy_id)
        return CoverageResult(
            found=c.found, active=c.active, collision=c.collision, deductible=c.deductible
        )

    def search_policy_clauses(
        query: Annotated[str, Field(min_length=2, max_length=500)],
        policy_id: str | None = None,
        k: Annotated[int, Field(ge=1, le=10)] = 5,
    ) -> ClauseResults:
        with tool_errors():
            if index is None:
                raise ValueError("policy search is not configured on this server")
            wording = None
            if policy_id is not None:
                record = policies.get_record(policy_id)
                if record is None:
                    raise ValueError(f"unknown policy {policy_id}")
                wording = record.wording
            hits = index().search(query, wording=wording, k=k)
            return ClauseResults(
                wording=wording,
                clauses=[
                    ClauseOut(
                        clause_id=h.clause_id, title=h.title, text=h.text, score=round(h.score, 4)
                    )
                    for h in hits
                ],
            )

    server.register(get_policy, "get_policy", "Look up a policy by id.")
    server.register(get_coverage, "get_coverage", "Check what a policy covers.")
    server.register(
        search_policy_clauses,
        "search_policy_clauses",
        "Search the policy wording; returns clauses with ids you can cite.",
    )
    return server
