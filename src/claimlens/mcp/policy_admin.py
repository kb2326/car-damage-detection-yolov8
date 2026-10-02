"""MCP `policy-admin` server (mock policy system)."""

from claimlens.mcp.base import AuditSink, ScopedServer
from claimlens.mcp.profiles import Profile
from claimlens.mcp.schemas import CoverageResult, PolicySummary
from claimlens.policy import PolicyRepository


def build_policy_admin(
    profile: Profile, policies: PolicyRepository, audit: AuditSink
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

    server.register(get_policy, "get_policy", "Look up a policy by id.")
    server.register(get_coverage, "get_coverage", "Check what a policy covers.")
    return server
