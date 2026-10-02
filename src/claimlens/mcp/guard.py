"""Pure checks shared by every MCP server: scopes, photo paths, human approval tokens."""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
from collections.abc import Sequence
from datetime import datetime, timedelta
from pathlib import Path

from claimlens.mcp.profiles import Profile

IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".webp"})


class GuardError(Exception):
    """A refusal that is safe to show to the client."""


class ScopeDenied(GuardError):  # noqa: N818 - spec name, shown to clients
    def __init__(self, profile: str, server: str, tool: str) -> None:
        super().__init__(f"ScopeDenied: profile {profile!r} may not call {server}.{tool}")


class PathRejected(GuardError):  # noqa: N818 - spec name, shown to clients
    pass


class ApprovalInvalid(GuardError):  # noqa: N818 - spec name, shown to clients
    pass


def check_scope(profile: Profile, server: str, tool: str) -> None:
    if not profile.allows(server, tool):
        raise ScopeDenied(profile.name, server, tool)


def safe_photo_path(path: str, roots: Sequence[Path]) -> Path:
    candidate = Path(path)
    resolved = (candidate if candidate.is_absolute() else Path.cwd() / candidate).resolve()
    if resolved.suffix.lower() not in IMAGE_SUFFIXES:
        raise PathRejected(f"PathRejected: not an image file: {path}")
    if not any(resolved.is_relative_to(root.resolve()) for root in roots):
        raise PathRejected(f"PathRejected: {path} is outside the allowed folders")
    if not resolved.is_file():
        raise PathRejected(f"PathRejected: photo not found: {path}")
    return resolved


def _sign(body: str, secret: str) -> str:
    return hmac.new(secret.encode(), body.encode(), hashlib.sha256).hexdigest()


def issue_approval(
    claim_id: str,
    amount_usd: int,
    secret: str,
    *,
    now: datetime,
    ttl: timedelta = timedelta(hours=24),
) -> str:
    body = f"{claim_id}|{amount_usd}|{int((now + ttl).timestamp())}"
    return base64.urlsafe_b64encode(f"{body}|{_sign(body, secret)}".encode()).decode()


def verify_approval(
    token: str, claim_id: str, amount_usd: int, secret: str, *, now: datetime
) -> None:
    if not secret:
        raise ApprovalInvalid("payments are disabled: set CLAIMLENS_APPROVAL_SECRET in .env")
    try:
        token_claim, token_amount, expiry, signature = (
            base64.urlsafe_b64decode(token.encode()).decode().split("|")
        )
        expires_at = int(expiry)
    except (ValueError, binascii.Error, UnicodeDecodeError):
        raise ApprovalInvalid("ApprovalInvalid: malformed token") from None
    body = f"{token_claim}|{token_amount}|{expiry}"
    if not hmac.compare_digest(signature, _sign(body, secret)):
        raise ApprovalInvalid("ApprovalInvalid: signature does not match")
    if token_claim != claim_id or token_amount != str(amount_usd):
        raise ApprovalInvalid("ApprovalInvalid: token is for a different claim or amount")
    if now.timestamp() > expires_at:
        raise ApprovalInvalid("ApprovalInvalid: token expired")
