"""Drive MCP servers through the SDK's in-memory client (same protocol as Claude Code)."""

from __future__ import annotations

from typing import Any

import anyio
from mcp.client import Client
from mcp.types import CallToolResult

from claimlens.mcp.base import AuditRecord, ScopedServer


class MemoryAudit:
    def __init__(self) -> None:
        self.records: list[AuditRecord] = []

    def __call__(self, record: AuditRecord) -> None:
        self.records.append(record)


def call(server: ScopedServer, tool: str, args: dict[str, Any]) -> CallToolResult:
    async def run() -> CallToolResult:
        async with Client(server) as client:
            return await client.call_tool(tool, args)

    return anyio.run(run)


def tool_names(server: ScopedServer) -> list[str]:
    async def run() -> list[str]:
        async with Client(server) as client:
            return sorted(t.name for t in (await client.list_tools()).tools)

    return anyio.run(run)


def error_text(result: CallToolResult) -> str:
    return " ".join(getattr(c, "text", "") for c in result.content)
