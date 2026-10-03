"""LangChain tools backed by our MCP servers. Scopes and audit stay in force: every call goes
through `ScopedServer` by the MCP in-memory client, as Claude Code's calls do over stdio."""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from typing import Any

import anyio
from langchain_core.tools import StructuredTool, ToolException
from mcp.client import Client

from claimlens.mcp.base import ScopedServer


def _text(result: Any) -> str:
    return "\n".join(getattr(block, "text", "") for block in result.content)


def _hide(schema: dict[str, Any], names: Sequence[str]) -> dict[str, Any]:
    out = copy.deepcopy(schema)
    for name in names:
        out.get("properties", {}).pop(name, None)
    out["required"] = [r for r in out.get("required", []) if r not in names]
    return out


def _make(server: ScopedServer, spec: Any, bound: Mapping[str, str]) -> StructuredTool:
    fixed = {k: v for k, v in bound.items() if k in spec.input_schema.get("properties", {})}

    def run(**kwargs: Any) -> str:
        async def call() -> Any:
            async with Client(server) as client:
                return await client.call_tool(spec.name, {**kwargs, **fixed})

        result = anyio.run(call)
        if result.is_error:
            raise ToolException(_text(result))
        return _text(result)

    return StructuredTool.from_function(
        func=run,
        name=spec.name,
        description=spec.description or spec.name,
        args_schema=_hide(spec.input_schema, list(fixed)),
    )


def load_tools(
    servers: Sequence[ScopedServer], *, allow: Sequence[str], bound: Mapping[str, str]
) -> list[StructuredTool]:
    """The allowed tools that each server's profile exposes, in `allow` order."""

    async def listed(server: ScopedServer) -> list[Any]:
        async with Client(server) as client:
            return list((await client.list_tools()).tools)

    found: dict[str, StructuredTool] = {}
    for server in servers:
        for spec in anyio.run(listed, server):
            if spec.name in allow:
                found[spec.name] = _make(server, spec, bound)
    return [found[name] for name in allow if name in found]
