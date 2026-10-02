import importlib.util
from pathlib import Path

import anyio
import pytest

from claimlens.cli import resolve_detector

ROOT = Path(__file__).resolve().parents[2]


def _ready() -> bool:
    try:
        spec = resolve_detector("fused", None, ROOT / "config")
    except ValueError:
        return False
    return (ROOT / spec.weights).is_file() and importlib.util.find_spec("ultralytics") is not None


pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not _ready(), reason="needs champions and vision group"),
]


def test_vision_server_over_real_stdio() -> None:
    from mcp.client import Client
    from mcp.client.stdio import StdioServerParameters

    params = StdioServerParameters(
        command="uv",
        args=["run", "--no-sync", "claimlens", "mcp", "vision", "--profile", "demo"],
        cwd=str(ROOT),
    )

    async def run() -> dict[str, object] | None:
        async with Client(params, read_timeout_seconds=300) as client:
            names = sorted(t.name for t in (await client.list_tools()).tools)
            assert names == ["assess_quality", "segment_damage", "segment_parts"]
            result = await client.call_tool(
                "segment_damage", {"photo": "tests/fixtures/images/dent_1.jpg"}
            )
            assert not result.is_error
            content: dict[str, object] | None = result.structured_content
            return content

    report = anyio.run(run)
    assert report is not None
    assert str(report["model_version"]).startswith("fused:")
