from pathlib import Path

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from claimlens.agent.recommendation import SUBMIT
from claimlens.tracing import setup_tracing, shutdown_tracing
from tests.unit.test_agent_graph import ANSWER, _call, _run


def test_tracing_is_off_by_default() -> None:
    assert setup_tracing(env={}) is None


def test_a_run_produces_one_trace_with_model_and_tool_spans(tmp_path: Path) -> None:
    exporter = InMemorySpanExporter()
    provider = setup_tracing(env={"CLAIMLENS_TRACING": "1"}, exporter=exporter)
    try:
        _run(
            tmp_path,
            [
                _call("search_policy_clauses", {"query": "delivery"}, "t1"),
                _call(SUBMIT, ANSWER, "t2"),
            ],
        )
        assert provider is not None
        provider.force_flush()
        spans = exporter.get_finished_spans()
        assert len({s.context.trace_id for s in spans}) == 1
        names = [s.name for s in spans]
        assert any("search_policy_clauses" in n for n in names)
        assert sum("GatewayChatModel" in n for n in names) == 2
        blob = " ".join(str(dict(s.attributes or {})) for s in spans)
        assert "sk-ant-" not in blob
    finally:
        shutdown_tracing()


def test_setup_is_idempotent() -> None:
    exporter = InMemorySpanExporter()
    first = setup_tracing(env={"CLAIMLENS_TRACING": "1"}, exporter=exporter)
    try:
        assert setup_tracing(env={"CLAIMLENS_TRACING": "1"}, exporter=exporter) is first
    finally:
        shutdown_tracing()


def test_runs_are_tagged_with_claim_and_versions(tmp_path: Path) -> None:
    from tests.unit.test_langgraph_agent import ESCALATE, _agent, _state

    exporter = InMemorySpanExporter()
    provider = setup_tracing(env={"CLAIMLENS_TRACING": "1"}, exporter=exporter)
    try:
        from claimlens.knowledge.clauses import load_wordings
        from claimlens.knowledge.embed import FakeEmbedder
        from claimlens.knowledge.index import PolicyIndex, build_index

        root = Path(__file__).resolve().parents[2]
        build_index(load_wordings(root / "knowledge" / "policies"), FakeEmbedder(), tmp_path / "ix")
        index = PolicyIndex.open(tmp_path / "ix", FakeEmbedder())
        state = _state()
        agent, _, _ = _agent(
            tmp_path, [_call(SUBMIT, {**ESCALATE, "policy_citations": []}, "t1")], index
        )
        agent.recommend(state)
        assert provider is not None
        provider.force_flush()
        blob = " ".join(str(dict(s.attributes or {})) for s in exporter.get_finished_spans())
        assert str(state.claim_id) in blob
        assert "triage-agent-v1+triage/v1" in blob
    finally:
        shutdown_tracing()


def test_cli_turns_tracing_on_only_when_asked(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from claimlens import cli

    calls: list[str] = []
    monkeypatch.setattr(cli, "setup_tracing", lambda: calls.append("on"))
    monkeypatch.setattr(cli, "shutdown_tracing", lambda: calls.append("off"))
    cli.main(["eval-gate", "--root", str(tmp_path)])
    assert calls == ["on", "off"]


def test_missing_tracing_group_warns_and_continues(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import sys

    monkeypatch.setitem(sys.modules, "openinference.instrumentation.langchain", None)
    assert setup_tracing(env={"CLAIMLENS_TRACING": "1"}) is None
    assert "uv sync --group tracing" in capsys.readouterr().err
