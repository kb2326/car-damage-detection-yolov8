"""Opt-in OpenTelemetry tracing of agent runs (OpenInference spans, viewable in Phoenix)."""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping
from typing import Any

_state: dict[str, Any] = {}


def setup_tracing(env: Mapping[str, str] = os.environ, exporter: Any = None) -> Any:
    """Turn tracing on when CLAIMLENS_TRACING=1. Returns the tracer provider, or None when off."""
    if env.get("CLAIMLENS_TRACING") != "1":
        return None
    if "provider" in _state:
        return _state["provider"]
    try:
        from openinference.instrumentation.langchain import LangChainInstrumentor
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor
    except ImportError:
        print(
            "warning: CLAIMLENS_TRACING=1 but tracing is not installed: "
            "uv sync --group tracing (continuing without tracing)",
            file=sys.stderr,
        )
        return None

    provider = TracerProvider(resource=Resource.create({"service.name": "claimlens"}))
    if exporter is not None:
        provider.add_span_processor(SimpleSpanProcessor(exporter))
    else:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        endpoint = env.get("PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006").rstrip("/")
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(f"{endpoint}/v1/traces")))
    instrumentor = LangChainInstrumentor()
    instrumentor.instrument(tracer_provider=provider)
    _state.update(provider=provider, instrumentor=instrumentor)
    return provider


def shutdown_tracing() -> None:
    if "provider" in _state:
        _state.pop("instrumentor").uninstrument()
        _state.pop("provider").shutdown()
