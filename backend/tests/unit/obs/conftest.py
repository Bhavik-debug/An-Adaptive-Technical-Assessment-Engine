"""Fixtures for testing observability without any observability infrastructure.

Two substitutions, mirroring the two things Day 4 produces:

* ``spans`` swaps the process tracer provider for one that keeps finished spans
  in a list.  That is the standard OpenTelemetry testing seam - the SDK ships
  ``InMemorySpanExporter`` for exactly this - and it means every assertion about
  tracing is about the attributes we set, not about a network protocol.
* ``captured_logs`` adds a handler that keeps each record's *rendered* output.
  Rendered, not raw, because the redaction happens in the formatter: asserting
  on ``record.msg`` would pass while the thing actually written to stdout still
  contained the secret.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import pytest
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from app.config import Settings
from app.obs import logging as obs_logging
from app.obs import tracing as obs_tracing
from tests.unit.llm.conftest import MemoryCache


@dataclass
class SpanRecorder:
    """The spans a test produced, with the lookups a test actually wants."""

    exporter: InMemorySpanExporter
    provider: TracerProvider

    @property
    def finished(self) -> tuple[ReadableSpan, ...]:
        return self.exporter.get_finished_spans()

    def named(self, name: str) -> ReadableSpan:
        matches = [s for s in self.finished if s.name == name]
        assert matches, f"no span named {name!r}; saw {[s.name for s in self.finished]}"
        return matches[-1]

    def attributes(self, name: str) -> dict[str, Any]:
        return dict(self.named(name).attributes or {})


@pytest.fixture
def spans(monkeypatch: pytest.MonkeyPatch) -> Iterator[SpanRecorder]:
    """Record every span this test produces, and export none of them anywhere."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    # Simple, not batched: a test must not have to wait for a flush interval.
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(obs_tracing, "_provider", provider)
    yield SpanRecorder(exporter=exporter, provider=provider)
    provider.shutdown()


@dataclass
class LogCapture:
    """Every line this test's logging produced, as it would have been written."""

    lines: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n".join(self.lines)

    def records(self) -> list[dict[str, Any]]:
        """The captured lines parsed back from JSON."""
        return [json.loads(line) for line in self.lines]

    def one(self, msg: str) -> dict[str, Any]:
        matches = [r for r in self.records() if r.get("msg") == msg]
        seen = [r.get("msg") for r in self.records()]
        assert matches, f"no log line with msg={msg!r}; saw {seen}"
        return matches[-1]


class _CaptureHandler(logging.Handler):
    def __init__(self, sink: LogCapture) -> None:
        super().__init__()
        self._sink = sink

    def emit(self, record: logging.LogRecord) -> None:
        self._sink.lines.append(self.format(record))


@pytest.fixture
def captured_logs(env) -> Iterator[LogCapture]:
    """Capture rendered log output through the real formatter and filter.

    The handler is attached to the root logger and removed afterwards, and it is
    deliberately not tagged as ours, so ``configure_logging`` running inside the
    test leaves it in place.
    """
    env()  # the standard test environment; a test may re-apply it with overrides
    settings = Settings()  # type: ignore[call-arg]  # values come from env
    sink = LogCapture()
    handler = _CaptureHandler(sink)
    handler.setFormatter(obs_logging.build_formatter(settings, obs_logging.get_redactor()))
    handler.addFilter(obs_logging.ContextFilter(service=settings.app_name, env=settings.app_env))

    root = logging.getLogger()
    previous_level = root.level
    root.addHandler(handler)
    root.setLevel(logging.DEBUG)
    try:
        yield sink
    finally:
        root.removeHandler(handler)
        root.setLevel(previous_level)


@pytest.fixture
def memory_cache() -> MemoryCache:
    """The Day-3 in-process cache double.

    Re-declared here rather than importing the neighbouring conftest as a
    plugin: pytest registers a conftest exactly once, and asking for it a second
    time by name is an error. The class is shared; only the fixture is local.
    """
    return MemoryCache()
