"""OpenTelemetry wiring - plan section 13.1 (``obs/``) and Day 4.

**What OpenTelemetry is.**  A vendor-neutral standard for describing *what a
program did and how long it took*.  Your code talks to the OTel API; an
*exporter* decides where that ends up.  Swapping Langfuse for Jaeger, Grafana
Tempo or a file is then a configuration change, not a code change.  That
indirection is the entire reason to use it rather than calling a vendor SDK: the
instrumentation you write today outlives the backend you picked today.

**The three objects that matter here.**

* A **span** is one timed operation: a name, a start, an end, and a bag of
  attributes.  ``llm.grade_answer`` with ``llm.cost_usd=0.00062`` is a span.
* A **trace** is a tree of spans that share a **trace id** - one request, from
  the HTTP handler down to the provider call, however deep it nests.
* A **TracerProvider** is the factory that makes spans and owns the pipeline
  that ships them.

**Exporters this build supports** (``OTEL_EXPORTER``):

``none``      spans are still created - so ``trace_id`` appears in every log
              line and the request can be followed through the logs - but
              nothing is shipped anywhere.  The default, and the plan's own
              Phase 1 cut-line.
``console``   print spans to stdout.  For seeing what instrumentation produces
              without running any infrastructure.
``otlp``      OTLP/HTTP to any collector you point it at.
``langfuse``  OTLP/HTTP to a self-hosted Langfuse, which ingests OpenTelemetry
              natively and renders LLM spans as generations with token and cost
              roll-ups.  Plan section 13.10's chosen backend.

**A failure to export must never fail a request.**  ``BatchSpanProcessor``
buffers spans on a background thread and drops them if the collector is down;
building an exporter that turns out to be unreachable degrades to no tracing,
with a warning.  Observability is not allowed to be the thing that takes the
interview down.
"""

from __future__ import annotations

import base64
import logging
from typing import TYPE_CHECKING, Any

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
    SimpleSpanProcessor,
    SpanExporter,
)
from opentelemetry.trace import format_span_id, format_trace_id

from app.config import Settings

if TYPE_CHECKING:  # pragma: no cover - typing only
    from fastapi import FastAPI

log = logging.getLogger(__name__)

#: Instrumentation scope name - shows up on every span we create ourselves.
TRACER_NAME = "app"

#: Path a Langfuse server exposes for OpenTelemetry ingestion.
LANGFUSE_OTEL_PATH = "/api/public/otel/v1/traces"

#: Never traced. These are polled every few seconds by the orchestrator and
#: would otherwise be 95% of the spans in the system, saying nothing.
EXCLUDED_URLS = "healthz,readyz"

_provider: TracerProvider | None = None
_global_set = False
#: ``None`` = this process never configured tracing; ``False`` = it configured
#: tracing and was told to switch it off. The two differ: the second must force
#: no-op tracers even though a provider may already be registered globally (a
#: worker and the API can share a process in a test run).
_enabled: bool | None = None


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def build_span_exporter(settings: Settings) -> SpanExporter | None:
    """The exporter the configuration asks for, or ``None`` for local-only spans.

    Import of the OTLP exporter is deferred: it pulls in protobuf and requests,
    and a build that exports nothing should not pay for them.
    """
    choice = settings.otel_exporter
    if choice == "none":
        return None
    if choice == "console":
        return ConsoleSpanExporter()

    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

    if choice == "langfuse":
        endpoint, headers = _langfuse_target(settings)
    else:
        # Validated at boot: `otlp` without an endpoint is a config error.
        endpoint = settings.otel_exporter_endpoint or ""
        headers = _parse_headers(settings.otel_exporter_headers_value)
    return OTLPSpanExporter(endpoint=endpoint, headers=headers)


def _langfuse_target(settings: Settings) -> tuple[str, dict[str, str]]:
    """Langfuse's OTLP endpoint and its HTTP Basic credentials.

    Langfuse authenticates OTLP ingestion with the project's public key as the
    username and the secret key as the password.  The header is assembled here
    and nowhere else, and the secret is read out of ``SecretStr`` at the last
    possible moment - it is never interpolated into a log line or a URL.
    """
    public = settings.langfuse_public_key or ""
    secret = settings.langfuse_secret_key.get_secret_value() if settings.langfuse_secret_key else ""
    token = base64.b64encode(f"{public}:{secret}".encode()).decode("ascii")
    endpoint = settings.langfuse_host.rstrip("/") + LANGFUSE_OTEL_PATH
    return endpoint, {"Authorization": f"Basic {token}"}


def _parse_headers(raw: str | None) -> dict[str, str]:
    """``a=1,b=2`` - the format the OTel specification uses for header lists."""
    if not raw:
        return {}
    headers: dict[str, str] = {}
    for part in raw.split(","):
        if "=" not in part:
            continue
        name, _, value = part.partition("=")
        if name.strip():
            headers[name.strip()] = value.strip()
    return headers


def build_tracer_provider(
    settings: Settings, *, exporter: SpanExporter | None = None
) -> TracerProvider:
    """A provider tagged with who and where this process is.

    Those resource attributes are what let one Langfuse project hold traces from
    a laptop and from the Phase 6 VM without the two being confused.
    """
    resource = Resource.create(
        {
            "service.name": settings.otel_service_name or settings.app_name,
            "service.version": settings.app_version,
            "deployment.environment.name": settings.app_env,
        }
    )
    provider = TracerProvider(resource=resource)
    if exporter is not None:
        # Console is synchronous so output appears in order while you watch it;
        # anything over the network is batched onto a background thread so an
        # unreachable collector costs a request nothing.
        processor = (
            SimpleSpanProcessor(exporter)
            if isinstance(exporter, ConsoleSpanExporter)
            else BatchSpanProcessor(exporter)
        )
        provider.add_span_processor(processor)
    return provider


# ---------------------------------------------------------------------------
# Process lifecycle
# ---------------------------------------------------------------------------


def init_tracing(settings: Settings) -> TracerProvider | None:
    """Build the tracer provider once per process. Idempotent.

    Returns ``None`` when tracing is switched off entirely, in which case
    ``get_tracer()`` hands back OTel's no-op tracer and every span in the
    codebase becomes a few nanoseconds of nothing.
    """
    global _provider, _global_set, _enabled
    if not settings.otel_enabled:
        _enabled = False
        log.info("tracing disabled by configuration (OTEL_ENABLED=false)")
        return None
    _enabled = True
    if _provider is not None:
        return _provider

    try:
        exporter = build_span_exporter(settings)
    except Exception as exc:  # noqa: BLE001 - see the module docstring
        log.warning(
            "tracing exporter %s could not be built; continuing without export: %s",
            settings.otel_exporter,
            exc,
        )
        exporter = None

    _provider = build_tracer_provider(settings, exporter=exporter)
    if not _global_set:
        # Set once. OTel refuses to replace a provider that is already global,
        # and a second attempt only produces a warning nobody can act on.
        trace.set_tracer_provider(_provider)
        _global_set = True
    log.info(
        "tracing ready: exporter=%s service=%s",
        settings.otel_exporter,
        settings.otel_service_name or settings.app_name,
    )
    return _provider


def shutdown_tracing() -> None:
    """Flush buffered spans and stop the exporter thread.

    Without this, spans produced in the last few seconds before shutdown are
    lost - which is precisely the window in which a crash happened.
    """
    global _provider, _enabled
    _enabled = None
    if _provider is not None:
        _provider.shutdown()
        _provider = None


def get_tracer(name: str = TRACER_NAME) -> trace.Tracer:
    """A tracer from *our* provider, or a no-op one if tracing is off.

    ``OTEL_ENABLED=false`` has to win over whatever provider happens to be
    registered globally, otherwise the switch does not switch anything off.
    """
    if _enabled is False:
        return trace.NoOpTracer()
    if _provider is not None:
        return _provider.get_tracer(name)
    return trace.get_tracer(name)


def instrument_app(app: FastAPI, settings: Settings) -> None:
    """Give every HTTP request a root span, automatically.

    This is *auto-instrumentation*: the library wraps the ASGI application and
    creates one span per request with method, route, and status code, and reads
    any inbound ``traceparent`` header so a trace that started in a browser or
    another service continues here rather than starting again.  That header
    handling is what "distributed tracing" means in practice, and it is the
    reason to use the maintained instrumentation instead of a hand-rolled
    middleware that would have to reimplement W3C context propagation.
    """
    if not settings.otel_enabled:
        return
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

    FastAPIInstrumentor.instrument_app(
        app,
        tracer_provider=_provider,
        excluded_urls=EXCLUDED_URLS,
    )


# ---------------------------------------------------------------------------
# Reading the current context - used by the log formatter and the middleware
# ---------------------------------------------------------------------------


def current_span() -> trace.Span:
    return trace.get_current_span()


def current_trace_id_hex() -> str | None:
    """The active trace id as 32 hex characters, or ``None`` outside a trace."""
    context = trace.get_current_span().get_span_context()
    if not context.is_valid:
        return None
    return format_trace_id(context.trace_id)


def current_span_id_hex() -> str | None:
    context = trace.get_current_span().get_span_context()
    if not context.is_valid:
        return None
    return format_span_id(context.span_id)


def set_span_attributes(attributes: dict[str, Any]) -> None:
    """Add attributes to whatever span is currently active.

    A no-op when there is no span, which is what makes it safe to call from code
    that runs both inside a request and from a script.
    """
    span = trace.get_current_span()
    if not span.get_span_context().is_valid:
        return
    for key, value in attributes.items():
        if value is not None:
            span.set_attribute(key, value)
