"""Exporter selection, graceful degradation, and the fail-fast boundary.

The distinction under test throughout: a **misconfiguration** stops the process
at boot, because someone made a mistake and should be told; a **collector that
is down** does not, because that is someone else's outage and an interview in
progress must survive it.
"""

from __future__ import annotations

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter

from app.config import ConfigError, get_settings
from app.obs import tracing as obs_tracing
from app.obs.tracing import (
    LANGFUSE_OTEL_PATH,
    build_span_exporter,
    build_tracer_provider,
    current_span_id_hex,
    current_trace_id_hex,
    get_tracer,
    init_tracing,
    shutdown_tracing,
)


@pytest.fixture(autouse=True)
def _no_leftover_provider():
    yield
    shutdown_tracing()


class TestExporterSelection:
    def test_the_default_exports_nothing(self, env):
        """The plan's own Phase 1 cut-line, and the local-development default.

        Spans are still created - which is what puts a trace id on every log
        line - they simply are not shipped anywhere. Local development must not
        require an observability stack to be running.
        """
        env()
        assert build_span_exporter(get_settings()) is None

    def test_console_exporter_needs_no_configuration(self, env):
        env({"OTEL_EXPORTER": "console"})
        assert isinstance(build_span_exporter(get_settings()), ConsoleSpanExporter)

    def test_otlp_exporter_points_where_it_was_told(self, env):
        env(
            {
                "OTEL_EXPORTER": "otlp",
                "OTEL_EXPORTER_ENDPOINT": "http://localhost:4318/v1/traces",
                "OTEL_EXPORTER_HEADERS": "x-api-key=abc,x-tenant=me",
            }
        )
        exporter = build_span_exporter(get_settings())
        assert exporter is not None
        assert exporter._endpoint == "http://localhost:4318/v1/traces"
        assert exporter._session.headers["x-api-key"] == "abc"

    def test_langfuse_exporter_targets_the_otel_ingestion_path_with_basic_auth(self, env):
        """Langfuse authenticates OTLP with public key as user, secret as password."""
        env(
            {
                "OTEL_EXPORTER": "langfuse",
                "LANGFUSE_HOST": "http://localhost:3000/",
                "LANGFUSE_PUBLIC_KEY": "pk-lf-1",
                "LANGFUSE_SECRET_KEY": "sk-lf-secret-value",
            }
        )
        exporter = build_span_exporter(get_settings())
        assert exporter is not None
        assert exporter._endpoint == "http://localhost:3000" + LANGFUSE_OTEL_PATH

        header = exporter._session.headers["Authorization"]
        assert header.startswith("Basic ")
        # The secret is base64-encoded, never interpolated into a URL and never
        # present in plain text anywhere it could be logged.
        assert "sk-lf-secret-value" not in header


class TestConfigurationIsFailFast:
    def test_otlp_without_an_endpoint_refuses_to_boot(self, env):
        env({"OTEL_EXPORTER": "otlp"})
        with pytest.raises(ConfigError) as exc:
            get_settings()
        assert "OTEL_EXPORTER_ENDPOINT is required" in str(exc.value)

    def test_langfuse_without_keys_refuses_to_boot_and_names_them(self, env):
        env({"OTEL_EXPORTER": "langfuse"})
        with pytest.raises(ConfigError) as exc:
            get_settings()
        message = str(exc.value)
        assert "LANGFUSE_PUBLIC_KEY" in message
        assert "LANGFUSE_SECRET_KEY" in message

    def test_an_unknown_exporter_name_refuses_to_boot(self, env):
        env({"OTEL_EXPORTER": "jaeger-but-misspelled"})
        with pytest.raises(ConfigError):
            get_settings()


class TestDegradation:
    def test_an_exporter_that_cannot_be_built_leaves_the_app_running(self, env, monkeypatch):
        """A broken observability backend must not be an outage.

        This is the whole reason the exporter is constructed inside a try: the
        alternative is an API that will not start because a trace collector
        moved.
        """
        env({"OTEL_EXPORTER": "console"})

        def _explode(_settings):
            raise RuntimeError("collector library is broken")

        monkeypatch.setattr(obs_tracing, "build_span_exporter", _explode)

        provider = init_tracing(get_settings())

        assert isinstance(provider, TracerProvider)
        with get_tracer().start_as_current_span("still works"):
            assert current_trace_id_hex() is not None

    def test_tracing_can_be_switched_off_entirely(self, env):
        env({"OTEL_ENABLED": "false"})
        assert init_tracing(get_settings()) is None
        # Spans become OTel's no-op implementation: the call sites throughout
        # the codebase stay exactly as they are and cost nothing.
        with get_tracer().start_as_current_span("nothing"):
            assert current_trace_id_hex() is None

    def test_init_is_idempotent(self, env):
        env()
        first = init_tracing(get_settings())
        assert init_tracing(get_settings()) is first


class TestSpanContextReading:
    def test_outside_a_span_there_are_no_ids(self, env):
        env()
        assert current_trace_id_hex() is None
        assert current_span_id_hex() is None

    def test_a_child_span_shares_its_parent_trace_id(self, env):
        """This is what "one trace per request" means in practice.

        The HTTP span and the LLM span several layers below it carry the same
        trace id, which is how the whole pipeline can be pulled up as one tree.
        """
        env()
        provider = build_tracer_provider(get_settings())
        tracer = provider.get_tracer("t")
        with tracer.start_as_current_span("request"):
            outer_trace, outer_span = current_trace_id_hex(), current_span_id_hex()
            with tracer.start_as_current_span("llm.grade_answer"):
                assert current_trace_id_hex() == outer_trace
                assert current_span_id_hex() != outer_span


class TestResourceAttributes:
    def test_a_span_says_which_service_and_environment_produced_it(self, env):
        """One Langfuse project holds traces from a laptop and from the VM."""
        env({"APP_ENV": "staging", "OTEL_SERVICE_NAME": "interviewer-api"})
        provider = build_tracer_provider(get_settings())
        attributes = provider.resource.attributes
        assert attributes["service.name"] == "interviewer-api"
        assert attributes["deployment.environment.name"] == "staging"
