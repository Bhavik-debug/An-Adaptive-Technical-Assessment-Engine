"""Request correlation, end to end through the real application.

These go through ``create_app()`` rather than exercising the middleware in
isolation, because half of what is being tested is the *ordering* of the
middleware stack: the correlation middleware has to run inside the
OpenTelemetry server span, or the request id lands on nothing.

``/openapi.json`` is the endpoint used for the ordinary cases: it is a real
routed endpoint that touches no database, so these tests stay in the unit suite
and need nothing running.
"""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import create_app
from app.obs.context import clear_context
from app.obs.middleware import REQUEST_ID_HEADER

ROUTED = "/openapi.json"


@pytest.fixture(autouse=True)
def _clean_context():
    clear_context()
    yield
    clear_context()


@pytest.fixture
def app(env, spans):
    # `spans` first: create_app instruments the app with whatever provider is
    # active, so the recorder has to be in place before the app is built.
    env()
    return create_app()


@pytest.fixture
async def client(app):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


class TestRequestId:
    async def test_every_response_carries_one(self, client):
        """So a bug report can quote an id instead of a timestamp."""
        response = await client.get("/healthz")
        assert response.headers[REQUEST_ID_HEADER]

    async def test_a_client_supplied_id_is_honoured(self, client):
        """What makes a request traceable across a proxy or the Phase 6 frontend."""
        response = await client.get("/healthz", headers={REQUEST_ID_HEADER: "trace-me-123"})
        assert response.headers[REQUEST_ID_HEADER] == "trace-me-123"

    @pytest.mark.parametrize(
        "hostile",
        [
            'evil" injected',
            "with a\ttab",
            "x" * 200,
            "",
        ],
    )
    async def test_a_hostile_id_is_replaced_rather_than_echoed(self, client, hostile):
        """An inbound header is untrusted input that ends up in every log line.

        A header carrying separator characters would otherwise let a caller
        forge log entries - the log-injection bug that makes an audit trail
        worthless.
        """
        response = await client.get("/healthz", headers={REQUEST_ID_HEADER: hostile})
        returned = response.headers[REQUEST_ID_HEADER]
        assert returned != hostile
        assert len(returned) == 32
        assert returned.isalnum()

    async def test_two_requests_get_different_ids(self, client):
        first = await client.get("/healthz")
        second = await client.get("/healthz")
        assert first.headers[REQUEST_ID_HEADER] != second.headers[REQUEST_ID_HEADER]


class TestLatencyAndOutcome:
    async def test_one_line_per_request_records_route_status_and_duration(
        self, client, captured_logs
    ):
        """ "How long did it take" is a first-class field, not something to grep for."""
        await client.get(ROUTED)

        record = captured_logs.one("http_request")
        assert record["http.method"] == "GET"
        # The route template, not the concrete URL: a path parameter can be a
        # user id, and grouping by template is what makes a percentile mean
        # anything.
        assert record["http.route"] == ROUTED
        assert record["http.status_code"] == 200
        assert isinstance(record["duration_ms"], int)

    async def test_the_request_log_line_carries_the_same_id_the_caller_got_back(
        self, client, captured_logs
    ):
        response = await client.get(ROUTED, headers={REQUEST_ID_HEADER: "corr-1"})
        assert response.headers[REQUEST_ID_HEADER] == "corr-1"
        assert captured_logs.one("http_request")["request_id"] == "corr-1"

    async def test_health_probes_do_not_fill_the_log_at_info(self, client, captured_logs):
        """A healthy stack writes a line every five seconds, forever, otherwise."""
        await client.get("/healthz")
        assert captured_logs.one("http_request")["level"] == "DEBUG"


class TestFailurePath:
    """A request that dies before producing a response is the one you most need."""

    @pytest.fixture
    def broken_app(self, app):
        @app.get("/boom")
        async def _boom() -> dict[str, str]:
            raise ValueError("deliberate failure")

        return app

    @pytest.fixture
    async def broken_client(self, broken_app):
        # raise_app_exceptions=False makes the transport behave like a real
        # server: the exception becomes a 500 rather than escaping into pytest.
        transport = ASGITransport(app=broken_app, raise_app_exceptions=False)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            yield c

    async def test_an_unhandled_error_is_still_logged_with_its_latency(
        self, broken_client, captured_logs
    ):
        await broken_client.get("/boom", headers={REQUEST_ID_HEADER: "boom-1"})

        record = captured_logs.one("http_request")
        assert record["level"] == "ERROR"
        assert record["request_id"] == "boom-1"
        assert record["http.route"] == "/boom"
        assert isinstance(record["duration_ms"], int)

    async def test_the_server_span_is_marked_as_failed(self, broken_client, spans):
        await broken_client.get("/boom")

        server = [s for s in spans.finished if s.kind.name == "SERVER"][-1]
        assert server.status.status_code.name == "ERROR"


class TestSpans:
    async def test_a_request_produces_a_server_span_carrying_the_request_id(self, client, spans):
        """The ordering test: the id has to land on the request's root span.

        If the correlation middleware ran *outside* the OpenTelemetry
        middleware, there would be no span open when it tries to set this, and
        the attribute would silently go nowhere.
        """
        await client.get(ROUTED, headers={REQUEST_ID_HEADER: "span-corr"})

        server_spans = [s for s in spans.finished if s.kind.name == "SERVER"]
        assert server_spans, f"no server span; saw {[s.name for s in spans.finished]}"
        assert (server_spans[-1].attributes or {})["request.id"] == "span-corr"

    async def test_health_probes_are_not_traced(self, client, spans):
        """They would be 95% of the spans in the system and say nothing."""
        await client.get("/healthz")
        assert [s for s in spans.finished if s.kind.name == "SERVER"] == []

    async def test_a_log_line_and_its_span_share_a_trace_id(self, client, captured_logs, spans):
        """The join that makes a trace and a log searchable as one thing."""
        await client.get(ROUTED)

        logged = captured_logs.one("http_request")["trace_id"]
        server = [s for s in spans.finished if s.kind.name == "SERVER"][-1]
        assert logged == f"{server.context.trace_id:032x}"
