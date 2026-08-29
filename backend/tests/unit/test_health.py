"""Liveness must never depend on Postgres or Redis; readiness must."""

from __future__ import annotations

import pytest
from asgi_lifespan import LifespanManager
from httpx import ASGITransport, AsyncClient

from app.main import create_app


@pytest.fixture
def app(env):
    env()
    return create_app()


@pytest.fixture
async def client(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


def _ok():
    async def _probe(timeout_s: float = 2.0) -> None:
        return None

    return _probe


def _down(message: str):
    async def _probe(timeout_s: float = 2.0) -> None:
        raise ConnectionError(message)

    return _probe


async def test_healthz_is_ok_without_any_dependency(client, monkeypatch):
    # Both dependencies are hard down; liveness must still be 200, otherwise the
    # orchestrator restarts a perfectly healthy process because Postgres blinked.
    monkeypatch.setattr("app.api.health.check_database", _down("no db"))
    monkeypatch.setattr("app.api.health.check_redis", _down("no redis"))

    response = await client.get("/healthz")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


async def test_readyz_is_ready_when_dependencies_answer(client, monkeypatch):
    monkeypatch.setattr("app.api.health.check_database", _ok())
    monkeypatch.setattr("app.api.health.check_redis", _ok())

    response = await client.get("/readyz")
    body = response.json()

    assert response.status_code == 200
    assert body["status"] == "ready"
    assert body["checks"]["postgres"]["status"] == "ok"
    assert body["checks"]["redis"]["status"] == "ok"


async def test_readyz_reports_the_llm_router_once_the_app_has_started(app, monkeypatch):
    """Day 3 wired the router into the lifespan, so readiness now covers it.

    The router is built at startup, which is why this test runs the lifespan
    rather than only the request path. Building it makes no network call, so
    this stays an offline unit test.
    """
    monkeypatch.setattr("app.api.health.check_database", _ok())
    monkeypatch.setattr("app.api.health.check_redis", _ok())

    async with LifespanManager(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as started:
            body = (await started.get("/readyz")).json()

    assert body["checks"]["llm_providers"]["status"] == "ok"
    assert "nvidia" in body["checks"]["llm_providers"]["detail"]


async def test_readyz_reports_llm_as_skipped_in_a_process_that_never_started(client, monkeypatch):
    """Honest reporting: a router that was never built is never "ok"."""
    monkeypatch.setattr("app.api.health.check_database", _ok())
    monkeypatch.setattr("app.api.health.check_redis", _ok())

    body = (await client.get("/readyz")).json()

    assert body["checks"]["llm_providers"]["status"] == "skipped"


@pytest.mark.parametrize("broken", ["postgres", "redis"])
async def test_readyz_is_503_when_a_dependency_is_down(client, monkeypatch, broken):
    probes = {"postgres": "check_database", "redis": "check_redis"}
    for name, attr in probes.items():
        monkeypatch.setattr(f"app.api.health.{attr}", _down("boom") if name == broken else _ok())

    response = await client.get("/readyz")
    body = response.json()

    assert response.status_code == 503
    assert body["status"] == "not_ready"
    assert body["checks"][broken]["status"] == "down"
    assert "ConnectionError" in body["checks"][broken]["detail"]


async def test_readyz_surfaces_a_hung_dependency_as_down(client, monkeypatch):
    async def _hang(timeout_s: float = 2.0) -> None:
        raise TimeoutError("probe exceeded readiness_timeout_s")

    monkeypatch.setattr("app.api.health.check_database", _hang)
    monkeypatch.setattr("app.api.health.check_redis", _ok())

    response = await client.get("/readyz")

    assert response.status_code == 503
    assert response.json()["checks"]["postgres"]["status"] == "down"


async def test_readyz_probes_dependencies_concurrently(client, monkeypatch):
    import asyncio

    async def _slow(timeout_s: float = 2.0) -> None:
        await asyncio.sleep(0.30)

    monkeypatch.setattr("app.api.health.check_database", _slow)
    monkeypatch.setattr("app.api.health.check_redis", _slow)

    started = asyncio.get_running_loop().time()
    response = await client.get("/readyz")
    elapsed = asyncio.get_running_loop().time() - started

    assert response.status_code == 200
    # Serial execution would take ~0.60s; concurrent takes ~0.30s.
    assert elapsed < 0.50, f"probes appear to run serially ({elapsed:.2f}s)"
