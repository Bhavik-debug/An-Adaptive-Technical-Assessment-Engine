"""Lifecycle wiring and what ``/readyz`` says about the LLM layer."""

from __future__ import annotations

import pytest

from app.api.health import _llm_check
from app.config import get_settings
from app.llm import runtime

from .conftest import FakeProvider, make_router, result


@pytest.fixture(autouse=True)
async def _clean_runtime():
    """The router is process state; no test may leak it into the next one."""
    await runtime.dispose_llm()
    yield
    await runtime.dispose_llm()


async def test_init_builds_a_router_and_is_idempotent(env):
    env()
    settings = get_settings()
    first = runtime.init_llm(settings)
    assert runtime.init_llm(settings) is first
    assert runtime.get_router() is first
    assert first.provider_names == ("nvidia",)


async def test_the_router_is_not_available_before_init():
    with pytest.raises(RuntimeError, match="not initialised"):
        runtime.get_router()


async def test_dispose_closes_every_provider(env):
    env()
    provider = FakeProvider("nvidia", [result("{}")])
    runtime._router = make_router(provider)  # noqa: SLF001 - lifecycle under test

    await runtime.dispose_llm()

    assert provider.closed is True
    with pytest.raises(RuntimeError):
        runtime.get_router()


async def test_cache_falls_back_to_a_no_op_when_redis_is_absent(env):
    """A script or a test has no Redis; that must not disable the LLM layer."""
    env()
    runtime.init_llm(get_settings())
    cache = runtime.get_response_cache()
    await cache.set("k", None)  # type: ignore[arg-type]
    assert await cache.get("k") is None


# --- readiness -------------------------------------------------------------


def test_readiness_reports_skipped_when_the_router_is_not_built():
    check = _llm_check()
    assert check.status == "skipped"
    assert check.blocks_readiness is False


async def test_readiness_is_ok_with_a_provider_in_rotation(env):
    env()
    runtime._router = make_router(FakeProvider("nvidia", [result("{}")]))  # noqa: SLF001

    check = _llm_check()

    assert check.status == "ok"
    assert "nvidia" in (check.detail or "")


async def test_readiness_goes_down_when_every_provider_is_circuit_broken(env):
    env()
    router = make_router(FakeProvider("nvidia", [result("{}")]), threshold=1)
    router.breaker.record_failure("nvidia", reason="401 rejected")
    runtime._router = router  # noqa: SLF001

    check = _llm_check()

    assert check.status == "down"
    assert check.blocks_readiness is True
    assert "401" in (check.detail or "")


async def test_readiness_never_makes_a_model_call(env):
    """A probe polled every few seconds must not spend the daily quota."""
    env()
    provider = FakeProvider("nvidia", [result("{}")])
    runtime._router = make_router(provider)  # noqa: SLF001

    for _ in range(5):
        _llm_check()

    assert provider.call_count == 0
