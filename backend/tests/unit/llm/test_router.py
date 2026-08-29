"""Routing, retry, failover and the circuit breaker.

The Phase 1 exit gate says failover must be *proven by a test*, not asserted in
a README. These are the tests. Two fake providers stand in for two vendors; the
router cannot tell the difference, which is the point of the seam.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from app.llm.errors import (
    AllProvidersFailedError,
    LLMConfigError,
    ProviderAuthError,
    ProviderBadRequestError,
    ProviderRateLimitedError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)
from app.llm.router import CircuitBreaker, ProviderRouter, build_router
from app.llm.structured import schema_spec
from app.llm.types import (
    ChatMessage,
    CompletionRequest,
    ModelTier,
    ReasoningPolicy,
    Role,
)

from .conftest import FakeProvider, make_router, result


class Tiny(BaseModel):
    ok: bool


def a_request() -> CompletionRequest:
    return CompletionRequest(
        messages=(ChatMessage(role=Role.USER, content="hello"),),
        json_schema=schema_spec(Tiny),
        tier=ModelTier.SMALL_FAST,
        temperature=0.0,
        top_p=1.0,
        max_output_tokens=128,
        timeout_s=10.0,
        reasoning=ReasoningPolicy(),
    )


# --- ordering and failover -------------------------------------------------


async def test_first_healthy_provider_wins_and_the_rest_are_untouched():
    primary = FakeProvider("primary", [result("ok", provider="primary")])
    secondary = FakeProvider("secondary", [result("ok", provider="secondary")])

    outcome = await make_router(primary, secondary).complete(a_request())

    assert outcome.result.provider == "primary"
    assert secondary.call_count == 0
    assert outcome.failover_count == 0


async def test_a_bad_api_key_fails_over_to_the_next_provider():
    """The Phase 1 gate, exactly: kill the primary, traffic keeps flowing."""
    primary = FakeProvider("primary", [ProviderAuthError("bad key", provider="primary")])
    secondary = FakeProvider("secondary", [result("ok", provider="secondary")])

    outcome = await make_router(primary, secondary).complete(a_request())

    assert outcome.result.provider == "secondary"
    assert outcome.failover_count == 1
    # Not retryable: one attempt, then move on rather than burn the budget.
    assert primary.call_count == 1


async def test_rate_limit_is_retried_on_the_same_provider_before_failing_over():
    primary = FakeProvider(
        "primary",
        [ProviderRateLimitedError("429", provider="primary"), result("ok", provider="primary")],
    )
    secondary = FakeProvider("secondary", [result("ok", provider="secondary")])

    outcome = await make_router(primary, secondary).complete(a_request())

    assert outcome.result.provider == "primary"
    assert primary.call_count == 2
    assert secondary.call_count == 0


async def test_persistent_rate_limiting_exhausts_attempts_then_fails_over():
    primary = FakeProvider("primary", [ProviderRateLimitedError("429", provider="primary")])
    secondary = FakeProvider("secondary", [result("ok", provider="secondary")])

    outcome = await make_router(primary, secondary, max_attempts_per_provider=3).complete(
        a_request()
    )

    assert primary.call_count == 3
    assert outcome.result.provider == "secondary"


@pytest.mark.parametrize(
    "failure",
    [
        ProviderTimeoutError("slow", provider="primary"),
        ProviderUnavailableError("503", provider="primary"),
        ProviderRateLimitedError("429", provider="primary"),
    ],
)
async def test_every_retryable_failure_class_fails_over(failure):
    primary = FakeProvider("primary", [failure])
    secondary = FakeProvider("secondary", [result("ok", provider="secondary")])

    outcome = await make_router(primary, secondary).complete(a_request())

    assert outcome.result.provider == "secondary"


async def test_bad_request_is_not_retried():
    primary = FakeProvider("primary", [ProviderBadRequestError("400", provider="primary")])
    secondary = FakeProvider("secondary", [result("ok", provider="secondary")])

    await make_router(primary, secondary, max_attempts_per_provider=3).complete(a_request())

    assert primary.call_count == 1


async def test_when_everything_fails_the_error_names_every_provider():
    primary = FakeProvider("primary", [ProviderAuthError("bad key", provider="primary")])
    secondary = FakeProvider("secondary", [ProviderUnavailableError("503", provider="secondary")])

    with pytest.raises(AllProvidersFailedError) as exc:
        await make_router(primary, secondary).complete(a_request())

    assert set(exc.value.failures) == {"primary", "secondary"}
    assert "bad key" in str(exc.value)
    assert "503" in str(exc.value)


async def test_a_router_with_no_providers_is_a_configuration_error():
    with pytest.raises(LLMConfigError):
        ProviderRouter(providers=[], breaker=CircuitBreaker(threshold=1, cooldown_s=1))


# --- backoff ---------------------------------------------------------------


async def test_retry_after_header_is_honoured_over_our_own_backoff():
    slept: list[float] = []

    async def record(seconds: float) -> None:
        slept.append(seconds)

    primary = FakeProvider(
        "primary",
        [
            ProviderRateLimitedError("429", provider="primary", retry_after_s=3.0),
            result("ok", provider="primary"),
        ],
    )
    router = make_router(primary)
    router.sleep = record

    await router.complete(a_request())
    assert slept == [3.0]


async def test_backoff_grows_and_is_jittered():
    slept: list[float] = []

    async def record(seconds: float) -> None:
        slept.append(seconds)

    primary = FakeProvider("primary", [ProviderUnavailableError("503", provider="primary")])
    secondary = FakeProvider("secondary", [result("ok", provider="secondary")])
    router = make_router(primary, secondary, max_attempts_per_provider=3)
    router.sleep = record
    router.jitter = lambda: 1.0  # jitter at its maximum makes the maths exact

    await router.complete(a_request())
    assert slept == [0.5, 1.0]


# --- circuit breaker -------------------------------------------------------


def test_breaker_opens_after_the_threshold_and_closes_after_the_cooldown(clock):
    breaker = CircuitBreaker(threshold=2, cooldown_s=30.0, clock=clock)

    breaker.record_failure("nvidia", reason="503")
    assert breaker.is_open("nvidia") is False

    breaker.record_failure("nvidia", reason="503")
    assert breaker.is_open("nvidia") is True

    clock.advance(29)
    assert breaker.is_open("nvidia") is True

    clock.advance(2)
    assert breaker.is_open("nvidia") is False


def test_one_success_resets_the_failure_count(clock):
    breaker = CircuitBreaker(threshold=2, cooldown_s=30.0, clock=clock)
    breaker.record_failure("nvidia", reason="503")
    breaker.record_success("nvidia")
    breaker.record_failure("nvidia", reason="503")
    assert breaker.is_open("nvidia") is False


async def test_an_open_provider_is_skipped_without_being_called(clock):
    primary = FakeProvider("primary", [ProviderUnavailableError("503", provider="primary")])
    secondary = FakeProvider("secondary", [result("ok", provider="secondary")])
    router = make_router(primary, secondary, max_attempts_per_provider=1, threshold=1, clock=clock)

    await router.complete(a_request())  # opens primary's circuit
    assert primary.call_count == 1

    await router.complete(a_request())
    assert primary.call_count == 1  # skipped entirely, no timeout paid
    assert secondary.call_count == 2


async def test_provider_recovers_once_the_cooldown_elapses(clock):
    primary = FakeProvider(
        "primary",
        [ProviderUnavailableError("503", provider="primary"), result("ok", provider="primary")],
    )
    secondary = FakeProvider("secondary", [result("ok", provider="secondary")])
    router = make_router(
        primary, secondary, max_attempts_per_provider=1, threshold=1, cooldown_s=30.0, clock=clock
    )

    await router.complete(a_request())
    clock.advance(31)
    outcome = await router.complete(a_request())

    assert outcome.result.provider == "primary"


# --- health ----------------------------------------------------------------


def test_health_reports_the_model_and_never_calls_the_api(clock):
    primary = FakeProvider("primary", [result("ok")], model="fake/big-1")
    router = make_router(primary, threshold=1, clock=clock)

    health = router.health()
    assert [(h.name, h.model, h.available) for h in health] == [("primary", "fake/big-1", True)]
    assert primary.call_count == 0

    router.breaker.record_failure("primary", reason="401 rejected")
    degraded = router.health()[0]
    assert degraded.available is False
    assert "401" in (degraded.detail or "")


# --- construction from settings --------------------------------------------


def test_build_router_from_settings_selects_nvidia(env):
    from app.config import get_settings

    env()
    router = build_router(get_settings())
    assert router.provider_names == ("nvidia",)
    assert router.providers[0].model_for(ModelTier.MID) == ("nvidia/nemotron-3.5-lightning-30b-a3b")


def test_build_router_honours_tuning_settings(env):
    from app.config import get_settings

    env({"LLM_MAX_ATTEMPTS_PER_PROVIDER": "4", "LLM_BREAKER_FAILURE_THRESHOLD": "7"})
    router = build_router(get_settings())
    assert router.max_attempts_per_provider == 4
    assert router.breaker.state("nvidia") == (False, None)
