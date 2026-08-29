"""The provider router: who to ask, how many times, and when to give up.

The plan calls for "a provider router with health-checking and 429 failover"
because the free-tier strategy is the whole cost model - one provider's quota
running out during a viva must degrade into a slower answer from somewhere else,
not into a 500.

The router owns three policies, deliberately kept out of the provider adapters:

* **order** - providers are tried highest-priority first, as configured;
* **retry** - a retryable failure gets another attempt on the same provider,
  with backoff, before the next provider is tried;
* **health** - a provider that keeps failing is taken out of rotation for a
  cooldown, so a dead provider costs one failed call per cooldown window rather
  than one failed call per request.

Today the configured order contains one provider.  The mechanism still matters:
it is what makes adding a second one a configuration change rather than a
rewrite, and it is what the failover test exercises.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from app.config import Settings
from app.llm.errors import AllProvidersFailedError, LLMConfigError, ProviderError
from app.llm.fixtures import FixtureStore
from app.llm.providers.nvidia import NvidiaProvider
from app.llm.providers.stub import MissingFixturePolicy, StubProvider
from app.llm.types import (
    CompletionRequest,
    CompletionResult,
    LLMProvider,
    ModelTier,
    ProviderAttempt,
    RouterOutcome,
)

log = logging.getLogger(__name__)

#: Backoff between attempts on the same provider: 0.5s, 1s, 2s ... capped.
_BACKOFF_BASE_S = 0.5
_BACKOFF_CAP_S = 8.0

#: The tier ``/readyz`` names when reporting which model a provider would use.
_HEALTH_TIER = ModelTier.SMALL_FAST


@dataclass
class _BreakerEntry:
    consecutive_failures: int = 0
    opened_until: float | None = None
    last_error: str | None = None


class CircuitBreaker:
    """Stops a known-dead provider from being retried on every request.

    A plain retry loop treats "this provider is momentarily busy" and "this
    provider's key was revoked an hour ago" identically, and pays the timeout
    for both, on every call.  The breaker separates them: N consecutive failures
    open the circuit, and while it is open the provider is skipped instantly.
    One success closes it.

    In-process state, on purpose.  It is an optimisation, not a correctness
    mechanism, and sharing it across replicas would need coordination that buys
    nothing here.
    """

    def __init__(
        self,
        *,
        threshold: int,
        cooldown_s: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._threshold = threshold
        self._cooldown_s = cooldown_s
        self._clock = clock
        self._entries: dict[str, _BreakerEntry] = {}

    def _entry(self, provider: str) -> _BreakerEntry:
        return self._entries.setdefault(provider, _BreakerEntry())

    def is_open(self, provider: str) -> bool:
        entry = self._entry(provider)
        if entry.opened_until is None:
            return False
        if self._clock() >= entry.opened_until:
            # Cooldown elapsed: let exactly one request through to find out
            # whether the provider recovered.
            entry.opened_until = None
            entry.consecutive_failures = 0
            return False
        return True

    def record_success(self, provider: str) -> None:
        self._entries[provider] = _BreakerEntry()

    def record_failure(self, provider: str, *, reason: str) -> None:
        entry = self._entry(provider)
        entry.consecutive_failures += 1
        entry.last_error = reason
        if entry.consecutive_failures >= self._threshold and entry.opened_until is None:
            entry.opened_until = self._clock() + self._cooldown_s
            log.warning(
                "llm provider %s taken out of rotation for %.0fs after %d failures: %s",
                provider,
                self._cooldown_s,
                entry.consecutive_failures,
                reason,
            )

    def state(self, provider: str) -> tuple[bool, str | None]:
        """``(is_open, last_error)`` - what ``/readyz`` reports."""
        entry = self._entry(provider)
        return self.is_open(provider), entry.last_error


@dataclass(frozen=True, slots=True)
class ProviderHealth:
    name: str
    model: str
    available: bool
    detail: str | None = None


def _default_jitter() -> float:
    # Not a security decision - this only spreads retries out in time, so the
    # standard generator is the right tool.
    return random.random()  # noqa: S311


@dataclass
class ProviderRouter:
    providers: Sequence[LLMProvider]
    breaker: CircuitBreaker
    max_attempts_per_provider: int = 2
    #: Injectable so tests exercise the backoff logic without waiting for it.
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep
    #: Jitter fraction in [0, 1). Injectable for the same reason.
    jitter: Callable[[], float] = _default_jitter

    def __post_init__(self) -> None:
        if not self.providers:
            raise LLMConfigError("the provider router was built with no providers")

    @property
    def provider_names(self) -> tuple[str, ...]:
        return tuple(p.name for p in self.providers)

    async def complete(self, request: CompletionRequest) -> RouterOutcome:
        """Try each provider in order until one answers, or raise."""
        attempts: list[ProviderAttempt] = []
        failures: dict[str, str] = {}

        for provider in self.providers:
            open_circuit, last_error = self.breaker.state(provider.name)
            if open_circuit:
                skipped = f"circuit open ({last_error or 'repeated failures'})"
                attempts.append(
                    ProviderAttempt(provider=provider.name, ok=False, skipped_reason=skipped)
                )
                failures[provider.name] = skipped
                continue

            result, reason = await self._try_provider(provider, request)
            if result is not None:
                attempts.append(ProviderAttempt(provider=provider.name, ok=True))
                return RouterOutcome(result=result, attempts=tuple(attempts))

            attempts.append(ProviderAttempt(provider=provider.name, ok=False, error=reason))
            failures[provider.name] = reason or "unknown failure"

        raise AllProvidersFailedError(failures)

    async def _try_provider(
        self, provider: LLMProvider, request: CompletionRequest
    ) -> tuple[CompletionResult | None, str | None]:
        """One provider, up to ``max_attempts_per_provider`` times."""
        reason: str | None = None
        for attempt in range(self.max_attempts_per_provider):
            try:
                result = await provider.complete(request)
            except ProviderError as exc:
                reason = str(exc)
                self.breaker.record_failure(provider.name, reason=reason)
                if not exc.retryable:
                    # A 401 or a 400 will say the same thing next time. Move on
                    # to the next provider instead of spending the attempts.
                    log.warning("llm provider %s failed permanently: %s", provider.name, exc)
                    return None, reason
                if attempt + 1 < self.max_attempts_per_provider:
                    delay = self._backoff(attempt, exc.retry_after_s)
                    log.info(
                        "llm provider %s attempt %d/%d failed (%s); retrying in %.2fs",
                        provider.name,
                        attempt + 1,
                        self.max_attempts_per_provider,
                        exc,
                        delay,
                    )
                    await self.sleep(delay)
                continue
            self.breaker.record_success(provider.name)
            return result, None
        return None, reason

    def _backoff(self, attempt: int, retry_after_s: float | None) -> float:
        """Exponential backoff with jitter, unless the server named a delay.

        Jitter is not decoration: without it, every worker that got rate limited
        at the same moment retries at the same moment, and the second burst
        looks exactly like the first one.
        """
        if retry_after_s is not None:
            return min(retry_after_s, _BACKOFF_CAP_S)
        # 2.0** rather than 2**: int.__pow__ is typed as returning Any, because
        # a negative exponent yields a float, and that Any leaks into the result.
        base = min(_BACKOFF_BASE_S * (2.0**attempt), _BACKOFF_CAP_S)
        return base * (0.5 + 0.5 * self.jitter())

    async def aclose(self) -> None:
        for provider in self.providers:
            await provider.aclose()

    def health(self) -> list[ProviderHealth]:
        """Configuration-level readiness, not a live API call.

        ``/readyz`` is polled continuously by the orchestrator.  Making a real
        completion on every probe would burn the free-tier quota the whole cost
        model depends on, so "reachable" here means *configured, and not
        currently circuit-broken*.  The honest live check is the opt-in smoke
        test, which a human runs.
        """
        out: list[ProviderHealth] = []
        for provider in self.providers:
            open_circuit, last_error = self.breaker.state(provider.name)
            out.append(
                ProviderHealth(
                    name=provider.name,
                    model=provider.model_for(_HEALTH_TIER),
                    available=not open_circuit,
                    detail=last_error if open_circuit else None,
                )
            )
        return out


def build_router(settings: Settings) -> ProviderRouter:
    """Construct the router the environment describes.

    Configuration faults surface here, at boot, naming the provider - the same
    fail-fast contract the rest of ``config.py`` follows.
    """
    providers = [_build_provider(name, settings) for name in settings.llm_providers]
    return ProviderRouter(
        providers=providers,
        breaker=CircuitBreaker(
            threshold=settings.llm_breaker_failure_threshold,
            cooldown_s=settings.llm_breaker_cooldown_s,
        ),
        max_attempts_per_provider=settings.llm_max_attempts_per_provider,
    )


def _build_provider(name: str, settings: Settings) -> LLMProvider:
    if name == "nvidia":
        if settings.nvidia_api_key is None:  # pragma: no cover - config validates this
            raise LLMConfigError("NVIDIA_API_KEY is not set")
        return NvidiaProvider(
            api_key=settings.nvidia_api_key.get_secret_value(),
            base_url=settings.nvidia_base_url,
            model=settings.nvidia_model,
        )
    if name == "stub":
        # The offline provider (Day 5). Constructed here, by the same factory,
        # from the same settings, and handed to the same router as any other -
        # which is the whole point. If the stub had its own path into
        # `call_structured`, an offline test would be exercising that path
        # rather than the one production uses, and would prove nothing.
        directory = Path(settings.llm_stub_fixture_dir) if settings.llm_stub_fixture_dir else None
        return StubProvider(
            store=FixtureStore(directory),
            on_missing=MissingFixturePolicy(settings.llm_stub_on_missing),
        )
    raise LLMConfigError(f"no adapter is registered for provider {name!r}")
