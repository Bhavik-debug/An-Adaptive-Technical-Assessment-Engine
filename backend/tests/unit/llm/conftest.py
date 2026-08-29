"""Test doubles for the LLM layer.

Nothing in ``tests/unit`` touches the network.  Two levels of substitution are
used, because they answer different questions:

* ``FakeProvider`` replaces a whole provider.  Used to test the router, the
  cache, and ``call_structured`` - the layers that must not care which vendor
  is underneath.
* ``fake_creator`` replaces the single SDK call inside ``NvidiaProvider``.  Used
  to test the NVIDIA adapter itself: what request body it builds and how it
  takes a response apart.  Substituting one callable rather than mocking HTTP
  keeps the assertions about *our* code rather than about httpx.

These are test doubles, not the Day 5 offline stub provider.  That one is a
shipped component that replays recorded fixtures; these live only in tests.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

import pytest

from app.llm.cache import CachedCall, ResponseCache
from app.llm.router import CircuitBreaker, ProviderRouter
from app.llm.types import (
    CompletionRequest,
    CompletionResult,
    LLMProvider,
    ModelTier,
)


class FakeProvider(LLMProvider):
    """A provider that returns, or raises, whatever the test scripted.

    ``script`` is consumed one entry per call; the last entry repeats forever so
    a test does not have to know how many attempts the router will make.
    """

    def __init__(
        self,
        name: str,
        script: Sequence[CompletionResult | BaseException],
        *,
        model: str = "fake/model-1",
    ) -> None:
        self.name = name
        self._script = list(script)
        self._model = model
        self.calls: list[CompletionRequest] = []
        self.closed = False

    def model_for(self, tier: ModelTier) -> str:
        return self._model

    async def complete(self, request: CompletionRequest) -> CompletionResult:
        self.calls.append(request)
        index = min(len(self.calls) - 1, len(self._script) - 1)
        step = self._script[index]
        if isinstance(step, BaseException):
            raise step
        return step

    async def aclose(self) -> None:
        self.closed = True

    @property
    def call_count(self) -> int:
        return len(self.calls)


def result(
    text: str,
    *,
    provider: str = "fake",
    model: str = "fake/model-1",
    input_tokens: int = 100,
    output_tokens: int = 20,
    **kwargs: Any,
) -> CompletionResult:
    """A successful completion carrying ``text`` as the answer."""
    return CompletionResult(
        text=text,
        provider=provider,
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        structured_mode="json_schema",
        **kwargs,
    )


def json_result(payload: dict[str, Any], **kwargs: Any) -> CompletionResult:
    return result(json.dumps(payload), **kwargs)


def make_router(
    *providers: LLMProvider,
    max_attempts_per_provider: int = 2,
    threshold: int = 3,
    cooldown_s: float = 30.0,
    clock: Callable[[], float] | None = None,
) -> ProviderRouter:
    """A router whose sleeping and jitter are removed, so tests are instant."""
    breaker = (
        CircuitBreaker(threshold=threshold, cooldown_s=cooldown_s, clock=clock)
        if clock is not None
        else CircuitBreaker(threshold=threshold, cooldown_s=cooldown_s)
    )

    async def _no_sleep(_seconds: float) -> None:
        return None

    return ProviderRouter(
        providers=list(providers),
        breaker=breaker,
        max_attempts_per_provider=max_attempts_per_provider,
        sleep=_no_sleep,
        jitter=lambda: 0.5,
    )


@dataclass
class MemoryCache(ResponseCache):
    """An in-process stand-in for the Redis cache, with call counters."""

    entries: dict[str, CachedCall] = field(default_factory=dict)
    reads: int = 0
    writes: int = 0

    async def get(self, key: str) -> CachedCall | None:
        self.reads += 1
        return self.entries.get(key)

    async def set(self, key: str, entry: CachedCall) -> None:
        self.writes += 1
        self.entries[key] = entry


class ExplodingCache(ResponseCache):
    """A cache backend that is completely broken, to prove calls still work."""

    async def get(self, key: str) -> CachedCall | None:
        raise RuntimeError("cache backend is down")

    async def set(self, key: str, entry: CachedCall) -> None:
        raise RuntimeError("cache backend is down")


@pytest.fixture
def memory_cache() -> MemoryCache:
    return MemoryCache()


class FakeClock:
    """A monotonic clock a test can move by hand."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def clock() -> Iterator[FakeClock]:
    yield FakeClock()
