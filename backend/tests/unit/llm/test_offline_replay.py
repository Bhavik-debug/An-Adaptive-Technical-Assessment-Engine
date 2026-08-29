"""Offline replay, end to end through the real chokepoint.

Plan section 3, Day 5. These are the tests that make the claim *"every future
test runs without API calls"* mean something, so they are deliberately written
against the **whole stack** - ``call_structured`` -> router -> provider - rather
than against the stub in isolation. A stub tested only in isolation proves the
stub works; these prove *the system* works when the stub is what the router
selected.

The headline test is ``test_a_real_nvidia_answer_replays_offline``: the fixture
in ``backend/fixtures/llm/connectivity_probe.json`` is a genuine Nemotron
response, recorded once from ``integrate.api.nvidia.com``. Replaying it exercises
JSON extraction, pydantic validation, cost arithmetic, cache handling and span
emission on real model output, with the network unplugged.

"With the network unplugged" is not a figure of speech: the ``no_network``
fixture makes any connection to a non-loopback address raise.
"""

from __future__ import annotations

import socket

import pytest
from pydantic import BaseModel, Field

from app.config import get_settings
from app.llm import prompts as prompts_module
from app.llm.client import call_structured
from app.llm.errors import AllProvidersFailedError
from app.llm.fixtures import DEFAULT_FIXTURE_DIR, Fixture, FixtureStore, RecordedError
from app.llm.probe import ProbeAnswer, probe_llm
from app.llm.prompts import PromptTemplate
from app.llm.providers.stub import STUB_MODEL, MissingFixturePolicy, StubProvider
from app.llm.router import CircuitBreaker, ProviderRouter, build_router
from app.llm.tasks import TaskName
from app.llm.types import CompletionRequest, CompletionResult, LLMProvider, ModelTier

from .conftest import MemoryCache

#: The token the shipped recording was made with. ``probe_llm`` normally mints a
#: fresh one per call so a cached or replayed answer cannot masquerade as a live
#: one - correct for the smoke test, and exactly why replay needs a fixed token.
REPLAY_TOKEN = "replay01"  # noqa: S105 - an echo token for a health probe


#: Hosts a test is still allowed to reach. Loopback only, and not as a
#: convenience: asyncio's Windows event loop builds its own wake-up pipe out of
#: a real socket pair on 127.0.0.1 at startup, so a guard that blocked
#: *everything* would break the event loop rather than the code under test.
#: Nothing this project talks to - NVIDIA, Postgres, Redis in CI - is loopback
#: from inside a unit test, so the guard still means what it says.
_LOOPBACK = {"127.0.0.1", "::1", "localhost", ""}


@pytest.fixture
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make any outbound connection to a non-loopback address raise.

    This is the difference between "we believe no call was made" and "no call
    *could* have been made". Applied to these tests specifically rather than
    globally, because integration tests legitimately open sockets.
    """
    real_connect = socket.socket.connect

    def _host_of(address: object) -> str | None:
        if isinstance(address, tuple) and address and isinstance(address[0], str):
            return address[0]
        return None  # AF_UNIX and friends: not an outbound network call

    def _guarded(self: socket.socket, address: object) -> object:
        host = _host_of(address)
        if host is not None and host not in _LOOPBACK and not host.startswith("127."):
            raise AssertionError(
                f"an offline test tried to connect to {host!r} - "
                "the stub provider must never reach the network"
            )
        return real_connect(self, address)  # type: ignore[arg-type]

    monkeypatch.setattr(socket.socket, "connect", _guarded)


def replay_router(
    *,
    on_missing: MissingFixturePolicy = MissingFixturePolicy.STRICT,
    max_attempts: int = 2,
) -> ProviderRouter:
    """A router holding one stub, reading the repository's real recordings."""
    return ProviderRouter(
        providers=[StubProvider(store=FixtureStore(DEFAULT_FIXTURE_DIR), on_missing=on_missing)],
        breaker=CircuitBreaker(threshold=3, cooldown_s=30.0),
        max_attempts_per_provider=max_attempts,
        sleep=_no_sleep,
        jitter=lambda: 0.5,
    )


async def _no_sleep(_seconds: float) -> None:
    return None


class _CapturingProvider(LLMProvider):
    """Wraps a provider and remembers the requests that went past.

    Used to learn the exact ``CompletionRequest`` the chokepoint assembles, so a
    test can file a fixture under its real key without re-implementing request
    assembly - which would be a second place that drifts.
    """

    def __init__(self, inner: StubProvider) -> None:
        self.name = inner.name
        self.inner = inner
        self.requests: list[CompletionRequest] = []

    def model_for(self, tier: ModelTier) -> str:
        return self.inner.model_for(tier)

    async def complete(self, request: CompletionRequest) -> CompletionResult:
        self.requests.append(request)
        return await self.inner.complete(request)

    async def aclose(self) -> None:
        await self.inner.aclose()


# ---------------------------------------------------------------------------
# The headline: a real recorded answer, replayed with the network unplugged
# ---------------------------------------------------------------------------


class TestReplayingARealRecording:
    def test_the_shipped_recording_loads(self):
        store = FixtureStore(DEFAULT_FIXTURE_DIR)
        assert len(store) >= 1, f"no fixtures found in {DEFAULT_FIXTURE_DIR}"

    async def test_a_real_nvidia_answer_replays_offline(self, env, no_network, memory_cache):
        """The Phase-1 exit-gate call, without the network.

        Everything downstream of the provider is the real thing: JSON
        extraction, schema validation, token accounting, cost arithmetic.
        """
        env()
        answer, meta = await call_structured(
            TaskName.CONNECTIVITY_PROBE,
            {"token": REPLAY_TOKEN},
            ProbeAnswer,
            router=replay_router(),
            cache=memory_cache,
            settings=get_settings(),
        )

        # The model really did echo the token - this is Nemotron's own output,
        # not something a test author typed.
        assert isinstance(answer, ProbeAnswer)
        assert answer.ok is True
        assert answer.echo == REPLAY_TOKEN

        assert meta.provider == "stub"
        assert meta.model == STUB_MODEL
        assert meta.structured_mode == "stub_replay"
        # Recorded token counts, so the cost model runs on real numbers.
        assert meta.input_tokens == 322
        assert meta.output_tokens == 19
        assert meta.price_known is True
        assert meta.schema_retry_count == 0
        assert meta.failover_count == 0

    async def test_replay_is_deterministic_across_runs(self, env, no_network):
        """Ten runs, one answer. The property every later phase leans on."""
        env()
        seen = set()
        for _ in range(10):
            answer, meta = await call_structured(
                TaskName.CONNECTIVITY_PROBE,
                {"token": REPLAY_TOKEN},
                ProbeAnswer,
                router=replay_router(),
                cache=MemoryCache(),
                settings=get_settings(),
            )
            seen.add((answer.model_dump_json(), meta.input_tokens, meta.output_tokens))

        assert len(seen) == 1

    async def test_the_probe_helper_itself_replays(self, env, no_network, memory_cache):
        """``probe_llm`` is the Phase-1 exit-gate call.

        Pointing it at the stub answers "does the whole chokepoint work?"
        offline - the same question the live smoke test answers with a network.
        """
        env()
        answer, meta = await probe_llm(
            REPLAY_TOKEN,
            settings=get_settings(),
            router=replay_router(),
            cache=memory_cache,
        )
        assert answer.echo == REPLAY_TOKEN
        assert meta.provider == "stub"

    async def test_a_missing_recording_fails_loudly_rather_than_inventing(
        self, env, no_network, memory_cache
    ):
        """Strict is the default, and this is why: an eval that silently ran on
        invented data reports a number that means nothing."""
        env()
        with pytest.raises(AllProvidersFailedError) as exc:
            await call_structured(
                TaskName.CONNECTIVITY_PROBE,
                {"token": "a-token-that-was-never-recorded"},
                ProbeAnswer,
                router=replay_router(),
                cache=memory_cache,
                settings=get_settings(),
            )
        assert "no recorded fixture" in str(exc.value)


# ---------------------------------------------------------------------------
# The router really selected it - configuration, not a separate code path
# ---------------------------------------------------------------------------


class TestProviderSelection:
    def test_configuration_alone_selects_the_stub(self, env):
        """``LLM_PROVIDER_ORDER=stub``, and nothing else changes."""
        env({"LLM_PROVIDER_ORDER": "stub", "NVIDIA_API_KEY": None})
        router = build_router(get_settings())

        assert router.provider_names == ("stub",)
        assert isinstance(router.providers[0], StubProvider)

    def test_the_stub_needs_no_api_key(self, env):
        """Test isolation, stated as a property: a clean checkout with no
        credential at all can still run the whole offline suite."""
        env({"LLM_PROVIDER_ORDER": "stub", "NVIDIA_API_KEY": None})
        settings = get_settings()

        assert settings.nvidia_api_key is None
        assert build_router(settings).provider_names == ("stub",)

    def test_configuration_still_selects_nvidia_for_real_work(self, env):
        from app.llm.providers.nvidia import NvidiaProvider

        env({"LLM_PROVIDER_ORDER": "nvidia"})
        assert isinstance(build_router(get_settings()).providers[0], NvidiaProvider)

    def test_both_can_be_ordered_together(self, env):
        """A deliberate degraded mode: real answers while the provider is up,
        recordings when it is not."""
        env({"LLM_PROVIDER_ORDER": "nvidia,stub"})
        assert build_router(get_settings()).provider_names == ("nvidia", "stub")

    def test_the_fixture_directory_is_configurable(self, env, tmp_path):
        env({"LLM_PROVIDER_ORDER": "stub", "LLM_STUB_FIXTURE_DIR": str(tmp_path)})
        provider = build_router(get_settings()).providers[0]

        assert isinstance(provider, StubProvider)
        assert provider.fixture_count == 0

    def test_the_miss_policy_is_configurable(self, env):
        env({"LLM_PROVIDER_ORDER": "stub", "LLM_STUB_ON_MISSING": "synthesize"})
        provider = build_router(get_settings()).providers[0]

        assert isinstance(provider, StubProvider)
        assert provider.on_missing is MissingFixturePolicy.SYNTHESIZE

    def test_strict_is_the_default_policy(self, env):
        env({"LLM_PROVIDER_ORDER": "stub"})
        provider = build_router(get_settings()).providers[0]

        assert isinstance(provider, StubProvider)
        assert provider.on_missing is MissingFixturePolicy.STRICT

    def test_readiness_reports_the_stub_without_calling_anything(self, env, no_network):
        """``/readyz`` must never spend quota. With the stub there is nothing to
        spend, but the contract is identical."""
        env({"LLM_PROVIDER_ORDER": "stub"})
        health = build_router(get_settings()).health()

        assert [h.name for h in health] == ["stub"]
        assert health[0].available is True
        assert health[0].model == STUB_MODEL


# ---------------------------------------------------------------------------
# Day 1-4 behaviour, exercised through the offline provider
# ---------------------------------------------------------------------------


class _Extraction(BaseModel):
    name: str = Field(description="the candidate's name")
    years: int


@pytest.fixture
def cacheable_task(monkeypatch):
    """A task with a prompt and temperature 0, and therefore a cache.

    The connectivity probe is deliberately uncacheable - a probe served from
    cache would prove nothing - so cache behaviour is exercised against a
    stand-in for a real Phase-5 task.
    """
    monkeypatch.setitem(
        prompts_module.PROMPTS,
        TaskName.RESUME_EXTRACTION,
        PromptTemplate(
            task=TaskName.RESUME_EXTRACTION,
            version="offline-v1",
            system="Extract fields.",
            user="Resume:\n$resume",
            required_inputs=("resume",),
        ),
    )
    return TaskName.RESUME_EXTRACTION


class TestExistingBehaviourStillWorksOffline:
    """Day 3 tested this machinery with a hand-written ``FakeProvider`` that
    lives only in the test suite. These re-check the same properties through the
    *shipped* offline provider, which is what every later phase will run
    against."""

    async def test_the_response_cache_still_works(
        self, env, no_network, memory_cache, cacheable_task
    ):
        env()
        router = replay_router(on_missing=MissingFixturePolicy.SYNTHESIZE)
        args = {"router": router, "cache": memory_cache, "settings": get_settings()}

        _, first = await call_structured(
            cacheable_task, {"resume": "Ada, 7 years"}, _Extraction, **args
        )
        _, second = await call_structured(
            cacheable_task, {"resume": "Ada, 7 years"}, _Extraction, **args
        )

        assert first.cache_hit is False
        assert second.cache_hit is True
        assert memory_cache.writes == 1
        # The provider answered once for two logical calls.
        provider = router.providers[0]
        assert isinstance(provider, StubProvider)
        assert provider.synthesized == 1

    async def test_cost_accounting_still_runs_on_a_replayed_call(
        self, env, no_network, memory_cache
    ):
        """``price_known=True`` with a zero cost - "free because nothing was
        bought", not "we have no idea what this cost"."""
        env()
        _, meta = await call_structured(
            TaskName.CONNECTIVITY_PROBE,
            {"token": REPLAY_TOKEN},
            ProbeAnswer,
            router=replay_router(),
            cache=memory_cache,
            settings=get_settings(),
        )

        assert meta.price_known is True
        assert meta.cost_usd == 0

    async def test_a_recorded_429_drives_the_real_retry_and_failover_path(
        self, env, no_network, memory_cache
    ):
        """A replayed *failure*.

        The retry loop, the backoff and the failover decision are all the
        production ones; only the source of the 429 is a recording. The request
        key is learned by capturing the request the chokepoint assembled, rather
        than by rebuilding it here - a second request builder would drift.
        """
        env()
        capturing = _CapturingProvider(
            StubProvider(
                store=FixtureStore(DEFAULT_FIXTURE_DIR),
                on_missing=MissingFixturePolicy.SYNTHESIZE,
            )
        )
        router = ProviderRouter(
            providers=[capturing],
            breaker=CircuitBreaker(threshold=10, cooldown_s=30.0),
            max_attempts_per_provider=2,
            sleep=_no_sleep,
            jitter=lambda: 0.5,
        )
        common = {"router": router, "cache": MemoryCache(), "settings": get_settings()}

        # 1. One ordinary call, purely to learn the exact request.
        await call_structured(
            TaskName.CONNECTIVITY_PROBE, {"token": "learn-the-key"}, ProbeAnswer, **common
        )
        captured = capturing.requests[-1]

        # 2. File a recorded 429 under that request's key.
        capturing.inner.add_fixture(
            Fixture(
                key=capturing.inner.key_for(captured),
                description="a recorded rate limit",
                error=RecordedError(kind="rate_limited", message="429 slow down", status_code=429),
            )
        )
        capturing.requests.clear()

        # 3. The same call now hits the recorded failure.
        with pytest.raises(AllProvidersFailedError) as exc:
            await call_structured(
                TaskName.CONNECTIVITY_PROBE, {"token": "learn-the-key"}, ProbeAnswer, **common
            )

        assert "429 slow down" in str(exc.value)
        # Retryable, so the router tried the same provider twice before giving up.
        assert len(capturing.requests) == 2

    async def test_an_llm_span_is_emitted_for_a_replayed_call(
        self, env, no_network, memory_cache, monkeypatch
    ):
        """Day-4 observability does not care which provider answered."""
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

        from app.obs import tracing as obs_tracing

        env()
        exporter = InMemorySpanExporter()
        tracer_provider = TracerProvider()
        tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))
        monkeypatch.setattr(obs_tracing, "_provider", tracer_provider)

        await call_structured(
            TaskName.CONNECTIVITY_PROBE,
            {"token": REPLAY_TOKEN},
            ProbeAnswer,
            router=replay_router(),
            cache=memory_cache,
            settings=get_settings(),
        )

        spans = [s for s in exporter.get_finished_spans() if s.name == "llm.connectivity_probe"]
        assert spans, "no span was emitted for a replayed call"
        attributes = dict(spans[-1].attributes or {})

        # Plan section 14.2's attribute set, unchanged by the provider swap.
        assert attributes["llm.provider"] == "stub"
        assert attributes["llm.input_tokens"] == 322
        assert attributes["llm.prompt_version"] == "v1"
        # And the field that says this was a replay rather than a live call.
        assert attributes["llm.structured_mode"] == "stub_replay"
