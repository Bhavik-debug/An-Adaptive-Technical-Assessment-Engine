"""``call_structured()`` - the contract every future caller depends on."""

from __future__ import annotations

import json
from decimal import Decimal

import pytest
from pydantic import BaseModel, Field

from app.config import get_settings
from app.llm import prompts as prompts_module
from app.llm.cache import CachedCall
from app.llm.client import call_structured
from app.llm.errors import (
    AllProvidersFailedError,
    PromptNotRegisteredError,
    SchemaValidationFailedError,
)
from app.llm.probe import ProbeAnswer
from app.llm.prompts import PromptTemplate
from app.llm.tasks import TaskName

from .conftest import ExplodingCache, FakeProvider, json_result, make_router, result


class Extraction(BaseModel):
    name: str = Field(description="The candidate's name.")
    years: int


@pytest.fixture
def cacheable_task(monkeypatch):
    """A task with a prompt, a temperature of 0, and therefore a cache.

    The connectivity probe is deliberately uncacheable, so cache behaviour is
    exercised against a stand-in for a real Phase-5 task instead.
    """
    template = PromptTemplate(
        task=TaskName.RESUME_EXTRACTION,
        version="test-v1",
        system="Extract fields.",
        user="Resume:\n$resume",
        required_inputs=("resume",),
    )
    monkeypatch.setitem(prompts_module.PROMPTS, TaskName.RESUME_EXTRACTION, template)
    return TaskName.RESUME_EXTRACTION


PROBE_JSON = {"ok": True, "echo": "abc123", "model_said": "nemotron"}


# --- the happy path --------------------------------------------------------


async def test_a_valid_response_becomes_a_validated_object(env, memory_cache):
    env()
    provider = FakeProvider("nvidia", [json_result(PROBE_JSON, provider="nvidia")])

    answer, meta = await call_structured(
        TaskName.CONNECTIVITY_PROBE,
        {"token": "abc123"},
        ProbeAnswer,
        router=make_router(provider),
        cache=memory_cache,
        settings=get_settings(),
    )

    assert isinstance(answer, ProbeAnswer)
    assert answer.echo == "abc123"
    assert meta.cache_hit is False
    assert meta.schema_retry_count == 0
    assert meta.failover_count == 0


async def test_metadata_carries_every_attribute_a_span_needs(env, memory_cache):
    env()
    provider = FakeProvider(
        "nvidia",
        [json_result(PROBE_JSON, provider="nvidia", input_tokens=612, output_tokens=148)],
    )

    _, meta = await call_structured(
        TaskName.CONNECTIVITY_PROBE,
        {"token": "abc123"},
        ProbeAnswer,
        router=make_router(provider),
        cache=memory_cache,
        settings=get_settings(),
    )

    # Plan section 14.2 calls these non-negotiable on every LLM span.
    attrs = meta.as_span_attributes()
    for required in (
        "llm.prompt_version",
        "llm.model",
        "llm.input_tokens",
        "llm.output_tokens",
        "llm.cost_usd",
        "llm.cache_hit",
        "llm.schema_retry_count",
    ):
        assert required in attrs
    assert meta.input_tokens == 612
    assert meta.output_tokens == 148
    assert meta.prompt_version == "v1"
    assert meta.task is TaskName.CONNECTIVITY_PROBE


async def test_trace_context_is_stamped_onto_the_metadata(env, memory_cache):
    import uuid

    from app.llm.types import TraceCtx

    env()
    session = uuid.uuid4()
    provider = FakeProvider("nvidia", [json_result(PROBE_JSON)])

    _, meta = await call_structured(
        TaskName.CONNECTIVITY_PROBE,
        {"token": "abc123"},
        ProbeAnswer,
        router=make_router(provider),
        cache=memory_cache,
        settings=get_settings(),
        trace=TraceCtx(session_id=session, plan="pro"),
    )

    assert meta.session_id == str(session)
    assert meta.plan == "pro"


async def test_the_task_decides_the_sampling_settings(env, memory_cache, cacheable_task):
    env()
    provider = FakeProvider("nvidia", [json_result({"name": "A", "years": 3})])
    router = make_router(provider)

    await call_structured(
        cacheable_task,
        {"resume": "..."},
        Extraction,
        router=router,
        cache=memory_cache,
        settings=get_settings(),
    )

    assert provider.calls[0].temperature == 0.0  # routing table, section 13.6


async def test_an_explicit_temperature_overrides_the_table(env, memory_cache, cacheable_task):
    env()
    provider = FakeProvider("nvidia", [json_result({"name": "A", "years": 3})])

    _, meta = await call_structured(
        cacheable_task,
        {"resume": "..."},
        Extraction,
        temperature=0.9,
        router=make_router(provider),
        cache=memory_cache,
        settings=get_settings(),
    )

    assert provider.calls[0].temperature == 0.9
    assert meta.temperature == 0.9


# --- schema retry ----------------------------------------------------------


async def test_invalid_output_is_repaired_on_a_second_attempt(env, memory_cache):
    env()
    provider = FakeProvider(
        "nvidia",
        [
            result("I think it went fine, honestly."),  # no JSON at all
            json_result(PROBE_JSON),
        ],
    )

    answer, meta = await call_structured(
        TaskName.CONNECTIVITY_PROBE,
        {"token": "abc123"},
        ProbeAnswer,
        router=make_router(provider),
        cache=memory_cache,
        settings=get_settings(),
    )

    assert answer.echo == "abc123"
    assert meta.schema_retry_count == 1
    assert provider.call_count == 2


async def test_the_repair_prompt_shows_the_model_its_mistake(env, memory_cache):
    env()
    provider = FakeProvider(
        "nvidia", [json_result({"ok": "not-a-bool", "echo": 1}), json_result(PROBE_JSON)]
    )

    await call_structured(
        TaskName.CONNECTIVITY_PROBE,
        {"token": "abc123"},
        ProbeAnswer,
        router=make_router(provider),
        cache=memory_cache,
        settings=get_settings(),
    )

    repair = provider.calls[1].messages
    assert any("not-a-bool" in m.content for m in repair)
    assert any("rejected" in m.content for m in repair)


async def test_output_that_never_validates_raises_after_the_configured_retries(env, memory_cache):
    env({"LLM_SCHEMA_MAX_RETRIES": "2"})
    provider = FakeProvider("nvidia", [result("still not json")])

    with pytest.raises(SchemaValidationFailedError) as exc:
        await call_structured(
            TaskName.CONNECTIVITY_PROBE,
            {"token": "abc123"},
            ProbeAnswer,
            router=make_router(provider),
            cache=memory_cache,
            settings=get_settings(),
        )

    assert exc.value.attempts == 3  # the first call plus two repairs
    assert provider.call_count == 3
    assert "still not json" in exc.value.raw_output


async def test_retries_can_be_switched_off(env, memory_cache):
    env({"LLM_SCHEMA_MAX_RETRIES": "0"})
    provider = FakeProvider("nvidia", [result("nope")])

    with pytest.raises(SchemaValidationFailedError):
        await call_structured(
            TaskName.CONNECTIVITY_PROBE,
            {"token": "abc123"},
            ProbeAnswer,
            router=make_router(provider),
            cache=memory_cache,
            settings=get_settings(),
        )
    assert provider.call_count == 1


async def test_a_truncated_response_says_so_instead_of_blaming_the_schema(env, memory_cache):
    env({"LLM_SCHEMA_MAX_RETRIES": "0"})
    provider = FakeProvider("nvidia", [result('{"ok": true, "ec', finish_reason="length")])

    with pytest.raises(SchemaValidationFailedError) as exc:
        await call_structured(
            TaskName.CONNECTIVITY_PROBE,
            {"token": "abc123"},
            ProbeAnswer,
            router=make_router(provider),
            cache=memory_cache,
            settings=get_settings(),
        )
    assert "output-token limit" in exc.value.last_error


async def test_tokens_from_every_attempt_are_counted(env, memory_cache):
    """A retry that is not billed in the trace is a retry that looks free."""
    env()
    provider = FakeProvider(
        "nvidia",
        [
            result("garbage", input_tokens=100, output_tokens=10),
            json_result(PROBE_JSON, input_tokens=180, output_tokens=30),
        ],
    )

    _, meta = await call_structured(
        TaskName.CONNECTIVITY_PROBE,
        {"token": "abc123"},
        ProbeAnswer,
        router=make_router(provider),
        cache=memory_cache,
        settings=get_settings(),
    )

    assert meta.input_tokens == 280
    assert meta.output_tokens == 40


# --- failover through the chokepoint ---------------------------------------


async def test_failover_is_reported_in_the_metadata(env, memory_cache):
    from app.llm.errors import ProviderAuthError

    env()
    dead = FakeProvider("primary", [ProviderAuthError("bad key", provider="primary")])
    alive = FakeProvider("secondary", [json_result(PROBE_JSON, provider="secondary")])

    _, meta = await call_structured(
        TaskName.CONNECTIVITY_PROBE,
        {"token": "abc123"},
        ProbeAnswer,
        router=make_router(dead, alive),
        cache=memory_cache,
        settings=get_settings(),
    )

    assert meta.provider == "secondary"
    assert meta.failover_count == 1


async def test_total_provider_failure_propagates(env, memory_cache):
    from app.llm.errors import ProviderUnavailableError

    env()
    dead = FakeProvider("nvidia", [ProviderUnavailableError("503", provider="nvidia")])

    with pytest.raises(AllProvidersFailedError):
        await call_structured(
            TaskName.CONNECTIVITY_PROBE,
            {"token": "abc123"},
            ProbeAnswer,
            router=make_router(dead),
            cache=memory_cache,
            settings=get_settings(),
        )


# --- caching ---------------------------------------------------------------


async def test_a_deterministic_task_is_computed_once(env, memory_cache, cacheable_task):
    env()
    provider = FakeProvider("nvidia", [json_result({"name": "Ada", "years": 7})])
    router = make_router(provider)
    args = dict(router=router, cache=memory_cache, settings=get_settings())

    first, first_meta = await call_structured(
        cacheable_task, {"resume": "Ada, 7 years"}, Extraction, **args
    )
    second, second_meta = await call_structured(
        cacheable_task, {"resume": "Ada, 7 years"}, Extraction, **args
    )

    assert provider.call_count == 1
    assert second == first
    assert first_meta.cache_hit is False
    assert second_meta.cache_hit is True


async def test_a_cache_hit_costs_no_tokens(env, memory_cache, cacheable_task):
    env()
    provider = FakeProvider(
        "nvidia", [json_result({"name": "Ada", "years": 7}, input_tokens=900, output_tokens=80)]
    )
    args = dict(router=make_router(provider), cache=memory_cache, settings=get_settings())

    await call_structured(cacheable_task, {"resume": "x"}, Extraction, **args)
    _, meta = await call_structured(cacheable_task, {"resume": "x"}, Extraction, **args)

    # Reporting the tokens it would have cost would double-count them.
    assert (meta.input_tokens, meta.output_tokens) == (0, 0)
    assert meta.cost_usd == Decimal("0")
    assert meta.structured_mode == "cache"


async def test_different_inputs_do_not_share_a_cache_entry(env, memory_cache, cacheable_task):
    env()
    provider = FakeProvider(
        "nvidia",
        [json_result({"name": "Ada", "years": 7}), json_result({"name": "Bob", "years": 2})],
    )
    args = dict(router=make_router(provider), cache=memory_cache, settings=get_settings())

    first, _ = await call_structured(cacheable_task, {"resume": "a"}, Extraction, **args)
    second, _ = await call_structured(cacheable_task, {"resume": "b"}, Extraction, **args)

    assert (first.name, second.name) == ("Ada", "Bob")
    assert provider.call_count == 2


async def test_a_nondeterministic_task_is_never_cached(env, memory_cache):
    """The probe must prove the network works, so a hit would prove nothing."""
    env()
    provider = FakeProvider("nvidia", [json_result(PROBE_JSON)])
    args = dict(router=make_router(provider), cache=memory_cache, settings=get_settings())

    await call_structured(TaskName.CONNECTIVITY_PROBE, {"token": "t"}, ProbeAnswer, **args)
    await call_structured(TaskName.CONNECTIVITY_PROBE, {"token": "t"}, ProbeAnswer, **args)

    assert provider.call_count == 2
    assert memory_cache.writes == 0


async def test_caching_can_be_disabled_globally(env, memory_cache, cacheable_task):
    env({"LLM_CACHE_ENABLED": "false"})
    provider = FakeProvider("nvidia", [json_result({"name": "Ada", "years": 7})])
    args = dict(router=make_router(provider), cache=memory_cache, settings=get_settings())

    await call_structured(cacheable_task, {"resume": "x"}, Extraction, **args)
    await call_structured(cacheable_task, {"resume": "x"}, Extraction, **args)

    assert provider.call_count == 2
    assert memory_cache.reads == 0


async def test_a_stale_cache_entry_is_ignored_rather_than_returned(
    env, memory_cache, cacheable_task
):
    env()
    provider = FakeProvider("nvidia", [json_result({"name": "Ada", "years": 7})])
    args = dict(router=make_router(provider), cache=memory_cache, settings=get_settings())

    await call_structured(cacheable_task, {"resume": "x"}, Extraction, **args)
    key = next(iter(memory_cache.entries))
    memory_cache.entries[key] = CachedCall(
        payload={"name": "Ada"},  # missing `years`: a shape an older build wrote
        provider="nvidia",
        model="old-model",
    )

    answer, meta = await call_structured(cacheable_task, {"resume": "x"}, Extraction, **args)
    assert answer.years == 7
    assert meta.cache_hit is False


async def test_the_cache_contract_is_that_it_never_raises(env, cacheable_task):
    """Where the degradation lives, and why it is not duplicated here.

    ``RedisResponseCache`` swallows a dead Redis and reports a miss - proven in
    test_cache_and_pricing.py. ``call_structured`` therefore trusts its cache
    and does not wrap it in a second try/except, which would hide a genuine bug
    in a future cache backend. This test pins that division of responsibility.
    """
    env()
    provider = FakeProvider("nvidia", [json_result({"name": "Ada", "years": 7})])

    with pytest.raises(RuntimeError, match="cache backend is down"):
        await call_structured(
            cacheable_task,
            {"resume": "x"},
            Extraction,
            router=make_router(provider),
            cache=ExplodingCache(),
            settings=get_settings(),
        )


# --- cost ------------------------------------------------------------------


async def test_cost_is_computed_and_flagged_when_the_price_is_known(env, memory_cache):
    env()
    provider = FakeProvider(
        "nvidia",
        [
            json_result(
                PROBE_JSON,
                model="nvidia/nemotron-3.5-lightning-30b-a3b",
                input_tokens=1000,
                output_tokens=500,
            )
        ],
    )

    _, meta = await call_structured(
        TaskName.CONNECTIVITY_PROBE,
        {"token": "abc123"},
        ProbeAnswer,
        router=make_router(provider),
        cache=memory_cache,
        settings=get_settings(),
    )

    assert meta.price_known is True
    assert meta.cost_usd == Decimal("0.000000")


async def test_an_unpriced_model_is_flagged_rather_than_reported_as_free(env, memory_cache):
    env()
    provider = FakeProvider("nvidia", [json_result(PROBE_JSON, model="some/unlisted-model")])

    _, meta = await call_structured(
        TaskName.CONNECTIVITY_PROBE,
        {"token": "abc123"},
        ProbeAnswer,
        router=make_router(provider),
        cache=memory_cache,
        settings=get_settings(),
    )

    assert meta.price_known is False


# --- reasoning containment -------------------------------------------------


async def test_reasoning_never_reaches_the_caller_or_the_metadata(env, memory_cache):
    """Requirement L: chain of thought stops at the provider boundary."""
    env()
    provider = FakeProvider(
        "nvidia",
        [
            json_result(
                PROBE_JSON,
                reasoning_text="The user is testing me. I should say abc123.",
                reasoning_tokens=140,
            )
        ],
    )

    answer, meta = await call_structured(
        TaskName.CONNECTIVITY_PROBE,
        {"token": "abc123"},
        ProbeAnswer,
        router=make_router(provider),
        cache=memory_cache,
        settings=get_settings(),
    )

    serialised = json.dumps({"answer": answer.model_dump(), "meta": meta.model_dump(mode="json")})
    assert "testing me" not in serialised
    # Only the accounting survives.
    assert meta.reasoning_tokens == 140


async def test_reasoning_is_off_for_todays_tasks(env, memory_cache):
    env()
    provider = FakeProvider("nvidia", [json_result(PROBE_JSON)])

    _, meta = await call_structured(
        TaskName.CONNECTIVITY_PROBE,
        {"token": "abc123"},
        ProbeAnswer,
        router=make_router(provider),
        cache=memory_cache,
        settings=get_settings(),
    )

    assert meta.reasoning_enabled is False
    assert provider.calls[0].reasoning.enabled is False


async def test_the_global_switch_can_force_reasoning_off(env, memory_cache, monkeypatch):
    """Even a task that asks for it yields to the kill switch."""
    from app.llm import tasks as tasks_module

    env({"LLM_REASONING_ENABLED": "false"})
    monkeypatch.setitem(
        prompts_module.PROMPTS,
        TaskName.DEEP_DIVE_AGENT,
        PromptTemplate(
            task=TaskName.DEEP_DIVE_AGENT,
            version="test-v1",
            system="Think.",
            user="$q",
            required_inputs=("q",),
        ),
    )
    assert tasks_module.TASK_SPECS[TaskName.DEEP_DIVE_AGENT].reasoning is True

    provider = FakeProvider("nvidia", [json_result({"name": "A", "years": 1})])
    await call_structured(
        TaskName.DEEP_DIVE_AGENT,
        {"q": "why"},
        Extraction,
        router=make_router(provider),
        cache=memory_cache,
        settings=get_settings(),
    )

    assert provider.calls[0].reasoning.enabled is False


# --- unregistered tasks ----------------------------------------------------


async def test_a_task_without_a_prompt_fails_loudly(env, memory_cache):
    env()
    provider = FakeProvider("nvidia", [json_result(PROBE_JSON)])

    with pytest.raises(PromptNotRegisteredError, match="grade_answer"):
        await call_structured(
            TaskName.GRADE_ANSWER,
            {},
            ProbeAnswer,
            router=make_router(provider),
            cache=memory_cache,
            settings=get_settings(),
        )
    assert provider.call_count == 0


async def test_a_missing_prompt_input_fails_before_any_call(env, memory_cache):
    env()
    provider = FakeProvider("nvidia", [json_result(PROBE_JSON)])

    with pytest.raises(KeyError, match="token"):
        await call_structured(
            TaskName.CONNECTIVITY_PROBE,
            {},
            ProbeAnswer,
            router=make_router(provider),
            cache=memory_cache,
            settings=get_settings(),
        )
    assert provider.call_count == 0
