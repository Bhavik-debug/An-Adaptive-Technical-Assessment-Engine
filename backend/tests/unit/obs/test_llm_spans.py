"""LLM observability: can one model call be followed, costed and trusted?

Plan section 14.2 names an attribute set as **non-negotiable on every LLM span**:
``prompt_version``, ``model``, ``input_tokens``, ``output_tokens``, ``cost_usd``,
``cache_hit``, ``schema_retry_count``, ``session_id``, ``plan``.  The reason it
matters is stated in the same paragraph: without ``prompt_version`` you cannot
attribute a quality regression to a prompt change.

These tests use the Day-3 fakes (``tests/unit/llm/conftest.py``) so that no
network, no provider and no key are involved, and assert on what came out of the
span recorder.  Nothing here re-implements the accounting - the point is that
Day 4 *reports* Day 3's numbers rather than computing its own.
"""

from __future__ import annotations

import uuid

import pytest
from pydantic import BaseModel, Field

from app.config import get_settings
from app.llm import TaskName, call_structured
from app.llm import prompts as prompts_module
from app.llm.errors import AllProvidersFailedError, ProviderAuthError, SchemaValidationFailedError
from app.llm.probe import ProbeAnswer
from app.llm.prompts import PromptTemplate
from app.llm.types import TraceCtx
from tests.unit.llm.conftest import FakeProvider, json_result, make_router


class Extraction(BaseModel):
    """A stand-in for a real Phase-5 schema, used for the cacheable task."""

    name: str = Field(description="The candidate's name.")
    years: int


PROBE_JSON = {"ok": True, "echo": "abc123", "model_said": "fake"}
SPAN = "llm.connectivity_probe"

#: Plan section 14.2, verbatim.
NON_NEGOTIABLE = (
    "llm.prompt_version",
    "llm.model",
    "llm.input_tokens",
    "llm.output_tokens",
    "llm.cost_usd",
    "llm.cache_hit",
    "llm.schema_retry_count",
)


@pytest.fixture
def cacheable_task(monkeypatch):
    """A task with a prompt, temperature 0, and therefore a cache.

    The connectivity probe is deliberately uncacheable - a probe served from
    cache would prove nothing - so cache behaviour is exercised against a
    stand-in for a real Phase-5 task, the same way ``tests/unit/llm`` does.
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


async def _probe(router, cache, *, trace=None, **kwargs):
    return await call_structured(
        TaskName.CONNECTIVITY_PROBE,
        {"token": "abc123"},
        ProbeAnswer,
        router=router,
        cache=cache,
        settings=get_settings(),
        **({"trace": trace} if trace else {}),
        **kwargs,
    )


class TestTheAttributeSet:
    async def test_a_successful_call_produces_a_span_with_every_required_attribute(
        self, env, spans, memory_cache
    ):
        env()
        provider = FakeProvider(
            "nvidia",
            [json_result(PROBE_JSON, provider="nvidia", input_tokens=612, output_tokens=148)],
        )

        await _probe(make_router(provider), memory_cache)

        attributes = spans.attributes(SPAN)
        for required in NON_NEGOTIABLE:
            assert required in attributes, f"{required} missing from the span"
        assert attributes["llm.input_tokens"] == 612
        assert attributes["llm.output_tokens"] == 148
        assert attributes["llm.provider"] == "nvidia"
        assert attributes["llm.task"] == "connectivity_probe"
        # Not part of the plan's list, but the reason a slow turn is diagnosable.
        assert attributes["llm.latency_ms"] >= 0

    async def test_the_span_reports_exactly_what_the_metadata_says(self, env, spans, memory_cache):
        """Day 4 must not recompute Day 3's accounting - only project it.

        If these two ever disagree, a cost report and a trace would tell
        different stories about the same call, and neither could be trusted.
        """
        env()
        provider = FakeProvider("nvidia", [json_result(PROBE_JSON)])

        _, meta = await _probe(make_router(provider), memory_cache)

        attributes = spans.attributes(SPAN)
        for key, value in meta.as_span_attributes().items():
            assert attributes[key] == value

    async def test_trace_context_reaches_the_span(self, env, spans, memory_cache):
        """``session_id`` and ``plan`` are on the plan's non-negotiable list.

        They are what turn a pile of spans into "this candidate's interview",
        and later "this plan tier's cost per interview".
        """
        env()
        session = uuid.uuid4()
        provider = FakeProvider("nvidia", [json_result(PROBE_JSON)])

        await _probe(
            make_router(provider),
            memory_cache,
            trace=TraceCtx(session_id=session, plan="pro"),
        )

        attributes = spans.attributes(SPAN)
        assert attributes["llm.session_id"] == str(session)
        assert attributes["llm.plan"] == "pro"

    async def test_the_span_also_speaks_the_standard_model_call_vocabulary(
        self, env, spans, memory_cache
    ):
        """So a general-purpose trace viewer renders this as a generation."""
        env()
        provider = FakeProvider(
            "nvidia", [json_result(PROBE_JSON, provider="nvidia", input_tokens=10)]
        )

        await _probe(make_router(provider), memory_cache)

        attributes = spans.attributes(SPAN)
        assert attributes["gen_ai.system"] == "nvidia"
        assert attributes["gen_ai.usage.input_tokens"] == 10
        assert attributes["langfuse.observation.type"] == "generation"


class TestPrivacy:
    """The rule that matters most: a prompt and an answer never leave the box."""

    async def test_no_prompt_answer_or_reasoning_appears_on_the_span(
        self, env, spans, memory_cache
    ):
        env()
        secret_answer = "the candidate said their phone number is 555-0100"
        provider = FakeProvider(
            "nvidia",
            [
                json_result(
                    {"ok": True, "echo": secret_answer, "model_said": "m"},
                    input_tokens=5,
                )
            ],
        )

        class Echo(BaseModel):
            ok: bool
            echo: str = Field(description="anything")
            model_said: str

        await call_structured(
            TaskName.CONNECTIVITY_PROBE,
            {"token": "a-candidate-answer-full-of-personal-detail"},
            Echo,
            router=make_router(provider),
            cache=memory_cache,
            settings=get_settings(),
        )

        rendered = repr(spans.attributes(SPAN))
        assert secret_answer not in rendered
        assert "a-candidate-answer-full-of-personal-detail" not in rendered

    async def test_no_prompt_or_answer_appears_in_the_log_line_either(
        self, env, captured_logs, memory_cache
    ):
        env()
        provider = FakeProvider("nvidia", [json_result(PROBE_JSON)])

        await _probe(make_router(provider), memory_cache)

        record = captured_logs.one("llm_call")
        assert "abc123" not in captured_logs.text
        # Metadata, though, is all there - that is the point of the line.
        assert record["llm.model"]
        assert record["llm.prompt_version"]


class TestTheThingsThatOnlyShowUpInProduction:
    async def test_a_cache_hit_is_visible_as_a_call_that_cost_nothing(
        self, env, spans, memory_cache, cacheable_task
    ):
        """A served-from-cache answer must not be invisible.

        Cache hit rate is an operational metric in plan section 12.4, and a
        cached answer is the first thing to check when a session behaves oddly.
        Day 3 returned early on a hit without emitting anything; Day 4 makes a
        hit a first-class, traced call that happens to have cost nothing.
        """
        env()
        provider = FakeProvider("nvidia", [json_result({"name": "Ada", "years": 7})])
        router = make_router(provider)
        args = {"router": router, "cache": memory_cache, "settings": get_settings()}

        await call_structured(cacheable_task, {"resume": "Ada, 7 years"}, Extraction, **args)
        await call_structured(cacheable_task, {"resume": "Ada, 7 years"}, Extraction, **args)

        grade_spans = [s for s in spans.finished if s.name == "llm.resume_extraction"]
        assert len(grade_spans) == 2
        first, second = (dict(s.attributes or {}) for s in grade_spans)
        assert first["llm.cache_hit"] is False
        assert second["llm.cache_hit"] is True
        # The provider was called exactly once for two logical calls.
        assert provider.call_count == 1
        assert second["llm.input_tokens"] == 0

    async def test_a_schema_repair_is_counted_on_the_span(self, env, spans, memory_cache):
        """The number that tells you a prompt has started drifting."""
        env()
        provider = FakeProvider("nvidia", [json_result({"nope": 1}), json_result(PROBE_JSON)])

        await _probe(make_router(provider), memory_cache)

        assert spans.attributes(SPAN)["llm.schema_retry_count"] == 1

    async def test_a_failover_is_counted_on_the_span(self, env, spans, memory_cache):
        """The Phase 1 exit gate, made observable rather than merely tested."""
        env()
        dead = FakeProvider(
            "nvidia", [ProviderAuthError("bad key", provider="nvidia", status_code=401)]
        )
        alive = FakeProvider("backup", [json_result(PROBE_JSON, provider="backup")])

        await _probe(make_router(dead, alive), memory_cache)

        attributes = spans.attributes(SPAN)
        assert attributes["llm.failover_count"] == 1
        assert attributes["llm.provider"] == "backup"


class TestFailurePaths:
    async def test_a_call_no_provider_answered_leaves_a_red_span(self, env, spans, memory_cache):
        env()
        dead = FakeProvider(
            "nvidia", [ProviderAuthError("bad key", provider="nvidia", status_code=401)]
        )

        with pytest.raises(AllProvidersFailedError):
            await _probe(make_router(dead), memory_cache)

        span = spans.named(SPAN)
        assert span.status.status_code.name == "ERROR"
        assert span.events[0].name == "exception"
        assert span.events[0].attributes["exception.type"] == "AllProvidersFailedError"

    async def test_a_call_nothing_validated_leaves_a_red_span(self, env, spans, memory_cache):
        env()
        provider = FakeProvider("nvidia", [json_result({"nope": 1})])

        with pytest.raises(SchemaValidationFailedError):
            await _probe(make_router(provider), memory_cache)

        span = spans.named(SPAN)
        assert span.status.status_code.name == "ERROR"
        assert span.events[0].attributes["exception.type"] == "SchemaValidationFailedError"

    async def test_a_failure_span_carries_no_model_output(self, env, spans, memory_cache):
        """The error path is where raw model output most easily escapes.

        ``SchemaValidationFailedError`` holds the raw output for tests to
        inspect; the span must carry the diagnosis, not the payload.
        """
        env()
        leaky = {"candidate_phone": "555-0100", "unexpected": "field"}
        provider = FakeProvider("nvidia", [json_result(leaky)])

        with pytest.raises(SchemaValidationFailedError):
            await _probe(make_router(provider), memory_cache)

        rendered = repr(spans.named(SPAN).events[0].attributes)
        assert "555-0100" not in rendered

    async def test_an_error_message_containing_a_credential_is_redacted_on_the_span(
        self, env, spans, memory_cache
    ):
        """Provider errors are exactly where a key gets pasted by accident.

        The adapter is careful not to include one; this is the layer that makes
        that a guarantee rather than a habit, because a span leaves the host.
        """
        env()
        dead = FakeProvider(
            "nvidia",
            [
                ProviderAuthError(
                    "rejected key nvapi-Abc123Def456Ghi789Jkl",
                    provider="nvidia",
                    status_code=401,
                )
            ],
        )

        with pytest.raises(AllProvidersFailedError):
            await _probe(make_router(dead), memory_cache)

        rendered = repr(spans.named(SPAN).events[0].attributes)
        assert "nvapi-Abc123" not in rendered
        assert "[REDACTED]" in rendered


class TestItStillWorksWithNoTracing:
    async def test_calls_succeed_when_tracing_was_never_initialised(
        self, env, memory_cache, monkeypatch
    ):
        """The chokepoint must not depend on observability being configured.

        Scripts, the Day 5 offline replay mode, and the smoke test all call
        ``call_structured`` in a process that never built a tracer.
        """
        from app.obs import tracing as obs_tracing

        env()
        monkeypatch.setattr(obs_tracing, "_provider", None)
        monkeypatch.setattr(obs_tracing, "_enabled", False)
        provider = FakeProvider("nvidia", [json_result(PROBE_JSON, provider="nvidia")])

        answer, meta = await _probe(make_router(provider), memory_cache)

        assert answer.echo == "abc123"
        assert meta.provider == "nvidia"
