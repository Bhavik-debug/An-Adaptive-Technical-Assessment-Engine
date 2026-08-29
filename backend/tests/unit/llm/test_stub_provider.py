"""The offline provider: determinism, replay, synthesis, and the line between them.

Plan section 3, Day 5: *"Offline replay mode (stub provider that serves recorded
fixtures) so every future test runs without API calls."*

The property this file exists to protect is not "the stub returns something".
It is that **a replayed answer and an invented one are never confusable**. Every
phase after this one will run its evals through this provider; the day someone
measures grading quality against synthesized labels and does not notice, the
number they report is meaningless. So the tests below spend most of their effort
on that distinction rather than on the happy path.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import BaseModel, Field

from app.llm.errors import (
    ProviderAuthError,
    ProviderRateLimitedError,
    ProviderTimeoutError,
)
from app.llm.fixtures import Fixture, FixtureStore, RecordedError, fixture_key
from app.llm.providers.stub import (
    MODE_SYNTHESIZED,
    STUB_MODEL,
    MissingFixtureError,
    MissingFixturePolicy,
    StubProvider,
    synthesize,
)
from app.llm.structured import schema_spec
from app.llm.types import (
    ChatMessage,
    CompletionRequest,
    ModelTier,
    ReasoningPolicy,
    Role,
)


class Answer(BaseModel):
    ok: bool
    echo: str
    count: int


def make_request(
    *,
    content: str = "say something",
    schema: type[BaseModel] = Answer,
    temperature: float = 0.0,
    tier: ModelTier = ModelTier.SMALL_FAST,
) -> CompletionRequest:
    return CompletionRequest(
        messages=(ChatMessage(role=Role.USER, content=content),),
        json_schema=schema_spec(schema),
        tier=tier,
        temperature=temperature,
        top_p=1.0,
        max_output_tokens=256,
        timeout_s=30.0,
        reasoning=ReasoningPolicy(enabled=False),
    )


def _empty_dir() -> Path:
    """A directory that does not exist.

    ``FixtureStore`` logs and stays empty rather than raising, which is exactly
    what a test wanting a blank slate needs - and it also asserts, implicitly,
    that a missing fixture directory is not a crash.
    """
    return Path(__file__).resolve().parent / "_no_such_fixture_dir"


# ---------------------------------------------------------------------------
# It is a provider like any other
# ---------------------------------------------------------------------------


class TestItImplementsTheProviderInterface:
    def test_it_names_itself_stub(self):
        assert StubProvider(store=FixtureStore(_empty_dir())).name == "stub"

    def test_every_tier_maps_to_one_model(self):
        """Not tier-dependent on purpose - see the docstring in stub.py.

        A per-tier model id would change the fixture key per tier, so the same
        recorded answer would need recording three times, and the stub is not
        modelling capability differences.
        """
        provider = StubProvider(store=FixtureStore(_empty_dir()))
        assert {provider.model_for(t) for t in ModelTier} == {STUB_MODEL}

    def test_the_stub_model_is_priced_so_cost_arithmetic_still_runs(self):
        """`price_known=False` means "unknown", not "free". Those differ."""
        from app.llm.pricing import price_call

        cost = price_call(model=STUB_MODEL, input_tokens=1000, output_tokens=500)
        assert cost.price_known is True
        assert cost.usd == 0

    async def test_closing_it_is_a_no_op_that_does_not_raise(self):
        await StubProvider(store=FixtureStore(_empty_dir())).aclose()


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------


class TestReplay:
    async def test_a_recorded_response_comes_back_verbatim(self):
        request = make_request()
        provider = StubProvider(store=FixtureStore(_empty_dir()))
        recorded = '{"ok":true,"echo":"hello","count":7}'
        provider.add_fixture(
            Fixture(
                key=provider.key_for(request),
                description="test",
                text=recorded,
                input_tokens=322,
                output_tokens=19,
            )
        )

        result = await provider.complete(request)

        assert result.text == recorded
        assert result.provider == "stub"
        assert result.model == STUB_MODEL
        # Recorded token counts survive, so cost arithmetic downstream is real
        # arithmetic on real numbers rather than zeroes.
        assert (result.input_tokens, result.output_tokens) == (322, 19)
        assert provider.replayed == 1
        assert provider.synthesized == 0

    async def test_a_replayed_answer_is_labelled_as_replayed(self):
        """This label is the whole safety mechanism - see the module docstring."""
        request = make_request()
        provider = StubProvider(store=FixtureStore(_empty_dir()))
        provider.add_fixture(
            Fixture(
                key=provider.key_for(request),
                description="t",
                text='{"ok":true,"echo":"","count":0}',
            )
        )

        result = await provider.complete(request)

        assert result.structured_mode == "stub_replay"

    async def test_a_fixture_cannot_make_a_replay_look_like_a_live_call(self):
        """Regression. `Fixture` used to carry `structured_mode` straight
        through to the result, so a recording claiming `json_schema` produced a
        `CallMeta` indistinguishable from a real NVIDIA call - in the log, on
        the span, everywhere. The whole safety story of the stub rests on that
        field, so the provider decides it and the data cannot influence it.
        """
        request = make_request()
        provider = StubProvider(store=FixtureStore(_empty_dir()))
        provider.add_fixture(
            Fixture(
                key=provider.key_for(request),
                description="a recording that would rather look live",
                text='{"ok":true,"echo":"","count":0}',
                recorded_structured_mode="json_schema",
            )
        )

        result = await provider.complete(request)

        assert result.structured_mode == "stub_replay"

    async def test_reasoning_text_is_never_replayed(self):
        """A recording is permanent; chain-of-thought is the last thing to make
        permanent. Only the count survives, which is all cost needs."""
        request = make_request()
        provider = StubProvider(store=FixtureStore(_empty_dir()))
        provider.add_fixture(
            Fixture(
                key=provider.key_for(request),
                description="t",
                text='{"ok":true,"echo":"","count":0}',
                reasoning_tokens=140,
            )
        )

        result = await provider.complete(request)

        assert result.reasoning_text is None
        assert result.reasoning_tokens == 140


class TestTheKeyTracksTheRequest:
    """A fixture that keeps being served after its prompt changed is worse than
    no fixture at all: it is a silent, permanent, wrong answer."""

    def test_the_same_request_always_produces_the_same_key(self):
        assert fixture_key(make_request(), model=STUB_MODEL) == fixture_key(
            make_request(), model=STUB_MODEL
        )

    @pytest.mark.parametrize(
        ("label", "changed"),
        [
            ("prompt text", lambda: make_request(content="something else")),
            ("temperature", lambda: make_request(temperature=0.7)),
            ("tier", lambda: make_request(tier=ModelTier.MID)),
        ],
    )
    def test_anything_that_would_change_the_answer_changes_the_key(self, label, changed):
        assert fixture_key(changed(), model=STUB_MODEL) != fixture_key(
            make_request(), model=STUB_MODEL
        )

    def test_a_changed_schema_changes_the_key(self):
        class Different(BaseModel):
            ok: bool
            echo: str
            count: int
            extra: str

        assert fixture_key(make_request(schema=Different), model=STUB_MODEL) != fixture_key(
            make_request(), model=STUB_MODEL
        )

    def test_the_timeout_does_not_change_the_key(self):
        """A deadline changes whether you get an answer, never which answer.

        Including it would invalidate every recording in the repository the day
        somebody tunes LLM_TIMEOUT_S.
        """
        base = make_request()
        slower = CompletionRequest(
            messages=base.messages,
            json_schema=base.json_schema,
            tier=base.tier,
            temperature=base.temperature,
            top_p=base.top_p,
            max_output_tokens=base.max_output_tokens,
            timeout_s=base.timeout_s * 10,
            reasoning=base.reasoning,
        )
        assert fixture_key(slower, model=STUB_MODEL) == fixture_key(base, model=STUB_MODEL)


# ---------------------------------------------------------------------------
# A miss
# ---------------------------------------------------------------------------


class TestMissingFixtures:
    async def test_strict_is_the_default_and_it_raises(self):
        """A suite that meant to replay must not silently run on invented data."""
        provider = StubProvider(store=FixtureStore(_empty_dir()))
        assert provider.on_missing is MissingFixturePolicy.STRICT

        with pytest.raises(MissingFixtureError) as exc:
            await provider.complete(make_request())

        message = str(exc.value)
        assert "no recorded fixture" in message
        # The key is in the message, because the next thing you want to do is
        # record that exact call.
        assert fixture_key(make_request(), model=STUB_MODEL) in message

    async def test_a_miss_is_a_provider_error_so_the_router_can_fail_over(self):
        from app.llm.errors import ProviderError

        provider = StubProvider(store=FixtureStore(_empty_dir()))
        with pytest.raises(ProviderError) as exc:
            await provider.complete(make_request())
        # Not retryable: the recording will still be missing next time.
        assert exc.value.retryable is False

    async def test_synthesize_produces_an_answer_instead(self):
        provider = StubProvider(
            store=FixtureStore(_empty_dir()), on_missing=MissingFixturePolicy.SYNTHESIZE
        )
        result = await provider.complete(make_request())

        assert result.structured_mode == MODE_SYNTHESIZED
        assert provider.synthesized == 1
        assert Answer.model_validate_json(result.text)

    async def test_a_synthesized_answer_is_labelled_as_synthesized(self):
        """The label that stops a quality measurement from silently being fiction."""
        provider = StubProvider(
            store=FixtureStore(_empty_dir()), on_missing=MissingFixturePolicy.SYNTHESIZE
        )
        result = await provider.complete(make_request())
        assert result.structured_mode != "stub_replay"
        assert "synthes" in result.structured_mode

    async def test_synthesized_strings_announce_themselves(self):
        """If one of these turns up in a report or a database row, its origin
        must be obvious without anyone having to investigate."""
        provider = StubProvider(
            store=FixtureStore(_empty_dir()), on_missing=MissingFixturePolicy.SYNTHESIZE
        )
        result = await provider.complete(make_request())
        assert "stub:" in result.text

    async def test_synthesis_is_deterministic(self):
        made = [
            await StubProvider(
                store=FixtureStore(_empty_dir()), on_missing=MissingFixturePolicy.SYNTHESIZE
            ).complete(make_request())
            for _ in range(3)
        ]
        assert len({r.text for r in made}) == 1

    async def test_different_requests_synthesize_different_answers(self):
        def run(content: str):
            return StubProvider(
                store=FixtureStore(_empty_dir()), on_missing=MissingFixturePolicy.SYNTHESIZE
            ).complete(make_request(content=content))

        first = await run("question one")
        second = await run("question two")
        assert first.text != second.text


# ---------------------------------------------------------------------------
# Recorded failures
# ---------------------------------------------------------------------------


class TestRecordedFailures:
    """Replaying a *failure* is what lets an offline test reach a retry, a
    failover or the circuit breaker through the real router - rather than
    through a fake provider that only exists in the test suite."""

    @pytest.mark.parametrize(
        ("kind", "expected"),
        [
            ("rate_limited", ProviderRateLimitedError),
            ("timeout", ProviderTimeoutError),
            ("auth", ProviderAuthError),
        ],
    )
    async def test_a_recorded_error_is_raised_as_the_real_exception(self, kind, expected):
        request = make_request()
        provider = StubProvider(store=FixtureStore(_empty_dir()))
        provider.add_fixture(
            Fixture(
                key=provider.key_for(request),
                description="a recorded failure",
                error=RecordedError(kind=kind, message="upstream said no", status_code=429),
            )
        )

        with pytest.raises(expected) as exc:
            await provider.complete(request)
        assert exc.value.provider == "stub"

    async def test_retryability_matches_the_real_error_class(self):
        """So the router's decision is the same offline as it is in production."""
        request = make_request()
        provider = StubProvider(store=FixtureStore(_empty_dir()))
        provider.add_fixture(
            Fixture(
                key=provider.key_for(request),
                description="429",
                error=RecordedError(kind="rate_limited", message="429", retry_after_s=0.25),
            )
        )
        with pytest.raises(ProviderRateLimitedError) as exc:
            await provider.complete(request)
        assert exc.value.retryable is True
        assert exc.value.retry_after_s == 0.25

    async def test_an_unknown_error_kind_does_not_import_anything(self):
        """A fixture is data. It must not be able to name an arbitrary class."""
        from app.llm.errors import ProviderResponseError

        request = make_request()
        provider = StubProvider(store=FixtureStore(_empty_dir()))
        provider.add_fixture(
            Fixture(
                key=provider.key_for(request),
                description="hostile",
                error=RecordedError(kind="os.system", message="rm -rf /"),
            )
        )
        with pytest.raises(ProviderResponseError) as exc:
            await provider.complete(request)
        assert "unknown error kind" in str(exc.value)


# ---------------------------------------------------------------------------
# Schema-driven synthesis
# ---------------------------------------------------------------------------


class TestSynthesisHandlesRealSchemas:
    """The shapes pydantic actually emits, since those are the only ones a
    provider will ever be handed."""

    def test_a_flat_model(self):
        value = synthesize(schema_spec(Answer).schema, seed="s")
        assert Answer.model_validate(value)

    def test_optional_fields_get_the_non_null_branch(self):
        class WithOptional(BaseModel):
            required: str
            optional: str | None = None

        value = synthesize(schema_spec(WithOptional).schema, seed="s")
        parsed = WithOptional.model_validate(value)
        # Filling it exercises the field; a test that wants a null records one.
        assert parsed.optional is not None

    def test_nested_models_resolve_through_defs(self):
        class Inner(BaseModel):
            key: str
            weight: int

        class Outer(BaseModel):
            title: str
            items: list[Inner]

        value = synthesize(schema_spec(Outer).schema, seed="s")
        parsed = Outer.model_validate(value)
        assert parsed.items and parsed.items[0].key

    def test_numeric_bounds_are_respected(self):
        class Bounded(BaseModel):
            confidence: float = Field(ge=0.0, le=1.0)
            level: int = Field(ge=0, le=4)

        for seed in ("a", "b", "c", "d", "e"):
            parsed = Bounded.model_validate(synthesize(schema_spec(Bounded).schema, seed=seed))
            assert 0.0 <= parsed.confidence <= 1.0
            assert 0 <= parsed.level <= 4

    def test_enums_take_the_first_value(self):
        from typing import Literal

        class Labelled(BaseModel):
            label: Literal["covered", "partial", "absent"]

        parsed = Labelled.model_validate(synthesize(schema_spec(Labelled).schema, seed="s"))
        assert parsed.label == "covered"

    def test_a_grading_shaped_schema_survives(self):
        """A dry run of the Phase-4 grader shape (plan section 7.2), so that the
        synthesizer is known to handle it before that phase depends on it."""
        from typing import Literal

        class Concept(BaseModel):
            key: str
            label: Literal["covered", "partial", "absent"]
            evidence: str | None

        class Rubric(BaseModel):
            structure: int = Field(ge=0, le=4)
            specificity: int = Field(ge=0, le=4)

        class Grade(BaseModel):
            concepts: list[Concept]
            rubric: Rubric
            grader_confidence: float = Field(ge=0.0, le=1.0)

        parsed = Grade.model_validate(synthesize(schema_spec(Grade).schema, seed="s"))
        assert parsed.concepts
        assert 0.0 <= parsed.grader_confidence <= 1.0
