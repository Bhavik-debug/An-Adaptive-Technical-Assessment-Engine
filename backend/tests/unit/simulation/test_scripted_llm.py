"""The scripted model (Day 25): its documented rule, and the stub's scripted mode."""

from __future__ import annotations

import json

import pytest

from app.bank.schema import ExpectedConcept
from app.config import get_settings
from app.grading.pipeline import grade_answer
from app.llm.fixtures import FixtureStore
from app.llm.providers.stub import MODE_SCRIPTED, MissingFixturePolicy, StubProvider
from app.simulation.scripted_llm import PROBE_TEXT, scripted_response
from tests.unit.llm.conftest import MemoryCache, make_router

CONCEPTS = [
    ExpectedConcept(key="hash_function", weight=3, hint="names a hash function and its role"),
    ExpectedConcept(key="collision_handling", weight=2, hint="explains how collisions are handled"),
    ExpectedConcept(key="load_factor", weight=1, hint="mentions resizing on load factor"),
]
STRONG = "A " + "detailed answer that discusses hashing, buckets, collisions and resizing " * 3
WEAK = "I'm not sure, honestly."


def stub(tmp_path):
    provider = StubProvider(store=FixtureStore(tmp_path), on_missing=MissingFixturePolicy.SCRIPTED)
    provider.set_responder(scripted_response)
    return provider


async def _grade(env, tmp_path, answer):
    env()
    return await grade_answer(
        question_text="How does a hash map work?",
        expected_concepts=CONCEPTS,
        answer_text=answer,
        router=make_router(stub(tmp_path), max_attempts_per_provider=1),
        cache=MemoryCache(),
        settings=get_settings(),
    )


async def test_a_strong_answer_covers_everything_through_the_real_pipeline(env, tmp_path):
    result = await _grade(env, tmp_path, STRONG)
    assert result.score == 1.0
    assert set(result.judgment.labels_by_key.values()) == {"covered"}
    assert result.verification.fabrication_count == 0  # evidence is a real substring
    assert {c.structured_mode for c in result.calls} == {MODE_SCRIPTED}


async def test_a_weak_answer_covers_nothing(env, tmp_path):
    result = await _grade(env, tmp_path, WEAK)
    assert result.score == 0.0
    assert set(result.judgment.labels_by_key.values()) == {"absent"}


def test_unknown_tasks_are_declined():
    from app.llm.types import CompletionRequest, JsonSchemaSpec

    class _R:  # minimal stand-in carrying only what the responder reads
        json_schema = JsonSchemaSpec(name="SomethingElse", schema={}, fingerprint="x")
        messages = ()

    assert scripted_response(_R()) is None  # type: ignore[arg-type]
    assert CompletionRequest  # imported for the type only


async def test_scripted_mode_without_a_responder_refuses(env, tmp_path):
    from app.llm.errors import AllProvidersFailedError

    env()
    provider = StubProvider(store=FixtureStore(tmp_path), on_missing=MissingFixturePolicy.SCRIPTED)
    with pytest.raises(AllProvidersFailedError):
        await grade_answer(
            question_text="Q?",
            expected_concepts=CONCEPTS,
            answer_text=STRONG,
            router=make_router(provider, max_attempts_per_provider=1),
            cache=MemoryCache(),
            settings=get_settings(),
        )


def test_probe_and_narrative_rules():
    from app.llm.types import ChatMessage, JsonSchemaSpec, Role

    class _R:
        def __init__(self, name, user):
            self.json_schema = JsonSchemaSpec(name=name, schema={}, fingerprint="x")
            self.messages = (ChatMessage(role=Role.USER, content=user),)

    assert json.loads(scripted_response(_R("FollowUpProbe", "")))["question"] == PROBE_TEXT  # type: ignore[arg-type]
    stats = json.dumps({"topics": [{"topic": "dsa"}, {"topic": "databases"}]})
    narrative = json.loads(scripted_response(_R("ReportNarrative", f"<stats>\n{stats}\n</stats>")))  # type: ignore[arg-type]
    assert "dsa, databases" in narrative["summary"]
    assert not any(ch.isdigit() for ch in narrative["summary"])
