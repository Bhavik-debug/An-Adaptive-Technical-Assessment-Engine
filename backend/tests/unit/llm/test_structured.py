"""Schema derivation, JSON recovery, and validation.

Validation is the only guarantee the system has that an LLM answer is usable,
so it is tested against the ways models actually misbehave rather than only
against clean input.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel, Field, ValidationError

from app.llm.structured import (
    JsonExtractionError,
    extract_json,
    format_validation_error,
    repair_messages,
    schema_instruction,
    schema_spec,
)
from app.llm.types import Role


class Grade(BaseModel):
    covered: list[str] = Field(description="Concept keys the answer covered.")
    confidence: float
    notes: str | None = None


class Nested(BaseModel):
    grade: Grade
    version: int


# --- schema derivation -----------------------------------------------------


def test_objects_are_closed_and_fully_required():
    spec = schema_spec(Grade)
    assert spec.schema["additionalProperties"] is False
    assert set(spec.schema["required"]) == {"covered", "confidence", "notes"}


def test_hardening_reaches_nested_definitions():
    spec = schema_spec(Nested)
    nested = spec.schema["$defs"]["Grade"]
    assert nested["additionalProperties"] is False
    assert set(nested["required"]) == {"covered", "confidence", "notes"}


def test_fingerprint_is_stable_and_field_sensitive():
    class Renamed(BaseModel):
        covered: list[str]
        confidence: float
        remarks: str | None = None

    assert schema_spec(Grade).fingerprint == schema_spec(Grade).fingerprint
    assert schema_spec(Grade).fingerprint != schema_spec(Renamed).fingerprint


def test_instruction_carries_the_schema_and_forbids_prose():
    text = schema_instruction(schema_spec(Grade))
    assert "confidence" in text
    assert "nothing else" in text


# --- extraction ------------------------------------------------------------


def test_clean_object_parses():
    assert extract_json('{"a": 1}') == {"a": 1}


def test_markdown_fence_is_stripped():
    raw = 'Here you go:\n```json\n{"a": 1}\n```\nHope that helps!'
    assert extract_json(raw) == {"a": 1}


def test_unlabelled_fence_is_stripped():
    assert extract_json('```\n{"a": 1}\n```') == {"a": 1}


def test_object_embedded_in_prose_is_recovered():
    assert extract_json('Sure. {"a": 1} That is the answer.') == {"a": 1}


def test_leading_prose_with_braces_does_not_confuse_the_scanner():
    raw = 'The set {not json} is irrelevant. {"a": 1}'
    assert extract_json(raw) == {"a": 1}


def test_empty_response_is_an_extraction_error():
    with pytest.raises(JsonExtractionError, match="empty"):
        extract_json("   ")


def test_prose_only_response_is_an_extraction_error():
    with pytest.raises(JsonExtractionError, match="no JSON"):
        extract_json("I'm sorry, I can't help with that.")


def test_truncated_object_is_an_extraction_error():
    with pytest.raises(JsonExtractionError):
        extract_json('{"covered": ["a", "b"')


# --- validation ------------------------------------------------------------


def test_validation_error_message_names_the_field():
    with pytest.raises(ValidationError) as exc:
        Grade.model_validate({"covered": "not-a-list", "confidence": "high"})
    message = format_validation_error(exc.value)
    assert "covered" in message
    assert "confidence" in message


def test_repair_messages_show_the_model_its_own_output():
    spec = schema_spec(Grade)
    messages = repair_messages(
        raw_output='{"covered": "oops"}', problem="covered: not a list", spec=spec
    )
    assert [m.role for m in messages] == [Role.ASSISTANT, Role.USER]
    assert '{"covered": "oops"}' in messages[0].content
    assert "covered: not a list" in messages[1].content
    assert "confidence" in messages[1].content


def test_repair_message_truncates_a_runaway_response():
    spec = schema_spec(Grade)
    messages = repair_messages(raw_output="x" * 10_000, problem="nope", spec=spec)
    assert len(messages[0].content) < 2_500
    assert "truncated" in messages[0].content
