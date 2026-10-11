"""GRADING_FAILED (Day 22): a durable record of a missing grade, not a state."""

from __future__ import annotations

import pytest

from app.interview import (
    EventLogCorrupted,
    EventType,
    InterviewState,
    InvalidEventPayload,
    fold,
    validate_payload,
)
from tests.unit.interview.scenario import HAPPY_PATH, SESSION_ID, TURN_1, TURN_3, as_stored

E = EventType


def _failed(turn=TURN_1, **over):
    payload = {
        "turn_id": str(turn),
        "error_code": "provider_timeout",
        "attempts": 3,
        "retries_exhausted": True,
    }
    payload.update(over)
    return (E.GRADING_FAILED, payload)


def test_a_failure_keeps_the_session_in_grading_and_ungraded():
    state = fold(SESSION_ID, as_stored([*HAPPY_PATH[:4], _failed()]))
    assert state.state is InterviewState.GRADING
    assert not state.current_graded
    assert state.current_turn_id == TURN_1
    assert state.ability == {}


def test_a_grade_after_a_failure_is_legal_requeue():
    steps = [*HAPPY_PATH[:4], _failed(), _failed(), HAPPY_PATH[4]]
    state = fold(SESSION_ID, as_stored(steps))
    assert state.current_graded and "arrays" in state.ability


def test_grading_cannot_be_left_on_a_failure_alone():
    steps = [*HAPPY_PATH[:4], _failed(), (E.SELECTION_STARTED, {})]
    with pytest.raises(EventLogCorrupted, match="not been graded"):
        fold(SESSION_ID, as_stored(steps))


def test_a_failure_for_another_turn_is_rejected():
    with pytest.raises(EventLogCorrupted, match="is being graded"):
        fold(SESSION_ID, as_stored([*HAPPY_PATH[:4], _failed(turn=TURN_3)]))


def test_a_failure_after_the_grade_is_rejected():
    with pytest.raises(EventLogCorrupted, match="already graded"):
        fold(SESSION_ID, as_stored([*HAPPY_PATH[:5], _failed()]))


def test_a_failure_outside_grading_is_rejected():
    with pytest.raises(EventLogCorrupted, match="not allowed from ASKING"):
        fold(SESSION_ID, as_stored([*HAPPY_PATH[:3], _failed()]))


@pytest.mark.parametrize(
    "over",
    [
        {"error_code": "Provider said: 'quota exceeded for key sk-...'"},
        {"error_code": ""},
        {"attempts": 0},
        {"retries_exhausted": "maybe"},
        {"message": "raw provider text"},
    ],
)
def test_payload_is_a_code_not_a_message(over):
    _, payload = _failed(**over)
    with pytest.raises(InvalidEventPayload):
        validate_payload(E.GRADING_FAILED, payload)
