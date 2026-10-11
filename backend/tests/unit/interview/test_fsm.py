"""The transition table: what is legal, what is not, and what ends.

The table is small enough to test exhaustively, so it is: every
(state, event) pair is checked against an independently written expectation,
not against the table itself.
"""

from __future__ import annotations

import itertools

import pytest

from app.interview import (
    TERMINAL_STATES,
    TRANSITIONS,
    EventType,
    IllegalTransition,
    InterviewState,
    allowed_events,
    is_terminal,
    next_state,
)
from app.interview.fsm import parse_event_type, parse_state, successors
from app.models.interview import SESSION_STATE_COMPLETED, SESSION_STATE_CREATED

S = InterviewState
E = EventType

#: Written out by hand from plan section 13.4 plus the documented additions:
#: the ANSWER_GRADED self-loop and SELECTING -> REPORTING on pool exhaustion
#: (Day 21), the GRADING_FAILED self-loop (Day 22), and PROBING -> SELECTING on
#: FOLLOW_UP_ABANDONED (Day 23).
EXPECTED: dict[tuple[InterviewState | None, EventType], InterviewState] = {
    (None, E.SESSION_CREATED): S.CREATED,
    (S.CREATED, E.PREPARATION_STARTED): S.PREPARING,
    (S.PREPARING, E.QUESTION_ASKED): S.ASKING,
    (S.ASKING, E.ANSWER_SUBMITTED): S.GRADING,
    (S.GRADING, E.ANSWER_GRADED): S.GRADING,
    (S.GRADING, E.GRADING_FAILED): S.GRADING,
    (S.GRADING, E.FOLLOW_UP_STARTED): S.PROBING,
    (S.GRADING, E.SELECTION_STARTED): S.SELECTING,
    (S.GRADING, E.REPORTING_STARTED): S.REPORTING,
    (S.PROBING, E.QUESTION_ASKED): S.ASKING,
    (S.PROBING, E.FOLLOW_UP_ABANDONED): S.SELECTING,
    (S.SELECTING, E.QUESTION_ASKED): S.ASKING,
    (S.SELECTING, E.REPORTING_STARTED): S.REPORTING,
    (S.REPORTING, E.SESSION_COMPLETED): S.COMPLETED,
}

ALL_SOURCES: list[InterviewState | None] = [None, *InterviewState]


def test_the_table_is_exactly_the_specification():
    assert dict(TRANSITIONS) == EXPECTED


@pytest.mark.parametrize(("source", "event"), list(itertools.product(ALL_SOURCES, EventType)))
def test_every_pair_is_either_legal_or_rejected(source, event):
    expected = EXPECTED.get((source, event))
    if expected is None:
        with pytest.raises(IllegalTransition) as exc:
            next_state(source, event)
        assert exc.value.state == source
        assert exc.value.event_type == event.value
    else:
        assert next_state(source, event) is expected


def test_the_plan_flow_is_reachable_in_order():
    path = [
        (E.SESSION_CREATED, S.CREATED),
        (E.PREPARATION_STARTED, S.PREPARING),
        (E.QUESTION_ASKED, S.ASKING),
        (E.ANSWER_SUBMITTED, S.GRADING),
        (E.SELECTION_STARTED, S.SELECTING),
        (E.REPORTING_STARTED, S.REPORTING),
        (E.SESSION_COMPLETED, S.COMPLETED),
    ]
    state: InterviewState | None = None
    for event, expected in path:
        state = next_state(state, event)
        assert state is expected


def test_both_probing_and_selecting_lead_back_to_asking():
    # Day 23: an abandoned follow-up continues with normal selection.
    assert successors(S.PROBING) == {S.ASKING, S.SELECTING}
    assert successors(S.SELECTING) == {S.ASKING, S.REPORTING}


class TestTerminal:
    def test_completed_is_the_only_terminal_state(self):
        assert TERMINAL_STATES == {S.COMPLETED}
        assert [s for s in InterviewState if is_terminal(s)] == [S.COMPLETED]

    @pytest.mark.parametrize("event", list(EventType))
    def test_nothing_leaves_completed(self, event):
        with pytest.raises(IllegalTransition, match="terminal state"):
            next_state(S.COMPLETED, event)

    def test_completed_accepts_no_events(self):
        assert allowed_events(S.COMPLETED) == frozenset()
        assert successors(S.COMPLETED) == frozenset()

    def test_every_non_terminal_state_has_a_way_forward(self):
        for state in InterviewState:
            if not is_terminal(state):
                assert allowed_events(state), state

    def test_every_state_can_reach_completed(self):
        """No dead ends: from anywhere, the interview can still finish."""
        for start in InterviewState:
            seen, frontier = {start}, [start]
            while frontier:
                for nxt in successors(frontier.pop()):
                    if nxt not in seen:
                        seen.add(nxt)
                        frontier.append(nxt)
            assert S.COMPLETED in seen, start


class TestMalformedInput:
    @pytest.mark.parametrize("raw", ["ACTIVE", "created", "", " ASKING", "DONE"])
    def test_unknown_state_strings_are_rejected(self, raw):
        with pytest.raises(ValueError, match="unknown interview state"):
            parse_state(raw)

    @pytest.mark.parametrize("raw", [None, 3, b"ASKING"])
    def test_non_string_states_are_rejected(self, raw):
        with pytest.raises(ValueError, match="must be a string"):
            parse_state(raw)

    @pytest.mark.parametrize("raw", ["session_created", "RESET", "", "SESSION_DELETED"])
    def test_unknown_event_types_are_rejected(self, raw):
        with pytest.raises(ValueError, match="unknown event type"):
            parse_event_type(raw)

    def test_next_state_refuses_raw_strings(self):
        """Strings must be parsed first; the table is keyed by enums only."""
        with pytest.raises(TypeError):
            next_state("CREATED", E.PREPARATION_STARTED)  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            next_state(S.CREATED, "PREPARATION_STARTED")  # type: ignore[arg-type]

    def test_the_table_cannot_be_mutated(self):
        with pytest.raises(TypeError):
            TRANSITIONS[(S.COMPLETED, E.SESSION_CREATED)] = S.CREATED  # type: ignore[index]


def test_state_values_match_the_day_2_model_constants():
    assert S.CREATED.value == SESSION_STATE_CREATED
    assert S.COMPLETED.value == SESSION_STATE_COMPLETED


def test_session_created_is_only_legal_on_an_empty_log():
    assert allowed_events(None) == {E.SESSION_CREATED}
    for state in InterviewState:
        assert E.SESSION_CREATED not in allowed_events(state)
