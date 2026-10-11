"""``fold(events)``: deterministic replay, and refusal of any log it cannot trust."""

from __future__ import annotations

import json
import uuid

import pytest

from app.ability import AbilityState
from app.interview import (
    STATE_SCHEMA_VERSION,
    EventLogCorrupted,
    EventType,
    IllegalTransition,
    InterviewState,
    InvalidEventPayload,
    SessionState,
    StoredEvent,
    apply,
    fold,
    validate_payload,
)
from tests.unit.interview.scenario import (
    HAPPY_PATH,
    SESSION_ID,
    TURN_1,
    TURN_3,
    as_stored,
    prefix,
)

E = EventType
S = InterviewState


def _fold(steps):
    return fold(SESSION_ID, as_stored(steps))


# ---------------------------------------------------------------------------
# The happy path
# ---------------------------------------------------------------------------


class TestReplay:
    def test_full_interview_folds_to_completed(self):
        state = fold(SESSION_ID, as_stored(HAPPY_PATH))
        assert state.state is S.COMPLETED
        assert state.is_terminal
        assert state.seq == len(HAPPY_PATH) - 1
        assert state.asked_question_ids == ("dsa-001", "db-004")
        assert state.follow_ups_asked == 1
        assert state.questions_asked == 3
        assert state.stop_reasons == ("item_budget_reached",)
        # The later grade on the same subtopic wins; other subtopics are kept.
        assert state.ability == {
            "arrays": AbilityState(theta=-0.05, rd=0.95, n_observations=2),
            "indexing": AbilityState(theta=0.3, rd=1.0, n_observations=1),
        }

    def test_initial_state(self):
        initial = SessionState.initial(SESSION_ID)
        assert initial.seq == -1
        assert initial.state is None
        assert initial.ability == {}

    def test_creation_alone_is_created_at_seq_0(self):
        state = fold(SESSION_ID, prefix(1))
        assert (state.state, state.seq) == (S.CREATED, 0)

    @pytest.mark.parametrize("n", range(1, len(HAPPY_PATH) + 1))
    def test_every_prefix_is_a_valid_state(self, n):
        state = fold(SESSION_ID, prefix(n))
        assert state.seq == n - 1

    def test_replay_is_deterministic(self):
        events = as_stored(HAPPY_PATH)
        folds = [fold(SESSION_ID, events) for _ in range(5)]
        assert all(f == folds[0] for f in folds)
        assert len({json.dumps(f.to_dict(), sort_keys=True) for f in folds}) == 1

    def test_fold_equals_incremental_apply(self):
        """Folding all at once == applying one at a time from any midpoint."""
        events = as_stored(HAPPY_PATH)
        for split in range(1, len(events)):
            state = fold(SESSION_ID, events[:split])
            for event in events[split:]:
                state = apply(state, event)
            assert state == fold(SESSION_ID, events)

    def test_fold_does_not_mutate_its_input(self):
        events = as_stored(HAPPY_PATH)
        snapshot = [(e.seq, e.type, json.dumps(dict(e.payload), sort_keys=True)) for e in events]
        fold(SESSION_ID, events)
        assert snapshot == [
            (e.seq, e.type, json.dumps(dict(e.payload), sort_keys=True)) for e in events
        ]

    def test_states_are_immutable(self):
        state = fold(SESSION_ID, prefix(5))
        with pytest.raises(AttributeError):
            state.seq = 99  # type: ignore[misc]
        with pytest.raises(TypeError):
            state.ability["arrays"] = AbilityState(theta=0.0, rd=1.0)  # type: ignore[index]

    def test_grade_without_ability_keeps_the_estimate(self):
        steps = list(HAPPY_PATH[:4]) + [
            (E.ANSWER_GRADED, {"turn_id": str(TURN_1), "score": 0.5}),
        ]
        state = _fold(steps)
        assert state.current_graded
        assert state.ability == {}

    def test_selecting_can_end_the_interview_when_the_pool_is_empty(self):
        steps = list(HAPPY_PATH[:5]) + [
            (E.SELECTION_STARTED, {}),
            (E.REPORTING_STARTED, {"reasons": ["pool_exhausted"]}),
            (E.SESSION_COMPLETED, {}),
        ]
        assert _fold(steps).state is S.COMPLETED


# ---------------------------------------------------------------------------
# Corruption: every one of these must raise, naming where
# ---------------------------------------------------------------------------


def _corrupt_at(steps, seq_expected):
    with pytest.raises(EventLogCorrupted) as exc:
        _fold(steps)
    assert exc.value.seq == seq_expected
    assert exc.value.session_id == SESSION_ID
    return exc.value


class TestCorruptedLogs:
    def test_empty_log(self):
        with pytest.raises(EventLogCorrupted, match="empty"):
            fold(SESSION_ID, [])

    def test_log_not_starting_with_creation(self):
        _corrupt_at([(E.PREPARATION_STARTED, {})], 0)

    def test_log_not_starting_at_seq_0(self):
        with pytest.raises(EventLogCorrupted, match="expected seq 0, got 1"):
            fold(SESSION_ID, as_stored(list(HAPPY_PATH[:2]), start=1))

    def test_gap_in_seq(self):
        events = prefix(4)
        events[2] = StoredEvent(seq=5, type=events[2].type, payload=events[2].payload)
        with pytest.raises(EventLogCorrupted, match="expected seq 2, got 5") as exc:
            fold(SESSION_ID, events)
        assert exc.value.seq == 5

    def test_duplicate_seq(self):
        events = prefix(3)
        events.insert(2, events[1])
        with pytest.raises(EventLogCorrupted, match="expected seq 2, got 1"):
            fold(SESSION_ID, events)

    def test_out_of_order_events(self):
        events = prefix(4)
        events[1], events[2] = events[2], events[1]
        with pytest.raises(EventLogCorrupted):
            fold(SESSION_ID, events)

    @pytest.mark.parametrize("seq", ["1", 1.0, None, True])
    def test_non_integer_seq(self, seq):
        events = prefix(1) + [StoredEvent(seq=seq, type="PREPARATION_STARTED", payload={})]  # type: ignore[arg-type]
        with pytest.raises(EventLogCorrupted, match="seq must be an integer"):
            fold(SESSION_ID, events)

    def test_unknown_event_type(self):
        err = _corrupt_at([(E.SESSION_CREATED, {}), ("SESSION_RESET", {})], 1)
        assert "unknown event type" in err.reason

    def test_illegal_transition(self):
        err = _corrupt_at(
            [(E.SESSION_CREATED, {}), (E.ANSWER_SUBMITTED, {"turn_id": str(TURN_1)})], 1
        )
        assert "not allowed from CREATED" in err.reason
        assert isinstance(err.__cause__, IllegalTransition)

    def test_event_after_completion(self):
        steps = [*HAPPY_PATH, (E.PREPARATION_STARTED, {})]
        err = _corrupt_at(steps, len(HAPPY_PATH))
        assert "terminal" in err.reason

    def test_second_completion(self):
        _corrupt_at([*HAPPY_PATH, (E.SESSION_COMPLETED, {})], len(HAPPY_PATH))

    def test_invalid_payload(self):
        steps = list(HAPPY_PATH[:3]) + [(E.ANSWER_SUBMITTED, {"turn_id": "not-a-uuid"})]
        err = _corrupt_at(steps, 3)
        assert isinstance(err.__cause__, InvalidEventPayload)

    def test_payload_with_unknown_field_is_rejected_not_ignored(self):
        steps = list(HAPPY_PATH[:3]) + [
            (E.ANSWER_SUBMITTED, {"turn_id": str(TURN_1), "answer_text": "secret"})
        ]
        err = _corrupt_at(steps, 3)
        assert "answer_text" in err.reason

    def test_payload_that_is_not_an_object(self):
        events = prefix(1) + [StoredEvent(seq=1, type="PREPARATION_STARTED", payload=[])]  # type: ignore[arg-type]
        with pytest.raises(EventLogCorrupted, match="JSON object"):
            fold(SESSION_ID, events)

    def test_the_same_question_cannot_be_asked_twice(self):
        steps = list(HAPPY_PATH[:10]) + [(E.QUESTION_ASKED, {"question_id": "dsa-001"})]
        err = _corrupt_at(steps, 10)
        assert "already asked" in err.reason

    def test_grade_for_the_wrong_turn(self):
        steps = list(HAPPY_PATH[:4]) + [(E.ANSWER_GRADED, {"turn_id": str(TURN_3), "score": 0.5})]
        err = _corrupt_at(steps, 4)
        assert "is being graded" in err.reason

    def test_grade_recorded_twice(self):
        steps = list(HAPPY_PATH[:5]) + [(E.ANSWER_GRADED, {"turn_id": str(TURN_1), "score": 0.9})]
        err = _corrupt_at(steps, 5)
        assert "already graded" in err.reason

    @pytest.mark.parametrize(
        "leave",
        [
            (E.SELECTION_STARTED, {}),
            (E.FOLLOW_UP_STARTED, {}),
            (E.REPORTING_STARTED, {"reasons": ["time_budget_reached"]}),
        ],
    )
    def test_grading_cannot_be_left_ungraded(self, leave):
        err = _corrupt_at([*HAPPY_PATH[:4], leave], 4)
        assert "not been graded" in err.reason

    def test_follow_up_outside_probing(self):
        steps = list(HAPPY_PATH[:10]) + [(E.QUESTION_ASKED, {"is_follow_up": True})]
        err = _corrupt_at(steps, 10)
        assert "only from PROBING" in err.reason

    def test_fresh_question_from_probing(self):
        steps = list(HAPPY_PATH[:6]) + [(E.QUESTION_ASKED, {"question_id": "dsa-002"})]
        _corrupt_at(steps, 6)

    def test_fresh_question_without_an_id(self):
        _corrupt_at([*HAPPY_PATH[:2], (E.QUESTION_ASKED, {})], 2)

    def test_follow_up_with_a_bank_id(self):
        steps = list(HAPPY_PATH[:6]) + [
            (E.QUESTION_ASKED, {"is_follow_up": True, "question_id": "dsa-001"})
        ]
        _corrupt_at(steps, 6)

    def test_no_partial_state_is_returned(self):
        """A corrupt event in the middle fails the whole fold - never 'the rest'."""
        events = as_stored(HAPPY_PATH)
        events[7] = StoredEvent(seq=7, type="BOGUS", payload={})
        with pytest.raises(EventLogCorrupted):
            fold(SESSION_ID, events)


# ---------------------------------------------------------------------------
# Payload contracts
# ---------------------------------------------------------------------------


class TestPayloads:
    @pytest.mark.parametrize("score", [-0.01, 1.01, float("nan"), float("inf"), "high"])
    def test_scores_outside_the_unit_interval(self, score):
        with pytest.raises(InvalidEventPayload):
            validate_payload(E.ANSWER_GRADED, {"turn_id": str(TURN_1), "score": score})

    @pytest.mark.parametrize(
        "ability",
        [
            {"subtopic_key": "x", "theta": 0.0, "rd": 0.0, "n_observations": 1},
            {"subtopic_key": "x", "theta": float("nan"), "rd": 1.0, "n_observations": 1},
            {"subtopic_key": "", "theta": 0.0, "rd": 1.0, "n_observations": 1},
            {"subtopic_key": "x", "theta": 0.0, "rd": 1.0, "n_observations": -1},
            {"subtopic_key": "x", "theta": 0.0, "rd": 1.0},
        ],
    )
    def test_invalid_ability_snapshots(self, ability):
        with pytest.raises(InvalidEventPayload):
            validate_payload(
                E.ANSWER_GRADED, {"turn_id": str(TURN_1), "score": 0.5, "ability": ability}
            )

    @pytest.mark.parametrize(
        "reasons", [[], ["Item Budget"], ["a" * 65], ["dup", "dup"], "item_budget_reached"]
    )
    def test_reporting_needs_valid_distinct_reasons(self, reasons):
        with pytest.raises(InvalidEventPayload):
            validate_payload(E.REPORTING_STARTED, {"reasons": reasons})

    @pytest.mark.parametrize("event", [E.SESSION_CREATED, E.SELECTION_STARTED, E.SESSION_COMPLETED])
    def test_empty_events_take_no_fields(self, event):
        validate_payload(event, {})
        with pytest.raises(InvalidEventPayload):
            validate_payload(event, {"note": "x"})

    def test_question_id_whitespace_is_rejected(self):
        with pytest.raises(InvalidEventPayload):
            validate_payload(E.QUESTION_ASKED, {"question_id": " dsa-001"})


# ---------------------------------------------------------------------------
# Serialisation (what the Redis cache stores)
# ---------------------------------------------------------------------------


class TestSerialisation:
    @pytest.mark.parametrize("n", range(1, len(HAPPY_PATH) + 1))
    def test_round_trip_through_json(self, n):
        state = fold(SESSION_ID, prefix(n))
        restored = SessionState.from_dict(json.loads(json.dumps(state.to_dict())))
        assert restored == state

    def test_version_is_embedded(self):
        assert fold(SESSION_ID, prefix(1)).to_dict()["v"] == STATE_SCHEMA_VERSION

    def test_other_version_is_rejected(self):
        data = fold(SESSION_ID, prefix(3)).to_dict()
        data["v"] = STATE_SCHEMA_VERSION + 1
        with pytest.raises(ValueError, match="schema version"):
            SessionState.from_dict(data)

    @pytest.mark.parametrize(
        "mutate",
        [
            lambda d: d.pop("seq"),
            lambda d: d.update(seq="3"),
            lambda d: d.update(state="ACTIVE"),
            lambda d: d.update(session_id="nope"),
            lambda d: d.update(ability=[]),
            lambda d: d.update(ability={"x": {"theta": 0.0, "rd": -1.0, "n_observations": 0}}),
            lambda d: d.update(asked_question_ids="dsa-001"),
            lambda d: d.update(current_graded="yes"),
            lambda d: d.update(current_turn_id="zzz"),
            lambda d: d.update(stop_reasons=[1]),
        ],
    )
    def test_malformed_dicts_are_rejected(self, mutate):
        data = fold(SESSION_ID, prefix(5)).to_dict()
        mutate(data)
        with pytest.raises(ValueError):
            SessionState.from_dict(data)

    def test_states_of_different_sessions_differ(self):
        a = fold(SESSION_ID, prefix(3))
        b = fold(uuid.uuid4(), prefix(3))
        assert a != b
