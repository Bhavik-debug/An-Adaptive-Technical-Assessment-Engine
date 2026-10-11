"""Day 23's follow-up bookkeeping in the fold: caps are invariants of the log."""

from __future__ import annotations

import json
import uuid
from types import MappingProxyType

import pytest

from app.interview import (
    EventLogCorrupted,
    EventType,
    IllegalTransition,
    InterviewState,
    SessionState,
    StoredEvent,
    apply,
    fold,
)
from app.interview.state import (
    MAX_FOLLOW_UPS_PER_ITEM,
    MAX_FOLLOW_UPS_PER_SESSION,
    STATE_SCHEMA_VERSION,
)
from tests.unit.interview.scenario import SESSION_ID, as_stored

E = EventType


class Log:
    """Builds a legal log step by step, so each test states only what matters."""

    def __init__(self) -> None:
        self.steps: list = [(E.SESSION_CREATED, {}), (E.PREPARATION_STARTED, {})]

    def ask(self, qid: str, text: str = "Bank question?") -> Log:
        self.steps.append((E.QUESTION_ASKED, {"question_id": qid, "text": text}))
        return self

    def answer_and_grade(self, delta: float | None = 0.2) -> Log:
        turn = str(uuid.uuid4())
        self.steps.append((E.ANSWER_SUBMITTED, {"turn_id": turn}))
        payload = {"turn_id": turn, "score": 0.5}
        if delta is not None:
            payload["delta_theta"] = delta
        self.steps.append((E.ANSWER_GRADED, payload))
        return self

    def follow_up(self, text: str = "Probe?") -> Log:
        self.steps.append((E.FOLLOW_UP_STARTED, {"reasons": ["high_uncertainty"]}))
        self.steps.append((E.QUESTION_ASKED, {"is_follow_up": True, "text": text}))
        return self

    def select(self, qid: str) -> Log:
        self.steps.append((E.SELECTION_STARTED, {}))
        return self.ask(qid)

    def fold(self) -> SessionState:
        return fold(SESSION_ID, as_stored(self.steps))


def test_a_follow_up_keeps_its_parent_item():
    state = Log().ask("q1").answer_and_grade().follow_up("Probe one?").fold()
    assert state.current_is_follow_up
    assert state.current_question_id is None
    assert state.current_item_id == "q1"
    assert state.current_question_text == "Probe one?"
    assert dict(state.follow_ups_by_item) == {"q1": 1}
    assert state.asked_question_ids == ("q1",)  # a follow-up is not a new item


def test_a_follow_up_of_a_follow_up_still_counts_against_the_original():
    log = Log().ask("q1").answer_and_grade().follow_up().answer_and_grade().follow_up()
    state = log.fold()
    assert dict(state.follow_ups_by_item) == {"q1": MAX_FOLLOW_UPS_PER_ITEM}
    assert not state.follow_up_room


def test_a_third_follow_up_on_one_item_is_illegal_in_the_log():
    log = Log().ask("q1").answer_and_grade()
    for _ in range(MAX_FOLLOW_UPS_PER_ITEM):
        log.follow_up().answer_and_grade()
    log.steps.append((E.FOLLOW_UP_STARTED, {"reasons": ["high_uncertainty"]}))
    with pytest.raises(EventLogCorrupted, match="already has 2 follow-ups"):
        log.fold()


def _probing(**fields) -> SessionState:
    """A forged PROBING state: unreachable through a legal log, which is the
    point - these guards are defence in depth behind FOLLOW_UP_STARTED's."""
    return SessionState(session_id=SESSION_ID, seq=10, state=InterviewState.PROBING, **fields)


def _follow_up_question(seq: int = 11) -> StoredEvent:
    return StoredEvent(
        seq=seq, type="QUESTION_ASKED", payload={"is_follow_up": True, "text": "?" * 12}
    )


def test_a_follow_up_question_beyond_the_item_cap_is_refused_by_apply():
    state = _probing(current_item_id="q1", follow_ups_by_item=MappingProxyType({"q1": 2}))
    with pytest.raises(IllegalTransition, match="already has 2 follow-ups"):
        apply(state, _follow_up_question())


def test_a_follow_up_question_beyond_the_session_cap_is_refused_by_apply():
    state = _probing(current_item_id="q7", follow_ups_asked=MAX_FOLLOW_UPS_PER_SESSION)
    with pytest.raises(IllegalTransition, match="already has 6 follow-ups"):
        apply(state, _follow_up_question())


def test_a_follow_up_question_with_no_item_is_refused_by_apply():
    with pytest.raises(IllegalTransition, match="no current item"):
        apply(_probing(current_item_id=None), _follow_up_question())


def test_the_session_cap_is_six_across_items():
    log = Log()
    for i in range(3):
        log.ask(f"q{i}") if i == 0 else log.select(f"q{i}")
        log.answer_and_grade().follow_up().answer_and_grade().follow_up().answer_and_grade()
    state = log.fold()
    assert state.follow_ups_asked == MAX_FOLLOW_UPS_PER_SESSION
    log.select("q9").answer_and_grade()
    assert not log.fold().follow_up_room
    log.steps.append((E.FOLLOW_UP_STARTED, {"reasons": ["ambiguous_band"]}))
    with pytest.raises(EventLogCorrupted, match="already has 6 follow-ups"):
        log.fold()


def test_theta_deltas_count_bank_items_only():
    log = Log().ask("q1").answer_and_grade(0.3).follow_up().answer_and_grade(0.01)
    log.select("q2").answer_and_grade(-0.2).select("q3").answer_and_grade(None)
    assert log.fold().theta_deltas == (0.3, -0.2)


def test_an_abandoned_follow_up_moves_to_selecting_and_counts_nothing():
    log = Log().ask("q1").answer_and_grade()
    log.steps.append(
        (E.FOLLOW_UP_STARTED, {"reasons": ["concept_absent"], "concept_key": "stampede"})
    )
    log.steps.append(
        (
            E.FOLLOW_UP_ABANDONED,
            {"error_code": "provider_auth", "attempts": 1, "retries_exhausted": False},
        )
    )
    state = log.fold()
    assert state.state is InterviewState.SELECTING
    assert state.follow_ups_asked == 0 and dict(state.follow_ups_by_item) == {}
    log.ask("q2")
    assert log.fold().current_item_id == "q2"


def test_abandon_is_only_legal_from_probing():
    log = Log().ask("q1").answer_and_grade()
    log.steps.append(
        (E.FOLLOW_UP_ABANDONED, {"error_code": "x_y", "attempts": 1, "retries_exhausted": True})
    )
    with pytest.raises(EventLogCorrupted, match="not allowed from GRADING"):
        log.fold()


def test_question_text_is_bounded():
    log = Log().ask("q1", text="x" * 2001)
    with pytest.raises(EventLogCorrupted):
        log.fold()


def test_version_2_state_round_trips():
    state = Log().ask("q1").answer_and_grade(0.25).follow_up("Why?").fold()
    again = SessionState.from_dict(json.loads(json.dumps(state.to_dict())))
    assert again == state
    assert state.to_dict()["v"] == STATE_SCHEMA_VERSION  # 2 on Day 23, 3 since Day 24 (priors)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d.update(follow_ups_by_item={"q1": -1}),
        lambda d: d.update(follow_ups_by_item=[]),
        lambda d: d.update(theta_deltas=["0.1"]),
        lambda d: d.update(current_item_id=7),
        lambda d: d.pop("current_question_text"),
    ],
)
def test_malformed_v2_fields_are_rejected(mutate):
    data = Log().ask("q1").answer_and_grade().follow_up().fold().to_dict()
    mutate(data)
    with pytest.raises(ValueError):
        SessionState.from_dict(data)
