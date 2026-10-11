"""SSE framing and the whitelist projection - pure, no I/O."""

from __future__ import annotations

import json

import pytest

from app.interview import EventType, fold
from app.interview.stream import (
    MAX_DATA_BYTES,
    SSE_NAMES,
    frame,
    keepalive,
    parse_last_event_id,
    to_message,
)
from tests.unit.interview.scenario import HAPPY_PATH, SESSION_ID, as_stored


def _parse(message: str) -> dict[str, str]:
    assert message.endswith("\n\n"), "an SSE message ends with a blank line"
    fields = {}
    for line in message[:-2].split("\n"):
        key, _, value = line.partition(": ")
        fields[key] = value
    return fields


def _messages() -> list[dict[str, str]]:
    events = as_stored(HAPPY_PATH)
    out = []
    for i, event in enumerate(events):
        state = fold(SESSION_ID, events[: i + 1])
        out.append(_parse(to_message(event, state)))
    return out


def test_every_log_event_becomes_one_well_formed_message():
    messages = _messages()
    assert [int(m["id"]) for m in messages] == list(range(len(HAPPY_PATH)))
    for m in messages:
        assert set(m) == {"id", "event", "data"}
        json.loads(m["data"])  # one line of valid JSON


def test_documented_names():
    names = [m["event"] for m in _messages()]
    assert names == [
        "state_changed",  # SESSION_CREATED
        "state_changed",  # PREPARATION_STARTED
        "next_question",
        "grading_started",
        "grade_ready",
        "follow_up",
        "next_question",
        "grading_started",
        "grade_ready",
        "state_changed",  # SELECTION_STARTED
        "next_question",
        "grading_started",
        "grade_ready",
        "state_changed",  # REPORTING_STARTED
        "session_complete",
    ]
    assert SSE_NAMES[EventType.GRADING_FAILED] == "grading_failed"


def test_data_is_a_whitelist_projection():
    """Ability snapshots, stop reasons and follow-up reasons never leave."""
    datas = [json.loads(m["data"]) for m in _messages()]
    # Day 23 adds "text": the question wording we showed (never candidate text).
    allowed = {
        "seq",
        "state",
        "turn_id",
        "score",
        "question_id",
        "is_follow_up",
        "text",
        "event_type",
        "report_id",  # Day 25: session_complete links the persisted report
    }
    for d in datas:
        assert set(d) <= allowed, d
    graded = datas[4]
    assert set(graded) == {"seq", "state", "turn_id", "score"}
    assert graded["state"] == "GRADING"
    assert "reasons" not in datas[5] and "reasons" not in datas[13]
    assert set(datas[5]) == {"seq", "state"}  # follow_up: no reason, no concept key
    # A follow-up's next_question has the same shape as any other: null id.
    assert datas[6]["question_id"] is None and datas[6]["is_follow_up"] is True
    assert datas[0] == {"seq": 0, "state": "CREATED", "event_type": "SESSION_CREATED"}


def test_grading_failed_projection():
    steps = [
        *HAPPY_PATH[:4],
        (
            EventType.GRADING_FAILED,
            {
                "turn_id": HAPPY_PATH[3][1]["turn_id"],
                "error_code": "provider_timeout",
                "attempts": 3,
                "retries_exhausted": True,
            },
        ),
    ]
    events = as_stored(list(steps))
    state = fold(SESSION_ID, events)
    m = _parse(to_message(events[-1], state))
    assert m["event"] == "grading_failed"
    assert json.loads(m["data"]) == {
        "seq": 4,
        "state": "GRADING",
        "turn_id": HAPPY_PATH[3][1]["turn_id"],
        "error_code": "provider_timeout",
        "retries_exhausted": True,
    }


def test_frame_rejects_oversized_data():
    with pytest.raises(ValueError):
        frame(event="x", data={"blob": "a" * MAX_DATA_BYTES})


def test_keepalive_is_a_comment():
    assert keepalive().startswith(":") and keepalive().endswith("\n\n")


@pytest.mark.parametrize(("raw", "want"), [(None, -1), ("", -1), (" 7 ", 7), ("-1", -1), ("0", 0)])
def test_last_event_id_parsing(raw, want):
    assert parse_last_event_id(raw) == want


@pytest.mark.parametrize("raw", ["abc", "1.5", "-2", "7; drop"])
def test_bad_last_event_ids_are_rejected(raw):
    with pytest.raises(ValueError):
        parse_last_event_id(raw)
