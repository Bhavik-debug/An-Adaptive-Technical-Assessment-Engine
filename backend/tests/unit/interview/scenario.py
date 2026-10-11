"""A complete, legal interview as an event list - the fixture most tests vary.

Every branch of the FSM appears once: a fresh question, a graded answer, a
follow-up (PROBING), a selection of a second item (SELECTING), the stop rule
(REPORTING) and completion. Deliberately small: three questions are enough to
exercise every edge, and a test that needs to see a failure should be able to
read the whole history it is failing on.
"""

from __future__ import annotations

import uuid
from typing import Any

from app.interview import EventType, StoredEvent

SESSION_ID = uuid.UUID("00000000-0000-4000-8000-000000000021")
TURN_1 = uuid.UUID("00000000-0000-4000-8000-0000000000a1")
TURN_2 = uuid.UUID("00000000-0000-4000-8000-0000000000a2")
TURN_3 = uuid.UUID("00000000-0000-4000-8000-0000000000a3")

#: (event type, payload) - no seq; positions are assigned by `as_stored`.
HAPPY_PATH: list[tuple[EventType, dict[str, Any]]] = [
    (EventType.SESSION_CREATED, {}),
    (EventType.PREPARATION_STARTED, {}),
    (EventType.QUESTION_ASKED, {"question_id": "dsa-001"}),
    (EventType.ANSWER_SUBMITTED, {"turn_id": str(TURN_1)}),
    (
        EventType.ANSWER_GRADED,
        {
            "turn_id": str(TURN_1),
            "score": 0.4,
            "ability": {"subtopic_key": "arrays", "theta": -0.25, "rd": 1.1, "n_observations": 1},
        },
    ),
    (EventType.FOLLOW_UP_STARTED, {"reasons": ["concept_absent"]}),
    (EventType.QUESTION_ASKED, {"is_follow_up": True}),
    (EventType.ANSWER_SUBMITTED, {"turn_id": str(TURN_2)}),
    (
        EventType.ANSWER_GRADED,
        {
            "turn_id": str(TURN_2),
            "score": 0.7,
            "ability": {"subtopic_key": "arrays", "theta": -0.05, "rd": 0.95, "n_observations": 2},
        },
    ),
    (EventType.SELECTION_STARTED, {}),
    (EventType.QUESTION_ASKED, {"question_id": "db-004"}),
    (EventType.ANSWER_SUBMITTED, {"turn_id": str(TURN_3)}),
    (
        EventType.ANSWER_GRADED,
        {
            "turn_id": str(TURN_3),
            "score": 1.0,
            "ability": {"subtopic_key": "indexing", "theta": 0.3, "rd": 1.0, "n_observations": 1},
        },
    ),
    (EventType.REPORTING_STARTED, {"reasons": ["item_budget_reached"]}),
    (EventType.SESSION_COMPLETED, {}),
]


def as_stored(
    steps: list[tuple[EventType | str, dict[str, Any]]], *, start: int = 0
) -> list[StoredEvent]:
    return [
        StoredEvent(seq=start + i, type=str(t), payload=payload)
        for i, (t, payload) in enumerate(steps)
    ]


def prefix(n: int) -> list[StoredEvent]:
    """The first ``n`` events of the happy path."""
    return as_stored(list(HAPPY_PATH[:n]))
