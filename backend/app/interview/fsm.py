"""The interview finite state machine - plan sections 3 (Day 21) and 13.4.

::

    CREATED -> PREPARING -> ASKING -> GRADING -> { PROBING   -> ASKING
                                                 | SELECTING -> ASKING }
                                          |            |
                                          +- stop -----+-> REPORTING -> COMPLETED

**Every legal move is one row of** :data:`TRANSITIONS`.  A transition is keyed
by ``(current state, event type)`` and yields the next state; anything not in
the table is illegal.  There is no other place in the codebase that decides
whether a session may move, and no LLM is ever consulted: the control flow is a
small closed graph with hard reliability requirements, which is exactly what
code is for (plan section 13.4, ADR-0003's argument).

How ``PROBING/SELECTING`` maps onto states
------------------------------------------

Plan section 3's Day-21 row writes ``PROBING/SELECTING`` as one stage; section
13.4's diagram resolves it into two sibling states that both lead back to
``ASKING``.  They are kept as two states because they are different decisions
with different successors' payloads: ``PROBING`` means the next question is a
follow-up on the *same* item (Day 23 decides *whether*), ``SELECTING`` means a
new bank item is being chosen (Days 12-15 decide *which*).  A single merged
state would push that distinction into payload flags, which is how an FSM
quietly stops being one.

Transitions beyond the diagram, and why each exists
---------------------------------------------------

* ``GRADING --ANSWER_GRADED--> GRADING`` - a *self-loop*, not a move.  The grade
  is recorded as its own event inside ``GRADING`` (plan section 11.2: every
  grade is in the log, which is what makes deterministic re-grading possible),
  and :mod:`app.interview.state` refuses to leave ``GRADING`` until it has
  happened.  The FSM itself only knows the move is legal in that state.
* ``GRADING --GRADING_FAILED--> GRADING`` - another self-loop (Day 22).  A
  grade that could not be produced after bounded retries is recorded as a fact
  in the log, so replay and the SSE stream both see it, *without* a new state:
  the session stays in ``GRADING`` with the turn ungraded, which is exactly
  true, and an operator can re-queue the turn (``ANSWER_GRADED`` is still legal
  afterwards).  A terminal ``FAILED`` state was rejected - it would end an
  interview over what may be a provider outage.
* ``PROBING --FOLLOW_UP_ABANDONED--> SELECTING`` (Day 23).  A follow-up was
  decided but its wording could not be generated (a permanent model error, or
  retries used up).  The follow-up is abandoned *on the record* and the
  interview continues with normal selection - never a fabricated question, and
  never a silent jump: the event names the error.  The alternative, a
  self-loop that leaves the session in ``PROBING`` until an operator acts,
  would block a candidate's interview on extra evidence the grade did not need.
* ``SELECTING --REPORTING_STARTED--> REPORTING`` - the item pool can be empty
  even after the stop rule said "continue" (plan section 8.7: "never crash,
  never repeat").  Without this edge a session whose bank is exhausted would be
  stuck in ``SELECTING`` forever.

Deliberately **not** here: an ``ABANDONED`` or ``FAILED`` state, an early-end
edge from ``ASKING`` (the ``/end`` endpoint of plan section 13.8), and any
retry edge for a failed grade.  None is specified for Day 21; each is a
decision for the day that owns its trigger, and adding one is a one-row change
to :data:`TRANSITIONS`.

Terminal behaviour
------------------

``COMPLETED`` has no outgoing transition.  Every event offered to a completed
session is rejected with :class:`IllegalTransition`, including a second
``SESSION_COMPLETED`` - a terminal state that accepted "complete again" would
make "is this session over?" depend on how many times someone asked.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from types import MappingProxyType


class InterviewState(StrEnum):
    """The eight states.  Values are the strings stored in
    ``interview_sessions.state`` - ``CREATED`` and ``COMPLETED`` already match
    the constants Day 2 put in :mod:`app.models.interview`."""

    CREATED = "CREATED"
    PREPARING = "PREPARING"
    ASKING = "ASKING"
    GRADING = "GRADING"
    PROBING = "PROBING"
    SELECTING = "SELECTING"
    REPORTING = "REPORTING"
    COMPLETED = "COMPLETED"


class EventType(StrEnum):
    """Every event the log may contain.  Stored in ``interview_events.type``."""

    SESSION_CREATED = "SESSION_CREATED"
    PREPARATION_STARTED = "PREPARATION_STARTED"
    QUESTION_ASKED = "QUESTION_ASKED"
    ANSWER_SUBMITTED = "ANSWER_SUBMITTED"
    ANSWER_GRADED = "ANSWER_GRADED"
    GRADING_FAILED = "GRADING_FAILED"
    FOLLOW_UP_STARTED = "FOLLOW_UP_STARTED"
    FOLLOW_UP_ABANDONED = "FOLLOW_UP_ABANDONED"
    SELECTION_STARTED = "SELECTION_STARTED"
    REPORTING_STARTED = "REPORTING_STARTED"
    SESSION_COMPLETED = "SESSION_COMPLETED"


TERMINAL_STATES: frozenset[InterviewState] = frozenset({InterviewState.COMPLETED})

#: ``(state, event) -> next state``.  ``None`` as the state is the empty log:
#: the only thing that may happen to a session with no history is its creation.
#: Read-only, so no caller can add an edge at runtime.
TRANSITIONS: Mapping[tuple[InterviewState | None, EventType], InterviewState] = MappingProxyType(
    {
        (None, EventType.SESSION_CREATED): InterviewState.CREATED,
        (InterviewState.CREATED, EventType.PREPARATION_STARTED): InterviewState.PREPARING,
        (InterviewState.PREPARING, EventType.QUESTION_ASKED): InterviewState.ASKING,
        (InterviewState.ASKING, EventType.ANSWER_SUBMITTED): InterviewState.GRADING,
        (InterviewState.GRADING, EventType.ANSWER_GRADED): InterviewState.GRADING,
        (InterviewState.GRADING, EventType.GRADING_FAILED): InterviewState.GRADING,
        (InterviewState.GRADING, EventType.FOLLOW_UP_STARTED): InterviewState.PROBING,
        (InterviewState.GRADING, EventType.SELECTION_STARTED): InterviewState.SELECTING,
        (InterviewState.GRADING, EventType.REPORTING_STARTED): InterviewState.REPORTING,
        (InterviewState.PROBING, EventType.QUESTION_ASKED): InterviewState.ASKING,
        (InterviewState.PROBING, EventType.FOLLOW_UP_ABANDONED): InterviewState.SELECTING,
        (InterviewState.SELECTING, EventType.QUESTION_ASKED): InterviewState.ASKING,
        (InterviewState.SELECTING, EventType.REPORTING_STARTED): InterviewState.REPORTING,
        (InterviewState.REPORTING, EventType.SESSION_COMPLETED): InterviewState.COMPLETED,
    }
)


class IllegalTransition(ValueError):
    """The event is well-formed but not allowed from the session's state.

    Carries both ends so a caller can say *why* (``"ANSWER_SUBMITTED from
    COMPLETED"``) without parsing a message.
    """

    def __init__(self, state: InterviewState | None, event_type: str, detail: str = "") -> None:
        self.state = state
        self.event_type = event_type
        where = state.value if state is not None else "<no events>"
        message = f"{event_type} is not allowed from {where}"
        if state in TERMINAL_STATES:
            message += " (terminal state)"
        if detail:
            message += f": {detail}"
        super().__init__(message)


def parse_state(raw: object) -> InterviewState:
    """A stored state string -> :class:`InterviewState`, or ``ValueError``."""
    if not isinstance(raw, str):
        raise ValueError(f"state must be a string, got {type(raw).__name__}")
    try:
        return InterviewState(raw)
    except ValueError:
        raise ValueError(f"unknown interview state {raw!r}") from None


def parse_event_type(raw: object) -> EventType:
    """A stored event-type string -> :class:`EventType`, or ``ValueError``."""
    if not isinstance(raw, str):
        raise ValueError(f"event type must be a string, got {type(raw).__name__}")
    try:
        return EventType(raw)
    except ValueError:
        raise ValueError(f"unknown event type {raw!r}") from None


def next_state(state: InterviewState | None, event_type: EventType) -> InterviewState:
    """The state ``event_type`` moves ``state`` to, or :class:`IllegalTransition`.

    The one function every caller goes through - appending, replaying and
    testing.  Pure: no I/O, no clock, no randomness.
    """
    if state is not None and not isinstance(state, InterviewState):
        raise TypeError(f"state must be an InterviewState or None, got {state!r}")
    if not isinstance(event_type, EventType):
        raise TypeError(f"event_type must be an EventType, got {event_type!r}")
    target = TRANSITIONS.get((state, event_type))
    if target is None:
        raise IllegalTransition(state, event_type.value)
    return target


def allowed_events(state: InterviewState | None) -> frozenset[EventType]:
    """Every event the FSM accepts from ``state``.  Empty for a terminal state."""
    return frozenset(event for (source, event) in TRANSITIONS if source == state)


def successors(state: InterviewState) -> frozenset[InterviewState]:
    """The states reachable from ``state`` in one move (self-loops included)."""
    return frozenset(target for (source, _), target in TRANSITIONS.items() if source == state)


def is_terminal(state: InterviewState) -> bool:
    return state in TERMINAL_STATES


__all__ = [
    "TERMINAL_STATES",
    "TRANSITIONS",
    "EventType",
    "IllegalTransition",
    "InterviewState",
    "allowed_events",
    "is_terminal",
    "next_state",
    "parse_event_type",
    "parse_state",
    "successors",
]
