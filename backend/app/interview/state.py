"""Session state as ``fold(events)`` - plan section 11.2.

::

    state_0     = SessionState.initial(session_id)        # seq -1, no state
    state_{n+1} = apply(state_n, event_n)                 # validated, pure
    state       = fold(session_id, events)                # = apply, left to right

``apply`` is the only function that changes a :class:`SessionState`, and it
does two kinds of checking:

* **The FSM** (:func:`app.interview.fsm.next_state`): is this event legal from
  this state at all?
* **Cross-event invariants** that a transition table cannot express, because
  they depend on what happened earlier in the log:

  - ``seq`` is exactly ``previous + 1``, starting at 0 - no gaps, no reordering;
  - a fresh question is a bank item that was never asked before in this session
    (plan section 8.3's "no repeats" hard constraint), and a follow-up has no
    bank id and is only asked from ``PROBING``;
  - a grade is for the turn that is actually being graded, and is recorded once;
  - ``GRADING`` cannot be left before the grade is recorded.

**Nothing here does I/O, reads a clock, or calls a model.**  The same events in
the same order produce an equal :class:`SessionState` every time, on any
machine - which is what replay, crash recovery and re-grading all rely on.

What the state holds is plan section 11.1's *working state*: FSM state, the
ability vector, asked ids and the current turn.  Budgets are not duplicated
here; the time and item budgets are on the blueprint (session row), and "items
used" is ``len(asked_question_ids)``.  Never store what you can derive.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any

from app.ability import AbilityState
from app.interview.events import (
    AnswerGraded,
    AnswerSubmitted,
    FollowUpStarted,
    GradingFailed,
    InvalidEventPayload,
    PreparationStarted,
    QuestionAsked,
    ReportingStarted,
    SelectionStarted,
    StoredEvent,
    validate_payload,
)
from app.interview.fsm import (
    EventType,
    IllegalTransition,
    InterviewState,
    next_state,
    parse_event_type,
)

#: Version of the *derived* state's shape.  Part of every cached copy; a cache
#: entry with another version is discarded and rebuilt from the log, so a deploy
#: that changes this dataclass can never read a stale shape out of Redis.
STATE_SCHEMA_VERSION = 3  # 2: Day 23 follow-up bookkeeping; 3: Day 24 priors

#: Plan section 8.5's caps.  Enforced HERE, as invariants of the log itself, as
#: well as by the policy: a FOLLOW_UP_STARTED or a follow-up QUESTION_ASKED that
#: would exceed either is an illegal event, so no code path - a retry, a
#: duplicate delivery, a race, a bug in the policy - can append one.
MAX_FOLLOW_UPS_PER_ITEM = 2
MAX_FOLLOW_UPS_PER_SESSION = 6


class EventLogCorrupted(ValueError):
    """The persisted log cannot be folded into a valid state.

    Raised by :func:`fold` - never "best effort".  Skipping a bad event and
    returning the rest would hand the caller a state that no real history
    produced, and every decision made from it would be wrong in a way nobody
    could see.
    """

    def __init__(self, session_id: uuid.UUID, seq: int | None, reason: str) -> None:
        self.session_id = session_id
        self.seq = seq
        self.reason = reason
        at = f" at seq {seq}" if seq is not None else ""
        super().__init__(f"event log for session {session_id} is corrupted{at}: {reason}")


@dataclass(frozen=True, slots=True)
class SessionState:
    """Everything replay knows about one session after ``seq``.

    ``state`` is ``None`` only for :meth:`initial` - the empty log, before
    ``SESSION_CREATED``.  Every persisted session has at least that event, so
    every state read from the database has a real FSM state.

    ``ability`` is keyed by subtopic, like :class:`app.selection.SelectionState`
    expects; it holds only subtopics an ``ANSWER_GRADED`` event has touched.

    Day 23 fields:

    ``current_item_id``
        The **bank item** the current question belongs to.  Equal to
        ``current_question_id`` for a fresh question; for a follow-up it is the
        item being probed (``current_question_id`` is ``None``).  This is what
        keeps a follow-up from becoming a new "original question" that would
        reset the per-item cap.
    ``follow_ups_by_item``
        Follow-ups actually asked, per bank item.
    ``current_question_text``
        The wording shown, from ``QUESTION_ASKED.text`` (``None`` for logs
        written before Day 23).
    ``theta_deltas``
        The theta step of every graded *bank-item* answer, in order - the
        stopping rule's "no new information" input.  Follow-up steps are
        excluded: a half-weight probe is not a new item (plan section 8.4 counts
        items).
    """

    session_id: uuid.UUID
    seq: int = -1
    state: InterviewState | None = None
    asked_question_ids: tuple[str, ...] = ()
    follow_ups_asked: int = 0
    current_question_id: str | None = None
    current_is_follow_up: bool = False
    current_turn_id: uuid.UUID | None = None
    current_graded: bool = False
    ability: Mapping[str, AbilityState] = field(default_factory=lambda: MappingProxyType({}))
    stop_reasons: tuple[str, ...] = ()
    current_item_id: str | None = None
    follow_ups_by_item: Mapping[str, int] = field(default_factory=lambda: MappingProxyType({}))
    current_question_text: str | None = None
    theta_deltas: tuple[float, ...] = ()
    #: Day 24: resume-seeded starting abilities from PREPARATION_STARTED.  Kept
    #: apart from ``ability`` (which holds only *observed* subtopics) so the
    #: stopping rule and reports never mistake a claim for a measurement.
    priors: Mapping[str, AbilityState] = field(default_factory=lambda: MappingProxyType({}))

    @classmethod
    def initial(cls, session_id: uuid.UUID) -> SessionState:
        return cls(session_id=session_id)

    @property
    def is_terminal(self) -> bool:
        return self.state is InterviewState.COMPLETED

    @property
    def questions_asked(self) -> int:
        """Bank items plus follow-ups."""
        return len(self.asked_question_ids) + self.follow_ups_asked

    def follow_ups_for(self, item_id: str | None) -> int:
        return 0 if item_id is None else self.follow_ups_by_item.get(item_id, 0)

    @property
    def follow_up_room(self) -> bool:
        """Whether one more follow-up on the current item fits both caps."""
        return (
            self.current_item_id is not None
            and self.follow_ups_for(self.current_item_id) < MAX_FOLLOW_UPS_PER_ITEM
            and self.follow_ups_asked < MAX_FOLLOW_UPS_PER_SESSION
        )

    # -- serialisation (the Redis cache) ------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """A JSON-ready dict.  Keys sorted wherever order is not meaningful."""
        return {
            "v": STATE_SCHEMA_VERSION,
            "session_id": str(self.session_id),
            "seq": self.seq,
            "state": self.state.value if self.state is not None else None,
            "asked_question_ids": list(self.asked_question_ids),
            "follow_ups_asked": self.follow_ups_asked,
            "current_question_id": self.current_question_id,
            "current_is_follow_up": self.current_is_follow_up,
            "current_turn_id": str(self.current_turn_id) if self.current_turn_id else None,
            "current_graded": self.current_graded,
            "ability": {
                key: {"theta": a.theta, "rd": a.rd, "n_observations": a.n_observations}
                for key, a in sorted(self.ability.items())
            },
            "stop_reasons": list(self.stop_reasons),
            "current_item_id": self.current_item_id,
            "follow_ups_by_item": dict(sorted(self.follow_ups_by_item.items())),
            "current_question_text": self.current_question_text,
            "theta_deltas": list(self.theta_deltas),
            "priors": {
                key: {"theta": a.theta, "rd": a.rd} for key, a in sorted(self.priors.items())
            },
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SessionState:
        """Inverse of :meth:`to_dict`.  ``ValueError`` on anything malformed,
        including a different :data:`STATE_SCHEMA_VERSION`."""
        try:
            if data["v"] != STATE_SCHEMA_VERSION:
                raise ValueError(f"state schema version {data['v']!r} != {STATE_SCHEMA_VERSION}")
            raw_state = data["state"]
            ability_raw = data["ability"]
            if not isinstance(ability_raw, Mapping):
                raise ValueError("ability must be an object")
            turn = data["current_turn_id"]
            seq = data["seq"]
            follow_ups = data["follow_ups_asked"]
            if type(seq) is not int or type(follow_ups) is not int:
                raise ValueError("seq and follow_ups_asked must be integers")
            asked = data["asked_question_ids"]
            reasons = data["stop_reasons"]
            if not isinstance(asked, list) or not all(isinstance(q, str) for q in asked):
                raise ValueError("asked_question_ids must be a list of strings")
            if not isinstance(reasons, list) or not all(isinstance(r, str) for r in reasons):
                raise ValueError("stop_reasons must be a list of strings")
            for flag in ("current_is_follow_up", "current_graded"):
                if not isinstance(data[flag], bool):
                    raise ValueError(f"{flag} must be a boolean")
            current_q = data["current_question_id"]
            if current_q is not None and not isinstance(current_q, str):
                raise ValueError("current_question_id must be a string or null")
            for name in ("current_item_id", "current_question_text"):
                if data[name] is not None and not isinstance(data[name], str):
                    raise ValueError(f"{name} must be a string or null")
            by_item = data["follow_ups_by_item"]
            if not isinstance(by_item, Mapping) or not all(
                isinstance(k, str) and type(v) is int and v >= 0 for k, v in by_item.items()
            ):
                raise ValueError("follow_ups_by_item must map strings to counts")
            priors_raw = data["priors"]
            if not isinstance(priors_raw, Mapping):
                raise ValueError("priors must be an object")
            deltas = data["theta_deltas"]
            if not isinstance(deltas, list) or not all(
                isinstance(d, int | float) and not isinstance(d, bool) for d in deltas
            ):
                raise ValueError("theta_deltas must be a list of numbers")
            return cls(
                session_id=uuid.UUID(data["session_id"]),
                seq=seq,
                state=InterviewState(raw_state) if raw_state is not None else None,
                asked_question_ids=tuple(asked),
                follow_ups_asked=follow_ups,
                current_question_id=current_q,
                current_is_follow_up=data["current_is_follow_up"],
                current_turn_id=uuid.UUID(turn) if turn is not None else None,
                current_graded=data["current_graded"],
                ability=MappingProxyType(
                    {
                        str(key): AbilityState(
                            theta=float(v["theta"]),
                            rd=float(v["rd"]),
                            n_observations=int(v["n_observations"]),
                        )
                        for key, v in sorted(ability_raw.items())
                    }
                ),
                stop_reasons=tuple(reasons),
                current_item_id=data["current_item_id"],
                follow_ups_by_item=MappingProxyType(dict(sorted(by_item.items()))),
                current_question_text=data["current_question_text"],
                theta_deltas=tuple(float(d) for d in deltas),
                priors=MappingProxyType(
                    {
                        str(k): AbilityState(theta=float(v["theta"]), rd=float(v["rd"]))
                        for k, v in sorted(priors_raw.items())
                    }
                ),
            )
        except (KeyError, TypeError, AttributeError) as exc:
            raise ValueError(f"malformed session state: {exc!r}") from exc


# ---------------------------------------------------------------------------
# apply / fold
# ---------------------------------------------------------------------------


def apply(state: SessionState, event: StoredEvent) -> SessionState:
    """One step of the fold.  Raises :class:`IllegalTransition` or
    :class:`InvalidEventPayload` (a request problem when appending) or
    ``ValueError`` for a malformed event; :func:`fold` converts all of them
    into :class:`EventLogCorrupted`.
    """
    if type(event.seq) is not int:
        raise ValueError(f"seq must be an integer, got {event.seq!r}")
    if event.seq != state.seq + 1:
        raise ValueError(f"expected seq {state.seq + 1}, got {event.seq}")

    event_type = parse_event_type(event.type)
    target = next_state(state.state, event_type)
    payload = validate_payload(event_type, event.payload)
    advanced = replace(state, seq=event.seq, state=target)

    if isinstance(payload, QuestionAsked):
        return _question_asked(state, advanced, payload)
    if isinstance(payload, AnswerSubmitted):
        return replace(advanced, current_turn_id=payload.turn_id, current_graded=False)
    if isinstance(payload, AnswerGraded):
        return _answer_graded(state, advanced, payload)
    if isinstance(payload, GradingFailed):
        _require_current_ungraded(state, event_type, payload.turn_id)
        return advanced
    if isinstance(payload, FollowUpStarted):
        _require_graded(state, event_type)
        _require_follow_up_room(state, event_type)
        return advanced
    if isinstance(payload, SelectionStarted):
        _require_graded(state, event_type)
        return advanced
    if isinstance(payload, ReportingStarted):
        # From SELECTING the grade was already required on the way in.
        if state.state is InterviewState.GRADING:
            _require_graded(state, event_type)
        return replace(advanced, stop_reasons=tuple(payload.reasons))
    if isinstance(payload, PreparationStarted):
        return replace(
            advanced,
            priors=MappingProxyType(
                {
                    key: AbilityState(theta=p.theta, rd=p.rd)
                    for key, p in sorted(payload.priors.items())
                }
            ),
        )
    # SESSION_CREATED, SESSION_COMPLETED: the transition is the whole effect.
    return advanced


def _question_asked(
    before: SessionState, advanced: SessionState, payload: QuestionAsked
) -> SessionState:
    from_probing = before.state is InterviewState.PROBING
    if payload.is_follow_up != from_probing:
        raise IllegalTransition(
            before.state,
            EventType.QUESTION_ASKED.value,
            "a follow-up is asked from PROBING, and only from PROBING",
        )
    if payload.is_follow_up:
        if payload.question_id is not None:
            raise InvalidEventPayload(
                EventType.QUESTION_ASKED.value, "a follow-up has no bank question_id"
            )
        _require_follow_up_room(before, EventType.QUESTION_ASKED)
        item = before.current_item_id
        assert item is not None  # noqa: S101 - _require_follow_up_room checked it
        counts = dict(before.follow_ups_by_item)
        counts[item] = counts.get(item, 0) + 1
        return replace(
            advanced,
            follow_ups_asked=before.follow_ups_asked + 1,
            follow_ups_by_item=MappingProxyType(dict(sorted(counts.items()))),
            current_question_id=None,
            current_is_follow_up=True,
            current_question_text=payload.text,
            current_turn_id=None,
            current_graded=False,
        )
    if payload.question_id is None:
        raise InvalidEventPayload(
            EventType.QUESTION_ASKED.value, "a fresh question needs its bank question_id"
        )
    if payload.question_id in before.asked_question_ids:
        raise IllegalTransition(
            before.state,
            EventType.QUESTION_ASKED.value,
            f"question {payload.question_id!r} was already asked in this session",
        )
    return replace(
        advanced,
        asked_question_ids=(*before.asked_question_ids, payload.question_id),
        current_question_id=payload.question_id,
        current_item_id=payload.question_id,
        current_question_text=payload.text,
        current_is_follow_up=False,
        current_turn_id=None,
        current_graded=False,
    )


def _require_current_ungraded(
    before: SessionState, event_type: EventType, turn_id: uuid.UUID
) -> None:
    if before.current_graded:
        raise IllegalTransition(before.state, event_type.value, "this turn is already graded")
    if turn_id != before.current_turn_id:
        raise IllegalTransition(
            before.state,
            event_type.value,
            f"event is for turn {turn_id}, but turn {before.current_turn_id} is being graded",
        )


def _answer_graded(
    before: SessionState, advanced: SessionState, payload: AnswerGraded
) -> SessionState:
    _require_current_ungraded(before, EventType.ANSWER_GRADED, payload.turn_id)
    ability = dict(before.ability)
    if payload.ability is not None:
        snap = payload.ability
        ability[snap.subtopic_key] = AbilityState(
            theta=snap.theta, rd=snap.rd, n_observations=snap.n_observations
        )
    deltas = before.theta_deltas
    if payload.delta_theta is not None and not before.current_is_follow_up:
        deltas = (*deltas, payload.delta_theta)
    return replace(
        advanced,
        current_graded=True,
        ability=MappingProxyType(dict(sorted(ability.items()))),
        theta_deltas=deltas,
    )


def _require_follow_up_room(before: SessionState, event_type: EventType) -> None:
    if before.current_item_id is None:
        raise IllegalTransition(
            before.state, event_type.value, "there is no current item to follow up on"
        )
    if before.follow_ups_for(before.current_item_id) >= MAX_FOLLOW_UPS_PER_ITEM:
        raise IllegalTransition(
            before.state,
            event_type.value,
            f"item {before.current_item_id!r} already has " f"{MAX_FOLLOW_UPS_PER_ITEM} follow-ups",
        )
    if before.follow_ups_asked >= MAX_FOLLOW_UPS_PER_SESSION:
        raise IllegalTransition(
            before.state,
            event_type.value,
            f"the session already has {MAX_FOLLOW_UPS_PER_SESSION} follow-ups",
        )


def _require_graded(before: SessionState, event_type: EventType) -> None:
    if not before.current_graded:
        raise IllegalTransition(
            before.state, event_type.value, "the current answer has not been graded yet"
        )


def fold(session_id: uuid.UUID, events: Iterable[StoredEvent]) -> SessionState:
    """Replay ``events`` (in ``seq`` order) from the initial state.

    Raises :class:`EventLogCorrupted` on the first event that cannot be applied
    - a gap or duplicate in ``seq``, an unknown type, an invalid payload, an
    illegal transition - naming the ``seq`` it happened at.  An empty log is
    also corrupted: every session is created with ``SESSION_CREATED``.
    """
    state = SessionState.initial(session_id)
    for event in events:
        try:
            state = apply(state, event)
        except (IllegalTransition, InvalidEventPayload, ValueError, TypeError) as exc:
            raise EventLogCorrupted(session_id, _safe_seq(event), str(exc)) from exc
    if state.state is None:
        raise EventLogCorrupted(session_id, None, "the log is empty")
    return state


def _safe_seq(event: object) -> int | None:
    seq = getattr(event, "seq", None)
    return seq if type(seq) is int else None


__all__ = [
    "MAX_FOLLOW_UPS_PER_ITEM",
    "MAX_FOLLOW_UPS_PER_SESSION",
    "STATE_SCHEMA_VERSION",
    "EventLogCorrupted",
    "SessionState",
    "apply",
    "fold",
]
