"""Postgres access for the event log: read it, and append to it safely.

The append protocol, all inside the caller's transaction::

    SELECT ... FROM interview_sessions WHERE id = :sid AND user_id = :uid
        FOR UPDATE                                  -- 1. ownership + row lock
    SELECT seq, type, payload FROM interview_events
        WHERE session_id = :sid ORDER BY seq        -- 2. the authoritative log
    fold -> current state; apply(new event)         -- 3. FSM + invariants, pure
    INSERT INTO interview_events (seq = head + 1)   -- 4. the fact
    UPDATE interview_sessions SET state = ...       -- 5. the projection

**Why a row lock, and why it is enough.**  Two requests appending to one
session would otherwise both fold the same head ``seq``, both validate a
transition from the same state, and both try to write ``seq + 1``.  The lock on
the session row makes the second wait until the first commits; it then folds a
log that already contains the first event and validates against *that*.  The
lock is per session, so unrelated sessions never contend, and it is a plain
Postgres lock inside the transaction we need anyway - no distributed lock, no
new infrastructure.  ``unique(session_id, seq)`` stays as the backstop: if any
future code path appends without taking the lock, the second insert fails
instead of silently forking the history (it surfaces as :class:`SeqConflict`).

**Why the write path folds from Postgres, never from Redis.**  Validation of a
durable write must not depend on a cache.  A session's log is tens of events
read by one ordered range scan over the ``(session_id, seq)`` unique index, so
the cost is a few milliseconds - the trade plan section 11.2 argues for.

**Why ``interview_sessions.state`` is also written.**  It is a *projection* of
the log, updated in the same transaction as the event, so it can never
disagree with it after commit.  It exists for the partial "unfinished sessions"
index Day 2 created, and for listing sessions without folding each one.  The
log stays authoritative: :func:`app.interview.service.load_state` never reads
the column.

Nothing here commits.  The caller owns the transaction boundary - see
:func:`app.interview.service.append_event` - so that a later day can make, say,
inserting a ``turns`` row and appending ``ANSWER_SUBMITTED`` one atomic unit.
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping
from typing import Any

import sqlalchemy as sa
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.interview.events import StoredEvent, canonical_payload, validate_payload
from app.interview.fsm import EventType, InterviewState, next_state
from app.interview.state import SessionState, apply, fold
from app.models.interview import InterviewEvent, InterviewSession

_SEQ_CONSTRAINT = "uq_interview_events_session_id_seq"


class SessionNotFound(LookupError):
    """No session with that id belongs to that user.

    One error for "does not exist" and "is someone else's", deliberately: a
    distinguishable answer would let a caller probe for other users' session
    ids (plan section 14.1, broken access control).
    """

    def __init__(self, session_id: uuid.UUID) -> None:
        self.session_id = session_id
        super().__init__(f"session {session_id} not found")


class SeqConflict(RuntimeError):
    """The log moved on since the caller last looked.

    Raised when ``expected_seq`` does not match the head, or when the unique
    ``(session_id, seq)`` constraint rejects an insert.  The caller should
    reload the state and decide again; retrying blindly could apply a decision
    made against a state that no longer exists.
    """

    def __init__(self, session_id: uuid.UUID, expected: int | None, actual: int | None) -> None:
        self.session_id = session_id
        self.expected = expected
        self.actual = actual
        super().__init__(
            f"session {session_id}: expected head seq {expected}, but the log is at {actual}"
        )


async def lock_session(
    db: AsyncSession, *, session_id: uuid.UUID, user_id: uuid.UUID
) -> InterviewSession:
    """The owned session row, locked ``FOR UPDATE`` until the transaction ends."""
    row = await db.scalar(
        sa.select(InterviewSession)
        .where(InterviewSession.id == session_id, InterviewSession.user_id == user_id)
        .with_for_update()
        .execution_options(populate_existing=True)
    )
    if row is None:
        raise SessionNotFound(session_id)
    return row


async def read_events(db: AsyncSession, session_id: uuid.UUID) -> list[StoredEvent]:
    """Every event of one session, in ``seq`` order."""
    rows = await db.execute(
        sa.select(InterviewEvent.seq, InterviewEvent.type, InterviewEvent.payload)
        .where(InterviewEvent.session_id == session_id)
        .order_by(InterviewEvent.seq)
    )
    return [StoredEvent(seq=seq, type=type_, payload=payload) for seq, type_, payload in rows]


async def head_seq(db: AsyncSession, *, session_id: uuid.UUID, user_id: uuid.UUID) -> int | None:
    """``max(seq)`` of an owned session (``-1`` if it has no events), or ``None``
    when the session does not exist or is not this user's.

    The ownership filter is here, not in the cache, so a cached state can never
    be served to someone who does not own the session.
    """
    result = await db.execute(
        sa.select(sa.func.coalesce(sa.func.max(InterviewEvent.seq), -1))
        .select_from(InterviewSession)
        .outerjoin(InterviewEvent, InterviewEvent.session_id == InterviewSession.id)
        .where(InterviewSession.id == session_id, InterviewSession.user_id == user_id)
        .group_by(InterviewSession.id)
    )
    value = result.scalar_one_or_none()
    return int(value) if value is not None else None


async def stage_new_session(
    db: AsyncSession,
    *,
    user_id: uuid.UUID,
    blueprint: Mapping[str, Any],
    resume_id: uuid.UUID | None = None,
    jd_id: uuid.UUID | None = None,
) -> SessionState:
    """Insert a session row and its ``SESSION_CREATED`` event (seq 0).  Not committed.

    ``resume_id`` / ``jd_id`` (Day 24) must already be verified as the user's.
    """
    session = InterviewSession(
        user_id=user_id, blueprint=dict(blueprint), resume_id=resume_id, jd_id=jd_id
    )
    session.state = InterviewState.CREATED.value
    db.add(session)
    await db.flush()  # assigns the id
    payload = canonical_payload(validate_payload(EventType.SESSION_CREATED, {}))
    event = StoredEvent(seq=0, type=EventType.SESSION_CREATED.value, payload=payload)
    state = apply(SessionState.initial(session.id), event)
    db.add(_row(session.id, event))
    await db.flush()
    return state


async def stage_append(
    db: AsyncSession,
    *,
    session_id: uuid.UUID,
    user_id: uuid.UUID,
    event_type: EventType,
    payload: Mapping[str, Any],
    expected_seq: int | None = None,
) -> SessionState:
    """Validate and insert one event under the session lock.  Not committed.

    Raises, leaving nothing written by this call:

    * :class:`SessionNotFound` - not this user's session;
    * :class:`app.interview.state.EventLogCorrupted` - the existing log cannot
      be folded (refuse to extend a history we cannot read);
    * :class:`SeqConflict` - ``expected_seq`` is not the head;
    * :class:`app.interview.events.InvalidEventPayload` /
      :class:`app.interview.fsm.IllegalTransition` - the event itself is wrong.
    """
    if not isinstance(event_type, EventType):
        raise TypeError(f"event_type must be an EventType, got {event_type!r}")
    session = await lock_session(db, session_id=session_id, user_id=user_id)
    current = fold(session_id, await read_events(db, session_id))
    if expected_seq is not None and expected_seq != current.seq:
        raise SeqConflict(session_id, expected_seq, current.seq)

    # Legality first, payload second - the same order as `apply`. A completed
    # session must answer "terminal state" whatever the payload looks like.
    next_state(current.state, event_type)
    canonical = canonical_payload(validate_payload(event_type, payload))
    event = StoredEvent(seq=current.seq + 1, type=event_type.value, payload=canonical)
    new_state = apply(current, event)

    if new_state.state is None:  # pragma: no cover - apply never returns the empty state
        raise RuntimeError("apply returned the pre-creation state")

    db.add(_row(session_id, event))
    session.state = new_state.state.value
    if new_state.is_terminal:
        session.ended_at = sa.func.now()  # the DB clock, not the app clock
    try:
        await db.flush()
    except IntegrityError as exc:
        # Only reachable if something appended without the lock above.
        if _SEQ_CONSTRAINT in str(exc.orig):
            raise SeqConflict(session_id, current.seq, None) from exc
        raise
    return new_state


def _row(session_id: uuid.UUID, event: StoredEvent) -> InterviewEvent:
    return InterviewEvent(
        session_id=session_id, seq=event.seq, type=event.type, payload=dict(event.payload)
    )


__all__ = [
    "SeqConflict",
    "SessionNotFound",
    "head_seq",
    "lock_session",
    "read_events",
    "stage_append",
    "stage_new_session",
]
