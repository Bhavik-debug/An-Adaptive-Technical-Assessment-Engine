"""Server-sent events over the event log: state transitions, never tokens.

Plan §3, Day 22: "SSE events carry **state transitions, not tokens**."  Every
SSE message is derived from one durable ``interview_events`` row:

    id: <seq>                       the row's per-session seq - stable forever
    event: <name>                   from SSE_NAMES below
    data: {"seq":..,"state":..,..}  a WHITELISTED projection of the payload
    <blank line>

**Why the stream is a view of the log and not a separate channel.**  Because
the log is durable and ordered, ``id`` can be the ``seq``, and reconnection is
exact: a client reconnecting with ``Last-Event-ID: 7`` receives every event with
``seq > 7`` in order, nothing twice, nothing missed - whatever happened while it
was away.  A pub/sub channel would lose whatever was published during the gap.

**What is never sent.**  Payloads are projected field by field, not forwarded:
no answer text (events do not carry it anyway), no judgment, no evidence spans,
no prompts, no provider messages, no stop/follow-up reasons.  The grade is one
number; a failure is one machine code.

**How it waits.**  It polls: one indexed ``max(seq)`` query per interval
(``SSE_POLL_INTERVAL_S``), each in its own short-lived database session, so an
open stream holds no connection while idle.  A comment line is sent every
``SSE_KEEPALIVE_S`` without events so proxies do not drop the connection.  The
stream ends after ``session_complete``, when the client disconnects, or if the
session disappears.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.interview.events import StoredEvent, validate_payload
from app.interview.fsm import EventType, InterviewState
from app.interview.state import SessionState, apply
from app.interview.store import head_seq, read_events

log = logging.getLogger(__name__)

#: Plan §13.8's event names.  ``grading_failed`` is Day 22's addition (a client
#: must be able to stop waiting).  Events with no entry are sent as
#: ``state_changed``.  ``follow_up`` and ``follow_up_abandoned`` are produced
#: from Day 23; ``session_complete`` from Day 25 (carries ``report_id``).
SSE_NAMES: Mapping[EventType, str] = {
    EventType.ANSWER_SUBMITTED: "grading_started",
    EventType.ANSWER_GRADED: "grade_ready",
    EventType.GRADING_FAILED: "grading_failed",
    EventType.FOLLOW_UP_STARTED: "follow_up",
    EventType.FOLLOW_UP_ABANDONED: "follow_up_abandoned",
    EventType.QUESTION_ASKED: "next_question",
    EventType.SESSION_COMPLETED: "session_complete",
}
STATE_CHANGED = "state_changed"
ERROR = "error"

#: The only payload fields that may leave the server, per event type.
_FIELDS: Mapping[EventType, tuple[str, ...]] = {
    EventType.ANSWER_SUBMITTED: ("turn_id",),
    EventType.ANSWER_GRADED: ("turn_id", "score"),
    EventType.GRADING_FAILED: ("turn_id", "error_code", "retries_exhausted"),
    # Day 23: the wording shown. Ours, not the candidate's - safe to send.
    EventType.QUESTION_ASKED: ("question_id", "is_follow_up", "text"),
    EventType.FOLLOW_UP_ABANDONED: ("error_code", "retries_exhausted"),
    # Day 25: the id of the persisted report - fetch it from GET .../report.
    EventType.SESSION_COMPLETED: ("report_id",),
    # FOLLOW_UP_STARTED deliberately sends nothing beyond seq/state: its reason
    # and concept_key would tell the candidate what the probe is looking for.
}

#: A message larger than this is a bug.  Sized for the largest field, a question
#: ``text`` of up to 2000 characters (``events.MAX_QUESTION_TEXT``), which JSON
#: escaping and multi-byte characters can roughly double.
MAX_DATA_BYTES = 8192


def frame(*, event: str, data: Mapping[str, Any], event_id: int | None = None) -> str:
    """One SSE message.  ``data`` is compact JSON, so it is one line."""
    body = json.dumps(data, separators=(",", ":"), sort_keys=True)
    if len(body.encode()) > MAX_DATA_BYTES:
        raise ValueError(f"SSE data for {event!r} is {len(body)} bytes")
    lines = [] if event_id is None else [f"id: {event_id}"]
    lines += [f"event: {event}", f"data: {body}"]
    return "\n".join(lines) + "\n\n"


def keepalive() -> str:
    return ": keepalive\n\n"


def to_message(event: StoredEvent, state_after: SessionState) -> str:
    """The SSE message for one already-applied log event."""
    event_type = EventType(event.type)
    data: dict[str, Any] = {"seq": event.seq, "state": str(state_after.state)}
    fields = _FIELDS.get(event_type, ())
    if fields:
        # From the validated model, not the stored dict: payloads are stored
        # without unset optionals (events.canonical_payload), and a client must
        # see the same shape either way - e.g. ``question_id: null`` on a
        # follow-up, not a missing key.
        full = validate_payload(event_type, event.payload).model_dump(mode="json")
        for field in fields:
            data[field] = full.get(field)
    if event_type not in SSE_NAMES:
        data["event_type"] = event_type.value
    return frame(event=SSE_NAMES.get(event_type, STATE_CHANGED), data=data, event_id=event.seq)


def parse_last_event_id(raw: str | None) -> int:
    """``Last-Event-ID`` -> the last seq the client has.  -1 = from the start."""
    if raw is None or raw.strip() == "":
        return -1
    value = int(raw.strip())  # ValueError for garbage; the route maps it to 400
    if value < -1:
        raise ValueError("Last-Event-ID must be >= -1")
    return value


async def event_stream(
    maker: async_sessionmaker[AsyncSession],
    *,
    session_id: uuid.UUID,
    user_id: uuid.UUID,
    last_event_id: int,
    poll_interval_s: float,
    keepalive_s: float,
    is_disconnected: Callable[[], Awaitable[bool]],
) -> AsyncIterator[str]:
    """Yield SSE messages for every event after ``last_event_id``, then follow.

    Ownership was checked by the route before the response started; it is
    re-checked on every poll by ``head_seq``, which filters on ``user_id``, so
    a deleted session ends the stream rather than leaking anything.
    """
    state = SessionState.initial(session_id)
    last_sent_at = time.monotonic()
    while True:
        if await is_disconnected():
            return
        async with maker() as db:
            head = await head_seq(db, session_id=session_id, user_id=user_id)
            if head is None:
                return
            new: list[StoredEvent] = []
            if head > state.seq:
                new = [e for e in await read_events(db, session_id) if e.seq > state.seq]
        for event in new:
            try:
                state = apply(state, event)
            except Exception as exc:  # noqa: BLE001 - reported to the client, not hidden
                log.error(
                    "event log unreadable while streaming",
                    extra={"session_id": str(session_id), "error": repr(exc)},
                )
                yield frame(event=ERROR, data={"code": "session_state_unavailable"})
                return
            if event.seq > last_event_id:
                yield to_message(event, state)
                last_sent_at = time.monotonic()
        if state.state is InterviewState.COMPLETED:
            return
        if time.monotonic() - last_sent_at >= keepalive_s:
            yield keepalive()
            last_sent_at = time.monotonic()
        await asyncio.sleep(poll_interval_s)


__all__ = [
    "ERROR",
    "MAX_DATA_BYTES",
    "SSE_NAMES",
    "STATE_CHANGED",
    "event_stream",
    "frame",
    "keepalive",
    "parse_last_event_id",
    "to_message",
]
