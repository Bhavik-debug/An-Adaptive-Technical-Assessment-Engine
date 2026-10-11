"""Submitting an answer: one logical turn per ``turn_id``, atomically with its event.

Plan §3, Day 22: "a client-supplied ``turn_id``, ``INSERT ... ON CONFLICT DO
NOTHING`` for idempotency".  The transaction, in order::

    SELECT ... FROM interview_sessions ... FOR UPDATE     ownership + per-session lock
    SELECT ... FROM turns WHERE (session_id, turn_id)     already submitted?
        yes, same answer      -> return it, write nothing         (a retry)
        yes, other answer     -> TurnIdConflict, write nothing    (key misuse)
    stage ANSWER_SUBMITTED                                the FSM decides legality
    INSERT INTO turns ... ON CONFLICT DO NOTHING          the turn, status pending
    COMMIT; then cache the new state                      (Day 21's commit_and_cache)

**The lock does the serialising; the unique constraint is the backstop.**  Two
simultaneous requests with one ``turn_id`` queue on the session row; the second
then finds the first's committed turn and returns it.  ``ON CONFLICT DO NOTHING``
is there for any path that ever inserts without the lock: it returns no row
instead of raising, the transaction is rolled back (taking the staged event with
it), and the caller re-reads and returns the winner's turn.

**The duplicate check comes before the FSM check**, deliberately: a retry
arriving after the original moved the session to ``GRADING`` must get the
original turn back, not "ANSWER_SUBMITTED is not allowed from GRADING".

**What is compared for a duplicate.**  The stored answer text against the
submitted one, exactly.  No normalisation: if the client sends different bytes
it sent a different submission, and guessing which differences "don't count" is
how an idempotency key ends up returning the grade for an answer nobody gave.
Neither text is logged.
"""

from __future__ import annotations

import datetime as dt
import logging
import uuid
from dataclasses import dataclass

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.interview.cache import SessionStateCache
from app.interview.fsm import EventType
from app.interview.service import commit_and_cache
from app.interview.state import SessionState
from app.interview.store import lock_session, stage_append
from app.models.interview import GRADING_PENDING, InterviewSession, Turn
from app.models.question import Question

log = logging.getLogger(__name__)


class TurnIdConflict(ValueError):
    """The ``turn_id`` was already used in this session for a different answer."""

    def __init__(self, session_id: uuid.UUID, turn_id: uuid.UUID) -> None:
        self.session_id = session_id
        self.turn_id = turn_id
        super().__init__(f"turn_id {turn_id} was already used with a different answer")


class TurnNotFound(LookupError):
    """No such turn in an owned session.  Same answer for 'not yours'."""

    def __init__(self, turn_id: uuid.UUID) -> None:
        self.turn_id = turn_id
        super().__init__(f"turn {turn_id} not found")


class FollowUpTurnsUnsupported(ValueError):
    """The question in front of the candidate is a follow-up asked *without*
    stored wording - only possible in a log written before Day 23 - so there is
    no question text to grade the answer against.  Refused rather than graded
    blind.  Follow-ups asked by the Day-23 worker always carry their text."""


@dataclass(frozen=True, slots=True)
class TurnRecord:
    """A turn as read inside a transaction.

    Returned instead of the ORM object, because the submission path ends with a
    rollback (to release the lock without writing) and a rollback expires every
    loaded instance - a live ``Turn`` would be unreadable by the time the route
    serialised it.  Deliberately has no ``answer_text``: nothing downstream of
    this service needs the answer, so nothing can leak it.
    """

    session_id: uuid.UUID
    turn_id: uuid.UUID
    question_id: str | None
    grading_status: str
    grading_attempts: int
    grading_error: str | None
    score: float | None
    created_at: dt.datetime
    graded_at: dt.datetime | None

    @classmethod
    def of(cls, turn: Turn) -> TurnRecord:
        return cls(
            session_id=turn.session_id,
            turn_id=turn.turn_id,
            question_id=turn.question_id,
            grading_status=turn.grading_status,
            grading_attempts=turn.grading_attempts,
            grading_error=turn.grading_error,
            score=turn.score,
            created_at=turn.created_at,
            graded_at=turn.graded_at,
        )


@dataclass(frozen=True, slots=True)
class Submission:
    turn: TurnRecord
    #: True for a new turn; False when an earlier identical submission was found.
    created: bool
    #: The session state after the submission (None when nothing was written).
    state: SessionState | None


async def submit_turn(
    maker: async_sessionmaker[AsyncSession],
    cache: SessionStateCache | None,
    *,
    session_id: uuid.UUID,
    user_id: uuid.UUID,
    turn_id: uuid.UUID,
    answer_text: str,
) -> Submission:
    """Record one answer, exactly once.  Commits.

    Raises ``SessionNotFound``, ``TurnIdConflict``, ``IllegalTransition`` (the
    session is not ``ASKING``), ``FollowUpTurnsUnsupported``, or
    ``EventLogCorrupted``; on any of them nothing is written.
    """
    async with maker() as db:
        try:
            outcome = await _submit(
                db, session_id=session_id, user_id=user_id, turn_id=turn_id, answer=answer_text
            )
        except BaseException:
            await db.rollback()
            raise
        if isinstance(outcome, Submission):
            await db.rollback()  # read-only path: release the lock
            return outcome
        turn_orm, staged = outcome
        if turn_orm is None:
            # ON CONFLICT DO NOTHING fired: someone inserted this turn_id without
            # the lock. Drop our staged event and return theirs.
            await db.rollback()
            log.warning("turn insert lost a race outside the session lock; re-reading")
        else:
            record = TurnRecord.of(turn_orm)
            state = await commit_and_cache(db, cache, staged)
            return Submission(turn=record, created=True, state=state)

    async with maker() as db:
        existing = await _find(db, session_id, turn_id)
        if existing is None:  # pragma: no cover - the conflict says it exists
            raise RuntimeError("turn conflict reported but no turn found")
        if existing.answer_text != answer_text:
            raise TurnIdConflict(session_id, turn_id)
        return Submission(turn=TurnRecord.of(existing), created=False, state=None)


async def _submit(
    db: AsyncSession, *, session_id: uuid.UUID, user_id: uuid.UUID, turn_id: uuid.UUID, answer: str
) -> Submission | tuple[Turn | None, SessionState]:
    await lock_session(db, session_id=session_id, user_id=user_id)

    existing = await _find(db, session_id, turn_id)
    if existing is not None:
        if existing.answer_text != answer:
            raise TurnIdConflict(session_id, turn_id)
        return Submission(turn=TurnRecord.of(existing), created=False, state=None)

    staged = await stage_append(
        db,
        session_id=session_id,
        user_id=user_id,
        event_type=EventType.ANSWER_SUBMITTED,
        payload={"turn_id": str(turn_id)},
    )
    # Day 23: the turn records what was actually shown (QUESTION_ASKED.text) and
    # the bank item it belongs to - for a follow-up, the item being probed, so
    # the worker grades it against that item's checklist.
    item_id = staged.current_item_id
    if item_id is None:
        raise RuntimeError("an answer was accepted for a question with no bank item")
    question_text = staged.current_question_text
    if question_text is None:
        if staged.current_is_follow_up:
            raise FollowUpTurnsUnsupported(
                "this follow-up was asked without stored wording (a pre-Day-23 log); "
                "there is no question text to grade the answer against"
            )
        question_text = await db.scalar(sa.select(Question.text).where(Question.id == item_id))
        if question_text is None:
            raise RuntimeError(f"question {item_id!r} was asked but is not in the bank")

    inserted = await db.execute(
        pg_insert(Turn)
        .values(
            session_id=session_id,
            turn_id=turn_id,
            question_id=item_id,
            question_text=question_text,
            answer_text=answer,
            is_follow_up=staged.current_is_follow_up,
            grading_status=GRADING_PENDING,
        )
        .on_conflict_do_nothing(constraint="uq_turns_session_id_turn_id")
        .returning(Turn.id)
    )
    turn_pk = inserted.scalar_one_or_none()
    if turn_pk is None:
        return None, staged
    turn = await db.get(Turn, turn_pk)
    if turn is None:  # pragma: no cover - just inserted in this transaction
        raise RuntimeError("inserted turn vanished")
    return turn, staged


async def _find(db: AsyncSession, session_id: uuid.UUID, turn_id: uuid.UUID) -> Turn | None:
    turn: Turn | None = await db.scalar(
        sa.select(Turn)
        .where(Turn.session_id == session_id, Turn.turn_id == turn_id)
        .execution_options(populate_existing=True)
    )
    return turn


async def get_turn(
    db: AsyncSession, *, session_id: uuid.UUID, user_id: uuid.UUID, turn_id: uuid.UUID
) -> TurnRecord:
    """One turn of an owned session, or ``TurnNotFound``."""
    turn = await db.scalar(
        sa.select(Turn)
        .join(InterviewSession, InterviewSession.id == Turn.session_id)
        .where(
            Turn.session_id == session_id,
            Turn.turn_id == turn_id,
            InterviewSession.user_id == user_id,
        )
        .execution_options(populate_existing=True)
    )
    if turn is None:
        raise TurnNotFound(turn_id)
    return TurnRecord.of(turn)


__all__ = [
    "FollowUpTurnsUnsupported",
    "Submission",
    "TurnIdConflict",
    "TurnNotFound",
    "TurnRecord",
    "get_turn",
    "submit_turn",
]
