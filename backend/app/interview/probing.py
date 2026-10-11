"""Wording decided follow-ups in the background - the PROBING state's job.

The decision was committed with the grade (``FOLLOW_UP_STARTED`` + a ``pending``
``follow_ups`` row, :mod:`app.interview.flow`).  This module does the only part
that needs a model - the wording - with the same machinery as grading
(:mod:`app.interview.grading`): a ``SKIP LOCKED`` claim with a lease, a bounded
:class:`~app.interview.grading.RetryPolicy`, the same error classification, and
a fence on ``attempts`` so a stale worker cannot finish someone else's attempt.

::

    pending --claim--> processing --valid probe--> ready      (+ QUESTION_ASKED follow-up)
       ^                   |
       +---transient-------+  (attempt < max: back off)
                           +--permanent / used up--> abandoned (+ FOLLOW_UP_ABANDONED,
                                                                then normal selection)

**No fabricated question, ever.**  A probe that fails validation
(:func:`app.wording.probe.check_probe`) or never arrives is not replaced with a
canned one.  The follow-up is abandoned on the record and the interview moves
on with the next bank item - the same thing that happens when the caps say no.
The grade, the ability estimate and the caps are untouched by any of this: the
wording is written *after* the decision and read by nothing that decides.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.blueprint import Blueprint
from app.interview.cache import SessionStateCache
from app.interview.flow import stage_next_bank_question
from app.interview.fsm import EventType, IllegalTransition, InterviewState
from app.interview.grading import RetryPolicy, classify_error
from app.interview.service import commit_and_cache
from app.interview.state import EventLogCorrupted, SessionState, fold
from app.interview.store import lock_session, read_events, stage_append
from app.llm.types import TraceCtx
from app.models.interview import (
    FOLLOW_UP_ABANDONED,
    FOLLOW_UP_PENDING,
    FOLLOW_UP_PROCESSING,
    FOLLOW_UP_READY,
    FollowUp,
    InterviewSession,
    Turn,
)
from app.models.question import Question
from app.privacy import ANSWER_KINDS, redact
from app.wording.probe import ProbeContext, ProbeRejected, generate_probe, seeds_from

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class FollowUpJob:
    follow_up_id: uuid.UUID
    session_id: uuid.UUID
    user_id: uuid.UUID
    attempt: int
    #: ``None`` when the bank item or the source answer is gone.
    context: ProbeContext | None


ProbeFn = Callable[[FollowUpJob], Awaitable[str]]


async def probe_with_model(job: FollowUpJob) -> str:
    """The production wording call (``FOLLOW_UP_PROBE`` through the chokepoint)."""
    if job.context is None:  # pragma: no cover - process_follow_up checks first
        raise ValueError("no probe context")
    text, _ = await generate_probe(
        job.context, trace=TraceCtx(session_id=job.session_id, user_id=job.user_id)
    )
    return text


_CLAIM = sa.text(
    """
    UPDATE follow_ups AS f
       SET status = 'processing',
           attempts = f.attempts + 1,
           lease_expires_at = now() + make_interval(secs => :lease_s),
           next_attempt_at = NULL
     WHERE f.id = (
           SELECT id FROM follow_ups
            WHERE (status = 'pending'
                   AND (next_attempt_at IS NULL OR next_attempt_at <= now()))
               OR (status = 'processing' AND lease_expires_at < now())
            ORDER BY created_at, id
            FOR UPDATE SKIP LOCKED
            LIMIT 1)
    RETURNING f.id
    """
)


def _major_claims(grade: Any) -> tuple[str, ...]:
    judgment = grade.get("judgment") if isinstance(grade, dict) else None
    errors = judgment.get("errors") if isinstance(judgment, dict) else None
    if not isinstance(errors, list):
        return ()
    return tuple(
        e["claim"]
        for e in errors
        if isinstance(e, dict) and e.get("severity") == "major" and isinstance(e.get("claim"), str)
    )


async def claim_next_follow_up(
    maker: async_sessionmaker[AsyncSession], policy: RetryPolicy
) -> FollowUpJob | None:
    async with maker() as db, db.begin():
        fid = (await db.execute(_CLAIM, {"lease_s": policy.lease_s})).scalar_one_or_none()
        if fid is None:
            return None
        row = await db.get(FollowUp, fid, populate_existing=True)
        session = await db.get(InterviewSession, row.session_id) if row else None
        if row is None or session is None:  # pragma: no cover - FK + same txn
            raise RuntimeError("claimed follow-up vanished")
        item = await db.get(Question, row.item_question_id) if row.item_question_id else None
        turn = await db.scalar(
            sa.select(Turn).where(
                Turn.session_id == row.session_id, Turn.turn_id == row.source_turn_id
            )
        )
        context = None
        if item is not None and turn is not None and turn.answer_text:
            hint = next(
                (
                    c.get("hint")
                    for c in item.expected_concepts or []
                    if isinstance(c, dict) and c.get("key") == row.concept_key
                ),
                None,
            )
            context = ProbeContext(
                question_text=item.text,
                # Day 24: contact details redacted before the model sees them.
                answer_text=redact(turn.answer_text, ANSWER_KINDS).text,
                reason=row.reason,
                concept_key=row.concept_key,
                concept_hint=hint if isinstance(hint, str) else None,
                error_claims=tuple(redact(c, ANSWER_KINDS).text for c in _major_claims(turn.grade)),
                seeds=seeds_from(item.follow_up_seeds),
            )
        return FollowUpJob(
            follow_up_id=row.id,
            session_id=row.session_id,
            user_id=session.user_id,
            attempt=row.attempts,
            context=context,
        )


async def _fenced(db: AsyncSession, job: FollowUpJob) -> FollowUp | None:
    row = await db.scalar(
        sa.select(FollowUp)
        .where(FollowUp.id == job.follow_up_id)
        .with_for_update()
        .execution_options(populate_existing=True)
    )
    if row is None or row.status != FOLLOW_UP_PROCESSING or row.attempts != job.attempt:
        log.warning("follow-up result discarded: attempt no longer owned")
        return None
    return row


class FollowUpOutcome:
    ASKED = "follow_up_asked"
    RETRY_SCHEDULED = "follow_up_retry_scheduled"
    ABANDONED = "follow_up_abandoned"
    STALE = "follow_up_stale"


async def process_follow_up(
    maker: async_sessionmaker[AsyncSession],
    cache: SessionStateCache | None,
    job: FollowUpJob,
    *,
    probe_fn: ProbeFn,
    policy: RetryPolicy,
) -> str:
    if job.attempt > policy.max_attempts:
        return await _abandon(maker, cache, job, "retry_exhausted", retries_exhausted=True)
    if job.context is None:
        return await _abandon(maker, cache, job, "context_unavailable", retries_exhausted=False)
    try:
        text = await asyncio.wait_for(probe_fn(job), timeout=policy.timeout_s)
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001 - classified, never swallowed
        if isinstance(exc, ProbeRejected):
            code, transient = f"probe_{exc.rule}", False
        else:
            code, transient = classify_error(exc)
        log.warning(
            "follow-up wording failed",
            extra={
                "follow_up.attempt": job.attempt,
                "follow_up.error_code": code,
                "follow_up.transient": transient,
                "follow_up.exception": type(exc).__name__,
            },
        )
        if transient and job.attempt < policy.max_attempts:
            return await _retry(maker, job, code, policy.delay_after(job.attempt))
        return await _abandon(maker, cache, job, code, retries_exhausted=transient)
    return await _ask(maker, cache, job, text)


async def _ask(
    maker: async_sessionmaker[AsyncSession],
    cache: SessionStateCache | None,
    job: FollowUpJob,
    text: str,
) -> str:
    async with maker() as db:
        try:
            await lock_session(db, session_id=job.session_id, user_id=job.user_id)
            row = await _fenced(db, job)
            if row is None:
                await db.rollback()
                return FollowUpOutcome.STALE
            staged = await stage_append(
                db,
                session_id=job.session_id,
                user_id=job.user_id,
                event_type=EventType.QUESTION_ASKED,
                payload={"is_follow_up": True, "text": text},
            )
            row.status = FOLLOW_UP_READY
            row.text = text
            row.generated_at = sa.func.now()
            row.lease_expires_at = None
            row.error = None
            await db.flush()
        except (IllegalTransition, EventLogCorrupted) as exc:
            await db.rollback()
            log.error("follow-up could not be asked", extra={"error": repr(exc)})
            return await _abandon(maker, cache, job, "state_conflict", retries_exhausted=False)
        except BaseException:
            await db.rollback()
            raise
        await commit_and_cache(db, cache, staged)
    return FollowUpOutcome.ASKED


async def _retry(
    maker: async_sessionmaker[AsyncSession], job: FollowUpJob, code: str, delay_s: float
) -> str:
    async with maker() as db, db.begin():
        row = await _fenced(db, job)
        if row is None:
            return FollowUpOutcome.STALE
        row.status = FOLLOW_UP_PENDING
        row.error = code
        row.lease_expires_at = None
        row.next_attempt_at = sa.func.now() + sa.func.make_interval(0, 0, 0, 0, 0, 0, delay_s)
    return FollowUpOutcome.RETRY_SCHEDULED


async def _abandon(
    maker: async_sessionmaker[AsyncSession],
    cache: SessionStateCache | None,
    job: FollowUpJob,
    code: str,
    *,
    retries_exhausted: bool,
) -> str:
    """Abandon on the record (``FOLLOW_UP_ABANDONED``) and continue with normal
    selection - in one transaction, so the session is never left in PROBING
    without a job."""
    async with maker() as db:
        try:
            session = await lock_session(db, session_id=job.session_id, user_id=job.user_id)
            row = await _fenced(db, job)
            if row is None:
                await db.rollback()
                return FollowUpOutcome.STALE
            row.status = FOLLOW_UP_ABANDONED
            row.error = code
            row.lease_expires_at = None
            current: SessionState = fold(job.session_id, await read_events(db, job.session_id))
            if current.state is not InterviewState.PROBING:
                # Nothing in the log is waiting on this follow-up.
                await db.commit()
                return FollowUpOutcome.ABANDONED
            abandoned = await stage_append(
                db,
                session_id=job.session_id,
                user_id=job.user_id,
                event_type=EventType.FOLLOW_UP_ABANDONED,
                payload={
                    "error_code": code,
                    "attempts": max(0, min(job.attempt, 1_000)),
                    "retries_exhausted": retries_exhausted,
                },
                expected_seq=current.seq,
            )
            staged = await stage_next_bank_question(
                db,
                abandoned,
                user_id=job.user_id,
                blueprint=Blueprint.from_dict(session.blueprint),
            )
        except BaseException:
            await db.rollback()
            raise
        await commit_and_cache(db, cache, staged)
    return FollowUpOutcome.ABANDONED


async def run_once_follow_up(
    maker: async_sessionmaker[AsyncSession],
    cache: SessionStateCache | None,
    *,
    probe_fn: ProbeFn,
    policy: RetryPolicy,
) -> str | None:
    job = await claim_next_follow_up(maker, policy)
    if job is None:
        return None
    return await process_follow_up(maker, cache, job, probe_fn=probe_fn, policy=policy)


__all__ = [
    "FollowUpJob",
    "FollowUpOutcome",
    "ProbeFn",
    "claim_next_follow_up",
    "probe_with_model",
    "process_follow_up",
    "run_once_follow_up",
]
