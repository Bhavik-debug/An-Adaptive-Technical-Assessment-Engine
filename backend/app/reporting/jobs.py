"""Building the report and completing the session - durably, exactly once.

The job is the ``reports`` row, created ``pending`` in the same transaction
that appended ``REPORTING_STARTED`` (``app.interview.flow``).  One attempt::

    claim (SKIP LOCKED, lease, attempts+1)                   commit
    stats  = aggregate(committed log, blueprint, questions)  read-only, no lock
    prose  = one FINAL_REPORT call over the stats            no transaction open
             (or the deterministic template - see narrative.py)
    ── one transaction ──────────────────────────────────────────────────────
    lock session; fence the report row (still ours, same attempt);
    the log must still be at the seq the stats were computed from;
    reports: status ready, stats, narrative, provenance, cost
    interview_sessions: cost_usd/total_tokens += the narrative call (once)
    SESSION_COMPLETED {report_id}            REPORTING -> COMPLETED
    commit; cache

**Ordering guarantee:** ``COMPLETED`` is appended in the same transaction that
persists the report, so a completed session always has its report and a
report marked ready always belongs to a completed session.

**Retries, bounded, never endless:** a transient model failure is retried with
backoff up to ``GRADING_MAX_ATTEMPTS``; after that - or on any permanent model
failure or rejected narrative - the report completes with the template
narrative, labelled as such.  The prose is optional; the statistics are not:
if the statistics cannot be built (the log is not in ``REPORTING``, or is
corrupted) the report is marked ``failed`` with a code and the session stays in
``REPORTING`` for an operator, never silently completed.

**What a retry can and cannot repeat:** the narrative call may be repeated
(external calls are not exactly-once - Day 22); the *committed* outcome - one
report row, one ``SESSION_COMPLETED``, one cost application - cannot, because
of the fence and the FSM (``COMPLETED`` accepts nothing).
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.blueprint import Blueprint
from app.interview.cache import SessionStateCache
from app.interview.fsm import EventType, InterviewState
from app.interview.grading import RetryPolicy, classify_error
from app.interview.service import commit_and_cache
from app.interview.state import EventLogCorrupted, fold
from app.interview.store import lock_session, read_events, stage_append
from app.llm.client import CallMeta, call_structured
from app.llm.tasks import TaskName
from app.llm.types import TraceCtx
from app.models.interview import InterviewSession
from app.models.question import Question
from app.models.reporting import (
    REPORT_FAILED,
    REPORT_PENDING,
    REPORT_PROCESSING,
    REPORT_READY,
    Report,
)
from app.reporting.narrative import (
    NarrativeRejected,
    ReportNarrative,
    check_numbers,
    narrative_inputs,
    template_narrative,
)
from app.reporting.stats import QuestionInfo, ReportNotReady, aggregate

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ReportJob:
    report_id: uuid.UUID
    session_id: uuid.UUID
    user_id: uuid.UUID
    attempt: int


#: stats -> (validated narrative, the call's metadata).  Production:
#: :func:`narrate_with_model`; tests inject failures.
NarrativeFn = Callable[[dict[str, Any], ReportJob], Awaitable[tuple[ReportNarrative, CallMeta]]]


async def narrate_with_model(
    stats: dict[str, Any], job: ReportJob
) -> tuple[ReportNarrative, CallMeta]:
    narrative, meta = await call_structured(
        TaskName.FINAL_REPORT,
        narrative_inputs(stats),
        ReportNarrative,
        trace=TraceCtx(session_id=job.session_id, user_id=job.user_id),
    )
    return check_numbers(narrative, stats), meta


_CLAIM = sa.text(
    """
    UPDATE reports AS r
       SET status = 'processing', attempts = r.attempts + 1,
           lease_expires_at = now() + make_interval(secs => :lease_s),
           next_attempt_at = NULL
     WHERE r.id = (
           SELECT id FROM reports
            WHERE (status = 'pending'
                   AND (next_attempt_at IS NULL OR next_attempt_at <= now()))
               OR (status = 'processing' AND lease_expires_at < now())
            ORDER BY created_at, id
            FOR UPDATE SKIP LOCKED
            LIMIT 1)
    RETURNING r.id, r.session_id, r.attempts
    """
)


async def claim_next_report(
    maker: async_sessionmaker[AsyncSession], policy: RetryPolicy
) -> ReportJob | None:
    async with maker() as db, db.begin():
        row = (await db.execute(_CLAIM, {"lease_s": policy.lease_s})).first()
        if row is None:
            return None
        user_id = await db.scalar(
            sa.select(InterviewSession.user_id).where(InterviewSession.id == row.session_id)
        )
    if user_id is None:  # pragma: no cover - FK
        raise RuntimeError("claimed report has no session")
    return ReportJob(
        report_id=row.id, session_id=row.session_id, user_id=user_id, attempt=row.attempts
    )


async def build_stats(
    maker: async_sessionmaker[AsyncSession], session_id: uuid.UUID
) -> dict[str, Any]:
    """Aggregate from the committed log.  Read-only; takes no lock."""
    async with maker() as db:
        session = await db.get(InterviewSession, session_id)
        if session is None:  # pragma: no cover - FK
            raise ReportNotReady("session vanished")
        events = await read_events(db, session_id)
        asked = [
            str(e.payload["question_id"])
            for e in events
            if e.type == EventType.QUESTION_ASKED and e.payload.get("question_id")
        ]
        rows = (
            await db.execute(
                sa.select(Question.id, Question.topic_key, Question.subtopic_key).where(
                    Question.id.in_(asked or [""])
                )
            )
        ).all()
        blueprint = Blueprint.from_dict(session.blueprint)
    return aggregate(
        events=events,
        session_id=session_id,
        blueprint=blueprint,
        questions={r.id: QuestionInfo(r.topic_key, r.subtopic_key) for r in rows},
    )


class ReportOutcome:
    COMPLETED = "report_completed"
    RETRY_SCHEDULED = "report_retry_scheduled"
    FAILED = "report_failed"
    STALE = "report_stale"


async def _set(
    maker: async_sessionmaker[AsyncSession], job: ReportJob, values: dict[str, Any]
) -> bool:
    async with maker() as db, db.begin():
        result = await db.execute(
            sa.update(Report)
            .where(
                Report.id == job.report_id,
                Report.status == REPORT_PROCESSING,
                Report.attempts == job.attempt,
            )
            .values(lease_expires_at=None, **values)
        )
        return bool(result.rowcount)  # type: ignore[attr-defined]


async def process_report(
    maker: async_sessionmaker[AsyncSession],
    cache: SessionStateCache | None,
    job: ReportJob,
    *,
    narrative_fn: NarrativeFn,
    policy: RetryPolicy,
) -> str:
    try:
        stats = await build_stats(maker, job.session_id)
    except (ReportNotReady, EventLogCorrupted) as exc:
        code = "not_reporting" if isinstance(exc, ReportNotReady) else "event_log_corrupted"
        log.error("report statistics unavailable", extra={"report.error_code": code})
        ok = await _set(maker, job, {"status": REPORT_FAILED, "error": code})
        return ReportOutcome.FAILED if ok else ReportOutcome.STALE

    narrative: ReportNarrative
    meta: CallMeta | None = None
    source, narrative_error = "model", None
    if job.attempt > policy.max_attempts:
        # Earlier attempts crashed mid-call. Do not call the model again.
        narrative, source, narrative_error = (
            template_narrative(stats),
            "template",
            "retry_exhausted",
        )
    else:
        try:
            narrative, meta = await asyncio.wait_for(
                narrative_fn(stats, job), timeout=policy.timeout_s
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - classified; never swallowed silently
            if isinstance(exc, NarrativeRejected):
                code, transient = f"narrative_{exc.rule}", False
            else:
                code, transient = classify_error(exc)
            log.warning(
                "report narrative failed",
                extra={"report.error_code": code, "report.exception": type(exc).__name__},
            )
            if transient and job.attempt < policy.max_attempts:
                ok = await _set(
                    maker,
                    job,
                    {
                        "status": REPORT_PENDING,
                        "error": code,
                        "next_attempt_at": sa.func.now()
                        + sa.func.make_interval(0, 0, 0, 0, 0, 0, policy.delay_after(job.attempt)),
                    },
                )
                return ReportOutcome.RETRY_SCHEDULED if ok else ReportOutcome.STALE
            narrative, source = template_narrative(stats), "template"
            narrative_error = code if not transient else "retry_exhausted"
    return await _complete(maker, cache, job, stats, narrative, source, narrative_error, meta)


async def _complete(
    maker: async_sessionmaker[AsyncSession],
    cache: SessionStateCache | None,
    job: ReportJob,
    stats: dict[str, Any],
    narrative: ReportNarrative,
    source: str,
    narrative_error: str | None,
    meta: CallMeta | None,
) -> str:
    cost = meta.cost_usd if meta else Decimal(0)
    tokens = (meta.input_tokens + meta.output_tokens) if meta else 0
    async with maker() as db:
        try:
            session = await lock_session(db, session_id=job.session_id, user_id=job.user_id)
            report = await db.scalar(
                sa.select(Report)
                .where(
                    Report.id == job.report_id,
                    Report.status == REPORT_PROCESSING,
                    Report.attempts == job.attempt,
                )
                .with_for_update()
                .execution_options(populate_existing=True)
            )
            current = fold(job.session_id, await read_events(db, job.session_id))
            if report is None or current.state is not InterviewState.REPORTING:
                await db.rollback()
                return ReportOutcome.STALE
            if (
                current.seq != stats["log_seq"]
            ):  # pragma: no cover - REPORTING admits no other event
                await db.rollback()
                return ReportOutcome.STALE
            staged = await stage_append(
                db,
                session_id=job.session_id,
                user_id=job.user_id,
                event_type=EventType.SESSION_COMPLETED,
                payload={"report_id": str(job.report_id)},
                expected_seq=current.seq,
            )
            report.status = REPORT_READY
            report.error = None
            report.stats = stats
            report.stats_seq = stats["log_seq"]
            report.narrative = narrative.model_dump(mode="json")
            report.narrative_source = source
            report.narrative_error = narrative_error
            report.cost_usd = cost
            report.total_tokens = tokens
            report.generated_at = sa.func.now()
            report.lease_expires_at = None
            session.cost_usd = InterviewSession.cost_usd + cost
            session.total_tokens = InterviewSession.total_tokens + tokens
            await db.flush()
        except BaseException:
            await db.rollback()
            raise
        await commit_and_cache(db, cache, staged)
    return ReportOutcome.COMPLETED


async def ensure_report_job(db: AsyncSession, session_id: uuid.UUID) -> None:
    """Idempotent: a pending report row for a session in REPORTING."""
    from sqlalchemy.dialects.postgresql import insert as pg_insert

    await db.execute(
        pg_insert(Report)
        .values(session_id=session_id, status=REPORT_PENDING)
        .on_conflict_do_nothing(constraint="uq_reports_session_id")
    )


async def requeue_failed_report(
    maker: async_sessionmaker[AsyncSession], *, session_id: uuid.UUID
) -> bool:
    """Operator action: retry a ``failed`` report with a fresh attempt budget."""
    async with maker() as db, db.begin():
        result = await db.execute(
            sa.update(Report)
            .where(Report.session_id == session_id, Report.status == REPORT_FAILED)
            .values(status=REPORT_PENDING, attempts=0, next_attempt_at=None, error=None)
        )
        return bool(result.rowcount)  # type: ignore[attr-defined]


async def run_once_report(
    maker: async_sessionmaker[AsyncSession],
    cache: SessionStateCache | None,
    *,
    narrative_fn: NarrativeFn,
    policy: RetryPolicy,
) -> str | None:
    job = await claim_next_report(maker, policy)
    if job is None:
        return None
    return await process_report(maker, cache, job, narrative_fn=narrative_fn, policy=policy)


__all__ = [
    "NarrativeFn",
    "ReportJob",
    "ReportOutcome",
    "build_stats",
    "claim_next_report",
    "ensure_report_job",
    "narrate_with_model",
    "process_report",
    "requeue_failed_report",
    "run_once_report",
]
