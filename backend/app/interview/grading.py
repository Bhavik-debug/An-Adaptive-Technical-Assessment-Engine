"""Grading submitted turns in the background - durably, at most once.

There is no queue technology in this repository, so the ``turns`` table *is*
the queue (migration 0004).  One job is one turn row::

    pending ──claim──► processing ──success──► graded      (+ ANSWER_GRADED, + cost)
       ▲                   │
       └──transient error──┤ (attempt < max: back off, retry)
                           └──permanent error / attempts used up──► failed  (+ GRADING_FAILED)

    processing with an expired lease  ──claim──►  processing   (a worker died mid-job)

**Claiming** is ``UPDATE ... WHERE id = (SELECT ... FOR UPDATE SKIP LOCKED)``:
any number of workers can poll at once, each row goes to exactly one of them, and
nobody waits on a row another worker holds.  The claim increments
``grading_attempts`` and commits *before* the model is called, so no database
connection or lock is held across a multi-second LLM call.

**Finishing is fenced.**  A worker may only write the result of the attempt it
claimed: the finishing transaction re-reads the row under ``FOR UPDATE`` and
proceeds only if it is still ``processing`` at the same ``grading_attempts``.  A
worker that stalled past its lease, was presumed dead, and then woke up finds
the attempt number moved on and discards its result.  Together with the FSM's
"a turn is graded once" rule this is what makes the guarantee hold:

    one logical turn -> at most one committed grade -> at most one cost applied

**What is *not* guaranteed: exactly one provider call.**  A crash between the
model answering and the commit, or a stale worker being fenced out, means the
model was paid for an answer nobody kept.  No design can make an external call
and a database commit atomic; this one makes the *committed* outcome unique and
keeps the duplicate calls bounded by ``grading_max_attempts``.

**What a success writes, in one transaction** (on top of Day 21's
``stage_append``, which takes the session lock and checks the FSM):

* ``ANSWER_GRADED`` with the score and the ability *after* ``update_ability``;
* the turn's ``grade`` / ``score`` / ``grader_confidence`` / ``latency_ms`` /
  ``graded_at`` and ``grading_status = 'graded'``;
* ``interview_sessions.cost_usd += Σ cost`` and ``total_tokens += Σ tokens`` over
  the grade's chokepoint calls - the only "charge" this codebase has before
  Phase 8's billing (see ``docs/turn-api.md`` §6).

The grading itself is ``app.grading.pipeline.grade_answer`` unchanged; nothing
here scores anything.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import logging
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.ability import STAKE_BANK_ITEM, STAKE_GENERATED_ITEM, AbilityState, update_ability
from app.bank.schema import ExpectedConcept
from app.blueprint import Blueprint
from app.config import Settings
from app.grading.grader import GradingContractError, GradingInputError
from app.grading.pipeline import GradeResult, grade_answer
from app.interview.cache import SessionStateCache
from app.interview.flow import stage_after_grade
from app.interview.fsm import EventType, IllegalTransition
from app.interview.service import commit_and_cache
from app.interview.state import EventLogCorrupted, fold
from app.interview.store import SeqConflict, lock_session, read_events, stage_append
from app.llm.errors import (
    AllProvidersFailedError,
    LLMConfigError,
    PromptNotRegisteredError,
    ProviderError,
    SchemaValidationFailedError,
)
from app.llm.types import TraceCtx
from app.models.interview import (
    GRADING_FAILED,
    GRADING_GRADED,
    GRADING_PENDING,
    GRADING_PROCESSING,
    InterviewSession,
    Turn,
)
from app.models.question import Question
from app.privacy import ANSWER_KINDS, redact

log = logging.getLogger(__name__)

#: How much longer than its own deadline a claim is held before another worker
#: may presume the claimant dead.
LEASE_MARGIN_S = 60.0


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """How many claims a turn gets, and how long to wait between them."""

    max_attempts: int = 3
    base_delay_s: float = 2.0
    max_delay_s: float = 30.0
    timeout_s: float = 300.0
    #: Day 25: was the fixed LEASE_MARGIN_S; same default.
    lease_margin_s: float = LEASE_MARGIN_S

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if self.base_delay_s <= 0 or self.max_delay_s <= 0 or self.timeout_s <= 0:
            raise ValueError("delays and timeout must be positive")

    @classmethod
    def from_settings(cls, settings: Settings) -> RetryPolicy:
        return cls(
            max_attempts=settings.grading_max_attempts,
            base_delay_s=settings.grading_retry_base_s,
            max_delay_s=settings.grading_retry_max_s,
            timeout_s=settings.grading_timeout_s,
            lease_margin_s=settings.grading_lease_margin_s,
        )

    @property
    def lease_s(self) -> float:
        return self.timeout_s + self.lease_margin_s

    def delay_after(self, attempt: int) -> float:
        """Backoff before the next claim, after ``attempt`` failed. Bounded."""
        return float(min(self.max_delay_s, self.base_delay_s * 2 ** max(0, attempt - 1)))


def classify_error(exc: BaseException) -> tuple[str, bool]:
    """``(error_code, transient)``.  The code is safe to store and show; the
    exception message is not (a provider error can quote the request).

    Transient means "the identical request might succeed later": the chokepoint's
    own ``retryable`` flag, a router that ran out of providers (it has already
    failed over and backed off once), and timeouts/dropped connections.  Every
    other failure is permanent - re-sending a request the grader rejected, or a
    reply that failed the schema after the chokepoint's own repair attempts,
    would only spend money to fail the same way.
    """
    if isinstance(exc, ProviderError):
        kind = type(exc).__name__.removeprefix("Provider").removesuffix("Error").lower()
        return f"provider_{kind or 'error'}", exc.retryable
    if isinstance(exc, AllProvidersFailedError):
        return "all_providers_failed", True
    if isinstance(exc, TimeoutError):
        return "timeout", True
    if isinstance(exc, ConnectionError):
        return "connection_error", True
    if isinstance(exc, SchemaValidationFailedError):
        return "schema_validation_failed", False
    if isinstance(exc, GradingInputError):
        return "ungradeable_input", False
    if isinstance(exc, GradingContractError):
        return "grader_contract_violation", False
    if isinstance(exc, LLMConfigError | PromptNotRegisteredError):
        return "llm_misconfigured", False
    return "internal_error", False


# ---------------------------------------------------------------------------
# The job
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GradingJob:
    """Everything one attempt needs, read at claim time."""

    turn_pk: uuid.UUID
    session_id: uuid.UUID
    user_id: uuid.UUID
    turn_id: uuid.UUID
    attempt: int
    question_id: str | None
    question_text: str
    answer_text: str
    expected_concepts: tuple[ExpectedConcept, ...]
    common_misconceptions: tuple[str, ...]
    subtopic_key: str | None
    difficulty_b: float | None
    discrimination_a: float | None
    #: Day 23: an answer to a follow-up probe.  Graded against the *parent* bank
    #: item's checklist (``question_id`` is the parent), with the probe appended
    #: to the question, and applied to theta at half stake (plan section 8.5).
    is_follow_up: bool = False
    #: The parent item's canonical text (``question_text`` is what was shown).
    item_text: str | None = None


#: The grading call.  Production: :func:`grade_with_pipeline`.  Tests inject a
#: fake to script transient and permanent failures.
GradeFn = Callable[[GradingJob], Awaitable[GradeResult]]


async def grade_with_pipeline(job: GradingJob) -> GradeResult:
    """Phase 4's grader, unchanged, through the LLM chokepoint.

    For a follow-up the grader is shown the original question *and* the probe,
    so it judges the answer in the context it was given.  The checklist is the
    parent item's: a probe has no checklist of its own (plan section 6.2 - the
    content of a question is canonical, only its framing is generated).
    """
    question_text = job.question_text
    if job.is_follow_up and job.item_text:
        question_text = f"{job.item_text}\n\nFollow-up question: {job.question_text}"
    return await grade_answer(
        question_text=question_text,
        expected_concepts=job.expected_concepts,
        # Day 24, plan section 14.3: "redaction pass before anything reaches an
        # LLM". Contact details only (ANSWER_KINDS): technical content - IPs,
        # versions, URLs of APIs - is left exactly as written. The evidence
        # check verifies spans against this same redacted text.
        answer_text=redact(job.answer_text, ANSWER_KINDS).text,
        common_misconceptions=job.common_misconceptions,
        trace=TraceCtx(session_id=job.session_id, turn_id=job.turn_id, user_id=job.user_id),
    )


_CLAIM = sa.text(
    """
    UPDATE turns AS t
       SET grading_status = 'processing',
           grading_attempts = t.grading_attempts + 1,
           lease_expires_at = now() + make_interval(secs => :lease_s),
           next_attempt_at = NULL
     WHERE t.id = (
           SELECT id FROM turns
            WHERE (grading_status = 'pending'
                   AND (next_attempt_at IS NULL OR next_attempt_at <= now()))
               OR (grading_status = 'processing' AND lease_expires_at < now())
            ORDER BY created_at, id
            FOR UPDATE SKIP LOCKED
            LIMIT 1)
    RETURNING t.id
    """
)


async def claim_next(
    maker: async_sessionmaker[AsyncSession], policy: RetryPolicy
) -> GradingJob | None:
    """Claim one claimable turn and commit the claim.  None when there is none."""
    async with maker() as db, db.begin():
        turn_pk = (await db.execute(_CLAIM, {"lease_s": policy.lease_s})).scalar_one_or_none()
        if turn_pk is None:
            return None
        turn = await db.get(Turn, turn_pk, populate_existing=True)
        session = await db.get(InterviewSession, turn.session_id) if turn else None
        if turn is None or session is None:  # pragma: no cover - FK + same txn
            raise RuntimeError("claimed turn vanished")
        question = await db.get(Question, turn.question_id) if turn.question_id else None
        return GradingJob(
            turn_pk=turn.id,
            session_id=turn.session_id,
            user_id=session.user_id,
            turn_id=turn.turn_id,
            attempt=turn.grading_attempts,
            question_id=turn.question_id,
            question_text=turn.question_text,
            answer_text=turn.answer_text or "",
            expected_concepts=tuple(
                ExpectedConcept.model_validate(c) for c in (question.expected_concepts or [])
            )
            if question
            else (),
            common_misconceptions=tuple(question.common_misconceptions or ()) if question else (),
            subtopic_key=question.subtopic_key if question else None,
            difficulty_b=question.difficulty_b if question else None,
            discrimination_a=question.discrimination_a if question else None,
            is_follow_up=turn.is_follow_up,
            item_text=question.text if question else None,
        )


# ---------------------------------------------------------------------------
# Processing one job
# ---------------------------------------------------------------------------


class Outcome:
    GRADED = "graded"
    RETRY_SCHEDULED = "retry_scheduled"
    FAILED = "failed"
    STALE = "stale"


async def process_job(
    maker: async_sessionmaker[AsyncSession],
    cache: SessionStateCache | None,
    job: GradingJob,
    *,
    grade_fn: GradeFn,
    policy: RetryPolicy,
) -> str:
    """Run one claimed attempt to its end state.  Returns an :class:`Outcome`."""
    if job.attempt > policy.max_attempts:
        # Only reachable by re-claiming an expired lease: earlier attempts
        # crashed mid-job. Do not call the model again.
        return await _fail(maker, cache, job, "retry_exhausted", retries_exhausted=True)
    if job.question_id is None or not job.expected_concepts:
        return await _fail(maker, cache, job, "question_unavailable", retries_exhausted=False)

    started = time.perf_counter()
    try:
        result = await asyncio.wait_for(grade_fn(job), timeout=policy.timeout_s)
    except asyncio.CancelledError:
        raise  # shutting down: the lease expires and another claim resumes it
    except Exception as exc:  # noqa: BLE001 - every failure is classified, none swallowed
        code, transient = classify_error(exc)
        log.warning(
            "grading attempt failed",
            extra={
                "grading.turn_id": str(job.turn_id),
                "grading.attempt": job.attempt,
                "grading.error_code": code,
                "grading.transient": transient,
                "grading.exception": type(exc).__name__,
            },
            exc_info=code == "internal_error",
        )
        if transient and job.attempt < policy.max_attempts:
            return await _schedule_retry(maker, job, code, policy.delay_after(job.attempt))
        return await _fail(maker, cache, job, code, retries_exhausted=transient)

    latency_ms = int((time.perf_counter() - started) * 1000)
    return await _succeed(maker, cache, job, result, latency_ms)


async def _fenced_turn(db: AsyncSession, job: GradingJob) -> Turn | None:
    """The turn, locked, if this worker still owns the attempt it claimed."""
    turn = await db.scalar(
        sa.select(Turn)
        .where(Turn.id == job.turn_pk)
        .with_for_update()
        .execution_options(populate_existing=True)
    )
    if (
        turn is None
        or turn.grading_status != GRADING_PROCESSING
        or turn.grading_attempts != job.attempt
    ):
        log.warning(
            "grading result discarded: attempt no longer owned",
            extra={"grading.turn_id": str(job.turn_id), "grading.attempt": job.attempt},
        )
        return None
    return turn


async def _succeed(
    maker: async_sessionmaker[AsyncSession],
    cache: SessionStateCache | None,
    job: GradingJob,
    result: GradeResult,
    latency_ms: int,
) -> str:
    async with maker() as db:
        try:
            # Session lock first, then the turn: the same order as submission.
            session = await lock_session(db, session_id=job.session_id, user_id=job.user_id)
            turn = await _fenced_turn(db, job)
            if turn is None:
                await db.rollback()
                return Outcome.STALE
            current = fold(job.session_id, await read_events(db, job.session_id))
            payload: dict[str, Any] = {"turn_id": str(job.turn_id), "score": result.score}
            ability = _ability_after(
                {**current.priors, **current.ability}, session.blueprint, job, result
            )
            if ability is not None:
                payload["ability"], payload["delta_theta"] = ability
            graded = await stage_append(
                db,
                session_id=job.session_id,
                user_id=job.user_id,
                event_type=EventType.ANSWER_GRADED,
                payload=payload,
                expected_seq=current.seq,
            )
            # Day 23: what comes next, decided in code and staged in this same
            # transaction - so a committed grade always has its decision.
            staged, decision = await stage_after_grade(
                db,
                graded,
                user_id=job.user_id,
                blueprint=Blueprint.from_dict(session.blueprint),
                judgment=result.judgment.model_dump(mode="json"),
                expected=[(c.key, c.weight) for c in job.expected_concepts],
                score=result.score,
                subtopic_key=job.subtopic_key,
                source_turn_id=job.turn_id,
            )
            turn.policy_decision = decision.trace()
            turn.grading_status = GRADING_GRADED
            turn.score = result.score
            turn.grader_confidence = result.grader_confidence
            turn.grade = _grade_record(result)
            turn.latency_ms = latency_ms
            turn.graded_at = sa.func.now()
            turn.lease_expires_at = None
            turn.grading_error = None
            cost, tokens = charge_for(result)
            session.cost_usd = InterviewSession.cost_usd + cost
            session.total_tokens = InterviewSession.total_tokens + tokens
            await db.flush()
        except (IllegalTransition, SeqConflict, EventLogCorrupted) as exc:
            # The log will not accept this grade (e.g. the turn is no longer the
            # one being graded). Never force it: record a permanent failure.
            await db.rollback()
            log.error(
                "grade could not be recorded",
                extra={"grading.turn_id": str(job.turn_id), "grading.exception": repr(exc)},
            )
            return await _fail(maker, cache, job, "state_conflict", retries_exhausted=False)
        except BaseException:
            await db.rollback()
            raise
        await commit_and_cache(db, cache, staged)
    log.info(
        "turn graded",
        extra={"grading.turn_id": str(job.turn_id), "grading.attempt": job.attempt},
    )
    return Outcome.GRADED


async def _schedule_retry(
    maker: async_sessionmaker[AsyncSession], job: GradingJob, code: str, delay_s: float
) -> str:
    async with maker() as db, db.begin():
        turn = await _fenced_turn(db, job)
        if turn is None:
            return Outcome.STALE
        turn.grading_status = GRADING_PENDING
        turn.grading_error = code
        turn.lease_expires_at = None
        turn.next_attempt_at = sa.func.now() + sa.func.make_interval(0, 0, 0, 0, 0, 0, delay_s)
    return Outcome.RETRY_SCHEDULED


async def _fail(
    maker: async_sessionmaker[AsyncSession],
    cache: SessionStateCache | None,
    job: GradingJob,
    code: str,
    *,
    retries_exhausted: bool,
) -> str:
    """Mark the turn failed and record ``GRADING_FAILED`` - atomically."""
    async with maker() as db:
        try:
            await lock_session(db, session_id=job.session_id, user_id=job.user_id)
            turn = await _fenced_turn(db, job)
            if turn is None:
                await db.rollback()
                return Outcome.STALE
            turn.grading_status = GRADING_FAILED
            turn.grading_error = code
            turn.lease_expires_at = None
            try:
                staged = await stage_append(
                    db,
                    session_id=job.session_id,
                    user_id=job.user_id,
                    event_type=EventType.GRADING_FAILED,
                    payload={
                        "turn_id": str(job.turn_id),
                        "error_code": code,
                        "attempts": job.attempt,
                        "retries_exhausted": retries_exhausted,
                    },
                )
            except (IllegalTransition, EventLogCorrupted) as exc:
                # The log cannot take the event (e.g. it is corrupted). The turn
                # row still records the failure, which is the part a client reads.
                log.error(
                    "GRADING_FAILED could not be appended; turn marked failed only",
                    extra={"grading.turn_id": str(job.turn_id), "grading.exception": repr(exc)},
                )
                await db.commit()
                return Outcome.FAILED
        except BaseException:
            await db.rollback()
            raise
        await commit_and_cache(db, cache, staged)
    return Outcome.FAILED


def _ability_after(
    ability: Any, blueprint_dict: dict[str, Any], job: GradingJob, result: GradeResult
) -> tuple[dict[str, Any], float] | None:
    """The subtopic's ability after this grade and the theta step, via Day 11's
    update - or None when the question carries no IRT parameters to update with.

    A follow-up answer uses ``STAKE_GENERATED_ITEM`` (0.5): "a probe is a hint,
    so succeeding after one is weaker evidence than succeeding unprompted"
    (plan section 8.5).
    """
    if job.subtopic_key is None or job.difficulty_b is None:
        return None
    before: AbilityState | None = ability.get(job.subtopic_key)
    if before is None:
        before = Blueprint.from_dict(blueprint_dict).prior_ability()
    update = update_ability(
        before,
        difficulty=job.difficulty_b,
        score=result.score,
        discrimination=job.discrimination_a or 1.0,
        grader_confidence=result.grader_confidence,
        stake=STAKE_GENERATED_ITEM if job.is_follow_up else STAKE_BANK_ITEM,
    )
    snapshot = {
        "subtopic_key": job.subtopic_key,
        "theta": update.after.theta,
        "rd": update.after.rd,
        "n_observations": update.after.n_observations,
    }
    return snapshot, update.delta_theta


def charge_for(result: GradeResult) -> tuple[Decimal, int]:
    """``(cost_usd, tokens)`` of one grade: every chokepoint call it made.

    Reasoning tokens are already inside ``output_tokens`` (``app.llm.pricing``),
    so they are not added again.  Cache hits report what the chokepoint reports.
    """
    cost = sum((call.cost_usd for call in result.calls), Decimal(0))
    tokens = sum(call.input_tokens + call.output_tokens for call in result.calls)
    return cost, tokens


def _grade_record(result: GradeResult) -> dict[str, Any]:
    """What is kept on the turn: the evidence for the number, never the prompt."""
    return {
        "score": result.score,
        "judgment": result.judgment.model_dump(mode="json"),
        "breakdown": dataclasses.asdict(result.breakdown),
        "gate": dataclasses.asdict(result.gate),
        "calls": [
            {
                "task": str(call.task),
                "prompt_version": call.prompt_version,
                "model": call.model,
                "input_tokens": call.input_tokens,
                "output_tokens": call.output_tokens,
                "cost_usd": str(call.cost_usd),
                "cache_hit": call.cache_hit,
            }
            for call in result.calls
        ],
    }


# ---------------------------------------------------------------------------
# Operations
# ---------------------------------------------------------------------------


async def requeue_failed_turn(
    maker: async_sessionmaker[AsyncSession], *, session_id: uuid.UUID, turn_id: uuid.UUID
) -> bool:
    """Put a ``failed`` turn back in the queue with a fresh attempt budget.

    An operator action, for after the cause (an outage, a bad key) is fixed.
    The session is still in ``GRADING`` with the turn ungraded, so a later
    ``ANSWER_GRADED`` is legal.  Returns False if the turn is not failed.
    """
    async with maker() as db, db.begin():
        result = await db.execute(
            sa.update(Turn)
            .where(
                Turn.session_id == session_id,
                Turn.turn_id == turn_id,
                Turn.grading_status == GRADING_FAILED,
            )
            .values(
                grading_status=GRADING_PENDING,
                grading_attempts=0,
                next_attempt_at=None,
                lease_expires_at=None,
            )
        )
        return bool(result.rowcount)  # type: ignore[attr-defined]


async def run_once(
    maker: async_sessionmaker[AsyncSession],
    cache: SessionStateCache | None,
    *,
    grade_fn: GradeFn,
    policy: RetryPolicy,
) -> str | None:
    """Claim and process at most one job.  None when the queue is empty."""
    job = await claim_next(maker, policy)
    if job is None:
        return None
    return await process_job(maker, cache, job, grade_fn=grade_fn, policy=policy)


async def run_worker(
    maker: async_sessionmaker[AsyncSession],
    cache: SessionStateCache | None,
    *,
    grade_fn: GradeFn,
    policy: RetryPolicy,
    poll_interval_s: float,
    stop: asyncio.Event,
    follow_ups: Callable[[], Awaitable[str | None]] | None = None,
) -> None:
    """Drain the queue until ``stop`` is set.  Never dies on one bad job.

    ``follow_ups`` (Day 23) claims and processes one follow-up wording job per
    iteration - ``app.interview.probing.run_once_follow_up`` bound to its
    arguments.  Passed in rather than imported, to keep the two modules free
    of an import cycle.
    """
    log.info("grading worker started")
    while not stop.is_set():
        outcome: str | None = None
        try:
            outcome = await run_once(maker, cache, grade_fn=grade_fn, policy=policy)
            if follow_ups is not None:
                outcome = (await follow_ups()) or outcome
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - a database blip must not kill the worker
            log.exception("grading worker iteration failed; backing off")
            outcome = None
        if outcome is None:
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(stop.wait(), timeout=poll_interval_s)
    log.info("grading worker stopped")


__all__ = [
    "GradeFn",
    "GradingJob",
    "Outcome",
    "RetryPolicy",
    "charge_for",
    "claim_next",
    "classify_error",
    "grade_with_pipeline",
    "process_job",
    "requeue_failed_turn",
    "run_once",
    "run_worker",
]
