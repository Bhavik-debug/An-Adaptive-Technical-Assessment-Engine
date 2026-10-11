"""Sessions: create, start (Day 23); submit, status, stream (Day 22).

Plan §13.8::

    POST /api/sessions/{id}/turns     -> 202 {turn_id}   [idempotent on turn_id]
    GET  /api/sessions/{id}/stream    -> SSE: state transitions, not tokens

plus ``GET /api/sessions/{id}/turns/{turn_id}``, the poll-able twin of the stream:
a client that cannot hold an SSE connection still has a documented way to learn
whether its answer was graded.

Every route resolves the caller from the bearer token and passes that user id
down; ownership is enforced in SQL (``app.interview.store``), and "not yours" is
the same 404 as "does not exist".  Business rules live in ``app.interview``;
this module only translates them to HTTP.
"""

from __future__ import annotations

import datetime as dt
import logging
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache
from typing import Annotated, Literal

import sqlalchemy as sa
from fastapi import APIRouter, Depends, Header, HTTPException, Request, Response, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.bank.taxonomy import Taxonomy, load_taxonomy
from app.blueprint import AssessmentSpec, BlueprintError, TopicRequest, build_blueprint
from app.db import get_sessionmaker
from app.deps import AppSettings, CurrentUser, SessionCache
from app.grading.grader import MAX_ANSWER_CHARS
from app.interview.events import InvalidEventPayload
from app.interview.flow import NoEligibleQuestion
from app.interview.fsm import IllegalTransition
from app.interview.lifecycle import start_session
from app.interview.service import create_session
from app.interview.state import EventLogCorrupted, SessionState
from app.interview.store import SeqConflict, SessionNotFound, head_seq
from app.interview.stream import event_stream, parse_last_event_id
from app.interview.turns import (
    FollowUpTurnsUnsupported,
    TurnIdConflict,
    TurnNotFound,
    TurnRecord,
    get_turn,
    submit_turn,
)
from app.models.ingestion import INGEST_READY, JobDescription, Resume
from app.models.interview import GRADING_GRADED, InterviewSession
from app.models.reporting import REPORT_FAILED, REPORT_READY, Report
from app.reporting.jobs import ensure_report_job

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/sessions", tags=["sessions"])

Maker = Annotated[async_sessionmaker[AsyncSession], Depends(get_sessionmaker)]


# ---------------------------------------------------------------------------
# Session creation and start (Day 23)
# ---------------------------------------------------------------------------


class TopicIn(BaseModel):
    """One ``app.blueprint.TopicRequest``, as JSON.  Validated by the builder."""

    model_config = ConfigDict(extra="forbid")

    topic_key: str = Field(min_length=1, max_length=64)
    jd_weight: float = 1.0
    subtopic_keys: list[str] = Field(default_factory=list, max_length=32)
    required: bool = True
    min_items: int = 0
    max_items: int | None = None


class SessionCreate(BaseModel):
    """``POST /api/sessions`` - an ``app.blueprint.AssessmentSpec``, as JSON.

    Every rule (known level and mode, topics in the taxonomy, budgets that fit)
    is the blueprint builder's; this model only shapes the input.
    """

    model_config = ConfigDict(extra="forbid")

    role: str = Field(min_length=1, max_length=120)
    #: Optional when ``jd_id`` is given: the topics then come from the JD's
    #: code-computed topic weights. Explicit topics always win.
    topics: list[TopicIn] = Field(default_factory=list, max_length=20)
    #: Day 24. Both must be the caller's own. The resume's *confirmed* profile
    #: seeds capped priors at /start; an unconfirmed one seeds nothing.
    resume_id: uuid.UUID | None = None
    jd_id: uuid.UUID | None = None
    level: str | None = None
    mode: str | None = None
    duration_minutes: int | None = None
    item_budget: int | None = None


class SessionCreated(BaseModel):
    session_id: uuid.UUID
    state: str
    seq: int
    blueprint: dict[str, object]
    resume_id: uuid.UUID | None = None
    jd_id: uuid.UUID | None = None
    start_url: str
    stream_url: str


class QuestionOut(BaseModel):
    """What the candidate is asked.  ``question_id`` is the bank item; for a
    follow-up it is ``None`` (the probe has no bank id)."""

    question_id: str | None
    is_follow_up: bool
    text: str | None


class SessionStarted(BaseModel):
    session_id: uuid.UUID
    state: str
    seq: int
    question: QuestionOut
    turns_url: str
    stream_url: str


@lru_cache(maxsize=1)
def _taxonomy() -> Taxonomy:
    """The committed taxonomy, read once per process."""
    return load_taxonomy()


def _question(state: SessionState) -> QuestionOut:
    return QuestionOut(
        question_id=state.current_question_id,
        is_follow_up=state.current_is_follow_up,
        text=state.current_question_text,
    )


def _spec(body: SessionCreate, jd_topics: dict[str, float] | None = None) -> AssessmentSpec:
    optional = {
        key: value
        for key, value in (
            ("level", body.level),
            ("mode", body.mode),
            ("duration_minutes", body.duration_minutes),
            ("item_budget", body.item_budget),
        )
        if value is not None
    }
    return AssessmentSpec(
        role=body.role,
        topics=tuple(
            TopicRequest(
                topic_key=t.topic_key,
                jd_weight=t.jd_weight,
                subtopic_keys=tuple(t.subtopic_keys),
                required=t.required,
                min_items=t.min_items,
                max_items=t.max_items,
            )
            for t in body.topics
        )
        if body.topics
        else tuple(
            TopicRequest(topic_key=key, jd_weight=weight)
            for key, weight in sorted((jd_topics or {}).items())
        ),
        **optional,  # type: ignore[arg-type]
    )


class TurnSubmission(BaseModel):
    """``POST /api/sessions/{id}/turns``."""

    model_config = ConfigDict(extra="forbid")

    #: Generated by the client, once per answer, and re-sent unchanged on retry.
    turn_id: uuid.UUID
    answer: str = Field(min_length=1, max_length=MAX_ANSWER_CHARS)

    @field_validator("answer")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("answer must not be blank")
        return v


GradingStatus = Literal["pending", "processing", "graded", "failed"]


class TurnStatus(BaseModel):
    """Where one turn stands.  ``score`` only when graded, ``error_code`` only
    when failed (or, while pending, the last transient error being retried)."""

    session_id: uuid.UUID
    turn_id: uuid.UUID
    question_id: str | None
    status: GradingStatus
    attempts: int
    score: float | None
    error_code: str | None
    submitted_at: dt.datetime
    graded_at: dt.datetime | None
    status_url: str
    stream_url: str


def _status(turn: TurnRecord) -> TurnStatus:
    base = f"/api/sessions/{turn.session_id}"
    graded = turn.grading_status == GRADING_GRADED
    return TurnStatus(
        session_id=turn.session_id,
        turn_id=turn.turn_id,
        question_id=turn.question_id,
        status=turn.grading_status,  # type: ignore[arg-type]  # CHECK-constrained
        attempts=turn.grading_attempts,
        score=turn.score if graded else None,
        error_code=None if graded else turn.grading_error,
        submitted_at=turn.created_at,
        graded_at=turn.graded_at,
        status_url=f"{base}/turns/{turn.turn_id}",
        stream_url=f"{base}/stream",
    )


@contextmanager
def _domain_errors() -> Iterator[None]:
    """The one mapping from ``app.interview`` errors to HTTP (docs/turn-api.md)."""
    try:
        yield
    except (SessionNotFound, TurnNotFound) as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Session not found") from exc
    except NoEligibleQuestion as exc:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            detail="No question in the bank fits this session's blueprint",
        ) from exc
    except TurnIdConflict as exc:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            detail="turn_id was already used for a different answer; generate a new turn_id",
        ) from exc
    except FollowUpTurnsUnsupported as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    except IllegalTransition as exc:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            detail=f"Not allowed in the session's current state ({exc})",
        ) from exc
    except SeqConflict as exc:
        raise HTTPException(
            status.HTTP_409_CONFLICT, detail="The session changed concurrently; retry"
        ) from exc
    except InvalidEventPayload as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    except EventLogCorrupted as exc:
        log.error("event log corrupted", extra={"error": str(exc)})
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Session state unavailable"
        ) from exc


@router.post(
    "",
    status_code=status.HTTP_201_CREATED,
    summary="Create a session from an assessment spec",
    responses={422: {"description": "Invalid spec (validated by the blueprint builder)"}},
)
async def create(
    body: SessionCreate, response: Response, user: CurrentUser, maker: Maker, cache: SessionCache
) -> SessionCreated:
    jd_topics: dict[str, float] | None = None
    async with maker() as db:
        # Day 24: the resume and JD must be the caller's (same 404 as missing).
        if body.resume_id is not None and not await db.scalar(
            sa.select(Resume.id).where(Resume.id == body.resume_id, Resume.user_id == user.id)
        ):
            raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Resume not found")
        if body.jd_id is not None:
            jd = await db.scalar(
                sa.select(JobDescription).where(
                    JobDescription.id == body.jd_id, JobDescription.user_id == user.id
                )
            )
            if jd is None:
                raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Job description not found")
            if not body.topics:
                if jd.status != INGEST_READY:
                    raise HTTPException(
                        status.HTTP_409_CONFLICT,
                        detail="The job description is not extracted yet; retry, or give topics",
                    )
                jd_topics = dict((jd.extracted or {}).get("topic_weights") or {})
    if not body.topics and not jd_topics:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Give topics, or a job description whose requirements map to topics",
        )
    try:
        # Against the committed taxonomy: an unknown topic is a 422 here, not an
        # empty question pool at /start.
        blueprint = build_blueprint(_spec(body, jd_topics), taxonomy=_taxonomy())
    except (BlueprintError, TypeError) as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    async with maker() as db:
        state = await create_session(
            db,
            cache,
            user_id=user.id,
            blueprint=blueprint,
            resume_id=body.resume_id,
            jd_id=body.jd_id,
        )
    base = f"/api/sessions/{state.session_id}"
    response.headers["Location"] = base
    return SessionCreated(
        session_id=state.session_id,
        state=str(state.state),
        seq=state.seq,
        blueprint=blueprint.to_dict(),
        resume_id=body.resume_id,
        jd_id=body.jd_id,
        start_url=f"{base}/start",
        stream_url=f"{base}/stream",
    )


@router.post(
    "/{session_id}/start",
    summary="Start the session and receive its first question",
    responses={
        404: {"description": "No such session for this user"},
        409: {"description": "Already past the first question, or nothing in the bank fits"},
    },
)
async def start(
    session_id: uuid.UUID, response: Response, user: CurrentUser, maker: Maker, cache: SessionCache
) -> SessionStarted:
    with _domain_errors():
        result = await start_session(maker, cache, session_id=session_id, user_id=user.id)
    if not result.started:
        response.headers["Idempotent-Replayed"] = "true"
    base = f"/api/sessions/{session_id}"
    return SessionStarted(
        session_id=session_id,
        state=str(result.state.state),
        seq=result.state.seq,
        question=_question(result.state),
        turns_url=f"{base}/turns",
        stream_url=f"{base}/stream",
    )


@router.post(
    "/{session_id}/turns",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit an answer (idempotent on turn_id)",
    responses={
        200: {"model": TurnStatus, "description": "A retry: the original turn"},
        404: {"description": "No such session for this user"},
        409: {"description": "turn_id reused for another answer, or not accepting answers"},
    },
)
async def submit(
    session_id: uuid.UUID,
    body: TurnSubmission,
    response: Response,
    user: CurrentUser,
    maker: Maker,
    cache: SessionCache,
) -> TurnStatus:
    with _domain_errors():
        submission = await submit_turn(
            maker,
            cache,
            session_id=session_id,
            user_id=user.id,
            turn_id=body.turn_id,
            answer_text=body.answer,
        )
    result = _status(submission.turn)
    response.headers["Location"] = result.status_url
    if not submission.created:
        # Same logical turn as before: report it, do not pretend it is new.
        response.status_code = status.HTTP_200_OK
        response.headers["Idempotent-Replayed"] = "true"
    return result


@router.get("/{session_id}/turns/{turn_id}", summary="Where one turn's grading stands")
async def turn_status(
    session_id: uuid.UUID, turn_id: uuid.UUID, user: CurrentUser, maker: Maker
) -> TurnStatus:
    with _domain_errors():
        async with maker() as db:
            turn = await get_turn(db, session_id=session_id, user_id=user.id, turn_id=turn_id)
    return _status(turn)


@router.get(
    "/{session_id}/stream",
    summary="Server-sent events: the session's state transitions",
    response_class=StreamingResponse,
)
async def stream(
    session_id: uuid.UUID,
    request: Request,
    user: CurrentUser,
    maker: Maker,
    settings: AppSettings,
    last_event_id: Annotated[str | None, Header(alias="Last-Event-ID")] = None,
) -> StreamingResponse:
    try:
        after = parse_last_event_id(last_event_id)
    except ValueError as exc:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST, detail="Last-Event-ID must be an integer seq"
        ) from exc
    # Checked before the response starts, so a stranger gets a real 404 rather
    # than a 200 followed by an empty stream.
    async with maker() as db:
        if await head_seq(db, session_id=session_id, user_id=user.id) is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Session not found")

    return StreamingResponse(
        event_stream(
            maker,
            session_id=session_id,
            user_id=user.id,
            last_event_id=after,
            poll_interval_s=settings.sse_poll_interval_s,
            keepalive_s=settings.sse_keepalive_s,
            is_disconnected=request.is_disconnected,
        ),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


__all__ = ["TurnStatus", "TurnSubmission", "router"]


# ---------------------------------------------------------------------------
# The report (Day 25)
# ---------------------------------------------------------------------------


class ReportOut(BaseModel):
    """``GET /api/sessions/{id}/report``.  ``stats`` are computed in code from
    the event log; ``narrative`` describes them and ``narrative_source`` says
    whether a model wrote it or it is the deterministic template."""

    report_id: uuid.UUID
    session_id: uuid.UUID
    status: str
    stats: dict[str, object] | None = None
    narrative: dict[str, object] | None = None
    narrative_source: str | None = None
    narrative_error: str | None = None
    error: str | None = None
    generated_at: dt.datetime | None = None


@router.get(
    "/{session_id}/report",
    summary="The interview report: 202 while it is being built, 200 when ready",
    responses={
        202: {"model": ReportOut, "description": "Being built"},
        404: {"description": "No such session for this user"},
        409: {"description": "The interview has not finished"},
        500: {"description": "Statistics could not be built; see `error`"},
    },
)
async def get_report(
    session_id: uuid.UUID, response: Response, user: CurrentUser, maker: Maker
) -> ReportOut:
    async with maker() as db:
        session = await db.scalar(
            sa.select(InterviewSession).where(
                InterviewSession.id == session_id, InterviewSession.user_id == user.id
            )
        )
        if session is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Session not found")
        if session.state not in ("REPORTING", "COMPLETED"):
            raise HTTPException(
                status.HTTP_409_CONFLICT, detail="The interview has not finished yet"
            )
        if session.state == "REPORTING":
            # Idempotent: a session that reached REPORTING before Day 25 (or by
            # any path that missed it) gets its job here, never a second one.
            await ensure_report_job(db, session_id)
            await db.commit()
        report = await db.scalar(sa.select(Report).where(Report.session_id == session_id))
    if report is None:  # pragma: no cover - ensured above for REPORTING
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Report missing")
    out = ReportOut(
        report_id=report.id,
        session_id=session_id,
        status=report.status,
        error=report.error,
        generated_at=report.generated_at,
    )
    if report.status == REPORT_READY:
        out.stats = report.stats
        out.narrative = report.narrative
        out.narrative_source = report.narrative_source
        out.narrative_error = report.narrative_error
        return out
    if report.status == REPORT_FAILED:
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        return out
    response.status_code = status.HTTP_202_ACCEPTED
    return out
