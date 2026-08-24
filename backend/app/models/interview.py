"""Interview sessions, their event log, and the turns inside them."""

from __future__ import annotations

import datetime as dt
import decimal
import uuid
from typing import Any

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, REAL, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, utc_now_column

# The FSM from plan section 13.4. Stored as text rather than a Postgres ENUM:
# adding a value to an ENUM is a migration, and the state machine will still be
# moving during Phase 5.
SESSION_STATE_CREATED = "CREATED"
SESSION_STATE_COMPLETED = "COMPLETED"


class InterviewSession(Base):
    __tablename__ = "interview_sessions"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        server_default=sa.text("gen_random_uuid()"),
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    # resume_id / jd_id arrive in Phase 5 with the tables they reference.

    # Topic quotas, item budget, time budget - computed once at session creation
    # by the blueprint builder (plan section 3, Day 15).
    blueprint: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    state: Mapped[str] = mapped_column(
        sa.Text, nullable=False, server_default=sa.literal(SESSION_STATE_CREATED)
    )

    started_at: Mapped[dt.datetime] = utc_now_column()
    ended_at: Mapped[dt.datetime | None] = mapped_column(sa.DateTime(timezone=True), nullable=True)

    # NUMERIC, not float: this is money-adjacent and gets summed. Binary floats
    # accumulate error under addition; NUMERIC is exact decimal arithmetic.
    cost_usd: Mapped[decimal.Decimal] = mapped_column(
        sa.Numeric(10, 6), nullable=False, server_default=sa.text("0")
    )
    total_tokens: Mapped[int] = mapped_column(
        sa.Integer, nullable=False, server_default=sa.text("0")
    )
    # True when the spend circuit breaker degraded this session (plan section 4.7).
    degraded: Mapped[bool] = mapped_column(sa.Boolean, nullable=False, server_default=sa.false())

    __table_args__ = (
        sa.Index("ix_interview_sessions_user_id_started_at", "user_id", "started_at"),
        # Partial index (plan section 13.7). In-flight sessions are a tiny
        # fraction of all rows, and "what am I currently in the middle of?" is a
        # hot query; indexing only the unfinished rows keeps it small.
        # The blueprint says WHERE state='ACTIVE'; the FSM has no literal ACTIVE
        # state, so the equivalent predicate is "not finished".
        sa.Index(
            "ix_interview_sessions_user_id_active",
            "user_id",
            postgresql_where=sa.column("state") != SESSION_STATE_COMPLETED,
        ),
    )


class InterviewEvent(Base):
    """The append-only log that IS the session (plan section 11.2).

    Session state is ``fold(events)``. Rows are never updated or deleted, which
    is what makes replay, deterministic re-grading, and crash recovery possible.
    """

    __tablename__ = "interview_events"

    # BIGSERIAL: a global insertion order across all sessions, useful for
    # debugging. `seq` below is the per-session ordering that actually matters.
    id: Mapped[int] = mapped_column(sa.BigInteger, primary_key=True, autoincrement=True)
    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        sa.ForeignKey("interview_sessions.id", ondelete="CASCADE"),
        nullable=False,
    )
    seq: Mapped[int] = mapped_column(sa.Integer, nullable=False)
    type: Mapped[str] = mapped_column(sa.Text, nullable=False)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[dt.datetime] = utc_now_column()

    __table_args__ = (
        # Both a correctness constraint (no two events claim the same position)
        # and the index that serves the only read pattern there is: "every event
        # for one session, in order" - a single ordered range scan.
        sa.UniqueConstraint("session_id", "seq", name="uq_interview_events_session_id_seq"),
    )


class Turn(Base):
    """One question-and-answer exchange."""

    __tablename__ = "turns"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        server_default=sa.text("gen_random_uuid()"),
    )
    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        sa.ForeignKey("interview_sessions.id", ondelete="CASCADE"),
        nullable=False,
    )
    # Supplied by the CLIENT, not generated here. That is the whole point: a
    # retried or double-clicked submission carries the same turn_id, the unique
    # constraint below rejects the duplicate, and the user is graded (and
    # charged) exactly once. Same idempotency pattern as billing_events.
    turn_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)

    question_id: Mapped[str | None] = mapped_column(
        sa.Text, sa.ForeignKey("questions.id", ondelete="SET NULL"), nullable=True
    )
    # The rendered wording is snapshotted, because the bank is versioned and an
    # item may be edited later. A report must show what was actually asked.
    question_text: Mapped[str] = mapped_column(sa.Text, nullable=False)
    answer_text: Mapped[str | None] = mapped_column(sa.Text, nullable=True)
    is_follow_up: Mapped[bool] = mapped_column(
        sa.Boolean, nullable=False, server_default=sa.false()
    )

    grade: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    score: Mapped[float | None] = mapped_column(REAL, nullable=True)
    grader_confidence: Mapped[float | None] = mapped_column(REAL, nullable=True)
    latency_ms: Mapped[int | None] = mapped_column(sa.Integer, nullable=True)
    created_at: Mapped[dt.datetime] = utc_now_column()

    __table_args__ = (
        sa.UniqueConstraint("session_id", "turn_id", name="uq_turns_session_id_turn_id"),
    )
