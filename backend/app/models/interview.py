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

# The FSM from plan section 13.4, defined in full in app/interview/fsm.py
# (InterviewState); these two constants predate it and match its values.
# Stored as text rather than a Postgres ENUM: adding a value to an ENUM is a
# migration, and the state machine will still be moving during Phase 5.
SESSION_STATE_CREATED = "CREATED"
SESSION_STATE_COMPLETED = "COMPLETED"

# Where a turn's grading stands (Day 22). Not FSM states: the session is in
# GRADING throughout; these describe the background job for one turn.
GRADING_PENDING = "pending"
GRADING_PROCESSING = "processing"
GRADING_GRADED = "graded"
GRADING_FAILED = "failed"


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
    # Day 24 (migration 0006): the resume and job description this session was
    # created with. SET NULL on delete: priors were already recorded in the
    # event log at /start, so deleting a resume rewrites nothing.
    resume_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), sa.ForeignKey("resumes.id", ondelete="SET NULL"), nullable=True
    )
    jd_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), sa.ForeignKey("job_descriptions.id", ondelete="SET NULL"), nullable=True
    )

    # Topic quotas, item budget, time budget - computed once at session creation
    # by the blueprint builder (plan section 3, Day 15). The shape written here
    # is `app.blueprint.Blueprint.to_dict()`, and `Blueprint.from_dict()` reads
    # it back and re-validates. Writing the row belongs to the session layer,
    # which is Phase 5.
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
        # Day 21 (migration 0003): seq counts from 0 - SESSION_CREATED is seq 0.
        # The same migration installs a trigger that rejects UPDATE and direct
        # DELETE on this table; triggers are not part of SQLAlchemy metadata, so
        # it is documented here rather than declared. See app/interview/store.py.
        sa.CheckConstraint("seq >= 0", name="seq_non_negative"),
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

    # --- The grading lifecycle (Day 22, migration 0004) ---------------------
    # The turn row is the durable grading queue; see app/interview/grading.py.
    grading_status: Mapped[str] = mapped_column(
        sa.Text, nullable=False, server_default=sa.literal(GRADING_PENDING)
    )
    grading_attempts: Mapped[int] = mapped_column(
        sa.Integer, nullable=False, server_default=sa.text("0")
    )
    next_attempt_at: Mapped[dt.datetime | None] = mapped_column(
        sa.DateTime(timezone=True), nullable=True
    )
    lease_expires_at: Mapped[dt.datetime | None] = mapped_column(
        sa.DateTime(timezone=True), nullable=True
    )
    # A short machine code (app.interview.grading.classify_error), never a
    # provider message - those can echo the prompt, and the prompt holds the answer.
    grading_error: Mapped[str | None] = mapped_column(sa.Text, nullable=True)
    graded_at: Mapped[dt.datetime | None] = mapped_column(sa.DateTime(timezone=True), nullable=True)
    # Day 23 (migration 0005): the follow-up policy's trace for this grade -
    # every rule's status and the numbers it saw. Keys and numbers only.
    policy_decision: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)

    __table_args__ = (
        sa.UniqueConstraint("session_id", "turn_id", name="uq_turns_session_id_turn_id"),
        sa.CheckConstraint(
            "grading_status IN ('pending', 'processing', 'graded', 'failed')",
            name="grading_status_valid",
        ),
        sa.CheckConstraint("grading_attempts >= 0", name="grading_attempts_non_negative"),
        # Only rows a worker can ever claim. See migration 0004.
        sa.Index(
            "ix_turns_claimable",
            "grading_status",
            "next_attempt_at",
            postgresql_where=sa.text("grading_status IN ('pending', 'processing')"),
        ),
    )


FOLLOW_UP_PENDING = "pending"
FOLLOW_UP_PROCESSING = "processing"
FOLLOW_UP_READY = "ready"
FOLLOW_UP_ABANDONED = "abandoned"


class FollowUp(Base):
    """One decided follow-up and the durable job that words it (Day 23).

    Created ``pending`` in the same transaction as the grade that triggered it;
    finished by the worker in ``app.interview.probing``. See migration 0005 for
    why the uniques make duplicates and the per-item cap impossible here too.
    """

    __tablename__ = "follow_ups"

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
    #: The client turn_id whose grade triggered this follow-up.
    source_turn_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    #: The bank item being probed - the parent, never the follow-up itself.
    item_question_id: Mapped[str | None] = mapped_column(
        sa.Text, sa.ForeignKey("questions.id", ondelete="SET NULL"), nullable=True
    )
    ordinal: Mapped[int] = mapped_column(sa.Integer, nullable=False)
    reason: Mapped[str] = mapped_column(sa.Text, nullable=False)
    concept_key: Mapped[str | None] = mapped_column(sa.Text, nullable=True)
    status: Mapped[str] = mapped_column(
        sa.Text, nullable=False, server_default=sa.literal(FOLLOW_UP_PENDING)
    )
    attempts: Mapped[int] = mapped_column(sa.Integer, nullable=False, server_default=sa.text("0"))
    next_attempt_at: Mapped[dt.datetime | None] = mapped_column(
        sa.DateTime(timezone=True), nullable=True
    )
    lease_expires_at: Mapped[dt.datetime | None] = mapped_column(
        sa.DateTime(timezone=True), nullable=True
    )
    #: A short machine code, never a provider message.
    error: Mapped[str | None] = mapped_column(sa.Text, nullable=True)
    #: The generated probe, once ready.
    text: Mapped[str | None] = mapped_column(sa.Text, nullable=True)
    created_at: Mapped[dt.datetime] = utc_now_column()
    generated_at: Mapped[dt.datetime | None] = mapped_column(
        sa.DateTime(timezone=True), nullable=True
    )

    __table_args__ = (
        sa.CheckConstraint(
            "status IN ('pending', 'processing', 'ready', 'abandoned')", name="status_valid"
        ),
        sa.CheckConstraint("ordinal IN (1, 2)", name="ordinal_within_cap"),
        sa.CheckConstraint("attempts >= 0", name="attempts_non_negative"),
        sa.UniqueConstraint(
            "session_id", "source_turn_id", name="uq_follow_ups_session_id_source_turn_id"
        ),
        sa.UniqueConstraint(
            "session_id",
            "item_question_id",
            "ordinal",
            name="uq_follow_ups_session_id_item_question_id_ordinal",
        ),
        sa.Index(
            "ix_follow_ups_claimable",
            "status",
            "next_attempt_at",
            postgresql_where=sa.text("status IN ('pending', 'processing')"),
        ),
    )
