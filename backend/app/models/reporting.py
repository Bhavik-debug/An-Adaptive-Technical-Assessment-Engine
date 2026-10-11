"""The interview report (Day 25, migration 0007).

Plan section 13.7: ``reports(id, session_id fk unique, summary jsonb, narrative
text, generated_at)``.  ``summary`` is ``stats`` here (the deterministic
aggregation) and the narrative is structured JSON with its provenance.

``session_id`` is UNIQUE: one report per session, however many times the report
is requested, retried or raced for.  The row is also the durable report job
(same lifecycle columns as ``turns``, ``follow_ups``, ``resumes``), created in
the same transaction that enters ``REPORTING``.
"""

from __future__ import annotations

import datetime as dt
import decimal
import uuid
from typing import Any

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, utc_now_column

REPORT_PENDING = "pending"
REPORT_PROCESSING = "processing"
REPORT_READY = "ready"
REPORT_FAILED = "failed"


class Report(Base):
    __tablename__ = "reports"

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
    status: Mapped[str] = mapped_column(
        sa.Text, nullable=False, server_default=sa.literal(REPORT_PENDING)
    )
    attempts: Mapped[int] = mapped_column(sa.Integer, nullable=False, server_default=sa.text("0"))
    next_attempt_at: Mapped[dt.datetime | None] = mapped_column(
        sa.DateTime(timezone=True), nullable=True
    )
    lease_expires_at: Mapped[dt.datetime | None] = mapped_column(
        sa.DateTime(timezone=True), nullable=True
    )
    #: A short machine code when the *report* (not just its prose) failed.
    error: Mapped[str | None] = mapped_column(sa.Text, nullable=True)
    #: The deterministic statistics (app.reporting.stats), and the log seq
    #: they were computed at.
    stats: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    stats_seq: Mapped[int | None] = mapped_column(sa.Integer, nullable=True)
    #: {summary, strengths, growth_areas, next_steps}
    narrative: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    #: "model" or "template" (the plan's cut-line) - never ambiguous.
    narrative_source: Mapped[str | None] = mapped_column(sa.Text, nullable=True)
    #: Why the model narrative was not used, when it was not.
    narrative_error: Mapped[str | None] = mapped_column(sa.Text, nullable=True)
    #: What the narrative call cost (also added once to the session's totals).
    cost_usd: Mapped[decimal.Decimal] = mapped_column(
        sa.Numeric(10, 6), nullable=False, server_default=sa.text("0")
    )
    total_tokens: Mapped[int] = mapped_column(
        sa.Integer, nullable=False, server_default=sa.text("0")
    )
    created_at: Mapped[dt.datetime] = utc_now_column()
    generated_at: Mapped[dt.datetime | None] = mapped_column(
        sa.DateTime(timezone=True), nullable=True
    )

    __table_args__ = (
        sa.UniqueConstraint("session_id", name="uq_reports_session_id"),
        sa.CheckConstraint(
            "status IN ('pending', 'processing', 'ready', 'failed')", name="status_valid"
        ),
        sa.CheckConstraint("attempts >= 0", name="attempts_non_negative"),
        sa.CheckConstraint(
            "narrative_source IS NULL OR narrative_source IN ('model', 'template')",
            name="narrative_source_valid",
        ),
        sa.CheckConstraint(
            "status <> 'ready' OR (stats IS NOT NULL AND narrative IS NOT NULL)",
            name="ready_has_content",
        ),
        sa.Index(
            "ix_reports_claimable",
            "status",
            "next_attempt_at",
            postgresql_where=sa.text("status IN ('pending', 'processing')"),
        ),
    )
