"""Reports: one per session, and the job that builds it.

Day 25. Plan section 13.7: ``reports(id uuid pk, session_id fk unique, summary
jsonb, narrative text, generated_at)``. ``summary`` is ``stats`` (the
deterministic aggregation) and the narrative is JSON with its provenance
(``narrative_source`` model | template). Same durable-job columns as 0004-0006.

Constraints that carry the guarantees:

* ``UNIQUE (session_id)`` - repeated, retried or concurrent report requests can
  never produce a second report for a session;
* ``ready`` implies ``stats`` and ``narrative`` are both present - a report
  cannot be marked done half-written;
* ``narrative_source`` is ``model`` or ``template``, so prose is never of
  unknown origin.

Additive; nothing existing changes.

Revision ID: 0007
Revises: 0006
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision = "0007"
down_revision = "0006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "reports",
        sa.Column(
            "id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")
        ),
        sa.Column(
            "session_id",
            UUID(as_uuid=True),
            sa.ForeignKey(
                "interview_sessions.id",
                ondelete="CASCADE",
                name=op.f("fk_reports_session_id_interview_sessions"),
            ),
            nullable=False,
        ),
        sa.Column("status", sa.Text(), nullable=False, server_default=sa.text("'pending'")),
        sa.Column("attempts", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("next_attempt_at", sa.DateTime(timezone=True)),
        sa.Column("lease_expires_at", sa.DateTime(timezone=True)),
        sa.Column("error", sa.Text()),
        sa.Column("stats", JSONB()),
        sa.Column("stats_seq", sa.Integer()),
        sa.Column("narrative", JSONB()),
        sa.Column("narrative_source", sa.Text()),
        sa.Column("narrative_error", sa.Text()),
        sa.Column("cost_usd", sa.Numeric(10, 6), nullable=False, server_default=sa.text("0")),
        sa.Column("total_tokens", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column("generated_at", sa.DateTime(timezone=True)),
        sa.UniqueConstraint("session_id", name="uq_reports_session_id"),
        sa.CheckConstraint(
            "status IN ('pending', 'processing', 'ready', 'failed')",
            name=op.f("ck_reports_status_valid"),
        ),
        sa.CheckConstraint("attempts >= 0", name=op.f("ck_reports_attempts_non_negative")),
        sa.CheckConstraint(
            "narrative_source IS NULL OR narrative_source IN ('model', 'template')",
            name=op.f("ck_reports_narrative_source_valid"),
        ),
        sa.CheckConstraint(
            "status <> 'ready' OR (stats IS NOT NULL AND narrative IS NOT NULL)",
            name=op.f("ck_reports_ready_has_content"),
        ),
    )
    op.create_index(
        "ix_reports_claimable",
        "reports",
        ["status", "next_attempt_at"],
        postgresql_where=sa.text("status IN ('pending', 'processing')"),
    )


def downgrade() -> None:
    op.drop_index(
        "ix_reports_claimable",
        table_name="reports",
        postgresql_where=sa.text("status IN ('pending', 'processing')"),
    )
    op.drop_table("reports")
