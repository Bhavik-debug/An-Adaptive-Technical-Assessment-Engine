"""Resumes, job descriptions, and the session's links to them.

Day 24. Plan section 13.7 lists ``resumes``, ``job_descriptions`` and
``interview_sessions.resume_id`` / ``jd_id`` (the latter deferred from Day 2,
"arrive in Phase 5 with the tables they reference").

Departures from the plan's sketch, each towards storing less (see
``app/models/ingestion.py``): no file blob and no raw text - only the
PII-redacted text; and a separate ``profile`` column for what the candidate
confirmed, which the parser never writes.

Each table carries the same durable-job columns as ``turns`` (0004) and
``follow_ups`` (0005), with the same CHECKs and a partial "claimable" index.

Safe on existing data: two new tables, and two NULLable FK columns on
``interview_sessions`` (no rewrite, no backfill).

Revision ID: 0006
Revises: 0005
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision = "0006"
down_revision = "0005"
branch_labels = None
depends_on = None

_STATUS = "status IN ('pending', 'processing', 'ready', 'failed')"


def _job_columns() -> list[sa.Column]:  # type: ignore[type-arg]
    return [
        sa.Column(
            "id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")
        ),
        sa.Column("status", sa.Text(), nullable=False, server_default=sa.text("'pending'")),
        sa.Column("attempts", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("next_attempt_at", sa.DateTime(timezone=True)),
        sa.Column("lease_expires_at", sa.DateTime(timezone=True)),
        sa.Column("error", sa.Text()),
        sa.Column("redacted_text", sa.Text(), nullable=False),
        sa.Column(
            "redaction_counts", JSONB(), nullable=False, server_default=sa.text("'{}'::jsonb")
        ),
        sa.Column("extracted", JSONB()),
        sa.Column("parser_version", sa.Text()),
        sa.Column("parsed_at", sa.DateTime(timezone=True)),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
    ]


def _user_fk(table: str) -> sa.Column:  # type: ignore[type-arg]
    return sa.Column(
        "user_id",
        UUID(as_uuid=True),
        sa.ForeignKey("users.id", ondelete="CASCADE", name=op.f(f"fk_{table}_user_id_users")),
        nullable=False,
    )


def upgrade() -> None:
    op.create_table(
        "resumes",
        *_job_columns(),
        _user_fk("resumes"),
        sa.Column("document_kind", sa.Text(), nullable=False),
        sa.Column("page_count", sa.Integer()),
        sa.Column("truncated", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("source_sha256", sa.Text(), nullable=False),
        sa.Column("profile", JSONB()),
        sa.Column("profile_updated_at", sa.DateTime(timezone=True)),
        sa.CheckConstraint(_STATUS, name=op.f("ck_resumes_status_valid")),
        sa.CheckConstraint("attempts >= 0", name=op.f("ck_resumes_attempts_non_negative")),
        sa.CheckConstraint(
            "document_kind IN ('pdf', 'text')", name=op.f("ck_resumes_document_kind_valid")
        ),
    )
    op.create_table(
        "job_descriptions",
        *_job_columns(),
        _user_fk("job_descriptions"),
        sa.CheckConstraint(_STATUS, name=op.f("ck_job_descriptions_status_valid")),
        sa.CheckConstraint("attempts >= 0", name=op.f("ck_job_descriptions_attempts_non_negative")),
    )
    for table in ("resumes", "job_descriptions"):
        op.create_index(f"ix_{table}_user_id_created_at", table, ["user_id", "created_at"])
        op.create_index(
            f"ix_{table}_claimable",
            table,
            ["status", "next_attempt_at"],
            postgresql_where=sa.text("status IN ('pending', 'processing')"),
        )
    op.add_column(
        "interview_sessions",
        sa.Column(
            "resume_id",
            UUID(as_uuid=True),
            sa.ForeignKey(
                "resumes.id",
                ondelete="SET NULL",
                name=op.f("fk_interview_sessions_resume_id_resumes"),
            ),
        ),
    )
    op.add_column(
        "interview_sessions",
        sa.Column(
            "jd_id",
            UUID(as_uuid=True),
            sa.ForeignKey(
                "job_descriptions.id",
                ondelete="SET NULL",
                name=op.f("fk_interview_sessions_jd_id_job_descriptions"),
            ),
        ),
    )


def downgrade() -> None:
    op.drop_column("interview_sessions", "jd_id")
    op.drop_column("interview_sessions", "resume_id")
    for table in ("job_descriptions", "resumes"):
        op.drop_index(
            f"ix_{table}_claimable",
            table_name=table,
            postgresql_where=sa.text("status IN ('pending', 'processing')"),
        )
        op.drop_index(f"ix_{table}_user_id_created_at", table_name=table)
        op.drop_table(table)
