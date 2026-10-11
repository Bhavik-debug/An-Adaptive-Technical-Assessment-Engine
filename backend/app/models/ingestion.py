"""Resumes and job descriptions (Day 24, migration 0006).

Plan section 13.7 sketches ``resumes(storage_key, raw_text, profile, ...)`` and
``job_descriptions(raw_text, requirements, ...)``.  Two deliberate departures,
both in the direction of keeping less:

* **No file and no raw text are stored.**  Only the *redacted* extracted text
  (``app.privacy``) is kept - there is no ``storage_key`` because there is no
  blob, and no ``raw_text`` because the identifiers in it are never needed again.
* **Suggestion and profile are separate columns.**  ``extracted`` is what the
  parser suggested; ``profile`` is what the candidate confirmed or typed.  The
  parser writes only ``extracted`` - never ``profile`` - so a later parse can
  never overwrite a candidate's edit, and only a confirmed profile seeds priors.

Each row is also its own durable extraction job (``status``, ``attempts``,
leases), exactly like ``turns`` (Day 22) and ``follow_ups`` (Day 23).
"""

from __future__ import annotations

import datetime as dt
import uuid
from typing import Any

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, utc_now_column

INGEST_PENDING = "pending"
INGEST_PROCESSING = "processing"
INGEST_READY = "ready"
INGEST_FAILED = "failed"

_STATUS_CHECK = "status IN ('pending', 'processing', 'ready', 'failed')"


class _IngestionJob:
    """Columns shared by both tables: the job lifecycle and the redacted text."""

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        server_default=sa.text("gen_random_uuid()"),
    )
    status: Mapped[str] = mapped_column(
        sa.Text, nullable=False, server_default=sa.literal(INGEST_PENDING)
    )
    attempts: Mapped[int] = mapped_column(sa.Integer, nullable=False, server_default=sa.text("0"))
    next_attempt_at: Mapped[dt.datetime | None] = mapped_column(
        sa.DateTime(timezone=True), nullable=True
    )
    lease_expires_at: Mapped[dt.datetime | None] = mapped_column(
        sa.DateTime(timezone=True), nullable=True
    )
    #: A short machine code, never a provider message or document content.
    error: Mapped[str | None] = mapped_column(sa.Text, nullable=True)
    #: The document text after PII redaction. The only copy of it that exists.
    redacted_text: Mapped[str] = mapped_column(sa.Text, nullable=False)
    #: ``{"email": 1, "phone": 2, ...}`` - counts only, never the values.
    redaction_counts: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")
    )
    #: The verified parser suggestion (app.ingestion.extraction), or NULL.
    extracted: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    parser_version: Mapped[str | None] = mapped_column(sa.Text, nullable=True)
    parsed_at: Mapped[dt.datetime | None] = mapped_column(sa.DateTime(timezone=True), nullable=True)
    created_at: Mapped[dt.datetime] = utc_now_column()


class Resume(_IngestionJob, Base):
    __tablename__ = "resumes"

    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    document_kind: Mapped[str] = mapped_column(sa.Text, nullable=False)
    page_count: Mapped[int | None] = mapped_column(sa.Integer, nullable=True)
    truncated: Mapped[bool] = mapped_column(sa.Boolean, nullable=False, server_default=sa.false())
    #: SHA-256 of the uploaded bytes - lets a client recognise a re-upload
    #: without the file being kept.
    source_sha256: Mapped[str] = mapped_column(sa.Text, nullable=False)
    #: What the candidate confirmed or typed (PATCH .../profile). Never written
    #: by the parser. NULL until the candidate acts.
    profile: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    profile_updated_at: Mapped[dt.datetime | None] = mapped_column(
        sa.DateTime(timezone=True), nullable=True
    )

    __table_args__ = (
        sa.CheckConstraint(_STATUS_CHECK, name="status_valid"),
        sa.CheckConstraint("attempts >= 0", name="attempts_non_negative"),
        sa.CheckConstraint("document_kind IN ('pdf', 'text')", name="document_kind_valid"),
        sa.Index("ix_resumes_user_id_created_at", "user_id", "created_at"),
        sa.Index(
            "ix_resumes_claimable",
            "status",
            "next_attempt_at",
            postgresql_where=sa.text("status IN ('pending', 'processing')"),
        ),
    )


class JobDescription(_IngestionJob, Base):
    __tablename__ = "job_descriptions"

    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )

    __table_args__ = (
        sa.CheckConstraint(_STATUS_CHECK, name="status_valid"),
        sa.CheckConstraint("attempts >= 0", name="attempts_non_negative"),
        sa.Index("ix_job_descriptions_user_id_created_at", "user_id", "created_at"),
        sa.Index(
            "ix_job_descriptions_claimable",
            "status",
            "next_attempt_at",
            postgresql_where=sa.text("status IN ('pending', 'processing')"),
        ),
    )
