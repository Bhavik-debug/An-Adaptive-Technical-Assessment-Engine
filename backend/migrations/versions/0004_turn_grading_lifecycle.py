"""Turns carry their own grading lifecycle; questions keep their misconceptions.

Day 22.

**1. ``turns`` becomes the durable grading queue.** The repository has no worker
or queue technology (plan §13.1 names arq; nothing installed it). Rather than add
one, the turn row itself records where its grading stands, and a worker claims
rows with ``FOR UPDATE SKIP LOCKED``. Postgres is already the source of truth, so
an accepted submission is durable the moment its transaction commits - an API
crash cannot lose it, because nothing about it lives only in memory.

* ``grading_status``  pending | processing | graded | failed  (CHECK-constrained)
* ``grading_attempts`` how many times a worker has claimed it (the retry bound and
  the fencing token: a worker may only finish the attempt it claimed)
* ``next_attempt_at``  when a pending row becomes claimable (retry backoff)
* ``lease_expires_at`` when a processing row is presumed abandoned (crash recovery)
* ``grading_error``    a short machine code, never a provider message
* ``graded_at``        when the grade committed

A partial index covers only the rows a worker can ever claim, so the claim query
stays an index scan however large ``turns`` grows.

Existing rows: there are none in any environment this project has run (verified
before writing this), but the migration must still be correct if there were.
They are backfilled as ``graded`` if they carry a score and ``failed`` with
``grading_error = 'legacy_row'`` otherwise, so no pre-Day-22 row is ever picked
up and graded by surprise.

**2. ``questions.common_misconceptions``.** A pre-existing Phase-2 defect found
on Day 22: the bank schema has ``common_misconceptions`` and the grader's Pass A
consumes them (plan §7.2), but ingest never wrote them, so a grader reading the
database would silently grade with less than the evaluation harness does.
NULLable; the next ``ingest_question_bank.py`` run fills it (ingest upserts every
column it writes).

Revision ID: 0004
Revises: 0003
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import ARRAY

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "turns",
        sa.Column("grading_status", sa.Text(), nullable=False, server_default=sa.text("'pending'")),
    )
    op.add_column(
        "turns",
        sa.Column("grading_attempts", sa.Integer(), nullable=False, server_default=sa.text("0")),
    )
    op.add_column("turns", sa.Column("next_attempt_at", sa.DateTime(timezone=True)))
    op.add_column("turns", sa.Column("lease_expires_at", sa.DateTime(timezone=True)))
    op.add_column("turns", sa.Column("grading_error", sa.Text()))
    op.add_column("turns", sa.Column("graded_at", sa.DateTime(timezone=True)))

    op.execute(
        "UPDATE turns SET grading_status = CASE WHEN score IS NOT NULL "
        "THEN 'graded' ELSE 'failed' END, "
        "grading_error = CASE WHEN score IS NOT NULL THEN NULL ELSE 'legacy_row' END"
    )

    op.create_check_constraint(
        op.f("ck_turns_grading_status_valid"),
        "turns",
        "grading_status IN ('pending', 'processing', 'graded', 'failed')",
    )
    op.create_check_constraint(
        op.f("ck_turns_grading_attempts_non_negative"), "turns", "grading_attempts >= 0"
    )
    op.create_index(
        "ix_turns_claimable",
        "turns",
        ["grading_status", "next_attempt_at"],
        postgresql_where=sa.text("grading_status IN ('pending', 'processing')"),
    )

    op.add_column("questions", sa.Column("common_misconceptions", ARRAY(sa.Text())))


def downgrade() -> None:
    op.drop_column("questions", "common_misconceptions")
    op.drop_index(
        "ix_turns_claimable",
        table_name="turns",
        postgresql_where=sa.text("grading_status IN ('pending', 'processing')"),
    )
    op.drop_constraint(op.f("ck_turns_grading_attempts_non_negative"), "turns", type_="check")
    op.drop_constraint(op.f("ck_turns_grading_status_valid"), "turns", type_="check")
    for column in (
        "graded_at",
        "grading_error",
        "lease_expires_at",
        "next_attempt_at",
        "grading_attempts",
        "grading_status",
    ):
        op.drop_column("turns", column)
