"""Follow-ups: one durable row per decided follow-up, and the decision on the turn.

Day 23.

**``follow_ups``** is to follow-up *wording* what ``turns`` became on Day 22 for
grading: the durable job.  The decision to follow up is taken in the same
transaction that commits the grade (``FOLLOW_UP_STARTED`` in the log), and that
transaction also inserts this row as ``pending``.  A worker claims it with
``FOR UPDATE SKIP LOCKED``, asks the model for the wording, and finishes it -
fenced on ``attempts`` exactly like grading - by appending the follow-up
``QUESTION_ASKED``.  A crash anywhere leaves a row a later worker resumes.

Constraints that make the caps and idempotency hold in the database itself, not
only in code:

* ``unique(session_id, source_turn_id)`` - one follow-up per graded turn, so a
  duplicate delivery of the same grade cannot create two;
* ``unique(session_id, item_question_id, ordinal)`` with ``ordinal IN (1, 2)`` -
  at most two follow-ups per bank item per session (plan section 8.5);
* ``status`` and ``attempts`` CHECKs, as on ``turns``.

The six-per-session cap is not expressible as a row constraint; it is an
invariant of the event log (``app.interview.state``), which every append is
validated against under the session lock.

**``turns.policy_decision``** stores the policy's trace for the grade - every
rule's status and the numbers it saw - so "why was there (no) follow-up?" is
answerable from the database.  Keys and numbers only, never answer text.

No existing rows are affected: ``follow_ups`` is new, and the turns column is
NULLable.

Revision ID: 0005
Revises: 0004
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "follow_ups",
        sa.Column(
            "id",
            UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(
            "session_id",
            UUID(as_uuid=True),
            sa.ForeignKey(
                "interview_sessions.id",
                ondelete="CASCADE",
                name=op.f("fk_follow_ups_session_id_interview_sessions"),
            ),
            nullable=False,
        ),
        sa.Column("source_turn_id", UUID(as_uuid=True), nullable=False),
        sa.Column(
            "item_question_id",
            sa.Text(),
            sa.ForeignKey(
                "questions.id",
                ondelete="SET NULL",
                name=op.f("fk_follow_ups_item_question_id_questions"),
            ),
            nullable=True,
        ),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("reason", sa.Text(), nullable=False),
        sa.Column("concept_key", sa.Text(), nullable=True),
        sa.Column("status", sa.Text(), nullable=False, server_default=sa.text("'pending'")),
        sa.Column("attempts", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("next_attempt_at", sa.DateTime(timezone=True)),
        sa.Column("lease_expires_at", sa.DateTime(timezone=True)),
        sa.Column("error", sa.Text()),
        sa.Column("text", sa.Text()),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column("generated_at", sa.DateTime(timezone=True)),
        sa.CheckConstraint(
            "status IN ('pending', 'processing', 'ready', 'abandoned')",
            name=op.f("ck_follow_ups_status_valid"),
        ),
        sa.CheckConstraint("ordinal IN (1, 2)", name=op.f("ck_follow_ups_ordinal_within_cap")),
        sa.CheckConstraint("attempts >= 0", name=op.f("ck_follow_ups_attempts_non_negative")),
        sa.UniqueConstraint(
            "session_id", "source_turn_id", name="uq_follow_ups_session_id_source_turn_id"
        ),
        sa.UniqueConstraint(
            "session_id",
            "item_question_id",
            "ordinal",
            name="uq_follow_ups_session_id_item_question_id_ordinal",
        ),
    )
    op.create_index(
        "ix_follow_ups_claimable",
        "follow_ups",
        ["status", "next_attempt_at"],
        postgresql_where=sa.text("status IN ('pending', 'processing')"),
    )
    op.add_column("turns", sa.Column("policy_decision", JSONB(), nullable=True))


def downgrade() -> None:
    op.drop_column("turns", "policy_decision")
    op.drop_index(
        "ix_follow_ups_claimable",
        table_name="follow_ups",
        postgresql_where=sa.text("status IN ('pending', 'processing')"),
    )
    op.drop_table("follow_ups")
