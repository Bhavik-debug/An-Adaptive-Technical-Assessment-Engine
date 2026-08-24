"""Core tables: users, topics, questions, sessions, events, turns, skill states.

The seven tables named in the Phase 1 Day 2 plan line. Deliberately NOT here:
``resumes``, ``job_descriptions``, ``question_embeddings``, ``question_stats``,
``reports`` and the billing tables - each arrives in the phase that first needs
it, as a forward migration.

``interview_sessions.resume_id`` / ``jd_id`` are likewise deferred to Phase 5,
because a foreign key cannot point at a table that does not exist yet.

Extensions (``citext``, ``vector``) are NOT created here. They need superuser
rights and are a property of the database server rather than of the application
schema, so they live in ``infra/postgres/init/`` and run once when the database
is first created.

Revision ID: 0001
Revises:
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "topics",
        sa.Column("key", sa.Text(), nullable=False),
        sa.Column("parent_key", sa.Text(), nullable=True),
        sa.Column("display_name", sa.Text(), nullable=False),
        sa.Column("domain", sa.Text(), nullable=False),
        sa.ForeignKeyConstraint(
            ["parent_key"],
            ["topics.key"],
            name=op.f("fk_topics_parent_key_topics"),
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("key", name=op.f("pk_topics")),
    )
    op.create_index("ix_topics_domain", "topics", ["domain"], unique=False)
    op.create_table(
        "users",
        sa.Column("id", sa.UUID(), server_default=sa.text("gen_random_uuid()"), nullable=False),
        sa.Column("email", postgresql.CITEXT(), nullable=False),
        sa.Column("password_hash", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_users")),
        sa.UniqueConstraint("email", name=op.f("uq_users_email")),
    )
    op.create_table(
        "interview_sessions",
        sa.Column("id", sa.UUID(), server_default=sa.text("gen_random_uuid()"), nullable=False),
        sa.Column("user_id", sa.UUID(), nullable=False),
        sa.Column("blueprint", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("state", sa.Text(), server_default=sa.text("'CREATED'"), nullable=False),
        sa.Column(
            "started_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("ended_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "cost_usd",
            sa.Numeric(precision=10, scale=6),
            server_default=sa.text("0"),
            nullable=False,
        ),
        sa.Column("total_tokens", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("degraded", sa.Boolean(), server_default=sa.text("false"), nullable=False),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.id"],
            name=op.f("fk_interview_sessions_user_id_users"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_interview_sessions")),
    )
    op.create_index(
        "ix_interview_sessions_user_id_active",
        "interview_sessions",
        ["user_id"],
        unique=False,
        postgresql_where=sa.text("state != 'COMPLETED'"),
    )
    op.create_index(
        "ix_interview_sessions_user_id_started_at",
        "interview_sessions",
        ["user_id", "started_at"],
        unique=False,
    )
    op.create_table(
        "questions",
        sa.Column("id", sa.Text(), nullable=False),
        sa.Column("topic_key", sa.Text(), nullable=False),
        sa.Column("subtopic_key", sa.Text(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("difficulty_b", sa.REAL(), nullable=False),
        sa.Column("discrimination_a", sa.REAL(), server_default=sa.text("1.0"), nullable=False),
        sa.Column("expected_concepts", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("reference_answer", sa.Text(), nullable=False),
        sa.Column("follow_up_seeds", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("anchor_terms", postgresql.ARRAY(sa.Text()), nullable=True),
        sa.Column("time_estimate_s", sa.Integer(), nullable=False),
        sa.Column("tags", postgresql.ARRAY(sa.Text()), nullable=True),
        sa.Column("source", sa.Text(), nullable=False),
        sa.Column("version", sa.Integer(), server_default=sa.text("1"), nullable=False),
        sa.Column(
            "tsv",
            postgresql.TSVECTOR(),
            sa.Computed("to_tsvector('english', \"text\")", persisted=True),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["subtopic_key"],
            ["topics.key"],
            name=op.f("fk_questions_subtopic_key_topics"),
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["topic_key"],
            ["topics.key"],
            name=op.f("fk_questions_topic_key_topics"),
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_questions")),
    )
    op.create_index(
        "ix_questions_subtopic_key_difficulty_b",
        "questions",
        ["subtopic_key", "difficulty_b"],
        unique=False,
    )
    op.create_index(
        "ix_questions_tags", "questions", ["tags"], unique=False, postgresql_using="gin"
    )
    op.create_index("ix_questions_tsv", "questions", ["tsv"], unique=False, postgresql_using="gin")
    op.create_table(
        "skill_states",
        sa.Column("user_id", sa.UUID(), nullable=False),
        sa.Column("subtopic_key", sa.Text(), nullable=False),
        sa.Column("theta", sa.REAL(), nullable=False),
        sa.Column("rd", sa.REAL(), nullable=False),
        sa.Column("n_observations", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("last_tested_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(
            ["subtopic_key"],
            ["topics.key"],
            name=op.f("fk_skill_states_subtopic_key_topics"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.id"],
            name=op.f("fk_skill_states_user_id_users"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("user_id", "subtopic_key", name=op.f("pk_skill_states")),
    )
    op.create_table(
        "interview_events",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("session_id", sa.UUID(), nullable=False),
        sa.Column("seq", sa.Integer(), nullable=False),
        sa.Column("type", sa.Text(), nullable=False),
        sa.Column("payload", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["session_id"],
            ["interview_sessions.id"],
            name=op.f("fk_interview_events_session_id_interview_sessions"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_interview_events")),
        sa.UniqueConstraint("session_id", "seq", name="uq_interview_events_session_id_seq"),
    )
    op.create_table(
        "turns",
        sa.Column("id", sa.UUID(), server_default=sa.text("gen_random_uuid()"), nullable=False),
        sa.Column("session_id", sa.UUID(), nullable=False),
        sa.Column("turn_id", sa.UUID(), nullable=False),
        sa.Column("question_id", sa.Text(), nullable=True),
        sa.Column("question_text", sa.Text(), nullable=False),
        sa.Column("answer_text", sa.Text(), nullable=True),
        sa.Column("is_follow_up", sa.Boolean(), server_default=sa.text("false"), nullable=False),
        sa.Column("grade", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("score", sa.REAL(), nullable=True),
        sa.Column("grader_confidence", sa.REAL(), nullable=True),
        sa.Column("latency_ms", sa.Integer(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["question_id"],
            ["questions.id"],
            name=op.f("fk_turns_question_id_questions"),
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["session_id"],
            ["interview_sessions.id"],
            name=op.f("fk_turns_session_id_interview_sessions"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_turns")),
        sa.UniqueConstraint("session_id", "turn_id", name="uq_turns_session_id_turn_id"),
    )


def downgrade() -> None:
    op.drop_table("turns")
    op.drop_table("interview_events")
    op.drop_table("skill_states")
    op.drop_index("ix_questions_tsv", table_name="questions", postgresql_using="gin")
    op.drop_index("ix_questions_tags", table_name="questions", postgresql_using="gin")
    op.drop_index("ix_questions_subtopic_key_difficulty_b", table_name="questions")
    op.drop_table("questions")
    op.drop_index("ix_interview_sessions_user_id_started_at", table_name="interview_sessions")
    op.drop_index(
        "ix_interview_sessions_user_id_active",
        table_name="interview_sessions",
        postgresql_where=sa.text("state != 'COMPLETED'"),
    )
    op.drop_table("interview_sessions")
    op.drop_table("users")
    op.drop_index("ix_topics_domain", table_name="topics")
    op.drop_table("topics")
