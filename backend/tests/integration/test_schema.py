"""What the migration actually produced in Postgres.

These assert on the live database rather than on the Python models, because the
models are the *intent* and this is the result. The two drifting apart is
exactly the failure mode migrations exist to prevent.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

BACKEND_DIR = Path(__file__).resolve().parents[2]

EXPECTED_TABLES = {
    "users",
    "topics",
    "questions",
    "interview_sessions",
    "interview_events",
    "turns",
    "skill_states",
}


async def test_every_core_table_exists(db_engine: AsyncEngine):
    async with db_engine.connect() as conn:
        rows = await conn.execute(
            text("SELECT tablename FROM pg_tables WHERE schemaname = 'public'")
        )
        tables = {r[0] for r in rows}
    assert EXPECTED_TABLES <= tables


async def test_alembic_stamped_the_head_revision(db_engine: AsyncEngine):
    """Whatever the newest migration is, the test database is at it.

    Read from the migrations directory rather than hard-coded, so adding a
    migration does not fail a test that has nothing to do with it - while still
    catching the real failure, which is a migration that did not apply.
    """
    versions = sorted((BACKEND_DIR / "migrations" / "versions").glob("[0-9]*.py"))
    expected = versions[-1].name.split("_", 1)[0]
    async with db_engine.connect() as conn:
        version = await conn.scalar(text("SELECT version_num FROM alembic_version"))
    assert version == expected


async def test_email_is_case_insensitive_and_unique(db_engine: AsyncEngine):
    """The citext column is what stops duplicate-account-by-case."""
    async with db_engine.begin() as conn:
        await conn.execute(
            text("INSERT INTO users (email, password_hash) VALUES ('Ada@Example.com', 'x')")
        )

    with pytest.raises(Exception) as exc:
        async with db_engine.begin() as conn:
            await conn.execute(
                text("INSERT INTO users (email, password_hash) VALUES ('ada@example.com', 'y')")
            )
    assert "uq_users_email" in str(exc.value)


async def test_email_lookup_ignores_case(db_engine: AsyncEngine):
    async with db_engine.begin() as conn:
        await conn.execute(
            text("INSERT INTO users (email, password_hash) VALUES ('Grace@Example.com', 'x')")
        )
    async with db_engine.connect() as conn:
        found = await conn.scalar(text("SELECT email FROM users WHERE email = 'GRACE@EXAMPLE.COM'"))
    assert found == "Grace@Example.com"


async def test_turns_are_idempotent_on_session_and_turn_id(db_engine: AsyncEngine):
    """The constraint that makes a double-submitted answer harmless."""
    async with db_engine.begin() as conn:
        user_id = await conn.scalar(
            text("INSERT INTO users (email, password_hash) VALUES ('t@x.com','h') RETURNING id")
        )
        session_id = await conn.scalar(
            text(
                "INSERT INTO interview_sessions (user_id, blueprint) "
                "VALUES (:uid, '{}'::jsonb) RETURNING id"
            ),
            {"uid": user_id},
        )
        turn_id = await conn.scalar(text("SELECT gen_random_uuid()"))
        await conn.execute(
            text(
                "INSERT INTO turns (session_id, turn_id, question_text) "
                "VALUES (:sid, :tid, 'q?')"
            ),
            {"sid": session_id, "tid": turn_id},
        )

    with pytest.raises(Exception) as exc:
        async with db_engine.begin() as conn:
            await conn.execute(
                text(
                    "INSERT INTO turns (session_id, turn_id, question_text) "
                    "VALUES (:sid, :tid, 'q? again')"
                ),
                {"sid": session_id, "tid": turn_id},
            )
    assert "uq_turns_session_id_turn_id" in str(exc.value)


async def test_event_sequence_is_unique_per_session(db_engine: AsyncEngine):
    """Two events cannot claim the same position in one session's log."""
    async with db_engine.begin() as conn:
        user_id = await conn.scalar(
            text("INSERT INTO users (email, password_hash) VALUES ('e@x.com','h') RETURNING id")
        )
        session_id = await conn.scalar(
            text(
                "INSERT INTO interview_sessions (user_id, blueprint) "
                "VALUES (:uid, '{}'::jsonb) RETURNING id"
            ),
            {"uid": user_id},
        )
        await conn.execute(
            text(
                "INSERT INTO interview_events (session_id, seq, type, payload) "
                "VALUES (:sid, 0, 'SESSION_CREATED', '{}'::jsonb)"
            ),
            {"sid": session_id},
        )

    with pytest.raises(Exception) as exc:
        async with db_engine.begin() as conn:
            await conn.execute(
                text(
                    "INSERT INTO interview_events (session_id, seq, type, payload) "
                    "VALUES (:sid, 0, 'OTHER', '{}'::jsonb)"
                ),
                {"sid": session_id},
            )
    assert "uq_interview_events_session_id_seq" in str(exc.value)


async def test_question_tsvector_is_generated_and_searchable(db_engine: AsyncEngine):
    """The full-text column maintains itself; Phase 2 relies on that."""
    async with db_engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO topics (key, display_name, domain) "
                "VALUES ('caching', 'Caching', 'Backend')"
            )
        )
        await conn.execute(
            text(
                "INSERT INTO questions (id, topic_key, subtopic_key, text, difficulty_b,"
                " expected_concepts, reference_answer, time_estimate_s, source) "
                "VALUES ('q1','caching','caching','How do you avoid a cache stampede?',"
                " 0.5, '[]'::jsonb, 'ref', 120, 'handwritten')"
            )
        )
    async with db_engine.connect() as conn:
        # 'stampede' stems to 'stamped' - proof the text really was analysed.
        found = await conn.scalar(
            text("SELECT id FROM questions WHERE tsv @@ to_tsquery('english', 'stampede')")
        )
    assert found == "q1"


async def test_deleting_a_user_cascades_to_their_sessions(db_engine: AsyncEngine):
    """A deletion request must not leave orphaned interview data behind."""
    async with db_engine.begin() as conn:
        user_id = await conn.scalar(
            text("INSERT INTO users (email, password_hash) VALUES ('c@x.com','h') RETURNING id")
        )
        await conn.execute(
            text(
                "INSERT INTO interview_sessions (user_id, blueprint) " "VALUES (:uid, '{}'::jsonb)"
            ),
            {"uid": user_id},
        )
        await conn.execute(text("DELETE FROM users WHERE id = :uid"), {"uid": user_id})
        remaining = await conn.scalar(text("SELECT count(*) FROM interview_sessions"))
    assert remaining == 0


async def test_partial_index_on_unfinished_sessions_exists(db_engine: AsyncEngine):
    async with db_engine.connect() as conn:
        definition = await conn.scalar(
            text(
                "SELECT indexdef FROM pg_indexes "
                "WHERE indexname = 'ix_interview_sessions_user_id_active'"
            )
        )
    assert definition is not None
    assert "WHERE" in definition.upper()
    assert "COMPLETED" in definition.upper()
