"""Migration 0005: follow_ups and turns.policy_decision - applies, constrains, rolls back."""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import AsyncEngine

BACKEND_DIR = Path(__file__).resolve().parents[2]


def _cfg(url: str) -> Config:
    cfg = Config(str(BACKEND_DIR / "alembic.ini"))
    cfg.set_main_option("script_location", str(BACKEND_DIR / "migrations"))
    cfg.set_main_option("sqlalchemy.url", url)
    return cfg


async def _exists(engine: AsyncEngine) -> tuple[bool, bool]:
    async with engine.connect() as conn:
        table = await conn.scalar(text("SELECT to_regclass('public.follow_ups') IS NOT NULL"))
        column = await conn.scalar(
            text(
                "SELECT count(*) FROM information_schema.columns "
                "WHERE table_name = 'turns' AND column_name = 'policy_decision'"
            )
        )
    return bool(table), bool(column)


async def _session(engine: AsyncEngine) -> uuid.UUID:
    async with engine.begin() as conn:
        uid = await conn.scalar(
            text("INSERT INTO users (email, password_hash) VALUES ('m5@x.com','h') RETURNING id")
        )
        return await conn.scalar(  # type: ignore[no-any-return]
            text(
                "INSERT INTO interview_sessions (user_id, blueprint) VALUES (:u,'{}') RETURNING id"
            ),
            {"u": uid},
        )


async def _insert(engine, sid, *, ordinal=1, source=None, status="pending", item=None):
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO follow_ups (session_id, source_turn_id, item_question_id, ordinal, "
                "reason, status) VALUES (:s, :t, :i, :o, 'high_uncertainty', :st)"
            ),
            {"s": sid, "t": source or uuid.uuid4(), "i": item, "o": ordinal, "st": status},
        )


async def test_round_trip(db_engine, migrated_database_url):
    cfg = _cfg(migrated_database_url)
    assert await _exists(db_engine) == (True, True)
    try:
        await asyncio.to_thread(command.downgrade, cfg, "0004")
        assert await _exists(db_engine) == (False, False)
    finally:
        await asyncio.to_thread(command.upgrade, cfg, "head")
    assert await _exists(db_engine) == (True, True)


@pytest.mark.parametrize(
    ("kwargs", "constraint"),
    [
        ({"ordinal": 3}, "ck_follow_ups_ordinal_within_cap"),
        ({"ordinal": 0}, "ck_follow_ups_ordinal_within_cap"),
        ({"status": "done"}, "ck_follow_ups_status_valid"),
    ],
)
async def test_check_constraints(db_engine, kwargs, constraint):
    sid = await _session(db_engine)
    with pytest.raises(DBAPIError, match=constraint):
        await _insert(db_engine, sid, **kwargs)


async def test_one_follow_up_per_graded_turn(db_engine):
    sid = await _session(db_engine)
    source = uuid.uuid4()
    await _insert(db_engine, sid, source=source)
    with pytest.raises(DBAPIError, match="uq_follow_ups_session_id_source_turn_id"):
        await _insert(db_engine, sid, source=source, ordinal=2)


async def test_one_row_per_item_ordinal(db_engine):
    async with db_engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO topics (key, parent_key, display_name, domain) VALUES "
                "('m5d', NULL, 'd', 'm5d'), ('m5t', 'm5d', 't', 'm5d'), ('m5s', 'm5t', 's', 'm5d')"
            )
        )
        await conn.execute(
            text(
                "INSERT INTO questions (id, topic_key, subtopic_key, text, difficulty_b, "
                "discrimination_a, expected_concepts, reference_answer, time_estimate_s, source) "
                "VALUES ('m5q', 'm5t', 'm5s', 'q', 0, 1, '[]', 'r', 60, 'test')"
            )
        )
    sid = await _session(db_engine)
    await _insert(db_engine, sid, item="m5q", ordinal=1)
    with pytest.raises(DBAPIError, match="uq_follow_ups_session_id_item_question_id_ordinal"):
        await _insert(db_engine, sid, item="m5q", ordinal=1)
