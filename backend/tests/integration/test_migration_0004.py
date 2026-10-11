"""Migration 0004 applies, constrains, backfills and rolls back."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import AsyncEngine

BACKEND_DIR = Path(__file__).resolve().parents[2]
TURN_COLUMNS = {
    "grading_status",
    "grading_attempts",
    "next_attempt_at",
    "lease_expires_at",
    "grading_error",
    "graded_at",
}


def _cfg(url: str) -> Config:
    cfg = Config(str(BACKEND_DIR / "alembic.ini"))
    cfg.set_main_option("script_location", str(BACKEND_DIR / "migrations"))
    cfg.set_main_option("sqlalchemy.url", url)
    return cfg


async def _columns(engine: AsyncEngine, table: str) -> set[str]:
    async with engine.connect() as conn:
        rows = await conn.execute(
            text("SELECT column_name FROM information_schema.columns WHERE table_name = :t"),
            {"t": table},
        )
        return {r[0] for r in rows}


async def _seed_turn(engine: AsyncEngine, score: float | None) -> None:
    async with engine.begin() as conn:
        uid = await conn.scalar(
            text("INSERT INTO users (email, password_hash) VALUES (:e,'h') RETURNING id"),
            {"e": f"m{score}@x.com"},
        )
        sid = await conn.scalar(
            text(
                "INSERT INTO interview_sessions (user_id, blueprint) VALUES (:u,'{}') RETURNING id"
            ),
            {"u": uid},
        )
        await conn.execute(
            text(
                "INSERT INTO turns (session_id, turn_id, question_text, score) "
                "VALUES (:s, gen_random_uuid(), 'q', :score)"
            ),
            {"s": sid, "score": score},
        )


async def test_0004_round_trip_and_legacy_backfill(db_engine, migrated_database_url):
    cfg = _cfg(migrated_database_url)
    assert TURN_COLUMNS <= await _columns(db_engine, "turns")
    assert "common_misconceptions" in await _columns(db_engine, "questions")
    try:
        await asyncio.to_thread(command.downgrade, cfg, "0003")
        assert not (TURN_COLUMNS & await _columns(db_engine, "turns"))
        assert "common_misconceptions" not in await _columns(db_engine, "questions")
        # Rows that existed before 0004 must never be picked up and graded.
        await _seed_turn(db_engine, 0.5)
        await _seed_turn(db_engine, None)
    finally:
        await asyncio.to_thread(command.upgrade, cfg, "head")
    async with db_engine.connect() as conn:
        rows = (
            await conn.execute(
                text("SELECT score, grading_status, grading_error FROM turns ORDER BY score")
            )
        ).all()
    assert [tuple(r) for r in rows] == [(0.5, "graded", None), (None, "failed", "legacy_row")]


@pytest.mark.parametrize(
    ("column", "value"), [("grading_status", "'done'"), ("grading_attempts", "-1")]
)
async def test_0004_constraints(db_engine, column, value):
    await _seed_turn(db_engine, None)
    with pytest.raises(DBAPIError, match="ck_turns_"):
        async with db_engine.begin() as conn:
            await conn.execute(text(f"UPDATE turns SET {column} = {value}"))  # noqa: S608
