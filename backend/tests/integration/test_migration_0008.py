"""Migration 0008: users.google_sub and optional password_hash - constrains, rolls back."""

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


def _cfg(url: str) -> Config:
    cfg = Config(str(BACKEND_DIR / "alembic.ini"))
    cfg.set_main_option("script_location", str(BACKEND_DIR / "migrations"))
    cfg.set_main_option("sqlalchemy.url", url)
    return cfg


async def _has_column(engine: AsyncEngine) -> bool:
    async with engine.connect() as conn:
        n = await conn.scalar(
            text(
                "SELECT count(*) FROM information_schema.columns "
                "WHERE table_name = 'users' AND column_name = 'google_sub'"
            )
        )
    return bool(n)


async def _insert(engine: AsyncEngine, email: str, password_hash: str | None, sub: str | None):
    async with engine.begin() as conn:
        await conn.execute(
            text("INSERT INTO users (email, password_hash, google_sub) VALUES (:e, :p, :s)"),
            {"e": email, "p": password_hash, "s": sub},
        )


async def test_round_trip(db_engine, migrated_database_url):
    cfg = _cfg(migrated_database_url)
    assert await _has_column(db_engine)
    await _insert(db_engine, "pw@x.com", "h", None)  # existing-style rows survive both ways
    try:
        await asyncio.to_thread(command.downgrade, cfg, "0007")
        assert not await _has_column(db_engine)
    finally:
        await asyncio.to_thread(command.upgrade, cfg, "head")
    assert await _has_column(db_engine)


async def test_downgrade_refuses_while_google_only_accounts_exist(db_engine, migrated_database_url):
    await _insert(db_engine, "g@x.com", None, "sub-1")
    with pytest.raises(RuntimeError, match="Refusing"):
        await asyncio.to_thread(command.downgrade, _cfg(migrated_database_url), "0007")
    assert await _has_column(db_engine)  # nothing was half-applied


async def test_every_account_needs_a_credential(db_engine):
    with pytest.raises(DBAPIError, match="ck_users_has_credential"):
        await _insert(db_engine, "none@x.com", None, None)


async def test_a_google_account_is_unique(db_engine):
    await _insert(db_engine, "a@x.com", None, "same-sub")
    with pytest.raises(DBAPIError, match="uq_users_google_sub"):
        await _insert(db_engine, "b@x.com", None, "same-sub")
