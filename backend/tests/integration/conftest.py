"""Integration fixtures: a real Postgres and a real Redis.

These tests need the compose stack running. When it is not, they SKIP rather
than fail, so ``pytest`` stays green on a laptop with Docker closed while still
giving full coverage when the stack is up.

The database is a throwaway created fresh for the run, and the schema is built
by running the actual Alembic migration - so every test run also verifies that
the migration applies cleanly to an empty database.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator, Iterator
from pathlib import Path

import asyncpg
import pytest
import pytest_asyncio
from alembic import command
from alembic.config import Config
from asgi_lifespan import LifespanManager
from httpx import ASGITransport, AsyncClient
from redis.asyncio import Redis
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

BACKEND_DIR = Path(__file__).resolve().parents[2]

# Host ports from docker-compose.yml. Overridable so CI can point elsewhere.
ADMIN_DSN = os.getenv(
    "TEST_ADMIN_DSN", "postgresql://interviewer:interviewer@localhost:5433/interviewer"
)
TEST_DB_NAME = os.getenv("TEST_DB_NAME", "interviewer_test")
# DB 15 keeps test keys away from anything a developer has in DB 0.
TEST_REDIS_URL = os.getenv("TEST_REDIS_URL", "redis://localhost:6380/15")

TABLES = (
    "turns",
    "interview_events",
    "interview_sessions",
    "skill_states",
    "questions",
    "topics",
    "users",
)


def _async_dsn(database: str) -> str:
    base = ADMIN_DSN.rsplit("/", 1)[0]
    return f"{base}/{database}".replace("postgresql://", "postgresql+asyncpg://", 1)


async def _recreate_database() -> None:
    conn = await asyncpg.connect(ADMIN_DSN, timeout=5)
    try:
        # FORCE terminates any leftover connections from an interrupted run.
        await conn.execute(f'DROP DATABASE IF EXISTS "{TEST_DB_NAME}" WITH (FORCE)')
        await conn.execute(f'CREATE DATABASE "{TEST_DB_NAME}"')
    finally:
        await conn.close()

    fresh = await asyncpg.connect(ADMIN_DSN.rsplit("/", 1)[0] + f"/{TEST_DB_NAME}", timeout=5)
    try:
        # Mirrors infra/postgres/init/001-extensions.sql. Extensions are not in
        # the migration on purpose (they need superuser), so a fresh database
        # has to be given them the same way the real one is.
        await fresh.execute("CREATE EXTENSION IF NOT EXISTS citext")
        await fresh.execute("CREATE EXTENSION IF NOT EXISTS vector")
    finally:
        await fresh.close()


async def _drop_database() -> None:
    conn = await asyncpg.connect(ADMIN_DSN, timeout=5)
    try:
        await conn.execute(f'DROP DATABASE IF EXISTS "{TEST_DB_NAME}" WITH (FORCE)')
    finally:
        await conn.close()


def _run_migrations(url: str) -> None:
    cfg = Config(str(BACKEND_DIR / "alembic.ini"))
    cfg.set_main_option("script_location", str(BACKEND_DIR / "migrations"))
    cfg.set_main_option("sqlalchemy.url", url)
    command.upgrade(cfg, "head")


@pytest.fixture(scope="session")
def migrated_database_url() -> Iterator[str]:
    """A fresh database with the real migration applied. Skips if Postgres is down.

    Synchronous on purpose: ``asyncio.run`` needs a thread with no running loop,
    and Alembic's ``env.py`` calls it internally.
    """
    url = _async_dsn(TEST_DB_NAME)
    try:
        asyncio.run(_recreate_database())
    except (OSError, asyncpg.PostgresError, TimeoutError) as exc:
        pytest.skip(
            f"Postgres not reachable at {ADMIN_DSN} ({type(exc).__name__}); "
            "run `docker compose up -d` to enable integration tests"
        )

    _run_migrations(url)
    try:
        yield url
    finally:
        asyncio.run(_drop_database())


@pytest.fixture(scope="session")
def redis_url() -> str:
    """Skips if Redis is down."""

    async def _ping() -> None:
        client = Redis.from_url(TEST_REDIS_URL, socket_connect_timeout=3)
        try:
            await client.ping()
        finally:
            await client.aclose()

    try:
        asyncio.run(_ping())
    except Exception as exc:  # noqa: BLE001 - any failure means "not available"
        pytest.skip(f"Redis not reachable at {TEST_REDIS_URL} ({type(exc).__name__})")
    return TEST_REDIS_URL


@pytest_asyncio.fixture
async def db_engine(migrated_database_url: str) -> AsyncIterator[AsyncEngine]:
    engine = create_async_engine(migrated_database_url, poolclass=None)
    yield engine
    await engine.dispose()


@pytest_asyncio.fixture(autouse=True)
async def _clean_state(request: pytest.FixtureRequest) -> AsyncIterator[None]:
    """Start every integration test from an empty database and an empty Redis."""
    if "migrated_database_url" not in request.fixturenames:
        yield
        return

    url = request.getfixturevalue("migrated_database_url")
    engine = create_async_engine(url)
    async with engine.begin() as conn:
        # RESTART IDENTITY resets the interview_events BIGSERIAL so ids are
        # predictable; CASCADE handles the foreign keys between these tables.
        await conn.execute(text(f"TRUNCATE {', '.join(TABLES)} RESTART IDENTITY CASCADE"))
    await engine.dispose()

    if "redis_url" in request.fixturenames:
        client = Redis.from_url(request.getfixturevalue("redis_url"))
        await client.flushdb()
        await client.aclose()

    yield


@pytest_asyncio.fixture
async def client(env, migrated_database_url: str, redis_url: str) -> AsyncIterator[AsyncClient]:
    """The real app, wired to the test database and Redis, with lifespan run."""
    from app.cache import dispose_redis
    from app.db import dispose_engine
    from app.main import create_app

    # A previous test may have left module-level clients pointing elsewhere.
    await dispose_engine()
    await dispose_redis()

    env({"DATABASE_URL": migrated_database_url, "REDIS_URL": redis_url})
    app = create_app()

    # httpx does not run ASGI startup/shutdown events; LifespanManager does,
    # which is what creates the engine and Redis client the endpoints use.
    async with LifespanManager(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as http:
            yield http

    await dispose_engine()
    await dispose_redis()
