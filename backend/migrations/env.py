"""Alembic environment.

Alembic runs this file for every migration command. Its job is to hand Alembic
a database connection plus the target metadata, then start a migration run.

Two project-specific decisions live here:

1. The URL comes from the application's ``Settings``, not from ``alembic.ini``.
   One source of truth means migrations can never be applied to a different
   database than the one the app talks to.
2. The engine is async, because the app's DSN is ``postgresql+asyncpg://``.
   Alembic's own migration machinery is synchronous, so we open an async
   connection and hand it to ``run_sync``.
"""

from __future__ import annotations

import asyncio

from alembic import context
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

from app.config import get_database_settings, get_logging_settings

# Importing the package registers every model on Base.metadata. Without this,
# autogenerate would compare an empty catalogue against the database and cheerfully
# propose dropping all your tables.
from app.models import Base
from app.obs import configure_logging

config = context.config

# Deliberately NOT `fileConfig(config.config_file_name)`, which is what Alembic's
# generated template does.
#
# `fileConfig` disables every existing logger and *replaces* the root handler.
# In the migrate container that would mean this one process emits a different,
# unredacted format from everything else in the system. Worse, the integration
# test suite runs migrations in-process: the first migration wiped the log
# configuration for every test that ran afterwards, and the observability tests
# silently captured nothing. That is the bug this line exists to prevent.
#
# `LoggingSettings` has no required fields, so this cannot fail - which matters
# because a migration about to die on a missing DATABASE_URL still has to be
# able to say so in a format someone can read (see app/config.py).
configure_logging(get_logging_settings())

target_metadata = Base.metadata


def _database_url() -> str:
    """Prefer an explicitly injected URL (tests), else the app's own settings.

    ``DatabaseSettings`` rather than the full ``Settings``: a migration touches
    only Postgres, and making a schema change - or a restore drill - depend on
    an LLM credential being present would be coupling with a real cost. The URL
    is still validated by the same field definition the API uses, so the two
    cannot drift apart.
    """
    injected = config.get_main_option("sqlalchemy.url")
    if injected:
        return injected
    return get_database_settings().database_url


def _configure(connection: Connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        # Detect column type changes, not just added/removed columns.
        compare_type=True,
        # Detect changes to server-side defaults too.
        compare_server_default=True,
        # Wrap each migration in its own transaction. Postgres has
        # transactional DDL, so a migration that fails halfway rolls back
        # completely rather than leaving a half-changed schema.
        transaction_per_migration=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_offline() -> None:
    """Emit SQL to stdout instead of executing it (``alembic upgrade --sql``).

    Useful when a DBA has to review the change before it touches production.
    """
    context.configure(
        url=_database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


async def _run_async_migrations() -> None:
    connectable = async_engine_from_config(
        {"sqlalchemy.url": _database_url()},
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    async with connectable.connect() as connection:
        await connection.run_sync(_configure)
    await connectable.dispose()


def run_migrations_online() -> None:
    asyncio.run(_run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
