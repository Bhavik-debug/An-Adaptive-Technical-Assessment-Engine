"""Running a migration must not destroy the process's logging configuration.

The regression this exists for was genuinely nasty, so it is worth stating in
full. Alembic's generated ``env.py`` calls ``fileConfig(alembic.ini)``, which
**disables every existing logger and replaces the root handler**. That is fine
for a one-shot CLI process. It is not fine here, because the integration suite
runs migrations *in process*: the first migration silently wiped the logging
configuration for every test that ran afterwards, and the Day-4 observability
tests then captured nothing and failed with no hint as to the cause.

It stayed hidden for a day because those tests only run migrations when Postgres
is actually reachable - so on a laptop with Docker closed, everything passed.

``tests/unit/obs/test_logging.py`` guards the source (``env.py`` must not call
``fileConfig``). This guards the behaviour, and it is the one that runs in CI
where Postgres always exists.
"""

from __future__ import annotations

import logging

from app.config import get_logging_settings
from app.obs import configure_logging

_OWNED = "_obs_owned_handler"


def _our_handlers() -> list[logging.Handler]:
    return [h for h in logging.getLogger().handlers if getattr(h, _OWNED, False)]


def test_a_migration_leaves_the_applications_log_handler_in_place(migrated_database_url):
    """``migrated_database_url`` has already run the real Alembic migration."""
    configure_logging(get_logging_settings())
    assert _our_handlers(), "configure_logging did not install its handler"

    # Re-run the migration machinery the same way the suite's fixture does.
    from .conftest import _run_migrations

    _run_migrations(migrated_database_url)

    assert _our_handlers(), (
        "running a migration removed the application's log handler - "
        "migrations/env.py must not call fileConfig()"
    )


def test_a_migration_does_not_disable_existing_loggers(migrated_database_url):
    """``fileConfig``'s other half: it sets ``logger.disabled = True`` on every
    logger that already existed, which silences them without removing them - a
    failure mode even harder to spot than a missing handler."""
    from .conftest import _run_migrations

    logger = logging.getLogger("app.llm.client")
    logger.disabled = False

    _run_migrations(migrated_database_url)

    assert logger.disabled is False
