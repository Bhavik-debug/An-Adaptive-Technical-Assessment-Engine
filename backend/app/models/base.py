"""Declarative base and shared column helpers.

Every table in the system inherits from ``Base``. ``Base.metadata`` is the
in-memory catalogue of every table SQLAlchemy knows about, and it is what
Alembic diffs against the live database to generate a migration.
"""

from __future__ import annotations

import datetime as dt

import sqlalchemy as sa
from sqlalchemy.orm import DeclarativeBase, mapped_column

# Postgres names constraints for you when you do not. Those generated names are
# unpredictable, which makes "drop this constraint" in a later migration a
# guessing game. Fixing the naming scheme up front means every index, unique and
# foreign key has a name we can predict and reference forever.
NAMING_CONVENTION = {
    "ix": "ix_%(table_name)s_%(column_0_N_name)s",
    "uq": "uq_%(table_name)s_%(column_0_N_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_N_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    metadata = sa.MetaData(naming_convention=NAMING_CONVENTION)


def utc_now_column(**kwargs: object) -> sa.orm.Mapped[dt.datetime]:
    """A timestamptz column defaulted by the database clock, not the app clock.

    ``now()`` is evaluated by Postgres, so every row is stamped from one clock
    even when several API replicas disagree about the time.
    """
    return mapped_column(
        sa.DateTime(timezone=True),
        server_default=sa.func.now(),
        nullable=False,
        **kwargs,  # type: ignore[arg-type]
    )
