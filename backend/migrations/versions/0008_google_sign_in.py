"""Google sign-in: users.google_sub, and password_hash becomes optional.

Day 27. An account created with "Continue with Google" has no password, and
none is invented for it, so ``password_hash`` may now be NULL.  What keeps that
safe is in the schema, not only in code:

* ``google_sub`` (Google's stable account id) is ``UNIQUE`` - two concurrent
  first sign-ins with one Google account cannot create two users;
* ``CHECK (password_hash IS NOT NULL OR google_sub IS NOT NULL)`` - no row can
  exist with no way to authenticate.

Existing rows all have a password hash, so they satisfy the CHECK unchanged.

Downgrade refuses while any Google-only account exists: making
``password_hash`` NOT NULL again would otherwise mean deleting accounts or
inventing passwords, and a migration must do neither silently.

Revision ID: 0008
Revises: 0007
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0008"
down_revision = "0007"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("users", sa.Column("google_sub", sa.Text(), nullable=True))
    op.create_unique_constraint(op.f("uq_users_google_sub"), "users", ["google_sub"])
    op.alter_column("users", "password_hash", existing_type=sa.Text(), nullable=True)
    op.create_check_constraint(
        op.f("ck_users_has_credential"),
        "users",
        "password_hash IS NOT NULL OR google_sub IS NOT NULL",
    )


def downgrade() -> None:
    bind = op.get_bind()
    google_only = bind.scalar(sa.text("SELECT count(*) FROM users WHERE password_hash IS NULL"))
    if google_only:
        raise RuntimeError(
            f"{google_only} account(s) have no password (Google sign-in only); "
            "downgrading past 0008 would have to delete them. Refusing."
        )
    op.drop_constraint(op.f("ck_users_has_credential"), "users", type_="check")
    op.alter_column("users", "password_hash", existing_type=sa.Text(), nullable=False)
    op.drop_constraint(op.f("uq_users_google_sub"), "users", type_="unique")
    op.drop_column("users", "google_sub")
