"""User accounts."""

from __future__ import annotations

import datetime as dt
import uuid

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import CITEXT, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, utc_now_column


class User(Base):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        server_default=sa.text("gen_random_uuid()"),
    )
    # CITEXT compares case-insensitively, so "Manas@x.com" and "manas@x.com"
    # collide on the unique index instead of becoming two accounts. Doing this
    # in the column type rather than by lower()-ing in application code means no
    # future code path can forget to normalise.
    email: Mapped[str] = mapped_column(CITEXT, unique=True, nullable=False)
    # An argon2id digest, never the password. Length varies with the parameters,
    # so this is TEXT rather than a fixed-width column. NULL for an account
    # created with Google (migration 0008): such an account has no password,
    # and none is invented for it - password login simply cannot succeed.
    password_hash: Mapped[str | None] = mapped_column(sa.Text, nullable=True)
    # Google's stable account id (the ID token's `sub`), for accounts created
    # with "Continue with Google". Unique; never the email, which can change.
    # CHECK ck_users_has_credential: every row has a password or a Google id.
    google_sub: Mapped[str | None] = mapped_column(sa.Text, unique=True, nullable=True)
    created_at: Mapped[dt.datetime] = utc_now_column()

    __table_args__ = (
        sa.CheckConstraint(
            "password_hash IS NOT NULL OR google_sub IS NOT NULL", name="has_credential"
        ),
    )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<User {self.email}>"
