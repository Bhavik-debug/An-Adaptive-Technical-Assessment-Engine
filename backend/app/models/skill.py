"""The candidate skill model (plan section 9)."""

from __future__ import annotations

import datetime as dt
import uuid

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import REAL, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base


class SkillState(Base):
    """Ability at SUBTOPIC level only.

    Topic- and domain-level numbers are computed on read by precision-weighted
    aggregation (plan section 9.2). Storing them too would create a second
    source of truth that silently drifts - never store a number you can derive.
    """

    __tablename__ = "skill_states"

    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        sa.ForeignKey("users.id", ondelete="CASCADE"),
        primary_key=True,
    )
    subtopic_key: Mapped[str] = mapped_column(
        sa.Text,
        sa.ForeignKey("topics.key", ondelete="CASCADE"),
        primary_key=True,
    )

    # theta: latent ability, on the same scale as question difficulty b.
    # rd: rating deviation - how unsure we are. High rd = wide error bars.
    theta: Mapped[float] = mapped_column(REAL, nullable=False)
    rd: Mapped[float] = mapped_column(REAL, nullable=False)
    n_observations: Mapped[int] = mapped_column(
        sa.Integer, nullable=False, server_default=sa.text("0")
    )
    # Drives RD inflation over time, which is where spaced repetition falls out
    # for free in V3 (plan section 9.5).
    last_tested_at: Mapped[dt.datetime | None] = mapped_column(
        sa.DateTime(timezone=True), nullable=True
    )
