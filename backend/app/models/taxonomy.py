"""The topic taxonomy: domain -> topic -> subtopic (plan section 9.1).

One self-referencing table rather than three, because the levels behave
identically and a three-table version would need three near-identical queries
for every read. ``domain`` is denormalised onto every row so that "all subtopics
in the Backend domain" is one indexed filter instead of a recursive walk.
"""

from __future__ import annotations

import sqlalchemy as sa
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base


class Topic(Base):
    __tablename__ = "topics"

    # A human-readable slug ('caching', 'cache_invalidation') rather than a UUID:
    # these keys appear in the question bank JSONL files, which are hand-authored
    # and reviewed in git, so they have to be readable by a person.
    key: Mapped[str] = mapped_column(sa.Text, primary_key=True)
    parent_key: Mapped[str | None] = mapped_column(
        sa.Text, sa.ForeignKey("topics.key", ondelete="RESTRICT"), nullable=True
    )
    display_name: Mapped[str] = mapped_column(sa.Text, nullable=False)
    domain: Mapped[str] = mapped_column(sa.Text, nullable=False)

    __table_args__ = (sa.Index("ix_topics_domain", "domain"),)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<Topic {self.key}>"
