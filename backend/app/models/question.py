"""The question bank (populated in Phase 2)."""

from __future__ import annotations

from typing import Any

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, REAL, TSVECTOR
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base


class Question(Base):
    __tablename__ = "questions"

    # Readable ids ('sys-cache-002'). The bank is a git-versioned dataset, and a
    # diff full of UUIDs is unreviewable.
    id: Mapped[str] = mapped_column(sa.Text, primary_key=True)

    topic_key: Mapped[str] = mapped_column(
        sa.Text, sa.ForeignKey("topics.key", ondelete="RESTRICT"), nullable=False
    )
    subtopic_key: Mapped[str] = mapped_column(
        sa.Text, sa.ForeignKey("topics.key", ondelete="RESTRICT"), nullable=False
    )

    text: Mapped[str] = mapped_column(sa.Text, nullable=False)

    # IRT parameters (plan section 5.8). b is difficulty on the same scale as
    # candidate ability theta; a is discrimination, how sharply this item
    # separates candidates around b.
    difficulty_b: Mapped[float] = mapped_column(REAL, nullable=False)
    discrimination_a: Mapped[float] = mapped_column(
        REAL, nullable=False, server_default=sa.text("1.0")
    )

    # The concept checklist the grader classifies against - the heart of the
    # project (plan section 6.3). JSONB rather than a child table because it is
    # always read whole, with the question, and never queried across rows.
    expected_concepts: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, nullable=False)
    reference_answer: Mapped[str] = mapped_column(sa.Text, nullable=False)
    follow_up_seeds: Mapped[list[dict[str, Any]] | None] = mapped_column(JSONB, nullable=True)

    anchor_terms: Mapped[list[str] | None] = mapped_column(ARRAY(sa.Text), nullable=True)
    time_estimate_s: Mapped[int] = mapped_column(sa.Integer, nullable=False)
    tags: Mapped[list[str] | None] = mapped_column(ARRAY(sa.Text), nullable=True)
    source: Mapped[str] = mapped_column(sa.Text, nullable=False)
    version: Mapped[int] = mapped_column(sa.Integer, nullable=False, server_default=sa.text("1"))

    # A GENERATED column: Postgres recomputes it on every write of `text`, so the
    # search index can never drift out of sync with the thing it indexes.
    # The two-argument form of to_tsvector is required - the one-argument form
    # depends on a session setting and is therefore not IMMUTABLE, which
    # Postgres refuses in a generated column.
    tsv: Mapped[str] = mapped_column(
        TSVECTOR,
        sa.Computed("to_tsvector('english', \"text\")", persisted=True),
        nullable=False,
    )

    __table_args__ = (
        # GIN is the index type for "which rows contain this token", the inverse
        # of B-tree's "where does this value sort". Both lexical search (Phase 2)
        # and tag filtering need that shape.
        sa.Index("ix_questions_tsv", "tsv", postgresql_using="gin"),
        sa.Index("ix_questions_tags", "tags", postgresql_using="gin"),
        sa.Index("ix_questions_subtopic_key_difficulty_b", "subtopic_key", "difficulty_b"),
    )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<Question {self.id}>"
