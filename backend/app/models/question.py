"""The question bank (populated in Phase 2)."""

from __future__ import annotations

from typing import Any

import sqlalchemy as sa
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, REAL, TSVECTOR
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base
from app.retrieval.embedding import EMBEDDING_DIM


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

    # A GENERATED column: Postgres recomputes it on every write, so the search
    # index can never drift out of sync with the thing it indexes.
    # The two-argument form of to_tsvector is required - the one-argument form
    # depends on a session setting and is therefore not IMMUTABLE, which
    # Postgres refuses in a generated column.
    # COALESCE: a row ingested before Day 8 has no search_document yet and keeps
    # exactly the old behaviour until the next ingest upgrades it, so adding the
    # column can never leave lexical search returning nothing.
    tsv: Mapped[str] = mapped_column(
        TSVECTOR,
        sa.Computed("to_tsvector('english', coalesce(search_document, \"text\"))", persisted=True),
        nullable=False,
    )

    # --- Retrieval (Day 8) --------------------------------------------------
    # The exact string both retrievers search: the question plus its concepts,
    # taxonomy and tags, built by app.retrieval.embedding.document_text() and
    # written by ingest. It is stored rather than recomputed so that `tsv` can
    # be generated from it, and so that what was indexed is inspectable in SQL.
    #
    # Why both arms must search the same text: fusing a vector ranking over
    # "question + concepts" with a lexical ranking over "question" alone ranks
    # two different corpora against each other. A query for "cache stampede"
    # found nothing lexically until this existed, because those words are in
    # the item's expected_concepts, not its prose.
    search_document: Mapped[str | None] = mapped_column(sa.Text, nullable=True)

    # Nullable, because a question exists before it is embedded: ingest writes
    # the row first and the vector second, and a bank ingested with
    # `--no-embed` is a legitimate state. Vector search filters these out
    # rather than pretending a missing vector is a distant one.
    embedding: Mapped[list[float] | None] = mapped_column(Vector(EMBEDDING_DIM), nullable=True)

    # The two columns that make re-embedding idempotent. `embedding_model` is
    # which model produced the vector; `embedding_text_sha256` is a hash of the
    # exact text that was embedded, including the recipe version. A row whose
    # pair still matches is skipped on re-ingest; a row whose question text,
    # concepts, tags, model or recipe changed no longer matches and is
    # re-embedded. Without them the only options are re-embedding everything
    # every run, or letting vectors silently go stale against edited questions.
    embedding_model: Mapped[str | None] = mapped_column(sa.Text, nullable=True)
    embedding_text_sha256: Mapped[str | None] = mapped_column(sa.Text, nullable=True)

    __table_args__ = (
        # GIN is the index type for "which rows contain this token", the inverse
        # of B-tree's "where does this value sort". Both lexical search (Phase 2)
        # and tag filtering need that shape.
        sa.Index("ix_questions_tsv", "tsv", postgresql_using="gin"),
        sa.Index("ix_questions_tags", "tags", postgresql_using="gin"),
        sa.Index("ix_questions_subtopic_key_difficulty_b", "subtopic_key", "difficulty_b"),
        # HNSW: a navigable small-world graph over the vectors, so a nearest-
        # neighbour search follows a few hundred edges instead of comparing the
        # query against every row. `vector_cosine_ops` because bge returns
        # L2-normalised vectors and cosine is the similarity the model was
        # trained for. Default m/ef_construction: tuning them without a
        # measured recall number would be guessing (plan section 3, Day 8).
        sa.Index(
            "ix_questions_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<Question {self.id}>"
