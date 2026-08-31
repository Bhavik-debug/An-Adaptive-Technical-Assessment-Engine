"""Retrieval columns: the search document, its tsvector, the vector and its index.

Day 8. Everything goes on the existing ``questions`` table - there is no second
questions table and no second database.

**Why ``search_document`` exists, and why ``tsv`` is rebuilt over it.**  Day 2's
``tsv`` was generated from ``text`` alone.  That was right when lexical search
was the only search.  It is wrong once a vector arm exists, because the two
retrievers would then be searching *different content*: the embedding covers the
question plus its concepts, taxonomy and tags, while ``tsv`` covered only the
question. Fusing two rankings over two different corpora compares apples with
oranges, and it showed up immediately in practice - a query for "cache stampede
thundering herd" returned nothing lexically, because those words live in the
item's ``expected_concepts``, not in its prose.

So ``search_document`` stores exactly the string that gets embedded, written by
ingest from the one shared recipe (``app.retrieval.embedding.document_text``),
and ``tsv`` is generated from it.

**Why COALESCE.**  ``tsv`` is generated from
``coalesce(search_document, text)``. A row that has not been re-ingested yet has
``search_document IS NULL`` and keeps exactly the old behaviour, so this
migration cannot leave lexical search returning nothing on a populated table.
Once ingest runs, every row upgrades itself.

**Why this is safe on a populated table.** The added columns are all NULLable
with no default, which PostgreSQL records as a catalogue change: no rewrite, no
existing row read or modified. ``tsv`` *is* dropped and recreated - but ``tsv``
is a GENERATED column, so it holds no source data of its own; it is recomputed
from ``text`` for every row on creation. The 60 questions keep every value they
had.

**Why the indexes are not built concurrently.** Days 1-29 are local, the table
has 60 rows, and ``CREATE INDEX CONCURRENTLY`` cannot run inside the transaction
Alembic wraps a migration in. A concurrent rebuild belongs in the Phase 6 runbook,
when this table is large and serving traffic.

Revision ID: 0002
Revises: 0001
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None

#: Must equal app.retrieval.embedding.EMBEDDING_DIM (BAAI/bge-small-en-v1.5).
#: Written literally rather than imported: a migration is a historical record of
#: what was applied, and it has to keep meaning the same thing after the
#: application code moves on. A test asserts the two agree.
EMBEDDING_DIM = 384

#: The generated expression for `tsv`, in both directions of this migration.
#: The two-argument form of to_tsvector is required - the one-argument form
#: depends on a session setting and is therefore not IMMUTABLE, which Postgres
#: refuses in a generated column.
_NEW_TSV = "to_tsvector('english', coalesce(search_document, \"text\"))"
_OLD_TSV = "to_tsvector('english', \"text\")"


def upgrade() -> None:
    # --- what gets searched ------------------------------------------------
    op.add_column("questions", sa.Column("search_document", sa.Text(), nullable=True))

    # Rebuild `tsv` over the search document. A generated column's expression
    # cannot be altered in place, so it is dropped and recreated; its GIN index
    # goes with it and is rebuilt afterwards.
    op.drop_index("ix_questions_tsv", table_name="questions", postgresql_using="gin")
    op.drop_column("questions", "tsv")
    op.add_column(
        "questions",
        sa.Column(
            "tsv",
            sa.dialects.postgresql.TSVECTOR(),
            sa.Computed(_NEW_TSV, persisted=True),
            nullable=False,
        ),
    )
    op.create_index("ix_questions_tsv", "questions", ["tsv"], unique=False, postgresql_using="gin")

    # --- the vector and its index ------------------------------------------
    # The `vector` extension itself is created by
    # infra/postgres/init/001-extensions.sql, which runs as superuser on an
    # empty data directory. Alembic runs as the application user and is
    # forward-only, so it must not try to CREATE EXTENSION.
    op.add_column("questions", sa.Column("embedding", Vector(EMBEDDING_DIM), nullable=True))
    op.add_column("questions", sa.Column("embedding_model", sa.Text(), nullable=True))
    op.add_column("questions", sa.Column("embedding_text_sha256", sa.Text(), nullable=True))

    # Hierarchical Navigable Small World. `vector_cosine_ops` fixes the distance
    # this index answers for: the `<=>` operator. An index built for one
    # operator is simply not used by a query written with another - it silently
    # falls back to a sequential scan - so the operator class and the query have
    # to be chosen together.
    op.create_index(
        "ix_questions_embedding_hnsw",
        "questions",
        ["embedding"],
        unique=False,
        postgresql_using="hnsw",
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )


def downgrade() -> None:
    op.drop_index("ix_questions_embedding_hnsw", table_name="questions", postgresql_using="hnsw")
    op.drop_column("questions", "embedding_text_sha256")
    op.drop_column("questions", "embedding_model")
    op.drop_column("questions", "embedding")

    op.drop_index("ix_questions_tsv", table_name="questions", postgresql_using="gin")
    op.drop_column("questions", "tsv")
    op.add_column(
        "questions",
        sa.Column(
            "tsv",
            sa.dialects.postgresql.TSVECTOR(),
            sa.Computed(_OLD_TSV, persisted=True),
            nullable=False,
        ),
    )
    op.create_index("ix_questions_tsv", "questions", ["tsv"], unique=False, postgresql_using="gin")
    op.drop_column("questions", "search_document")
