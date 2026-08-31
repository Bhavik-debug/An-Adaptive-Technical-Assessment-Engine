"""The one number three files have to agree about.

``EMBEDDING_DIM`` appears in the application code, in the SQLAlchemy column, and
literally in the migration - the migration cannot import it, because a migration
is a historical record of what was applied and must keep meaning the same thing
after the application moves on.

That is the right call and it creates exactly one risk: the three drift apart.
A mismatch is not subtle - every insert fails - but it fails at ingest time
against a real database, which is much later than here.
"""

from __future__ import annotations

import re
from pathlib import Path

from app.models.question import Question
from app.retrieval.embedding import EMBEDDING_DIM

MIGRATION = Path(__file__).resolve().parents[3] / "migrations" / "versions"


def test_the_model_column_uses_the_declared_dimension():
    column = Question.__table__.c.embedding
    assert column.type.dim == EMBEDDING_DIM


def test_the_migration_uses_the_declared_dimension():
    source = (MIGRATION / "0002_question_embeddings.py").read_text(encoding="utf-8")
    match = re.search(r"^EMBEDDING_DIM = (\d+)$", source, re.MULTILINE)
    assert match, "the migration no longer declares EMBEDDING_DIM"
    assert int(match.group(1)) == EMBEDDING_DIM


def test_the_embedding_column_is_nullable():
    """A question exists before it is embedded; `--no-embed` is a legitimate state."""
    assert Question.__table__.c.embedding.nullable is True


def test_the_search_document_column_is_nullable():
    """Rows ingested before Day 8 have none, and `tsv` coalesces back to `text`."""
    assert Question.__table__.c.search_document.nullable is True


def test_the_tsvector_is_generated_from_the_search_document():
    """If this ever reverts to `text` alone, the two retrievers search different corpora."""
    computed = Question.__table__.c.tsv.computed
    assert computed is not None
    expression = str(computed.sqltext)
    assert "search_document" in expression
    assert "coalesce" in expression.lower()
