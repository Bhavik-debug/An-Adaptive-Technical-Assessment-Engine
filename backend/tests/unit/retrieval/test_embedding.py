"""The embedding input recipe, the fingerprint, and the deterministic embedder.

Nothing here loads a model. The real model has its own opt-in suite
(``test_real_model.py``, ``pytest -m embeddings``).
"""

from __future__ import annotations

import math

import pytest

from app.retrieval.embedders import HashingEmbedder
from app.retrieval.embedding import (
    EMBEDDING_DIM,
    EMBEDDING_MODEL_ID,
    document_text,
    humanise_key,
    normalise_whitespace,
    query_text,
    text_fingerprint,
)

DOC = {
    "text": "Explain why a B-tree index makes lookups faster.",
    "topic_key": "databases",
    "subtopic_key": "indexing",
    "concept_keys": ["btree_index", "index_read_path"],
    "tags": ["databases", "indexing"],
}


class TestTheRecipe:
    def test_it_is_deterministic(self):
        assert document_text(**DOC) == document_text(**DOC)

    def test_it_contains_the_question_the_concepts_the_taxonomy_and_the_tags(self):
        built = document_text(**DOC)
        assert "B-tree index" in built
        assert "btree index" in built, "concept keys are humanised"
        assert "databases / indexing" in built
        assert "Tags:" in built

    def test_snake_case_keys_become_words(self):
        """An embedding model handles 'cache invalidation'; 'cache_invalidation' is a fragment."""
        assert humanise_key("cache_invalidation") == "cache invalidation"
        assert "_" not in document_text(**DOC)

    def test_changing_the_question_changes_the_document(self):
        other = document_text(**{**DOC, "text": "Something else entirely."})
        assert other != document_text(**DOC)

    def test_changing_a_concept_changes_the_document(self):
        other = document_text(**{**DOC, "concept_keys": ["btree_index", "write_amplification"]})
        assert other != document_text(**DOC)

    def test_reordering_concepts_changes_the_document(self):
        """No sorting is applied; the author's order is stable and meaningful."""
        other = document_text(**{**DOC, "concept_keys": ["index_read_path", "btree_index"]})
        assert other != document_text(**DOC)

    def test_empty_concepts_and_tags_are_simply_omitted(self):
        built = document_text(**{**DOC, "concept_keys": [], "tags": []})
        assert "Concepts:" not in built
        assert "Tags:" not in built
        assert "Topic:" in built

    def test_the_id_and_difficulty_are_not_in_it(self):
        """Embedding an opaque id or a number is embedding noise."""
        built = document_text(**DOC)
        assert "db-index-001" not in built
        assert "difficulty" not in built.lower()


class TestWhitespace:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [("  a  b ", "a b"), ("a\n\nb", "a b"), ("a\tb", "a b"), ("a b", "a b")],
    )
    def test_runs_of_whitespace_collapse(self, raw, expected):
        assert normalise_whitespace(raw) == expected

    def test_reformatting_the_question_does_not_change_the_document(self):
        """Otherwise re-wrapping the JSONL would re-embed the whole bank."""
        wrapped = document_text(
            **{**DOC, "text": "Explain why a B-tree index\n  makes lookups faster."}
        )
        assert wrapped == document_text(**DOC)


class TestTheQuerySide:
    def test_a_query_is_normalised_but_not_decorated(self):
        """A query has no concepts or tags; inventing them would move it away from the documents."""
        assert query_text("  how do   indexes work? ") == "how do indexes work?"
        assert "Concepts:" not in query_text("indexes")

    def test_it_shares_normalisation_with_the_document_side(self):
        assert query_text("a\n\nb") == normalise_whitespace("a\n\nb")


class TestTheFingerprint:
    def test_the_same_text_and_model_give_the_same_fingerprint(self):
        assert text_fingerprint("x", model_id="m") == text_fingerprint("x", model_id="m")

    def test_a_different_text_gives_a_different_fingerprint(self):
        assert text_fingerprint("x", model_id="m") != text_fingerprint("y", model_id="m")

    def test_a_different_model_gives_a_different_fingerprint(self):
        """Changing the model must invalidate every stored vector."""
        assert text_fingerprint("x", model_id="m1") != text_fingerprint("x", model_id="m2")

    def test_it_is_a_sha256_hex_digest(self):
        fingerprint = text_fingerprint("x", model_id="m")
        assert len(fingerprint) == 64
        assert set(fingerprint) <= set("0123456789abcdef")


class TestTheHashingEmbedder:
    """The test seam. Word overlap only - it is not, and must not be claimed as, semantic."""

    def test_it_produces_the_declared_dimension(self):
        vector = HashingEmbedder().embed_query("hello world")
        assert len(vector) == EMBEDDING_DIM

    def test_vectors_are_l2_normalised_like_the_real_model(self):
        vector = HashingEmbedder().embed_query("database index performance")
        assert math.sqrt(sum(v * v for v in vector)) == pytest.approx(1.0)

    def test_it_is_deterministic_within_a_process(self):
        embedder = HashingEmbedder()
        assert embedder.embed_query("abc") == embedder.embed_query("abc")

    def test_it_is_deterministic_across_instances(self):
        """SHA-256, not Python's per-process-salted ``hash()``.

        A vector stored in the database has to still mean the same thing after
        the process that wrote it restarts.
        """
        assert HashingEmbedder().embed_query("abc") == HashingEmbedder().embed_query("abc")

    def test_shared_words_score_higher_than_unrelated_ones(self):
        embedder = HashingEmbedder()
        base = embedder.embed_query("database index performance")
        overlapping = embedder.embed_query("database index tuning")
        unrelated = embedder.embed_query("binary tree traversal")
        assert _dot(base, overlapping) > _dot(base, unrelated)

    def test_an_empty_text_still_yields_a_usable_unit_vector(self):
        """A zero vector has no direction, so cosine distance against it is undefined."""
        vector = HashingEmbedder().embed_query("")
        assert math.sqrt(sum(v * v for v in vector)) == pytest.approx(1.0)

    def test_documents_are_embedded_in_the_order_given(self):
        embedder = HashingEmbedder()
        vectors = embedder.embed_documents(["alpha", "beta"])
        assert vectors == [embedder.embed_query("alpha"), embedder.embed_query("beta")]

    def test_no_documents_gives_no_vectors(self):
        assert HashingEmbedder().embed_documents([]) == []

    def test_its_model_id_is_not_the_real_model_id(self):
        """A row embedded by the stand-in must never look like a real embedding."""
        assert HashingEmbedder().model_id != EMBEDDING_MODEL_ID


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b, strict=True))
