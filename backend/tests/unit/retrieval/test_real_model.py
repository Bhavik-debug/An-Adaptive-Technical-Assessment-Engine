"""The opt-in suite: does the real model behave the way the architecture assumes?

    pip install -e "./backend[embeddings]"
    cd backend && pytest -m embeddings

Excluded from the default run (see `addopts` in pyproject.toml), because it
loads a 67 MB model that CI has no reason to download.

**These are the assertions the stand-in cannot make.** ``HashingEmbedder``
matches words, so a test claiming it understands a paraphrase would be asserting
something false. Semantic behaviour is checked here against the real thing, or
nowhere.

The assertions are deliberately *relative* - "this is closer than that" - never
absolute thresholds on a similarity value. bge similarities live in a narrow
high band (two unrelated technical questions still score ~0.6), so a threshold
like "> 0.8 means related" would be a number invented to make a test pass, and
would break on any model update. Ranking is what retrieval uses; ranking is what
is tested.
"""

from __future__ import annotations

import math

import pytest

from app.retrieval.embedders import FastEmbedEmbedder
from app.retrieval.embedding import EMBEDDING_DIM, EMBEDDING_MODEL_ID

pytestmark = pytest.mark.embeddings


@pytest.fixture(scope="module")
def embedder():
    model = FastEmbedEmbedder(EMBEDDING_MODEL_ID)
    try:
        model.embed_query("warm up")
    except Exception as exc:  # noqa: BLE001 - report why rather than fail opaquely
        pytest.skip(f"real embedding model unavailable: {exc}")
    return model


def dot(a, b):
    return sum(x * y for x, y in zip(a, b, strict=True))


class TestTheContractTheSchemaDependsOn:
    def test_the_dimension_matches_the_vector_column(self):
        """384 is compiled into the migration and the model; a mismatch breaks every insert."""
        assert len(FastEmbedEmbedder(EMBEDDING_MODEL_ID).embed_query("x")) == EMBEDDING_DIM

    def test_vectors_are_l2_normalised(self, embedder):
        """Why cosine distance is the right operator and similarity is the dot product."""
        for vector in embedder.embed_documents(["a short text", "another, longer piece of text"]):
            assert math.sqrt(sum(v * v for v in vector)) == pytest.approx(1.0, abs=1e-5)

    def test_it_is_deterministic(self, embedder):
        assert embedder.embed_query("database indexes") == embedder.embed_query("database indexes")

    def test_batching_does_not_change_a_vector(self, embedder):
        """Ingest embeds in batches; search embeds one query. They must agree."""
        together = embedder.embed_documents(["first text", "second text"])
        assert together[1] == pytest.approx(embedder.embed_query("second text"), abs=1e-6)

    def test_embed_documents_preserves_order(self, embedder):
        vectors = embedder.embed_documents(["alpha text", "beta text"])
        assert vectors[0] == pytest.approx(embedder.embed_query("alpha text"), abs=1e-6)


class TestItActuallyUnderstandsMeaning:
    """The claim the whole vector arm rests on."""

    def test_a_paraphrase_beats_an_unrelated_question(self, embedder):
        a, b, c = embedder.embed_documents(
            [
                "Why do database indexes improve query performance?",
                "What is the purpose of an index in a relational database?",
                "Describe Floyd's cycle-detection algorithm for a linked list.",
            ]
        )
        assert dot(a, b) > dot(a, c)

    def test_a_query_with_no_shared_words_still_finds_the_right_question(self, embedder):
        """The case lexical search cannot handle, and the reason vectors are here."""
        query = embedder.embed_query("How can a relational database speed up slow lookups?")
        indexes, dp = embedder.embed_documents(
            [
                "Explain why a B-tree index on a column makes lookups on that column faster.",
                "Contrast top-down memoisation with bottom-up tabulation in dynamic programming.",
            ]
        )
        assert dot(query, indexes) > dot(query, dp)

    def test_within_domain_questions_are_closer_than_across_domain_ones(self, embedder):
        db_one, db_two, dsa = embedder.embed_documents(
            [
                "Compare READ COMMITTED with REPEATABLE READ isolation levels.",
                "Explain what write-ahead logging guarantees when a database commits.",
                "Given a sorted array, remove duplicates in place using two pointers.",
            ]
        )
        assert dot(db_one, db_two) > dot(db_one, dsa)


class TestTheThingsWeDecidedRatherThanAssumed:
    def test_query_embed_and_embed_agree_for_this_model(self, embedder):
        """bge offers an optional query instruction prefix; fastembed applies none.

        `query_text()` therefore adds no prefix either. If a future fastembed
        release starts applying one, this fails and the decision gets revisited
        with evidence - which is the point of pinning an assumption to a test.
        """
        from fastembed import TextEmbedding

        model = TextEmbedding(model_name=EMBEDDING_MODEL_ID, cache_dir=str(embedder._cache_dir))
        text = "how do database indexes work"
        as_query = [float(x) for x in next(iter(model.query_embed(text)))]
        as_document = [float(x) for x in next(iter(model.embed([text])))]
        assert as_query == pytest.approx(as_document, abs=1e-6)

    def test_similarities_sit_in_a_narrow_high_band(self, embedder):
        """Documents why no absolute threshold is used anywhere in this codebase.

        Two questions from completely different domains still score well above
        0.4, so "similarity > 0.5 means relevant" would be meaningless.
        """
        db, dsa = embedder.embed_documents(
            [
                "Explain what each letter of ACID guarantees in a relational database.",
                "Describe how a heap solves the top-k problem over a large stream.",
            ]
        )
        assert (
            0.3 < dot(db, dsa) < 0.9
        ), "unrelated pairs are not near zero, which is normal for bge"
