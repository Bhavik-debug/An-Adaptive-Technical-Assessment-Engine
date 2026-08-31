"""The opt-in suite: does the real cross-encoder behave the way Day 9 assumes?

    pip install -e "./backend[embeddings]"
    cd backend && pytest -m embeddings

Excluded from the default run (see ``addopts`` in pyproject.toml), because it
loads a ~1 GB model that CI has no reason to download. Same pattern as Day 8's
``test_real_model.py``.

**These are the assertions the stand-in cannot make.** ``LexicalOverlapReranker``
counts shared words, so a test claiming it judges *relevance* would be asserting
something false. Relevance behaviour is checked here against the real model, or
nowhere.

Assertions are deliberately **relative** - "this pair scores above that pair" -
and never absolute thresholds. The score is a raw logit whose scale is a
property of the model's training, not a probability, and a threshold like
"> 0.5 means relevant" would be a number invented to make a test pass.
"""

from __future__ import annotations

import math

import pytest

from app.retrieval.rerank import rerank_candidates
from app.retrieval.rerankers import FastEmbedCrossEncoder
from app.retrieval.search import QuestionRef, RetrievedQuestion

pytestmark = pytest.mark.embeddings

MODEL_ID = "BAAI/bge-reranker-base"

QUERY = "How can database reads be made faster?"
RELEVANT = "Explain why a B-tree index on a column makes lookups on that column faster."
UNRELATED = "Describe Floyd's cycle-detection algorithm for a singly linked list."
MIDDLING = "Compare READ COMMITTED with REPEATABLE READ isolation levels."


@pytest.fixture(scope="module")
def reranker():
    model = FastEmbedCrossEncoder(MODEL_ID)
    try:
        model.score_pairs("warm up", ["a document"])
    except Exception as exc:  # noqa: BLE001 - say why rather than fail opaquely
        pytest.skip(f"real cross-encoder unavailable: {exc}")
    return model


def candidate(qid: str, text: str) -> RetrievedQuestion:
    return RetrievedQuestion(
        question=QuestionRef(
            id=qid,
            text=text,
            topic_key="databases",
            subtopic_key="indexing",
            difficulty_b=0.0,
            tags=(),
            search_document=text,
        ),
        rrf_score=0.01,
    )


class TestItLoadsAndRuns:
    def test_it_returns_one_score_per_document(self, reranker):
        scores = reranker.score_pairs(QUERY, [RELEVANT, UNRELATED, MIDDLING])
        assert len(scores) == 3

    def test_scores_are_finite_floats(self, reranker):
        for score in reranker.score_pairs(QUERY, [RELEVANT, UNRELATED]):
            assert isinstance(score, float)
            assert math.isfinite(score)

    def test_no_documents_returns_no_scores_without_loading(self):
        """An empty candidate set must not cost a model load."""
        assert FastEmbedCrossEncoder(MODEL_ID).score_pairs(QUERY, []) == []

    def test_it_is_deterministic(self, reranker):
        """Fixed weights, CPU, no sampling - the same pair must score the same twice."""
        first = reranker.score_pairs(QUERY, [RELEVANT, UNRELATED])
        second = reranker.score_pairs(QUERY, [RELEVANT, UNRELATED])
        assert first == second

    def test_batching_does_not_change_a_score(self, reranker):
        """Reranking 40 candidates batches them; the score must not depend on that."""
        alone = reranker.score_pairs(QUERY, [MIDDLING])[0]
        together = reranker.score_pairs(QUERY, [RELEVANT, MIDDLING, UNRELATED])[1]
        assert alone == pytest.approx(together, abs=1e-4)

    def test_scores_are_returned_in_document_order(self, reranker):
        """`rerank_candidates` zips scores to candidates positionally; order is the contract."""
        forward = reranker.score_pairs(QUERY, [RELEVANT, UNRELATED])
        backward = reranker.score_pairs(QUERY, [UNRELATED, RELEVANT])
        assert forward[0] == pytest.approx(backward[1], abs=1e-4)
        assert forward[1] == pytest.approx(backward[0], abs=1e-4)


class TestItJudgesRelevance:
    """The claim the whole reranking stage rests on."""

    def test_a_relevant_candidate_outscores_an_unrelated_one(self, reranker):
        relevant, unrelated = reranker.score_pairs(QUERY, [RELEVANT, UNRELATED])
        assert relevant > unrelated

    def test_the_same_document_scores_higher_for_a_query_it_answers(self, reranker):
        """Scores are only comparable within one query; this checks the other axis."""
        dsa_query = "How do I detect a loop in a linked list?"
        for_db_query = reranker.score_pairs(QUERY, [UNRELATED])[0]
        for_dsa_query = reranker.score_pairs(dsa_query, [UNRELATED])[0]
        assert for_dsa_query > for_db_query

    def test_it_reorders_a_deliberately_wrong_candidate_order(self, reranker):
        """End to end: the unrelated candidate is handed over first and must not stay there."""
        candidates = [
            candidate("dsa-lists-002", UNRELATED),
            candidate("db-iso-001", MIDDLING),
            candidate("db-index-001", RELEVANT),
        ]
        outcome = rerank_candidates(reranker, QUERY, candidates)
        assert outcome.reranked is True
        assert outcome.results[0].id == "db-index-001"
        assert outcome.results[0].retrieval_rank == 3
        assert outcome.results[0].rank_delta == 2

    def test_concepts_in_the_document_are_visible_to_the_model(self, reranker):
        """`rerank_document` shows the model the same text stage 1 indexed.

        "thundering herd" is in one bank item's `expected_concepts` and nowhere
        in its prose. If appending the concept line did not raise that item's
        score for that query, showing it would be pointless - so this asserts
        the reason the choice was made.
        """
        bare = (
            "A popular cache entry with a 60-second TTL expires at peak traffic, and the "
            "database immediately receives thousands of identical queries. Name this failure."
        )
        with_concepts = bare + "\nConcepts: cache stampede, thundering herd, request coalescing"
        without, with_ = reranker.score_pairs("thundering herd", [bare, with_concepts])
        assert with_ > without


class TestScoreSemantics:
    def test_the_score_is_not_bounded_to_zero_and_one(self, reranker):
        """Documents why nothing in this codebase treats it as a probability.

        bge-reranker-base emits a raw logit. An irrelevant pair scores well
        below zero, which no probability can do - so any code applying a
        [0, 1] threshold to this number would be wrong.
        """
        scores = reranker.score_pairs(QUERY, [RELEVANT, UNRELATED])
        assert min(scores) < 0.0 or max(scores) > 1.0

    def test_no_transformation_is_applied_between_model_and_result(self, reranker):
        """`rerank_score` is exactly what the model returned."""
        raw = reranker.score_pairs(QUERY, [RELEVANT])[0]
        outcome = rerank_candidates(reranker, QUERY, [candidate("x", RELEVANT)])
        assert outcome.results[0].rerank_score == pytest.approx(raw, abs=1e-6)
