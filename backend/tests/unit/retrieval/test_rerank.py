"""Reranking: the ordering logic, the fallback contract, and the stand-in.

No database, no model. ``rerank_candidates`` takes candidates and a scorer and
returns a reordered list — keeping it that way is what makes these assertions
worth anything.

The real cross-encoder is asserted separately in ``test_real_reranker.py``,
behind ``pytest -m embeddings``.
"""

from __future__ import annotations

import pytest

from app.retrieval.rerank import (
    DEFAULT_CANDIDATE_K,
    DEFAULT_RERANK_FINAL_K,
    rerank_candidates,
    rerank_document,
)
from app.retrieval.rerankers import LexicalOverlapReranker
from app.retrieval.search import QuestionRef, RetrievedQuestion


def ref(qid: str, text: str = "some question text", *, document: str | None = None) -> QuestionRef:
    return QuestionRef(
        id=qid,
        text=text,
        topic_key="databases",
        subtopic_key="indexing",
        difficulty_b=0.5,
        tags=("databases",),
        search_document=document,
    )


def candidate(
    qid: str, text: str = "some question text", *, rrf: float = 0.01
) -> RetrievedQuestion:
    return RetrievedQuestion(question=ref(qid, text), rrf_score=rrf, vector_rank=1)


class FixedScorer:
    """Returns whatever scores the test asked for, in candidate order."""

    def __init__(self, scores: list[float], *, model_id: str = "fixed") -> None:
        self._scores = scores
        self._model_id = model_id
        self.calls = 0

    @property
    def model_id(self) -> str:
        return self._model_id

    def score_pairs(self, query, documents):
        self.calls += 1
        return list(self._scores[: len(documents)])


class ExplodingScorer:
    """A model that cannot run — the fallback path."""

    def __init__(self, exc: Exception | None = None) -> None:
        self._exc = exc or RuntimeError("model file is corrupt")

    @property
    def model_id(self) -> str:
        return "exploding"

    def score_pairs(self, query, documents):
        raise self._exc


class WrongLengthScorer:
    """A model that returns the wrong number of scores — a contract violation."""

    @property
    def model_id(self) -> str:
        return "wrong-length"

    def score_pairs(self, query, documents):
        return [1.0]


class TestOrdering:
    def test_candidates_are_reordered_by_descending_score(self):
        candidates = [candidate("a"), candidate("b"), candidate("c")]
        outcome = rerank_candidates(FixedScorer([0.1, 0.9, 0.5]), "q", candidates)
        assert [r.id for r in outcome.results] == ["b", "c", "a"]

    def test_the_retrieval_rank_is_preserved_alongside_the_new_one(self):
        """ "The reranker moved this from 3rd to 1st" is most of the value."""
        candidates = [candidate("a"), candidate("b"), candidate("c")]
        outcome = rerank_candidates(FixedScorer([0.1, 0.2, 0.9]), "q", candidates)
        top = outcome.results[0]
        assert top.id == "c"
        assert top.retrieval_rank == 3
        assert top.rerank_rank == 1
        assert top.rank_delta == 2

    def test_rank_delta_is_negative_for_a_demoted_candidate(self):
        candidates = [candidate("a"), candidate("b")]
        outcome = rerank_candidates(FixedScorer([0.1, 0.9]), "q", candidates)
        demoted = next(r for r in outcome.results if r.id == "a")
        assert demoted.rank_delta == -1

    def test_an_order_the_reranker_agrees_with_is_left_alone(self):
        candidates = [candidate("a"), candidate("b"), candidate("c")]
        outcome = rerank_candidates(FixedScorer([0.9, 0.5, 0.1]), "q", candidates)
        assert [r.id for r in outcome.results] == ["a", "b", "c"]
        assert all(r.rank_delta == 0 for r in outcome.results)

    def test_negative_scores_order_correctly(self):
        """bge-reranker returns logits; irrelevant pairs score below zero."""
        candidates = [candidate("a"), candidate("b"), candidate("c")]
        outcome = rerank_candidates(FixedScorer([-8.0, -1.0, -11.0]), "q", candidates)
        assert [r.id for r in outcome.results] == ["b", "a", "c"]

    def test_the_raw_score_is_recorded_untransformed(self):
        """No sigmoid, no normalisation — a transformation would invent precision."""
        outcome = rerank_candidates(FixedScorer([-3.25]), "q", [candidate("a")])
        assert outcome.results[0].rerank_score == -3.25


class TestTieBreaking:
    def test_equal_scores_fall_back_to_the_retrieval_order(self):
        """Where the reranker is indifferent, stage 1's opinion decides."""
        candidates = [candidate("z"), candidate("a"), candidate("m")]
        outcome = rerank_candidates(FixedScorer([1.0, 1.0, 1.0]), "q", candidates)
        assert [r.id for r in outcome.results] == ["z", "a", "m"], "not sorted by id"

    def test_a_partial_tie_only_affects_the_tied_candidates(self):
        candidates = [candidate("a"), candidate("b"), candidate("c")]
        outcome = rerank_candidates(FixedScorer([0.5, 0.9, 0.5]), "q", candidates)
        assert [r.id for r in outcome.results] == ["b", "a", "c"]

    def test_ordering_is_repeatable(self):
        candidates = [candidate(c) for c in "abcdef"]
        scores = [0.5, 0.5, 0.9, 0.1, 0.9, 0.5]
        first = rerank_candidates(FixedScorer(scores), "q", candidates)
        second = rerank_candidates(FixedScorer(scores), "q", candidates)
        assert [r.id for r in first.results] == [r.id for r in second.results]


class TestTopK:
    def test_only_final_k_are_returned(self):
        candidates = [candidate(c) for c in "abcdef"]
        outcome = rerank_candidates(FixedScorer([1, 2, 3, 4, 5, 6]), "q", candidates, final_k=2)
        assert len(outcome.results) == 2
        assert [r.id for r in outcome.results] == ["f", "e"]

    def test_every_candidate_is_scored_even_though_few_are_returned(self):
        """The point of the design: a candidate ranked 30th can still win."""
        candidates = [candidate(c) for c in "abcdef"]
        outcome = rerank_candidates(FixedScorer([1, 2, 3, 4, 5, 6]), "q", candidates, final_k=1)
        assert outcome.candidates_scored == 6
        assert outcome.results[0].id == "f"
        assert outcome.results[0].retrieval_rank == 6

    def test_truncated_candidates_are_counted_not_hidden(self):
        candidates = [candidate(c) for c in "abcdef"]
        outcome = rerank_candidates(FixedScorer([1, 2, 3, 4, 5, 6]), "q", candidates, final_k=2)
        assert outcome.truncated == 4

    def test_a_final_k_larger_than_the_candidate_set_returns_everything(self):
        outcome = rerank_candidates(
            FixedScorer([1, 2]), "q", [candidate("a"), candidate("b")], final_k=99
        )
        assert len(outcome.results) == 2

    def test_the_plan_defaults_are_forty_into_eight(self):
        """Plan 5.3: "the bi-encoder to go from 150 to 40 and the cross-encoder 40 to 8"."""
        assert DEFAULT_CANDIDATE_K == 40
        assert DEFAULT_RERANK_FINAL_K == 8
        assert DEFAULT_CANDIDATE_K > DEFAULT_RERANK_FINAL_K


class TestEmptyInput:
    def test_no_candidates_returns_no_results(self):
        outcome = rerank_candidates(FixedScorer([]), "q", [])
        assert outcome.results == []

    def test_no_candidates_does_not_call_the_model(self):
        """Loading a cross-encoder to score nothing is pure waste."""
        scorer = FixedScorer([])
        rerank_candidates(scorer, "q", [])
        assert scorer.calls == 0

    def test_an_empty_run_still_counts_as_reranked(self):
        """Nothing failed; there was simply nothing to order."""
        outcome = rerank_candidates(FixedScorer([]), "q", [])
        assert outcome.reranked is True
        assert outcome.fallback_reason is None


class TestFallback:
    def test_a_failing_model_serves_the_hybrid_order(self):
        candidates = [candidate("a"), candidate("b"), candidate("c")]
        outcome = rerank_candidates(ExplodingScorer(), "q", candidates)
        assert [r.id for r in outcome.results] == ["a", "b", "c"]

    def test_a_failing_model_is_never_reported_as_reranked(self):
        """The whole point: never silently pretend reranking happened."""
        outcome = rerank_candidates(ExplodingScorer(), "q", [candidate("a")])
        assert outcome.reranked is False
        assert outcome.fallback_reason is not None

    def test_the_fallback_reason_names_the_error(self):
        outcome = rerank_candidates(
            ExplodingScorer(ValueError("no weights")), "q", [candidate("a")]
        )
        assert "ValueError" in outcome.fallback_reason
        assert "no weights" in outcome.fallback_reason

    def test_fallback_scores_are_none_not_zero(self):
        """A zero reads like a measurement. None does not."""
        outcome = rerank_candidates(ExplodingScorer(), "q", [candidate("a")])
        assert outcome.results[0].rerank_score is None

    def test_fallback_respects_final_k(self):
        candidates = [candidate(c) for c in "abcdef"]
        outcome = rerank_candidates(ExplodingScorer(), "q", candidates, final_k=2)
        assert [r.id for r in outcome.results] == ["a", "b"]

    def test_fallback_ranks_are_the_retrieval_ranks(self):
        candidates = [candidate("a"), candidate("b")]
        outcome = rerank_candidates(ExplodingScorer(), "q", candidates)
        assert all(r.rerank_rank == r.retrieval_rank for r in outcome.results)
        assert all(r.rank_delta == 0 for r in outcome.results)

    def test_a_model_returning_the_wrong_number_of_scores_falls_back(self):
        """A length mismatch would silently pair scores with the wrong candidates."""
        candidates = [candidate("a"), candidate("b"), candidate("c")]
        outcome = rerank_candidates(WrongLengthScorer(), "q", candidates)
        assert outcome.reranked is False
        assert "1 scores for 3 candidates" in outcome.fallback_reason
        assert [r.id for r in outcome.results] == ["a", "b", "c"]


class TestWhatTheModelIsShown:
    def test_the_search_document_is_used_when_present(self):
        """The same text stage 1 searched — see the docstring on rerank_document."""
        question = ref("a", "bare text", document="bare text\nConcepts: cache stampede")
        assert "cache stampede" in rerank_document(question)

    def test_it_falls_back_to_the_question_text(self):
        assert rerank_document(ref("a", "bare text")) == "bare text"

    def test_the_model_receives_that_document_not_the_id_or_difficulty(self):
        seen: list[str] = []

        class Capturing:
            model_id = "capturing"

            def score_pairs(self, query, documents):
                seen.extend(documents)
                return [0.0] * len(documents)

        question = ref("db-index-001", "why indexes help", document="why indexes help\nTags: db")
        rerank_candidates(Capturing(), "q", [RetrievedQuestion(question=question, rrf_score=0.01)])
        assert seen == ["why indexes help\nTags: db"]
        assert "db-index-001" not in seen[0], "the opaque id is not shown to the model"
        assert "0.5" not in seen[0], "difficulty is a filter, not text"


class TestMetadata:
    def test_the_model_id_is_recorded(self):
        outcome = rerank_candidates(FixedScorer([1.0], model_id="bge-x"), "q", [candidate("a")])
        assert outcome.model_id == "bge-x"

    def test_timings_are_reported_for_each_stage(self):
        outcome = rerank_candidates(FixedScorer([1.0]), "q", [candidate("a")])
        assert set(outcome.timings_ms) == {"score", "sort", "total"}
        assert all(v >= 0 for v in outcome.timings_ms.values())


class TestTheLexicalOverlapStandIn:
    """The test seam. Word overlap only — it is not, and must not be claimed as, relevance."""

    def test_more_query_words_present_scores_higher(self):
        reranker = LexicalOverlapReranker()
        scores = reranker.score_pairs(
            "database index performance",
            [
                "database index performance tuning guide",
                "a completely unrelated linked list question",
            ],
        )
        assert scores[0] > scores[1]

    def test_it_is_deterministic(self):
        reranker = LexicalOverlapReranker()
        args = ("database index", ["an index on a database column", "trees"])
        assert reranker.score_pairs(*args) == reranker.score_pairs(*args)

    def test_it_is_deterministic_across_instances(self):
        args = ("database index", ["an index on a database column"])
        assert LexicalOverlapReranker().score_pairs(*args) == LexicalOverlapReranker().score_pairs(
            *args
        )

    def test_one_score_per_document_in_order(self):
        scores = LexicalOverlapReranker().score_pairs("x", ["a", "b", "c"])
        assert len(scores) == 3

    def test_no_documents_gives_no_scores(self):
        assert LexicalOverlapReranker().score_pairs("x", []) == []

    def test_an_empty_query_scores_everything_zero(self):
        assert LexicalOverlapReranker().score_pairs("", ["anything at all"]) == [0.0]

    def test_a_stop_word_only_query_scores_everything_zero(self):
        """Otherwise "how does the" would rank by how many articles a text contains."""
        assert LexicalOverlapReranker().score_pairs("how does the", ["the a and of"]) == [0.0]

    def test_an_empty_document_scores_zero(self):
        assert LexicalOverlapReranker().score_pairs("database", [""]) == [0.0]

    def test_padding_a_document_does_not_help_it(self):
        """The length penalty: a longer text does not win by containing more words."""
        reranker = LexicalOverlapReranker()
        tight, padded = reranker.score_pairs(
            "database index",
            ["database index", "database index " + ("filler " * 50)],
        )
        assert tight > padded

    def test_its_model_id_is_not_a_real_model_id(self):
        """A result scored by the stand-in must never look like a real rerank."""
        assert "lexical" in LexicalOverlapReranker().model_id

    @pytest.mark.parametrize("score", LexicalOverlapReranker().score_pairs("a b", ["a b c"]))
    def test_scores_are_finite_numbers(self, score):
        assert isinstance(score, float)


class TestBackendSelection:
    """Settings wiring: which reranker a process gets, and whether it is reused."""

    def test_the_overlap_backend_is_selected_by_settings(self, env):
        from app.config import Settings
        from app.retrieval.rerankers import LexicalOverlapReranker, build_reranker

        env({"RERANK_BACKEND": "overlap"})
        assert isinstance(build_reranker(Settings()), LexicalOverlapReranker)

    def test_the_fastembed_backend_is_selected_by_settings(self, env):
        """Constructing it must not load the model - loading is deferred."""
        from app.config import Settings
        from app.retrieval.rerankers import FastEmbedCrossEncoder, build_reranker

        env({"RERANK_BACKEND": "fastembed"})
        reranker = build_reranker(Settings())
        assert isinstance(reranker, FastEmbedCrossEncoder)
        assert reranker.model_id == "BAAI/bge-reranker-base"

    def test_the_real_model_instance_is_reused_across_calls(self, env):
        """Loading costs seconds and scoring costs milliseconds; reloading per
        query would make the two-stage design pointless."""
        from app.config import Settings
        from app.retrieval.rerankers import _cached_cross_encoder, get_reranker

        _cached_cross_encoder.cache_clear()
        env({"RERANK_BACKEND": "fastembed"})
        settings = Settings()
        assert get_reranker(settings) is get_reranker(settings)
        _cached_cross_encoder.cache_clear()

    def test_the_stand_in_is_not_cached(self, env):
        """It is cheap enough that caching would only add a way to be surprised."""
        from app.config import Settings
        from app.retrieval.rerankers import get_reranker

        env({"RERANK_BACKEND": "overlap"})
        settings = Settings()
        assert get_reranker(settings) is not get_reranker(settings)

    def test_the_default_k_values_match_the_plan(self, env):
        from app.config import Settings

        env({})
        settings = Settings()
        assert settings.rerank_candidate_k == 40
        assert settings.rerank_final_k == 8
        assert settings.rerank_candidate_k > settings.rerank_final_k
        assert settings.rerank_enabled is True
