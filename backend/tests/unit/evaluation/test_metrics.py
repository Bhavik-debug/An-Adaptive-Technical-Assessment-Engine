"""The metrics, against hand-worked examples.

No database, no models, no fixtures. Every expected value here was computed by
hand from the definition, not by running the code and recording what it said -
which is the only way a metric test proves anything.
"""

from __future__ import annotations

import math

import pytest

from app.evaluation.metrics import (
    EmptyRelevantSet,
    dcg_at_k,
    first_relevant_rank,
    ideal_dcg_at_k,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)


class TestRecallAtK:
    def test_the_relevant_result_is_inside_k(self):
        """retrieved=[B, A, C], relevant={A} -> A is 2nd, so it is in the top 3."""
        assert recall_at_k(["B", "A", "C"], {"A"}, 3) == 1.0

    def test_the_relevant_result_is_outside_k(self):
        """Same list, k=1: only B is considered."""
        assert recall_at_k(["B", "A", "C"], {"A"}, 1) == 0.0

    def test_at_rank_one(self):
        assert recall_at_k(["A", "B"], {"A"}, 1) == 1.0

    def test_nothing_relevant_was_retrieved(self):
        assert recall_at_k(["X", "Y", "Z"], {"A"}, 10) == 0.0

    def test_an_empty_retrieved_list_scores_zero(self):
        assert recall_at_k([], {"A"}, 10) == 0.0

    def test_two_of_three_relevant_found(self):
        """|{A,B} intersect {A,B,C}| / |{A,B,C}| = 2/3."""
        assert recall_at_k(["A", "X", "B"], {"A", "B", "C"}, 10) == pytest.approx(2 / 3)

    def test_all_relevant_found(self):
        assert recall_at_k(["A", "B", "C"], {"A", "B", "C"}, 3) == 1.0

    def test_the_k_boundary_is_inclusive(self):
        """A is exactly 3rd, so recall@3 counts it and recall@2 does not."""
        assert recall_at_k(["X", "Y", "A"], {"A"}, 3) == 1.0
        assert recall_at_k(["X", "Y", "A"], {"A"}, 2) == 0.0

    def test_k_larger_than_the_list_is_harmless(self):
        assert recall_at_k(["A"], {"A"}, 100) == 1.0

    def test_a_duplicate_cannot_inflate_the_score(self):
        """A retriever returning the same id twice must not score 2/2."""
        assert recall_at_k(["A", "A"], {"A", "B"}, 10) == 0.5

    def test_an_empty_relevant_set_is_rejected(self):
        """Dividing by zero, or quietly returning 1.0, would both be lies."""
        with pytest.raises(EmptyRelevantSet):
            recall_at_k(["A"], set(), 10)

    @pytest.mark.parametrize("k", [0, -1])
    def test_a_non_positive_k_is_rejected(self, k):
        with pytest.raises(ValueError, match="at least 1"):
            recall_at_k(["A"], {"A"}, k)


class TestFirstRelevantRank:
    def test_it_is_one_based(self):
        assert first_relevant_rank(["A", "B"], {"A"}) == 1

    def test_it_finds_the_earliest_relevant_result(self):
        assert first_relevant_rank(["X", "B", "A"], {"A", "B"}) == 2

    def test_none_when_nothing_relevant_appears(self):
        assert first_relevant_rank(["X", "Y"], {"A"}) is None

    def test_none_for_an_empty_list(self):
        assert first_relevant_rank([], {"A"}) is None

    def test_an_empty_relevant_set_is_rejected(self):
        with pytest.raises(EmptyRelevantSet):
            first_relevant_rank(["A"], set())


class TestReciprocalRank:
    @pytest.mark.parametrize(
        ("retrieved", "expected"),
        [
            (["A", "X", "Y"], 1.0),
            (["X", "A", "Y"], 0.5),
            (["X", "Y", "A"], 1 / 3),
            (["X", "Y", "Z", "A"], 0.25),
            (["a1", "a2", "a3", "a4", "a5", "a6", "a7", "a8", "a9", "A"], 0.1),
        ],
    )
    def test_the_reciprocal_of_the_first_relevant_rank(self, retrieved, expected):
        assert reciprocal_rank(retrieved, {"A"}) == pytest.approx(expected)

    def test_zero_when_nothing_relevant_is_found(self):
        assert reciprocal_rank(["X", "Y"], {"A"}) == 0.0

    def test_zero_for_an_empty_retrieved_list(self):
        assert reciprocal_rank([], {"A"}) == 0.0

    def test_only_the_first_relevant_result_counts(self):
        """Adding a second relevant result lower down changes nothing."""
        assert reciprocal_rank(["X", "A"], {"A"}) == reciprocal_rank(["X", "A", "B"], {"A", "B"})

    def test_k_truncates_before_looking(self):
        """A hit at rank 12 does not count for k=10."""
        retrieved = [f"x{i}" for i in range(11)] + ["A"]
        assert reciprocal_rank(retrieved, {"A"}) == pytest.approx(1 / 12)
        assert reciprocal_rank(retrieved, {"A"}, k=10) == 0.0

    def test_an_empty_relevant_set_is_rejected(self):
        with pytest.raises(EmptyRelevantSet):
            reciprocal_rank(["A"], set())


class TestDcg:
    def test_the_worked_example_from_the_docstring(self):
        """grades={A:2, B:1}, retrieved=[B, A, C]:

        rank 1  B  gain 2^1-1 = 1  / log2(2) = 1.000
        rank 2  A  gain 2^2-1 = 3  / log2(3) = 1.893
        rank 3  C  gain 0
        """
        expected = 1 / math.log2(2) + 3 / math.log2(3)
        assert dcg_at_k(["B", "A", "C"], {"A": 2, "B": 1}, 3) == pytest.approx(expected)

    def test_a_higher_grade_earlier_scores_more(self):
        grades = {"A": 2, "B": 1}
        assert dcg_at_k(["A", "B"], grades, 2) > dcg_at_k(["B", "A"], grades, 2)

    def test_ungraded_results_contribute_nothing(self):
        assert dcg_at_k(["X", "Y"], {"A": 2}, 2) == 0.0

    def test_results_beyond_k_are_ignored(self):
        assert dcg_at_k(["X", "A"], {"A": 2}, 1) == 0.0

    def test_an_empty_list_scores_zero(self):
        assert dcg_at_k([], {"A": 2}, 10) == 0.0


class TestIdealDcg:
    def test_it_sorts_the_grades_best_first(self):
        """grades {A:1, B:2} -> ideal order is [2, 1]."""
        expected = 3 / math.log2(2) + 1 / math.log2(3)
        assert ideal_dcg_at_k({"A": 1, "B": 2}, 10) == pytest.approx(expected)

    def test_it_is_truncated_at_k(self):
        assert ideal_dcg_at_k({"A": 2, "B": 2}, 1) == pytest.approx(3.0)

    def test_zero_grades_are_ignored(self):
        assert ideal_dcg_at_k({"A": 0}, 10) == 0.0


class TestNdcg:
    def test_the_ideal_ordering_scores_one(self):
        assert ndcg_at_k(["A", "B"], {"A": 2, "B": 1}, 10) == pytest.approx(1.0)

    def test_a_worse_ordering_scores_below_one(self):
        assert ndcg_at_k(["B", "A"], {"A": 2, "B": 1}, 10) < 1.0

    def test_retrieving_nothing_relevant_scores_zero(self):
        assert ndcg_at_k(["X", "Y"], {"A": 2}, 10) == 0.0

    def test_it_is_bounded_between_zero_and_one(self):
        for retrieved in ([], ["A"], ["X", "A", "B"], ["B", "X", "A"]):
            value = ndcg_at_k(retrieved, {"A": 2, "B": 1}, 10)
            assert 0.0 <= value <= 1.0

    def test_the_hand_worked_value(self):
        """retrieved=[B, A], grades={A:2, B:1}.

        DCG   = 1/log2(2) + 3/log2(3) = 1.0000 + 1.8928 = 2.8928
        iDCG  = 3/log2(2) + 1/log2(3) = 3.0000 + 0.6309 = 3.6309
        nDCG  = 0.7967
        """
        dcg = 1 / math.log2(2) + 3 / math.log2(3)
        idcg = 3 / math.log2(2) + 1 / math.log2(3)
        assert ndcg_at_k(["B", "A"], {"A": 2, "B": 1}, 10) == pytest.approx(dcg / idcg)
        assert ndcg_at_k(["B", "A"], {"A": 2, "B": 1}, 10) == pytest.approx(0.7967, abs=1e-4)

    def test_grades_matter_not_just_membership(self):
        """Both orderings find both questions; only the ordering differs."""
        assert ndcg_at_k(["A", "B"], {"A": 2, "B": 1}, 10) != ndcg_at_k(
            ["B", "A"], {"A": 2, "B": 1}, 10
        )

    def test_all_grades_zero_is_rejected(self):
        """Same reason as an empty relevant set: there is nothing to score against."""
        with pytest.raises(EmptyRelevantSet):
            ndcg_at_k(["A"], {"A": 0}, 10)

    def test_an_empty_grade_map_is_rejected(self):
        with pytest.raises(EmptyRelevantSet):
            ndcg_at_k(["A"], {}, 10)


class TestTheMetricsDisagree:
    """Each metric answers a different question - a test that they are not redundant."""

    def test_perfect_recall_with_poor_mrr(self):
        """Everything found, but the first hit is 5th."""
        retrieved = ["X", "Y", "Z", "W", "A", "B"]
        grades = {"A": 2, "B": 2}
        assert recall_at_k(retrieved, {"A", "B"}, 10) == 1.0
        assert reciprocal_rank(retrieved, {"A", "B"}) == pytest.approx(0.2)
        assert ndcg_at_k(retrieved, grades, 10) < 0.6

    def test_perfect_mrr_with_poor_recall(self):
        """The best result is first, but two other relevant ones are missing."""
        retrieved = ["A", "X", "Y"]
        assert reciprocal_rank(retrieved, {"A", "B", "C"}) == 1.0
        assert recall_at_k(retrieved, {"A", "B", "C"}, 10) == pytest.approx(1 / 3)
