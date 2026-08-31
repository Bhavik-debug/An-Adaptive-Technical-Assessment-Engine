"""Reciprocal Rank Fusion, tested as the pure function it is.

No database, no model, no I/O - fusion is arithmetic over two lists of ids, and
keeping it that way is what makes these assertions worth anything.
"""

from __future__ import annotations

import pytest

from app.retrieval.rrf import DEFAULT_RRF_K, reciprocal_rank_fusion


def keys(items):
    return [item.key for item in items]


class TestTheScore:
    def test_one_list_ranks_in_the_order_given(self):
        fused = reciprocal_rank_fusion({"vector": ["a", "b", "c"]})
        assert keys(fused) == ["a", "b", "c"]

    def test_the_score_is_one_over_k_plus_rank(self):
        fused = reciprocal_rank_fusion({"vector": ["a", "b"]}, k=60.0)
        assert fused[0].score == pytest.approx(1 / 61)
        assert fused[1].score == pytest.approx(1 / 62)

    def test_scores_from_both_lists_are_added(self):
        fused = reciprocal_rank_fusion({"vector": ["a"], "lexical": ["a"]}, k=60.0)
        assert fused[0].score == pytest.approx(2 / 61)

    def test_appearing_in_both_beats_being_first_in_one(self):
        """The whole reason to fuse: agreement between two retrievers is evidence."""
        fused = reciprocal_rank_fusion(
            {"vector": ["only_vector", "in_both"], "lexical": ["in_both"]}
        )
        assert keys(fused)[0] == "in_both"

    def test_the_worked_example_from_the_docstring(self):
        fused = reciprocal_rank_fusion(
            {"vector": ["A", "B", "C", "D"], "lexical": ["C", "A", "E", "F"]}, k=60.0
        )
        assert keys(fused)[:2] == ["A", "C"]
        assert fused[0].score == pytest.approx(1 / 61 + 1 / 62)
        assert fused[1].score == pytest.approx(1 / 63 + 1 / 61)


class TestNothingIsLost:
    def test_the_union_of_both_lists_is_returned(self):
        """Fusion truncates nothing; `search.py` does that, and reports it."""
        fused = reciprocal_rank_fusion({"vector": ["a", "b"], "lexical": ["c", "d"]})
        assert set(keys(fused)) == {"a", "b", "c", "d"}

    def test_an_empty_source_is_allowed(self):
        fused = reciprocal_rank_fusion({"vector": ["a"], "lexical": []})
        assert keys(fused) == ["a"]

    def test_all_sources_empty_gives_an_empty_result(self):
        assert reciprocal_rank_fusion({"vector": [], "lexical": []}) == []

    def test_sources_may_have_different_lengths(self):
        fused = reciprocal_rank_fusion({"vector": ["a", "b", "c"], "lexical": ["b"]})
        assert keys(fused)[0] == "b"


class TestEvidenceIsPreserved:
    def test_each_item_records_its_rank_in_every_source(self):
        fused = reciprocal_rank_fusion({"vector": ["a", "b"], "lexical": ["b", "a"]})
        by_key = {item.key: item for item in fused}
        assert by_key["a"].ranks == {"vector": 1, "lexical": 2}
        assert by_key["b"].ranks == {"vector": 2, "lexical": 1}

    def test_a_source_that_missed_a_document_is_absent_not_zero(self):
        fused = reciprocal_rank_fusion({"vector": ["a"], "lexical": ["b"]})
        by_key = {item.key: item for item in fused}
        assert "lexical" not in by_key["a"].ranks
        assert by_key["a"].sources == ("vector",)

    def test_sources_is_sorted_so_it_is_stable(self):
        fused = reciprocal_rank_fusion({"vector": ["a"], "lexical": ["a"]})
        assert fused[0].sources == ("lexical", "vector")


class TestDeterminism:
    def test_ties_are_broken_by_best_rank_then_id(self):
        """Symmetric evidence: both score 1/61 + 1/62, both have a best rank of 1."""
        fused = reciprocal_rank_fusion({"vector": ["b", "a"], "lexical": ["a", "b"]})
        assert fused[0].score == pytest.approx(fused[1].score)
        assert keys(fused) == ["a", "b"], "the id decides, ascending"

    def test_a_better_best_rank_wins_before_the_id(self):
        """Three items tie on score; the two with a rank-1 appearance come first.

        With k = 1: a rank-1 appearance is worth 1/2, and a rank-3 appearance in
        each of two lists is worth 1/4 + 1/4 - also 1/2. So `a_pair` ties with
        `q_first` and `z_first` on score, and is placed below both because its
        best rank is 3 rather than 1. Note that `a_pair` sorts *first*
        alphabetically, so only the best-rank rule can produce this order.
        """
        fused = reciprocal_rank_fusion(
            {
                "vector": ["z_first", "filler_v", "a_pair"],
                "lexical": ["q_first", "filler_l", "a_pair"],
            },
            k=1.0,
        )
        by_key = {item.key: item for item in fused}
        assert by_key["a_pair"].score == pytest.approx(by_key["z_first"].score)
        assert by_key["a_pair"].best_rank == 3
        assert keys(fused)[:3] == ["q_first", "z_first", "a_pair"]

    def test_the_result_does_not_depend_on_source_insertion_order(self):
        one = reciprocal_rank_fusion({"vector": ["a", "b"], "lexical": ["b", "c"]})
        two = reciprocal_rank_fusion({"lexical": ["b", "c"], "vector": ["a", "b"]})
        assert keys(one) == keys(two)
        assert [i.score for i in one] == [i.score for i in two]

    def test_repeated_runs_are_identical(self):
        rankings = {"vector": ["a", "b", "c"], "lexical": ["c", "d", "a"]}
        assert keys(reciprocal_rank_fusion(rankings)) == keys(reciprocal_rank_fusion(rankings))

    def test_a_duplicate_within_one_list_keeps_its_best_rank_and_scores_once(self):
        fused = reciprocal_rank_fusion({"vector": ["a", "b", "a"]}, k=60.0)
        by_key = {item.key: item for item in fused}
        assert by_key["a"].ranks == {"vector": 1}
        assert by_key["a"].score == pytest.approx(1 / 61), "not counted twice"


class TestTheConstant:
    def test_the_default_is_the_published_value(self):
        assert DEFAULT_RRF_K == 60.0

    def test_a_large_k_flattens_the_difference_between_adjacent_ranks(self):
        """k is damping: the bigger it is, the less a single first place is worth."""
        tight = reciprocal_rank_fusion({"v": ["a", "b"]}, k=1.0)
        flat = reciprocal_rank_fusion({"v": ["a", "b"]}, k=1000.0)
        assert tight[0].score / tight[1].score > flat[0].score / flat[1].score

    def test_k_decides_whether_one_first_place_beats_two_third_places(self):
        """The clearest statement of what k actually controls.

        `solo` is ranked 1st by one retriever and not returned by the other.
        `both` is ranked 3rd by each. Which should win is a judgement about how
        much weight a single confident vote deserves against agreement between
        two - and k is exactly that dial.

        At the published k = 60 the ranks barely differ, so appearing twice
        dominates and `both` wins. At k = 0.5 the gap between rank 1 and rank 3
        is enormous, so `solo` wins. This is why k = 60 is the right default
        here: agreement between two unrelated retrievers is the signal worth
        rewarding.
        """
        rankings = {"vector": ["solo", "x", "both"], "lexical": ["y", "z", "both"]}

        damped = reciprocal_rank_fusion(rankings, k=60.0)
        assert keys(damped)[0] == "both"

        sharp = reciprocal_rank_fusion(rankings, k=0.5)
        assert keys(sharp)[0] == "solo"

    @pytest.mark.parametrize("bad", [0.0, -1.0])
    def test_a_non_positive_k_is_rejected(self, bad):
        """k=0 would make rank 0 a division by zero, and k<0 inverts the ordering."""
        with pytest.raises(ValueError, match="must be positive"):
            reciprocal_rank_fusion({"v": ["a"]}, k=bad)
