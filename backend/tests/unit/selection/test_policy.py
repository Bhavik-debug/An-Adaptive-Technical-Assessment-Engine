"""Ranking and epsilon-greedy choice.

Nothing here is flaky. Every test either scripts the ``random()`` draw
explicitly (``ScriptedRandom``) or seeds a real ``random.Random``, so the
exploitation path, the exploration path and the top-5 restriction are all
asserted deterministically - which is the entire reason the generator is a
parameter instead of the module-level ``random``.
"""

from __future__ import annotations

import random

import pytest

from app.selection.objective import score_items
from app.selection.policy import (
    EPSILON,
    EXPLORATION_POOL_SIZE,
    choose_next,
    epsilon_greedy_select,
    rank_items,
)
from tests.unit.selection.conftest import ScriptedRandom, item, state

# Ten items whose only difference is difficulty, so the score ordering is the
# information ordering: b == theta scores highest, and it degrades outwards.
BANK = [item(f"q{i:02d}", b=round(-2.0 + 0.4 * i, 2), seconds=60) for i in range(11)]


def ranked_bank():
    current = state(theta=0.0, targets={"systems": 20}, time_left=1800.0)
    return rank_items(score_items(BANK, current))


class TestEpsilonAndPool:
    def test_epsilon_is_one_tenth(self):
        assert EPSILON == 0.10

    def test_the_exploration_pool_is_five(self):
        assert EXPLORATION_POOL_SIZE == 5


class TestRanking:
    def test_highest_score_first(self):
        totals = [entry.total for entry in ranked_bank()]
        assert totals == sorted(totals, reverse=True)

    def test_the_item_matched_to_theta_ranks_first(self):
        """theta is 0.0, and q05 is the item with b == 0.0."""
        assert ranked_bank()[0].item_id == "q05"

    def test_ties_are_broken_by_id_so_a_run_is_reproducible(self):
        current = state(theta=0.0, targets={"systems": 9})
        tied = [item("q_b", b=0.5), item("q_a", b=0.5), item("q_c", b=0.5)]
        assert [e.item_id for e in rank_items(score_items(tied, current))] == [
            "q_a",
            "q_b",
            "q_c",
        ]

    def test_ranking_an_empty_list_is_an_empty_list(self):
        assert rank_items([]) == []


class TestExploitation:
    def test_a_draw_at_or_above_epsilon_takes_the_argmax(self):
        ranked = ranked_bank()
        result = epsilon_greedy_select(ranked, rng=ScriptedRandom([0.10]))
        assert result is not None
        assert result.explored is False
        assert result.chosen is ranked[0]

    def test_a_draw_far_above_epsilon_takes_the_argmax(self):
        ranked = ranked_bank()
        result = epsilon_greedy_select(ranked, rng=ScriptedRandom([0.99]))
        assert result is not None and result.item_id == ranked[0].item_id

    def test_epsilon_zero_never_explores(self):
        ranked = ranked_bank()
        for seed in range(50):
            result = epsilon_greedy_select(ranked, rng=random.Random(seed), epsilon=0.0)
            assert result is not None
            assert result.explored is False
            assert result.item_id == ranked[0].item_id


class TestExploration:
    def test_a_draw_below_epsilon_takes_the_exploration_branch(self):
        result = epsilon_greedy_select(ranked_bank(), rng=ScriptedRandom([0.0]))
        assert result is not None and result.explored is True

    def test_exploration_only_ever_chooses_from_the_top_five(self):
        ranked = ranked_bank()
        top_five = {entry.item_id for entry in ranked[:EXPLORATION_POOL_SIZE]}
        assert len(ranked) > EXPLORATION_POOL_SIZE  # the restriction has something to do

        chosen = set()
        for seed in range(200):
            result = epsilon_greedy_select(ranked, rng=ScriptedRandom([0.0], seed=seed))
            assert result is not None
            chosen.add(result.item_id)
        assert chosen <= top_five

    def test_exploration_can_reach_every_one_of_the_top_five(self):
        """Uniformly from the top 5 - not "usually the first one"."""
        ranked = ranked_bank()
        top_five = {entry.item_id for entry in ranked[:EXPLORATION_POOL_SIZE]}
        chosen = {
            epsilon_greedy_select(ranked, rng=ScriptedRandom([0.0], seed=seed)).item_id
            for seed in range(200)
        }
        assert chosen == top_five

    def test_an_item_outside_the_top_five_is_never_reachable(self):
        ranked = ranked_bank()
        outside = {entry.item_id for entry in ranked[EXPLORATION_POOL_SIZE:]}
        assert outside  # there is something outside to exclude
        for seed in range(200):
            result = epsilon_greedy_select(ranked, rng=ScriptedRandom([0.0], seed=seed))
            assert result is not None and result.item_id not in outside

    def test_epsilon_one_always_explores(self):
        for seed in range(20):
            result = epsilon_greedy_select(ranked_bank(), rng=random.Random(seed), epsilon=1.0)
            assert result is not None and result.explored is True

    def test_a_pool_smaller_than_five_explores_within_what_there_is(self):
        current = state(theta=0.0, targets={"systems": 9})
        ranked = rank_items(score_items(BANK[:3], current))
        result = epsilon_greedy_select(ranked, rng=ScriptedRandom([0.0]))
        assert result is not None
        assert len(result.top) == 3
        assert result.item_id in {entry.item_id for entry in ranked}

    def test_a_single_candidate_is_chosen_by_either_branch(self):
        current = state(theta=0.0)
        ranked = rank_items(score_items([item("only")], current))
        for draw in (0.0, 0.99):
            result = epsilon_greedy_select(ranked, rng=ScriptedRandom([draw]))
            assert result is not None and result.item_id == "only"


class TestDeterminism:
    def test_the_same_seed_gives_the_same_sequence_of_choices(self):
        ranked = ranked_bank()

        def run(seed: int) -> list[tuple[str, bool]]:
            rng = random.Random(seed)
            out = []
            for _ in range(40):
                result = epsilon_greedy_select(ranked, rng=rng)
                assert result is not None
                out.append((result.item_id, result.explored))
            return out

        assert run(12345) == run(12345)

    def test_different_seeds_do_diverge(self):
        """Exploration is real: two seeds must not produce identical runs."""
        ranked = ranked_bank()

        def run(seed: int) -> list[str]:
            rng = random.Random(seed)
            return [epsilon_greedy_select(ranked, rng=rng).item_id for _ in range(60)]

        assert run(1) != run(2)

    def test_exactly_one_draw_is_made_per_exploitation(self):
        """A stable draw count is what makes a seeded run reproducible."""
        rng = ScriptedRandom([0.9, 0.9, 0.9])
        ranked = ranked_bank()
        for _ in range(3):
            epsilon_greedy_select(ranked, rng=rng)
        assert rng.calls == 3

    def test_roughly_one_selection_in_ten_explores(self):
        """Not a distributional assertion with a tolerance - a fixed seed and an
        exact count, so it can never flake."""
        ranked = ranked_bank()
        rng = random.Random(2024)
        explored = sum(1 for _ in range(1000) if epsilon_greedy_select(ranked, rng=rng).explored)
        assert explored == 97


class TestValidation:
    def test_an_empty_pool_returns_none_rather_than_raising(self):
        assert epsilon_greedy_select([]) is None

    @pytest.mark.parametrize("bad", [-0.1, 1.1])
    def test_an_epsilon_outside_zero_and_one_is_rejected(self, bad):
        with pytest.raises(ValueError, match=r"epsilon must be in \[0, 1\]"):
            epsilon_greedy_select(ranked_bank(), epsilon=bad)

    def test_a_pool_size_below_one_is_rejected(self):
        with pytest.raises(ValueError, match="pool_size must be at least 1"):
            epsilon_greedy_select(ranked_bank(), pool_size=0)


class TestChooseNext:
    def test_it_filters_before_it_scores(self):
        """An ineligible item cannot be chosen however well it would score.

        `q_perfect` sits exactly on theta, so it would be the argmax - and it
        has already been asked, so it must not appear at all.
        """
        current = state(theta=0.0, targets={"systems": 9}, asked=[item("q_perfect", b=0.0)])
        candidates = [item("q_perfect", b=0.0), item("q_other", b=0.6)]
        result = choose_next(current, candidates, rng=ScriptedRandom([0.9]))
        assert result is not None
        assert result.item_id == "q_other"
        assert result.pool_size == 1

    def test_it_returns_none_when_nothing_survives_the_hard_filter(self):
        current = state(theta=0.0, time_left=30.0)
        assert choose_next(current, [item("q1", seconds=300)]) is None

    def test_pool_size_reports_the_survivors_not_the_candidates(self):
        current = state(theta=0.0, targets={"systems": 9}, time_left=1800.0)
        candidates = BANK + [item("way_too_hard", b=2.9)]
        result = choose_next(current, candidates, rng=ScriptedRandom([0.9]))
        assert result is not None
        assert result.pool_size == len([i for i in BANK if abs(i.difficulty_b) <= 1.5])

    def test_the_top_five_are_carried_for_the_event_log(self):
        current = state(theta=0.0, targets={"systems": 20}, time_left=1800.0)
        result = choose_next(current, BANK, rng=ScriptedRandom([0.9]))
        assert result is not None
        assert len(result.top) == EXPLORATION_POOL_SIZE
        assert result.top[0].item_id == result.item_id
