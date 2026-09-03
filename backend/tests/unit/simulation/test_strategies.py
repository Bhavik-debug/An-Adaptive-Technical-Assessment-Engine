"""The three policies, and the properties that make the comparison mean anything.

The load-bearing tests here are the ones about what a strategy *cannot* do:
the adaptive policy must be the shipped Day 12 selector rather than a
look-alike, the fixed policy must not react to theta, and none of the three may
see the simulator's ground truth.
"""

from __future__ import annotations

import dataclasses
import inspect
import random

import pytest

from app.ability import AbilityState
from app.selection import (
    DEFAULT_WEIGHTS,
    DIFFICULTY_WINDOW,
    EPSILON,
    SelectionState,
    choose_next,
    filter_eligible,
)
from app.simulation.environment import build_candidate, build_environment
from app.simulation.strategies import (
    ADAPTIVE,
    FIXED,
    RANDOM,
    STRATEGY_ORDER,
    AdaptiveStrategy,
    FixedStrategy,
    RandomStrategy,
    build_strategies,
    fixed_sequence,
)
from tests.unit.simulation.conftest import TINY

ENV = build_environment(TINY)
BANK = ENV.bank
QUOTAS = TINY.quotas


def state(**overrides) -> SelectionState:
    kwargs = {
        "ability": {},
        "coverage_targets": QUOTAS,
        "jd_weights": TINY.jd_weights,
        "time_left_s": TINY.time_budget_s,
        "asked": (),
        "resume": None,
    }
    kwargs.update(overrides)
    return SelectionState(**kwargs)


class TestTheInterface:
    def test_all_three_take_only_state_bank_and_rng(self):
        """No candidate argument: a strategy has no reference through which the
        simulator's ground truth could be read, even by accident."""
        for strategy in build_strategies(BANK, QUOTAS).values():
            params = list(inspect.signature(strategy.select).parameters)
            assert params == ["state", "bank", "rng"]

    def test_the_registry_covers_the_reported_order(self):
        assert set(build_strategies(BANK, QUOTAS)) == set(STRATEGY_ORDER)
        assert STRATEGY_ORDER[0] == ADAPTIVE


class TestAdaptiveIsTheProductionSelector:
    def test_it_returns_exactly_what_choose_next_returns(self):
        """Not "behaves similarly" - the same call, so there is no second
        implementation of the selector to drift from the first."""
        current = state()
        expected = choose_next(current, BANK, rng=random.Random(4))
        actual = AdaptiveStrategy().select(current, BANK, random.Random(4))
        assert expected is not None
        assert actual is not None
        assert actual.id == expected.item.id

    def test_it_uses_the_production_weights_by_default(self):
        assert AdaptiveStrategy().weights == DEFAULT_WEIGHTS
        assert AdaptiveStrategy().epsilon == EPSILON
        assert AdaptiveStrategy().difficulty_window == DIFFICULTY_WINDOW

    def test_it_respects_every_hard_constraint(self):
        asked = tuple(BANK[:3])
        current = state(asked=asked, time_left_s=200.0)
        chosen = AdaptiveStrategy().select(current, BANK, random.Random(1))
        assert chosen is not None
        assert chosen.id not in {i.id for i in asked}
        assert current.quota_remaining(chosen.topic_key) > 0
        assert abs(chosen.difficulty_b - current.theta_for(chosen.subtopic_key)) <= 1.5
        assert chosen.time_estimate_s <= current.time_left_s

    def test_it_prefers_an_item_near_the_current_estimate(self):
        """The information term, visible end to end: with theta pinned high the
        selector must stop offering easy questions."""
        high = state(ability={s: AbilityState(theta=2.0, rd=0.5) for s in TINY.subtopics})
        chosen = AdaptiveStrategy().select(high, BANK, random.Random(2))
        assert chosen is not None
        assert chosen.difficulty_b > 0.5

    def test_an_ablated_objective_changes_what_it_picks(self):
        """If it did not, the ablation would be measuring nothing.

        Constructed so the outcome is arithmetic rather than luck: theta is
        equal everywhere, so the information term cannot discriminate between
        subtopics, and the contest is JD weight against resume affinity.
        `queues` sits in the lowest-weighted topic (systems, 0.5) and
        `arrays` in the highest (algorithms, 0.9), a JD gap worth
        0.25 x 0.4 = 0.10; a resume affinity of 1.0 on `queues` is worth
        0.15 x 1.0 = 0.15 and must therefore flip the choice.
        """
        from app.selection import ResumeProfile

        big = build_environment(dataclasses.replace(TINY, items_per_subtopic=32))
        level = state(
            resume=ResumeProfile(topic_affinity={"queues": 1.0}),
            ability={s: AbilityState(theta=0.4, rd=0.8) for s in TINY.subtopics},
        )
        full = AdaptiveStrategy(epsilon=0.0)
        without = AdaptiveStrategy(weights=DEFAULT_WEIGHTS.without("resume"), epsilon=0.0)

        with_resume = full.select(level, big.bank, random.Random(0))
        no_resume = without.select(level, big.bank, random.Random(0))
        assert with_resume is not None and no_resume is not None
        assert with_resume.subtopic_key == "queues"
        assert no_resume.topic_key == "algorithms"

    def test_it_returns_none_when_nothing_is_eligible(self):
        assert AdaptiveStrategy().select(state(time_left_s=0.0), BANK, random.Random(0)) is None


class TestRandom:
    def test_it_draws_from_the_same_eligible_pool_the_adaptive_policy_scores(self):
        current = state()
        eligible = {i.id for i in filter_eligible(BANK, current)}
        for seed in range(40):
            chosen = RandomStrategy().select(current, BANK, random.Random(seed))
            assert chosen is not None and chosen.id in eligible

    def test_it_is_seeded_and_reproducible(self):
        current = state()
        first = RandomStrategy().select(current, BANK, random.Random(9))
        second = RandomStrategy().select(current, BANK, random.Random(9))
        assert first is not None and first.id == second.id

    def test_different_seeds_reach_different_items(self):
        current = state()
        picks = {RandomStrategy().select(current, BANK, random.Random(s)).id for s in range(60)}
        assert len(picks) > 5

    def test_it_does_not_rank_by_the_day_twelve_objective(self):
        """The point of the baseline: over many seeds it must not concentrate on
        whatever the weighted score would have chosen."""
        current = state()
        best = choose_next(current, BANK, rng=random.Random(0), epsilon=0.0)
        assert best is not None
        picks = [RandomStrategy().select(current, BANK, random.Random(s)).id for s in range(100)]
        assert picks.count(best.item.id) < 20

    def test_it_returns_none_when_nothing_is_eligible(self):
        assert RandomStrategy().select(state(time_left_s=0.0), BANK, random.Random(0)) is None

    def test_the_draw_does_not_depend_on_bank_order(self):
        current = state()
        shuffled = list(BANK)
        random.Random(3).shuffle(shuffled)
        assert (
            RandomStrategy().select(current, BANK, random.Random(5)).id
            == RandomStrategy().select(current, shuffled, random.Random(5)).id
        )


class TestFixedSequence:
    def test_it_covers_the_whole_bank_exactly_once(self):
        sequence = fixed_sequence(BANK, QUOTAS)
        assert len(sequence) == len(BANK)
        assert len({i.id for i in sequence}) == len(BANK)

    def test_it_is_deterministic(self):
        assert [i.id for i in fixed_sequence(BANK, QUOTAS)] == [
            i.id for i in fixed_sequence(BANK, QUOTAS)
        ]

    def test_it_does_not_depend_on_the_order_of_the_bank(self):
        shuffled = list(BANK)
        random.Random(1).shuffle(shuffled)
        assert [i.id for i in fixed_sequence(BANK, QUOTAS)] == [
            i.id for i in fixed_sequence(shuffled, QUOTAS)
        ]

    def test_the_paper_satisfies_the_blueprint_by_construction(self):
        """The first `budget` items already meet every quota, so the topic cap
        never binds for this policy and cannot disadvantage it."""
        sequence = fixed_sequence(BANK, QUOTAS)
        head = sequence[: sum(QUOTAS.values())]
        for topic, quota in QUOTAS.items():
            assert sum(1 for i in head if i.topic_key == topic) == quota

    def test_the_paper_spans_the_difficulty_range_rather_than_taking_the_easiest(self):
        """Read literally, "easy -> hard" over a 192-item bank means the twenty
        easiest questions in existence - a deliberately feeble baseline. The
        paper takes evenly spaced percentiles instead."""
        big = build_environment(dataclasses.replace(TINY, items_per_subtopic=32))
        sequence = fixed_sequence(big.bank, QUOTAS)
        head = sequence[: sum(QUOTAS.values())]
        assert min(i.difficulty_b for i in head) < -1.5
        assert max(i.difficulty_b for i in head) > 1.5

    def test_within_a_topic_the_paper_runs_easy_to_hard(self):
        sequence = fixed_sequence(BANK, QUOTAS)
        head = sequence[: sum(QUOTAS.values())]
        for topic in QUOTAS:
            bs = [i.difficulty_b for i in head if i.topic_key == topic]
            assert bs == sorted(bs)


class TestFixedIsNotAdaptive:
    def test_the_same_session_gets_the_same_item_whatever_theta_says(self):
        """The defining property. Two states differing only in the ability
        estimate - one candidate who has aced everything, one who has failed
        everything - must produce the same choice."""
        strategy = FixedStrategy(sequence=fixed_sequence(BANK, QUOTAS))
        genius = state(ability={s: AbilityState(theta=2.9, rd=0.3) for s in TINY.subtopics})
        novice = state(ability={s: AbilityState(theta=-2.9, rd=0.3) for s in TINY.subtopics})
        assert strategy.select(genius, BANK, random.Random(0)).id == (
            strategy.select(novice, BANK, random.Random(0)).id
        )

    def test_it_ignores_the_random_generator_entirely(self):
        strategy = FixedStrategy(sequence=fixed_sequence(BANK, QUOTAS))
        current = state()
        picks = {strategy.select(current, BANK, random.Random(s)).id for s in range(20)}
        assert len(picks) == 1

    def test_it_still_honours_the_constraints_that_do_not_depend_on_theta(self):
        strategy = FixedStrategy(sequence=fixed_sequence(BANK, QUOTAS))
        first = strategy.select(state(), BANK, random.Random(0))
        assert first is not None
        after = state(asked=(first,))
        assert strategy.select(after, BANK, random.Random(0)).id != first.id

    def test_it_skips_an_item_that_does_not_fit_the_clock(self):
        strategy = FixedStrategy(sequence=fixed_sequence(BANK, QUOTAS))
        tight = state(time_left_s=60.0)
        chosen = strategy.select(tight, BANK, random.Random(0))
        if chosen is not None:
            assert chosen.time_estimate_s <= 60

    def test_it_does_not_apply_the_difficulty_window(self):
        """Applying it would make the "fixed" policy adapt, since the window is
        a function of the running estimate."""
        strategy = FixedStrategy(sequence=fixed_sequence(BANK, QUOTAS))
        extreme = state(ability={s: AbilityState(theta=2.9, rd=0.3) for s in TINY.subtopics})
        chosen = strategy.select(extreme, BANK, random.Random(0))
        assert chosen is not None
        assert abs(chosen.difficulty_b - 2.9) > DIFFICULTY_WINDOW

    def test_an_empty_paper_returns_none(self):
        assert FixedStrategy().select(state(), BANK, random.Random(0)) is None


class TestNoGroundTruthLeaks:
    def test_two_candidates_who_differ_only_in_ability_get_the_same_first_item(self):
        """The strongest available check. Before any answer exists there is no
        legitimate signal distinguishing two candidates with the same resume, so
        a policy that picked differently could only be reading ground truth."""
        weak = build_candidate(TINY, 0)
        strong = dataclasses.replace(weak, id="strong", true_theta={s: 2.5 for s in TINY.subtopics})
        assert weak.true_theta != strong.true_theta
        for strategy in build_strategies(BANK, QUOTAS).values():
            first = strategy.select(state(resume=weak.resume), BANK, random.Random(0))
            second = strategy.select(state(resume=strong.resume), BANK, random.Random(0))
            assert first is not None and second is not None
            assert first.id == second.id, strategy.name

    def test_the_selection_state_carries_no_candidate(self):
        """Structural, not behavioural: there is no field to leak through."""
        fields = {f.name for f in dataclasses.fields(SelectionState)}
        assert "candidate" not in fields
        assert not any("true" in name or "truth" in name for name in fields)

    @pytest.mark.parametrize("name", [ADAPTIVE, RANDOM, FIXED])
    def test_no_strategy_imports_the_response_model_or_the_population(self, name):
        import app.simulation.strategies as module

        source = inspect.getsource(module)
        assert "true_theta" not in source
        assert "graded_score" not in source
        assert "SyntheticCandidate" not in source
