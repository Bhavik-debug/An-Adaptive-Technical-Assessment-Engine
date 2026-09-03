"""The session loop, the experiment, and the ablation.

The invariants asserted here are the ones that make the published numbers
trustworthy: a session obeys Day 12's constraints turn by turn, the three
policies really do sit the same exam, an ablation really does remove the
component it names, and running the whole thing twice gives the same answer.

No test here asserts that adaptive beats random. That is an empirical
hypothesis about a synthetic world, not a software invariant, and pinning it
would turn a change in the environment into a test failure that looks like a
bug.
"""

from __future__ import annotations

import dataclasses
import math

import pytest

from app.selection import COMPONENT_NAMES, DEFAULT_WEIGHTS, ineligibility_reason
from app.simulation.config import MAIN_CONFIG
from app.simulation.environment import build_environment
from app.simulation.response import graded_score
from app.simulation.runner import (
    ABLATION_FULL,
    STOP_POOL_EXHAUSTED,
    ablation_variants,
    run_ablation,
    run_experiment,
    run_session,
    summarise_sessions,
)
from app.simulation.strategies import ADAPTIVE, FIXED, RANDOM, AdaptiveStrategy, build_strategies
from tests.unit.simulation.conftest import TINY

ENV = build_environment(TINY)
POLICIES = build_strategies(ENV.bank, TINY.quotas)


def run_all(config=TINY, environment=ENV):
    return run_experiment(config, environment=environment)


class TestOneSession:
    def test_it_asks_at_most_the_item_budget(self):
        for policy in POLICIES.values():
            for candidate in ENV.candidates:
                assert run_session(ENV, candidate, policy).items_asked <= TINY.item_budget

    def test_it_never_repeats_a_question(self):
        for policy in POLICIES.values():
            for candidate in ENV.candidates:
                result = run_session(ENV, candidate, policy)
                assert len(set(result.asked_ids)) == len(result.asked_ids)

    def test_it_always_records_a_stopping_reason(self):
        for policy in POLICIES.values():
            for candidate in ENV.candidates:
                assert run_session(ENV, candidate, policy).stop_reasons

    def test_it_never_overspends_the_clock(self):
        for policy in POLICIES.values():
            for candidate in ENV.candidates:
                assert run_session(ENV, candidate, policy).time_used_s <= TINY.time_budget_s

    def test_it_never_exceeds_a_topic_quota(self):
        for policy in POLICIES.values():
            for candidate in ENV.candidates:
                result = run_session(ENV, candidate, policy)
                by_topic = {item: 0 for item in TINY.topics}
                for item_id in result.asked_ids:
                    subtopic = item_id.rsplit("-", 1)[0]
                    by_topic[TINY.parent_of[subtopic]] += 1
                for topic, quota in TINY.quotas.items():
                    assert by_topic[topic] <= quota

    def test_every_adaptive_choice_was_eligible_when_it_was_made(self):
        """Replays Day 12's own predicate against the state at each turn - the
        simulation cannot quietly bypass a hard constraint."""
        from app.selection import SelectionState

        candidate = ENV.candidates[0]
        result = run_session(ENV, candidate, POLICIES[ADAPTIVE])
        by_id = ENV.items_by_id
        from app.ability import update_ability

        ability: dict = {}
        asked: tuple = ()
        time_left = TINY.time_budget_s
        for item_id in result.asked_ids:
            state = SelectionState(
                ability=ability,
                coverage_targets=TINY.quotas,
                jd_weights=TINY.jd_weights,
                time_left_s=time_left,
                asked=asked,
                resume=candidate.resume,
            )
            item = by_id[item_id]
            assert ineligibility_reason(item, state) is None
            update = update_ability(
                state.ability_for(item.subtopic_key),
                difficulty=item.difficulty_b,
                score=graded_score(TINY.seed, candidate, item),
            )
            ability = {**ability, item.subtopic_key: update.after}
            asked = (*asked, item)
            time_left -= item.time_estimate_s

    def test_trajectories_line_up_with_the_item_count(self):
        result = run_session(ENV, ENV.candidates[0], POLICIES[ADAPTIVE])
        assert len(result.mae_trajectory) == result.items_asked + 1
        assert len(result.rd_trajectory) == result.items_asked + 1
        assert len(result.difficulty_gaps) == result.items_asked

    def test_the_final_error_matches_the_end_of_the_trajectory(self):
        result = run_session(ENV, ENV.candidates[1], POLICIES[RANDOM])
        assert result.mae == pytest.approx(result.mae_trajectory[-1])

    def test_ground_truth_is_recorded_but_was_never_in_the_loop(self):
        candidate = ENV.candidates[2]
        result = run_session(ENV, candidate, POLICIES[ADAPTIVE])
        assert result.ground_truth_theta == dict(candidate.true_theta)
        assert set(result.final_theta) == set(TINY.subtopics)

    def test_unmeasured_subtopics_read_back_as_the_cold_start_prior(self):
        from app.simulation.config import INITIAL_RD, INITIAL_THETA

        result = run_session(ENV, ENV.candidates[0], POLICIES[ADAPTIVE])
        untouched = set(TINY.subtopics) - {i.rsplit("-", 1)[0] for i in result.asked_ids}
        for subtopic in untouched:
            assert result.final_theta[subtopic] == INITIAL_THETA
            assert result.final_rd[subtopic] == INITIAL_RD

    def test_an_empty_pool_is_reported_rather_than_raised(self):
        starved = dataclasses.replace(TINY, time_budget_s=1.0)
        environment = build_environment(starved)
        policy = AdaptiveStrategy()
        result = run_session(environment, environment.candidates[0], policy)
        assert result.items_asked == 0
        assert result.stop_reasons == (STOP_POOL_EXHAUSTED,)
        assert result.primary_stop_reason == STOP_POOL_EXHAUSTED

    def test_it_is_reproducible(self):
        first = run_session(ENV, ENV.candidates[0], POLICIES[ADAPTIVE])
        second = run_session(ENV, ENV.candidates[0], POLICIES[ADAPTIVE])
        assert first == second


class TestFairness:
    def test_every_policy_sees_the_same_bank_and_the_same_people(self):
        result = run_all()
        for name in (ADAPTIVE, RANDOM, FIXED):
            sessions = result.sessions_for(name)
            assert [s.candidate_id for s in sessions] == [c.id for c in ENV.candidates]
            assert [s.ground_truth_theta for s in sessions] == [
                dict(c.true_theta) for c in ENV.candidates
            ]

    def test_the_same_question_always_gets_the_same_answer(self):
        """Common random numbers: any difference between policies is caused by
        which questions they chose, never by luckier grading."""
        candidate = ENV.candidates[0]
        item = ENV.bank[7]
        assert graded_score(TINY.seed, candidate, item) == graded_score(TINY.seed, candidate, item)

    def test_every_policy_starts_from_the_same_cold_start(self):
        result = run_all()
        for session in result.sessions:
            assert (
                session.mae_trajectory[0] == pytest.approx(result.sessions[0].mae_trajectory[0])
                or session.candidate_id != result.sessions[0].candidate_id
            )
        by_candidate: dict[str, set[float]] = {}
        for session in result.sessions:
            by_candidate.setdefault(session.candidate_id, set()).add(
                round(session.mae_trajectory[0], 12)
            )
        assert all(len(values) == 1 for values in by_candidate.values())

    def test_every_policy_gets_the_same_budgets(self):
        result = run_all()
        for session in result.sessions:
            assert session.items_asked <= TINY.item_budget
            assert session.time_used_s <= TINY.time_budget_s


class TestExperiment:
    def test_it_runs_every_policy_over_every_candidate(self):
        result = run_all()
        assert len(result.sessions) == 3 * TINY.candidate_count
        assert len(result.summaries) == 3

    def test_summaries_are_in_the_reported_order(self):
        assert [s.strategy for s in run_all().summaries] == [ADAPTIVE, RANDOM, FIXED]

    def test_summary_lookup(self):
        result = run_all()
        assert result.summary(ADAPTIVE).strategy == ADAPTIVE
        with pytest.raises(KeyError, match="no summary"):
            result.summary("nope")

    def test_no_session_is_dropped_from_a_summary(self):
        result = run_all()
        for summary in result.summaries:
            assert summary.n_sessions == TINY.candidate_count
            assert summary.mae.n == TINY.candidate_count
            assert sum(summary.stop_reasons.values()) == TINY.candidate_count

    def test_metrics_are_finite_and_in_range(self):
        for summary in run_all().summaries:
            assert math.isfinite(summary.mae.mean)
            assert summary.mae.mean >= 0.0
            assert 0.0 <= summary.coverage_compliance <= 1.0
            assert 0.0 <= summary.subtopics_measured.mean <= len(TINY.subtopics)
            assert all(math.isfinite(v) for v in summary.mae_curve)

    def test_the_whole_experiment_is_reproducible(self):
        """Same seed in, byte-identical results out."""
        first, second = run_all(), run_all()
        assert [dataclasses.astuple(s) for s in first.sessions] == [
            dataclasses.astuple(s) for s in second.sessions
        ]

    def test_it_rebuilds_the_same_environment_from_the_config_alone(self):
        assert [dataclasses.astuple(s) for s in run_experiment(TINY).sessions] == [
            dataclasses.astuple(s) for s in run_all().sessions
        ]

    def test_a_different_seed_gives_different_results(self):
        other = dataclasses.replace(TINY, seed=TINY.seed + 1)
        assert [s.mae for s in run_experiment(other).sessions] != [
            s.mae for s in run_all().sessions
        ]

    def test_summarising_nothing_is_rejected(self):
        with pytest.raises(ValueError, match="no sessions to summarise"):
            summarise_sessions("x", [])


class TestAblation:
    def test_it_covers_the_full_objective_and_one_removal_per_component(self):
        variants = ablation_variants(COMPONENT_NAMES)
        assert set(variants) == {ABLATION_FULL, *(f"no_{c}" for c in COMPONENT_NAMES)}
        assert len(variants) == 7

    def test_the_full_variant_is_the_production_objective_untouched(self):
        assert ablation_variants(COMPONENT_NAMES)[ABLATION_FULL] == DEFAULT_WEIGHTS

    @pytest.mark.parametrize("component", COMPONENT_NAMES)
    def test_each_variant_zeroes_exactly_the_component_it_names(self, component):
        weights = ablation_variants(COMPONENT_NAMES)[f"no_{component}"]
        assert getattr(weights, component) == 0.0
        for other in COMPONENT_NAMES:
            if other != component:
                assert getattr(weights, other) == getattr(DEFAULT_WEIGHTS, other)

    def test_the_production_defaults_are_never_mutated(self):
        """The ablation varies a copy. If it edited module state, every later
        caller in the process would silently get an ablated objective."""
        ablation_variants(COMPONENT_NAMES)
        run_ablation(TINY, environment=ENV)
        assert DEFAULT_WEIGHTS.information == 0.40
        assert DEFAULT_WEIGHTS.jd == 0.25
        assert DEFAULT_WEIGHTS.resume == 0.15
        assert DEFAULT_WEIGHTS.coverage == 0.15
        assert DEFAULT_WEIGHTS.redundancy == 0.10
        assert DEFAULT_WEIGHTS.time == 0.05

    def test_an_unknown_component_is_rejected(self):
        with pytest.raises(ValueError, match="unknown component"):
            DEFAULT_WEIGHTS.without("nonsense")

    def test_it_produces_one_summary_per_variant(self):
        result = run_ablation(TINY, environment=ENV)
        assert len(result.summaries) == 7
        assert result.summary(ABLATION_FULL).n_sessions == TINY.candidate_count
        with pytest.raises(KeyError, match="no summary for variant"):
            result.summary("nope")

    def test_the_full_variant_reproduces_the_headline_adaptive_run(self):
        """Same environment, same seeds, same policy - so the ablation's
        baseline row and the comparison table's adaptive row must agree."""
        headline = run_all().summary(ADAPTIVE)
        baseline = run_ablation(TINY, environment=ENV).summary(ABLATION_FULL)
        assert baseline.mae.mean == pytest.approx(headline.mae.mean)
        assert baseline.items_asked.mean == pytest.approx(headline.items_asked.mean)

    def test_it_is_reproducible(self):
        first = run_ablation(TINY, environment=ENV)
        second = run_ablation(TINY, environment=ENV)
        assert [s.mae.mean for s in first.summaries] == [s.mae.mean for s in second.summaries]


class TestWeightsValidation:
    def test_a_negative_weight_is_rejected(self):
        """It would flip a term's meaning - "prefer redundant questions" - which
        is never what an ablation means."""
        with pytest.raises(ValueError, match="must not be negative"):
            dataclasses.replace(DEFAULT_WEIGHTS, redundancy=-0.1)

    def test_a_non_finite_weight_is_rejected(self):
        with pytest.raises(ValueError, match="must be finite"):
            dataclasses.replace(DEFAULT_WEIGHTS, information=math.inf)

    def test_scaling_every_weight_cannot_change_a_decision(self):
        """Why the ablation needs no renormalisation: within one selection all
        items share the weights, so a common factor cancels in the argmax."""
        import random

        from app.selection import SelectionState, choose_next

        state = SelectionState(
            ability={},
            coverage_targets=MAIN_CONFIG.quotas,
            jd_weights=MAIN_CONFIG.jd_weights,
            time_left_s=1800.0,
        )
        big = build_environment(MAIN_CONFIG)
        doubled = dataclasses.replace(
            DEFAULT_WEIGHTS,
            information=0.80,
            jd=0.50,
            resume=0.30,
            coverage=0.30,
            redundancy=0.20,
            time=0.10,
        )
        for seed in range(10):
            base = choose_next(state, big.bank, rng=random.Random(seed))
            scaled = choose_next(state, big.bank, rng=random.Random(seed), weights=doubled)
            assert base is not None and scaled is not None
            assert base.item.id == scaled.item.id
