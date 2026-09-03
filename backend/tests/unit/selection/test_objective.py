"""The six-term weighted objective, term by term and then as a whole.

Each component is tested in isolation first, because the point of splitting the
score into named terms was to be able to say *which* one is wrong. Then the
complete formula is checked against a hand-computed value, and the six
monotonicity properties ("more redundancy never helps") are swept over a grid.
"""

from __future__ import annotations

import dataclasses
import itertools
import math

import pytest

from app.ability import AbilityState
from app.selection.objective import (
    COMPONENT_NAMES,
    COVERAGE_WEIGHT,
    DEFAULT_WEIGHTS,
    INFORMATION_WEIGHT,
    JD_WEIGHT,
    REDUNDANCY_PENALTY,
    RESUME_WEIGHT,
    TIME_PENALTY,
    cosine_similarity,
    coverage_deficit,
    redundancy,
    resume_affinity,
    score_item,
    score_items,
    time_cost,
)
from app.selection.state import ResumeProfile
from tests.unit.selection.conftest import item, state, unit_vector

CACHING = unit_vector(1.0, 0.0, 0.0)
NEARLY_CACHING = unit_vector(0.99, 0.14, 0.0)
UNRELATED = unit_vector(0.0, 0.0, 1.0)


# ---------------------------------------------------------------------------
# 0. The weights themselves
# ---------------------------------------------------------------------------


class TestWeights:
    def test_they_are_exactly_the_plans_numbers(self):
        """plan section 8.3. If one of these ever changes, it changes here first."""
        assert INFORMATION_WEIGHT == 0.40
        assert JD_WEIGHT == 0.25
        assert RESUME_WEIGHT == 0.15
        assert COVERAGE_WEIGHT == 0.15
        assert REDUNDANCY_PENALTY == 0.10
        assert TIME_PENALTY == 0.05

    def test_the_score_is_bounded_by_the_weights_and_not_by_one(self):
        """The positive weights sum to 0.95, not 1.0 - the plan does not make
        them a probability distribution, and nothing here pretends they are.
        The consequence worth pinning: a score lives in [-0.15, 0.95], so
        `total` is a ranking key and never a percentage."""
        assert INFORMATION_WEIGHT + JD_WEIGHT + RESUME_WEIGHT + COVERAGE_WEIGHT == pytest.approx(
            0.95
        )
        assert REDUNDANCY_PENALTY + TIME_PENALTY == pytest.approx(0.15)

    def test_information_carries_the_largest_single_share(self):
        assert INFORMATION_WEIGHT > max(JD_WEIGHT, RESUME_WEIGHT, COVERAGE_WEIGHT)
        assert INFORMATION_WEIGHT < 1.0  # ... and not the whole thing


# ---------------------------------------------------------------------------
# 1. cosine similarity
# ---------------------------------------------------------------------------


class TestCosineSimilarity:
    def test_identical_vectors_score_one(self):
        assert cosine_similarity(CACHING, CACHING) == pytest.approx(1.0)

    def test_orthogonal_vectors_score_zero(self):
        assert cosine_similarity(CACHING, UNRELATED) == pytest.approx(0.0)

    def test_opposite_vectors_score_minus_one(self):
        assert cosine_similarity((1.0, 0.0), (-1.0, 0.0)) == pytest.approx(-1.0)

    def test_it_normalises_rather_than_assuming_unit_length(self):
        assert cosine_similarity((3.0, 0.0), (5.0, 0.0)) == pytest.approx(1.0)

    def test_a_zero_vector_has_no_direction_and_scores_zero(self):
        assert cosine_similarity((0.0, 0.0), (1.0, 0.0)) == 0.0

    def test_mismatched_lengths_are_an_error(self):
        with pytest.raises(ValueError, match="same length"):
            cosine_similarity((1.0, 0.0), (1.0, 0.0, 0.0))


# ---------------------------------------------------------------------------
# 2. resume affinity
# ---------------------------------------------------------------------------


class TestResumeAffinity:
    def test_no_resume_is_neutral(self):
        assert resume_affinity(item(), None) == 0.0

    def test_an_empty_profile_is_neutral(self):
        assert resume_affinity(item(), ResumeProfile()) == 0.0

    def test_an_explicit_subtopic_score_is_used(self):
        profile = ResumeProfile(topic_affinity={"caching": 0.8})
        assert resume_affinity(item(), profile) == pytest.approx(0.8)

    def test_an_explicit_topic_score_is_used_when_the_subtopic_has_none(self):
        profile = ResumeProfile(topic_affinity={"systems": 0.6})
        assert resume_affinity(item(), profile) == pytest.approx(0.6)

    def test_the_subtopic_wins_over_the_topic(self):
        """Specific beats general: the narrower statement is the better evidence."""
        profile = ResumeProfile(topic_affinity={"systems": 0.2, "caching": 0.9})
        assert resume_affinity(item(), profile) == pytest.approx(0.9)

    def test_it_falls_back_to_vector_similarity(self):
        profile = ResumeProfile(embedding=CACHING)
        assert resume_affinity(item(embedding=NEARLY_CACHING), profile) == pytest.approx(
            0.99, abs=1e-3
        )

    def test_an_unrelated_resume_scores_near_zero(self):
        profile = ResumeProfile(embedding=UNRELATED)
        assert resume_affinity(item(embedding=CACHING), profile) == pytest.approx(0.0, abs=1e-9)

    def test_a_negative_cosine_is_clamped_to_zero_not_turned_into_a_penalty(self):
        profile = ResumeProfile(embedding=(-1.0, 0.0, 0.0))
        assert resume_affinity(item(embedding=CACHING), profile) == 0.0

    def test_a_question_with_no_vector_scores_zero_rather_than_guessing(self):
        profile = ResumeProfile(embedding=CACHING)
        assert resume_affinity(item(embedding=None), profile) == 0.0

    @pytest.mark.parametrize("bad", [-0.1, 1.1, math.nan])
    def test_an_out_of_range_explicit_score_is_rejected_at_construction(self, bad):
        with pytest.raises(ValueError, match=r"must be in \[0, 1\]"):
            ResumeProfile(topic_affinity={"caching": bad})


# ---------------------------------------------------------------------------
# 3. coverage deficit
# ---------------------------------------------------------------------------


class TestCoverageDeficit:
    def test_an_untouched_topic_is_fully_deficient(self):
        assert coverage_deficit("systems", state(targets={"systems": 3})) == 1.0

    def test_a_met_quota_has_no_deficit(self):
        asked = [item(f"q{i}") for i in range(3)]
        assert coverage_deficit("systems", state(targets={"systems": 3}, asked=asked)) == 0.0

    def test_a_partly_served_topic_is_proportionally_deficient(self):
        asked = [item("q1")]
        assert coverage_deficit(
            "systems", state(targets={"systems": 4}, asked=asked)
        ) == pytest.approx(0.75)

    def test_a_larger_deficit_ranks_higher(self):
        behind = state(targets={"systems": 4}, asked=[item("q1")])
        nearly_done = state(targets={"systems": 4}, asked=[item(f"q{i}") for i in range(3)])
        assert coverage_deficit("systems", behind) > coverage_deficit("systems", nearly_done)

    def test_an_exhausted_topic_gets_no_priority(self):
        asked = [item(f"q{i}") for i in range(3)]
        assert coverage_deficit("systems", state(targets={"systems": 3}, asked=asked)) == 0.0

    def test_an_over_served_topic_is_floored_at_zero_not_negative(self):
        asked = [item(f"q{i}") for i in range(5)]
        assert coverage_deficit("systems", state(targets={"systems": 3}, asked=asked)) == 0.0

    def test_a_topic_the_blueprint_never_asked_for_has_no_deficit(self):
        assert coverage_deficit("algorithms", state(targets={"systems": 3})) == 0.0

    def test_a_zero_target_is_not_a_division_by_zero(self):
        assert coverage_deficit("systems", state(targets={"systems": 0})) == 0.0

    def test_only_the_topics_own_asked_items_count(self):
        """An item asked in a different topic must not reduce this topic's deficit."""
        elsewhere = [item("q1", topic="algorithms", subtopic="sorting")]
        assert coverage_deficit("systems", state(targets={"systems": 2}, asked=elsewhere)) == 1.0


# ---------------------------------------------------------------------------
# 4. redundancy
# ---------------------------------------------------------------------------


class TestRedundancy:
    def test_nothing_asked_yet_means_no_redundancy(self):
        assert redundancy(item(embedding=CACHING), []) == 0.0

    def test_a_near_identical_question_is_highly_redundant(self):
        asked = [item("q0", embedding=CACHING)]
        assert redundancy(item("q1", embedding=NEARLY_CACHING), asked) > 0.95

    def test_an_identical_question_is_maximally_redundant(self):
        asked = [item("q0", embedding=CACHING)]
        assert redundancy(item("q1", embedding=CACHING), asked) == pytest.approx(1.0)

    def test_an_unrelated_question_is_not_redundant(self):
        asked = [item("q0", embedding=CACHING)]
        assert redundancy(item("q1", embedding=UNRELATED), asked) == pytest.approx(0.0, abs=1e-9)

    def test_it_takes_the_maximum_not_the_mean(self):
        """One collision among many unrelated asked items must still register."""
        asked = [
            item("q0", embedding=UNRELATED),
            item("q1", embedding=UNRELATED),
            item("q2", embedding=CACHING),
        ]
        assert redundancy(item("q3", embedding=CACHING), asked) == pytest.approx(1.0)

    def test_a_candidate_with_no_vector_contributes_no_similarity(self):
        asked = [item("q0", embedding=CACHING)]
        assert redundancy(item("q1", embedding=None), asked) == 0.0

    def test_asked_items_with_no_vectors_are_skipped_not_treated_as_similar(self):
        asked = [item("q0", embedding=None)]
        assert redundancy(item("q1", embedding=CACHING), asked) == 0.0

    def test_a_negative_cosine_is_clamped_to_zero(self):
        asked = [item("q0", embedding=(-1.0, 0.0, 0.0))]
        assert redundancy(item("q1", embedding=CACHING), asked) == 0.0


# ---------------------------------------------------------------------------
# 5. time cost
# ---------------------------------------------------------------------------


class TestTimeCost:
    def test_it_is_the_fraction_of_the_remaining_time_the_item_consumes(self):
        assert time_cost(item(seconds=120), state(time_left=600.0)) == pytest.approx(0.2)

    def test_an_item_that_exactly_fills_the_time_costs_one(self):
        assert time_cost(item(seconds=300), state(time_left=300.0)) == pytest.approx(1.0)

    def test_the_same_item_costs_more_later_in_the_session(self):
        early = time_cost(item(seconds=120), state(time_left=1800.0))
        late = time_cost(item(seconds=120), state(time_left=300.0))
        assert late > early

    def test_no_time_left_is_handled_rather_than_dividing_by_zero(self):
        assert time_cost(item(seconds=120), state(time_left=0.0)) == 1.0


# ---------------------------------------------------------------------------
# 6. The whole weighted score
# ---------------------------------------------------------------------------


class TestScoreItem:
    def test_the_exact_formula_by_hand(self):
        """A fully worked example, every term non-trivial.

        theta 0.0, b 0.0        -> p 0.5, info 0.25, normalised 1.0
        jd_weight               -> 0.8
        resume (explicit)       -> 0.5
        coverage 1 of 4 served  -> 0.75
        redundancy (identical)  -> 1.0
        time 120 / 600          -> 0.2

        0.40*1.0 + 0.25*0.8 + 0.15*0.5 + 0.15*0.75 - 0.10*1.0 - 0.05*0.2
          = 0.400 + 0.200 + 0.075 + 0.1125 - 0.100 - 0.010 = 0.6775
        """
        asked = [item("q0", embedding=CACHING)]
        current = state(
            theta=0.0,
            targets={"systems": 4},
            jd={"systems": 0.8},
            asked=asked,
            time_left=600.0,
            resume=ResumeProfile(topic_affinity={"caching": 0.5}),
        )
        scored = score_item(item("q1", b=0.0, seconds=120, embedding=CACHING), current)

        assert scored.p == pytest.approx(0.5)
        assert scored.information == pytest.approx(1.0)
        assert scored.jd_weight == pytest.approx(0.8)
        assert scored.resume_affinity == pytest.approx(0.5)
        assert scored.coverage_deficit == pytest.approx(0.75)
        assert scored.redundancy == pytest.approx(1.0)
        assert scored.time_cost == pytest.approx(0.2)
        assert scored.total == pytest.approx(0.6775)

    def test_the_total_is_the_sum_of_the_weighted_contributions(self):
        scored = score_item(item(b=0.4), state(theta=-0.3))
        assert sum(scored.contributions.values()) == pytest.approx(scored.total)

    def test_the_contributions_carry_the_signs_the_plan_gives_them(self):
        asked = [item("q0", embedding=CACHING)]
        current = state(asked=asked, targets={"systems": 4}, jd={"systems": 0.5})
        contributions = score_item(item("q1", embedding=CACHING), current).contributions
        assert contributions["information"] > 0
        assert contributions["jd"] > 0
        assert contributions["coverage"] > 0
        assert contributions["redundancy"] < 0
        assert contributions["time"] < 0

    def test_score_items_preserves_the_input_order(self):
        items = [item("q3"), item("q1"), item("q2")]
        assert [s.item_id for s in score_items(items, state())] == ["q3", "q1", "q2"]

    def test_an_unmeasured_subtopic_uses_the_cold_start_prior(self):
        """theta 0 from PRIOR_ABILITY, so a b of 0 is still a coin flip."""
        current = state(ability={})
        assert score_item(item(b=0.0), current).p == pytest.approx(0.5)


class TestEachComponentMovesTheScoreTheRightWay:
    """One term varied at a time, everything else held fixed."""

    def test_more_information_raises_the_score(self):
        current = state(theta=0.0)
        matched = score_item(item(b=0.0), current).total
        mismatched = score_item(item(b=1.4), current).total
        assert matched > mismatched

    def test_a_higher_jd_weight_raises_the_score(self):
        low = score_item(item(), state(jd={"systems": 0.2})).total
        high = score_item(item(), state(jd={"systems": 0.9})).total
        assert high > low
        assert high - low == pytest.approx(JD_WEIGHT * 0.7)

    def test_a_higher_resume_affinity_raises_the_score(self):
        low = score_item(item(), state(resume=ResumeProfile(topic_affinity={"caching": 0.1})))
        high = score_item(item(), state(resume=ResumeProfile(topic_affinity={"caching": 0.9})))
        assert high.total > low.total
        assert high.total - low.total == pytest.approx(RESUME_WEIGHT * 0.8)

    def test_a_higher_coverage_deficit_raises_the_score(self):
        served = state(targets={"systems": 4}, asked=[item(f"q{i}") for i in range(3)])
        unserved = state(targets={"systems": 4})
        assert score_item(item("qx"), unserved).total > score_item(item("qx"), served).total

    def test_more_redundancy_lowers_the_score(self):
        current = state(asked=[item("q0", embedding=CACHING)])
        similar = score_item(item("q1", embedding=CACHING), current).total
        different = score_item(item("q1", embedding=UNRELATED), current).total
        assert similar < different
        assert different - similar == pytest.approx(REDUNDANCY_PENALTY, abs=1e-6)

    def test_a_more_expensive_item_lowers_the_score(self):
        current = state(time_left=600.0)
        cheap = score_item(item(seconds=60), current).total
        expensive = score_item(item(seconds=600), current).total
        assert expensive < cheap
        assert cheap - expensive == pytest.approx(TIME_PENALTY * (1.0 - 0.1))


class TestObjectiveProperties:
    """Deterministic grids, in the style of tests/unit/test_ability.py."""

    def test_more_redundancy_never_improves_a_score_all_else_equal(self):
        for theta, b, seconds in itertools.product([-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0], [30, 300]):
            current = state(theta=theta, asked=[item("q0", embedding=CACHING)])
            more = score_item(item("q1", b=b, seconds=seconds, embedding=CACHING), current)
            less = score_item(item("q1", b=b, seconds=seconds, embedding=UNRELATED), current)
            assert more.total <= less.total

    def test_a_higher_time_cost_never_improves_a_score_all_else_equal(self):
        for theta, b in itertools.product([-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]):
            current = state(theta=theta, time_left=600.0)
            for cheap, dear in itertools.combinations([30, 120, 300, 600], 2):
                assert (
                    score_item(item(b=b, seconds=dear), current).total
                    <= score_item(item(b=b, seconds=cheap), current).total
                )

    def test_every_component_stays_inside_its_declared_range(self):
        asked = [item("q0", embedding=CACHING), item("q9", embedding=UNRELATED)]
        for theta, b, seconds in itertools.product(
            [-3.0, -1.0, 0.0, 1.0, 3.0], [-3.0, 0.0, 3.0], [30, 120, 600]
        ):
            current = state(
                ability={"caching": AbilityState(theta=theta, rd=0.9)},
                targets={"systems": 4},
                jd={"systems": 0.6},
                asked=asked,
                time_left=600.0,
                resume=ResumeProfile(embedding=CACHING),
            )
            scored = score_item(item("qx", b=b, seconds=seconds, embedding=NEARLY_CACHING), current)
            assert 0.0 <= scored.information <= 1.0
            assert 0.0 <= scored.jd_weight <= 1.0
            assert 0.0 <= scored.resume_affinity <= 1.0
            assert 0.0 <= scored.coverage_deficit <= 1.0
            assert 0.0 <= scored.redundancy <= 1.0
            assert 0.0 < scored.time_cost <= 1.0
            assert math.isfinite(scored.total)


# ---------------------------------------------------------------------------
# 7. ObjectiveWeights - the Day 13 ablation hook, and its regression guard
# ---------------------------------------------------------------------------


class TestObjectiveWeights:
    """Day 13 needs to vary the weights without mutating module state.

    The whole point of the type is that production behaviour is unchanged, so
    most of these tests assert that nothing moved.
    """

    def test_the_defaults_are_the_module_constants(self):
        assert DEFAULT_WEIGHTS.information == INFORMATION_WEIGHT
        assert DEFAULT_WEIGHTS.jd == JD_WEIGHT
        assert DEFAULT_WEIGHTS.resume == RESUME_WEIGHT
        assert DEFAULT_WEIGHTS.coverage == COVERAGE_WEIGHT
        assert DEFAULT_WEIGHTS.redundancy == REDUNDANCY_PENALTY
        assert DEFAULT_WEIGHTS.time == TIME_PENALTY

    def test_omitting_weights_scores_exactly_as_passing_the_defaults(self):
        """The regression guard for the refactor: the shipped call path and the
        new parameter must agree bit for bit, or Day 12's numbers moved."""
        current = state(theta=-0.3, asked=[item("q0", embedding=CACHING)])
        for b in (-2.0, -0.5, 0.0, 0.75, 2.0):
            candidate = item("qx", b=b, embedding=NEARLY_CACHING)
            assert (
                score_item(candidate, current).total
                == score_item(candidate, current, weights=DEFAULT_WEIGHTS).total
            )

    def test_a_breakdown_records_the_weights_it_was_scored_with(self):
        scored = score_item(item(), state(), weights=DEFAULT_WEIGHTS.without("jd"))
        assert scored.weights.jd == 0.0
        assert scored.contributions["jd"] == 0.0

    def test_the_components_are_still_computed_when_a_weight_is_zero(self):
        """An ablated run must still report what the switched-off term said."""
        current = state(jd={"systems": 0.8})
        scored = score_item(item(), current, weights=DEFAULT_WEIGHTS.without("jd"))
        assert scored.jd_weight == pytest.approx(0.8)
        assert scored.contributions["jd"] == 0.0

    def test_without_zeroes_only_the_named_component(self):
        without = DEFAULT_WEIGHTS.without("coverage")
        assert without.coverage == 0.0
        assert without.information == INFORMATION_WEIGHT
        assert without.time == TIME_PENALTY

    def test_without_does_not_mutate_the_original(self):
        DEFAULT_WEIGHTS.without("information")
        assert DEFAULT_WEIGHTS.information == 0.40

    def test_an_unknown_component_is_rejected_rather_than_ignored(self):
        """An ablation that quietly ablated nothing would publish the full
        objective's numbers under another label."""
        with pytest.raises(ValueError, match="unknown component"):
            DEFAULT_WEIGHTS.without("informations")

    def test_component_names_match_the_contribution_keys(self):
        keys = set(score_item(item(), state()).contributions)
        assert set(COMPONENT_NAMES) == keys

    @pytest.mark.parametrize("field", COMPONENT_NAMES)
    def test_a_negative_weight_is_rejected(self, field):
        with pytest.raises(ValueError, match="must not be negative"):
            dataclasses.replace(DEFAULT_WEIGHTS, **{field: -0.1})

    @pytest.mark.parametrize("bad", [math.inf, math.nan])
    def test_a_non_finite_weight_is_rejected(self, bad):
        with pytest.raises(ValueError, match="must be finite"):
            dataclasses.replace(DEFAULT_WEIGHTS, information=bad)

    def test_zeroing_a_weight_removes_that_term_from_the_total(self):
        current = state(jd={"systems": 0.8})
        full = score_item(item(), current).total
        without = score_item(item(), current, weights=DEFAULT_WEIGHTS.without("jd")).total
        assert full - without == pytest.approx(JD_WEIGHT * 0.8)
