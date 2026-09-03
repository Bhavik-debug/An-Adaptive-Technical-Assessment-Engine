"""The ability model, tested as the pure arithmetic it is.

No database, no LLM, no clock - the whole point of `app/ability.py` is that the
adaptive engine's core can be exercised exhaustively in milliseconds, so these
tests import one module and nothing else.

The "property" classes at the bottom sweep a deterministic grid rather than
using Hypothesis, which is not a dependency of this project. The grid is dense
enough to be meaningful (thousands of parameter combinations) and, unlike a
randomised search, it fails identically on every machine and in every CI run -
which for a numerical core is worth more than the extra coverage would be.
"""

from __future__ import annotations

import dataclasses
import itertools
import math

import pytest

from app.ability import (
    BASE_LEARNING_RATE,
    F_RD_CAP,
    MAX_THETA_STEP,
    RD_MAX,
    RD_MIN,
    STAKE_BANK_ITEM,
    STAKE_GENERATED_ITEM,
    THETA_MAX,
    THETA_MIN,
    AbilityState,
    KFactor,
    aggregate_ability,
    k_factor,
    probability_correct,
    roll_up,
    update_ability,
    update_uncertainty,
)

# A grid over the ranges the plan declares (theta and b in -3..3, a in 0.5..2.0),
# plus a little headroom on either side to catch boundary handling.
THETAS = [-4.0, -3.0, -1.5, -0.4, 0.0, 0.4, 1.5, 3.0, 4.0]
DIFFICULTIES = [-3.0, -1.2, 0.0, 0.7, 2.0, 3.0]
DISCRIMINATIONS = [0.25, 0.5, 1.0, 1.6, 2.0]
RDS = [0.30, 0.45, 0.6, 0.9, 1.2, 1.3]
SCORES = [0.0, 0.25, 0.5, 0.75, 1.0]


# ---------------------------------------------------------------------------
# 1. The 2PL probability
# ---------------------------------------------------------------------------


class TestProbability:
    def test_matched_question_is_a_coin_flip(self):
        """theta == b is the definition of "perfectly matched": p is exactly 0.5."""
        assert probability_correct(0.0, 0.0) == 0.5
        assert probability_correct(1.7, 1.7) == 0.5
        assert probability_correct(-2.5, -2.5, discrimination=2.0) == 0.5

    def test_the_shape_from_the_plan(self):
        """plan section 5.9: theta-b of -2, 0, +2 reads 0.12, 0.50, 0.88."""
        assert probability_correct(-1.0, 1.0) == pytest.approx(0.1192, abs=5e-4)
        assert probability_correct(1.0, 1.0) == pytest.approx(0.5)
        assert probability_correct(3.0, 1.0) == pytest.approx(0.8808, abs=5e-4)

    def test_higher_ability_raises_p(self):
        assert probability_correct(1.0, 0.0) > probability_correct(0.0, 0.0)

    def test_higher_difficulty_lowers_p(self):
        assert probability_correct(0.0, 1.0) < probability_correct(0.0, 0.0)

    def test_discrimination_sharpens_the_curve_around_the_match_point(self):
        """High `a` is a cliff, low `a` a gentle slope. Both pass through 0.5."""
        gentle = probability_correct(1.0, 0.0, discrimination=0.5)
        sharp = probability_correct(1.0, 0.0, discrimination=2.0)
        assert 0.5 < gentle < sharp

    def test_extreme_gaps_saturate_without_overflowing(self):
        """The stable sigmoid: exp(800) would raise OverflowError; this must not."""
        assert probability_correct(400.0, -400.0) == 1.0
        assert probability_correct(-400.0, 400.0) == 0.0

    @pytest.mark.parametrize("theta", [-3.0, 0.0, 3.0])
    def test_symmetry_about_the_match_point(self, theta):
        assert probability_correct(theta, theta + 1.0) == pytest.approx(
            1.0 - probability_correct(theta, theta - 1.0)
        )

    @pytest.mark.parametrize("bad", [0.0, -1.0])
    def test_non_positive_discrimination_is_rejected(self, bad):
        """A negative-`a` item is broken, not a modelling case to absorb."""
        with pytest.raises(ValueError, match="discrimination must be positive"):
            probability_correct(0.0, 0.0, discrimination=bad)

    @pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
    def test_non_finite_inputs_are_rejected(self, bad):
        with pytest.raises(ValueError, match="must be finite"):
            probability_correct(bad, 0.0)
        with pytest.raises(ValueError, match="must be finite"):
            probability_correct(0.0, bad)


# ---------------------------------------------------------------------------
# 2. The state
# ---------------------------------------------------------------------------


class TestAbilityState:
    def test_precision_is_one_over_rd_squared(self):
        assert AbilityState(theta=0.0, rd=0.5).precision == pytest.approx(4.0)
        assert AbilityState(theta=0.0, rd=1.0).precision == pytest.approx(1.0)

    def test_state_is_frozen(self):
        state = AbilityState(theta=0.0, rd=1.0)
        with pytest.raises(dataclasses.FrozenInstanceError):
            state.theta = 1.0  # type: ignore[misc]

    @pytest.mark.parametrize("bad_rd", [0.0, -0.5])
    def test_rd_must_be_positive(self, bad_rd):
        with pytest.raises(ValueError, match="rd must be positive"):
            AbilityState(theta=0.0, rd=bad_rd)

    def test_theta_must_be_finite(self):
        with pytest.raises(ValueError, match="theta must be finite"):
            AbilityState(theta=math.nan, rd=1.0)

    def test_observation_count_must_not_be_negative(self):
        with pytest.raises(ValueError, match="n_observations"):
            AbilityState(theta=0.0, rd=1.0, n_observations=-1)


# ---------------------------------------------------------------------------
# 3. The K-factor
# ---------------------------------------------------------------------------


class TestKFactor:
    def test_the_worked_example_from_the_plan(self):
        """plan section 5.9: RD 0.90, confidence 0.85, bank item -> K = 0.765."""
        k = k_factor(0.90, grader_confidence=0.85, stake=STAKE_BANK_ITEM)
        assert k.base == BASE_LEARNING_RATE
        assert k.quality == 0.85
        assert k.uncertainty == pytest.approx(1.5)
        assert k.stake == 1.0
        assert k.value == pytest.approx(0.765)

    def test_k_is_exactly_the_four_specified_factors(self):
        """K = K0 * f_conf * f_rd * f_stake, and nothing else."""
        k = k_factor(0.6, grader_confidence=0.5, stake=0.5)
        assert isinstance(k, KFactor)
        assert [f.name for f in dataclasses.fields(k)] == [
            "base",
            "quality",
            "uncertainty",
            "stake",
        ]
        assert k.value == pytest.approx(k.base * k.quality * k.uncertainty * k.stake)
        assert k.value == pytest.approx(BASE_LEARNING_RATE * 0.5 * 1.0 * 0.5)

    def test_k_is_decomposed_not_a_single_constant(self):
        """The point of the dataclass: every factor is separately inspectable."""
        k = k_factor(0.9, grader_confidence=0.5, stake=0.5)
        assert (k.base, k.quality, k.uncertainty, k.stake) == (BASE_LEARNING_RATE, 0.5, 1.5, 0.5)

    def test_a_less_confident_grade_shrinks_k(self):
        confident = k_factor(0.9, grader_confidence=1.0)
        unsure = k_factor(0.9, grader_confidence=0.4)
        assert unsure.value < confident.value

    def test_a_generated_item_moves_theta_less_than_a_bank_item(self):
        bank = k_factor(0.9, stake=STAKE_BANK_ITEM)
        generated = k_factor(0.9, stake=STAKE_GENERATED_ITEM)
        assert generated.value == pytest.approx(bank.value / 2)

    def test_higher_uncertainty_raises_k_until_the_cap(self):
        """Learning-rate decay: big steps while RD is large, small once it is not."""
        assert k_factor(0.3).uncertainty == pytest.approx(0.5)
        assert k_factor(0.6).uncertainty == pytest.approx(1.0)
        assert k_factor(0.9).uncertainty == pytest.approx(1.5)

    def test_the_uncertainty_factor_is_capped(self):
        """A cold-start RD of 1.3 would otherwise give an unstable first step."""
        assert k_factor(1.3).uncertainty == pytest.approx(F_RD_CAP)
        assert k_factor(5.0).uncertainty == pytest.approx(F_RD_CAP)

    def test_k_takes_no_discrimination_argument(self):
        """The plan's K has four factors. `a` belongs to `p` and to the RD update."""
        with pytest.raises(TypeError, match="discrimination"):
            k_factor(0.9, discrimination=2.0)  # type: ignore[call-arg]

    def test_zero_confidence_means_no_learning(self):
        assert k_factor(0.9, grader_confidence=0.0).value == 0.0

    @pytest.mark.parametrize("bad", [-0.1, 1.1])
    def test_confidence_outside_zero_one_is_rejected(self, bad):
        with pytest.raises(ValueError, match="grader_confidence must be in"):
            k_factor(0.9, grader_confidence=bad)

    def test_non_positive_rd_is_rejected(self):
        with pytest.raises(ValueError, match="rd must be positive"):
            k_factor(0.0)


class TestDiscriminationEntersOnlyWhereSpecified:
    """`a` shapes p and shrinks RD. It must never scale K directly.

    Pinning this is the whole point of the four-factor decomposition: the step
    size reflects how much the *evidence* is worth (grader confidence, RD, bank
    vs generated), not which item happened to produce it.
    """

    def test_k_is_identical_however_sharp_the_item_is(self):
        gentle = update_ability(
            AbilityState(theta=0.4, rd=0.9), difficulty=1.1, score=0.8, discrimination=0.5
        )
        sharp = update_ability(
            AbilityState(theta=0.4, rd=0.9), difficulty=1.1, score=0.8, discrimination=2.0
        )
        assert gentle.k == sharp.k
        assert gentle.k.value == sharp.k.value

    def test_a_matched_item_moves_theta_identically_at_any_discrimination(self):
        """theta == b pins p at 0.5 for every `a`, so delta_theta cannot differ."""
        steps = {
            a: update_ability(
                AbilityState(theta=0.5, rd=0.9), difficulty=0.5, score=0.9, discrimination=a
            ).delta_theta
            for a in DISCRIMINATIONS
        }
        assert len(set(round(step, 12) for step in steps.values())) == 1

    def test_but_the_same_item_still_shrinks_rd_by_more_when_sharper(self):
        gentle = update_ability(
            AbilityState(theta=0.5, rd=0.9), difficulty=0.5, score=0.9, discrimination=0.5
        )
        sharp = update_ability(
            AbilityState(theta=0.5, rd=0.9), difficulty=0.5, score=0.9, discrimination=2.0
        )
        assert sharp.after.rd < gentle.after.rd < 0.9

    def test_and_it_still_reaches_theta_indirectly_through_p(self):
        """Off the match point `a` changes the prediction, hence (score - p)."""
        gentle = update_ability(
            AbilityState(theta=0.0, rd=0.6), difficulty=1.0, score=0.7, discrimination=0.5
        )
        sharp = update_ability(
            AbilityState(theta=0.0, rd=0.6), difficulty=1.0, score=0.7, discrimination=2.0
        )
        assert not gentle.was_capped and not sharp.was_capped
        assert sharp.p < gentle.p  # the sharp item predicts failure more firmly
        assert sharp.delta_theta > gentle.delta_theta  # so beating it is more surprising
        assert sharp.k.value == pytest.approx(gentle.k.value)  # but K itself is untouched


# ---------------------------------------------------------------------------
# 4. The uncertainty update
# ---------------------------------------------------------------------------


class TestUncertaintyUpdate:
    def test_the_worked_example_from_the_plan(self):
        """RD 0.90 after a p = 0.3775 observation becomes 0.825."""
        assert update_uncertainty(0.90, p=0.3775) == pytest.approx(0.825, abs=1e-3)

    def test_rd_decreases_after_evidence(self):
        assert update_uncertainty(1.0, p=0.5) < 1.0
        assert update_uncertainty(0.6, p=0.4) < 0.6

    def test_a_matched_question_shrinks_rd_most(self):
        """Information peaks at p = 0.5; a foregone conclusion teaches nothing."""
        matched = update_uncertainty(1.0, p=0.5)
        lopsided = update_uncertainty(1.0, p=0.95)
        assert matched < lopsided < 1.0

    def test_a_certain_outcome_is_not_evidence(self):
        """p of exactly 0 or 1 adds zero precision, so RD is unchanged."""
        assert update_uncertainty(1.0, p=1.0) == pytest.approx(1.0)
        assert update_uncertainty(1.0, p=0.0) == pytest.approx(1.0)

    def test_a_sharper_item_shrinks_rd_more(self):
        assert update_uncertainty(1.0, p=0.5, discrimination=2.0) < update_uncertainty(
            1.0, p=0.5, discrimination=0.5
        )

    def test_repeated_updates_converge_downwards_and_stop_at_the_floor(self):
        rd = RD_MAX
        seen = [rd]
        for _ in range(50):
            rd = update_uncertainty(rd, p=0.5)
            assert 0.0 < rd <= seen[-1]
            seen.append(rd)
        assert rd == pytest.approx(RD_MIN)

    def test_rd_never_falls_below_the_floor(self):
        assert update_uncertainty(RD_MIN, p=0.5) == pytest.approx(RD_MIN)

    def test_result_is_positive_and_finite_across_the_grid(self):
        for rd, p, a in itertools.product(RDS, [0.0, 0.01, 0.5, 0.99, 1.0], DISCRIMINATIONS):
            new_rd = update_uncertainty(rd, p=p, discrimination=a)
            assert math.isfinite(new_rd)
            assert new_rd > 0.0
            assert new_rd <= rd + 1e-12  # evidence never increases uncertainty

    @pytest.mark.parametrize("bad_p", [-0.01, 1.01, math.nan])
    def test_p_outside_zero_one_is_rejected(self, bad_p):
        with pytest.raises(ValueError, match="p must be in"):
            update_uncertainty(1.0, p=bad_p)


# ---------------------------------------------------------------------------
# 5. The ability update
# ---------------------------------------------------------------------------


class TestAbilityUpdate:
    def test_the_worked_example_from_the_plan(self):
        """plan section 5.9, end to end: theta 0.30 -> 0.547, RD 0.90 -> 0.825."""
        before = AbilityState(theta=0.30, rd=0.90, n_observations=3)
        result = update_ability(
            before,
            difficulty=0.80,
            score=0.70,
            grader_confidence=0.85,
            stake=STAKE_BANK_ITEM,
        )
        assert result.p == pytest.approx(0.3775, abs=5e-4)
        assert result.k.value == pytest.approx(0.765)
        assert result.delta_theta == pytest.approx(0.247, abs=1e-3)
        assert result.after.theta == pytest.approx(0.547, abs=1e-3)
        assert result.after.rd == pytest.approx(0.825, abs=1e-3)
        assert result.after.n_observations == 4

    def test_beating_the_prediction_raises_theta(self):
        before = AbilityState(theta=0.0, rd=0.9)
        result = update_ability(before, difficulty=0.0, score=1.0)
        assert result.after.theta > before.theta
        assert result.delta_theta > 0

    def test_missing_the_prediction_lowers_theta(self):
        before = AbilityState(theta=0.0, rd=0.9)
        result = update_ability(before, difficulty=0.0, score=0.0)
        assert result.after.theta < before.theta
        assert result.delta_theta < 0

    def test_scoring_exactly_the_prediction_leaves_theta_alone(self):
        """theta moves on *surprise*, not on the answer being good or bad."""
        before = AbilityState(theta=0.0, rd=0.9)
        result = update_ability(before, difficulty=0.0, score=0.5)
        assert result.p == 0.5
        assert result.delta_theta == pytest.approx(0.0)
        assert result.after.theta == pytest.approx(before.theta)

    def test_a_confirmed_prediction_still_shrinks_rd(self):
        """Nothing learned about *where* they are; something learned all the same."""
        before = AbilityState(theta=0.0, rd=0.9)
        result = update_ability(before, difficulty=0.0, score=0.5)
        assert result.after.rd < before.rd

    def test_acing_a_hard_question_moves_theta_more_than_acing_an_easy_one(self):
        """Bigger surprise, bigger step. This is the (score - p) term doing its job."""
        before = AbilityState(theta=0.0, rd=0.9)
        hard = update_ability(before, difficulty=2.0, score=1.0)
        easy = update_ability(before, difficulty=-2.0, score=1.0)
        assert hard.delta_theta > easy.delta_theta > 0

    def test_a_less_confident_grade_moves_theta_less(self):
        before = AbilityState(theta=0.0, rd=0.9)
        sure = update_ability(before, difficulty=0.0, score=1.0, grader_confidence=1.0)
        unsure = update_ability(before, difficulty=0.0, score=1.0, grader_confidence=0.3)
        assert unsure.delta_theta < sure.delta_theta

    def test_a_generated_item_moves_theta_half_as_far(self):
        before = AbilityState(theta=0.0, rd=0.9)
        bank = update_ability(before, difficulty=0.0, score=1.0, stake=STAKE_BANK_ITEM)
        gen = update_ability(before, difficulty=0.0, score=1.0, stake=STAKE_GENERATED_ITEM)
        assert gen.delta_theta == pytest.approx(bank.delta_theta / 2)

    def test_an_early_answer_moves_theta_more_than_a_late_one(self):
        """The f_rd learning-rate schedule, observed from the outside."""
        early = update_ability(AbilityState(theta=0.0, rd=1.2), difficulty=0.0, score=1.0)
        late = update_ability(AbilityState(theta=0.0, rd=0.35), difficulty=0.0, score=1.0)
        assert early.delta_theta > late.delta_theta

    def test_the_step_is_capped_per_turn(self):
        """plan section 8.7: cap |delta_theta| so difficulty cannot oscillate."""
        before = AbilityState(theta=-1.0, rd=RD_MAX)
        result = update_ability(before, difficulty=2.0, score=1.0, discrimination=2.0)
        assert result.raw_delta_theta > MAX_THETA_STEP
        assert result.delta_theta == pytest.approx(MAX_THETA_STEP)
        assert result.was_capped

    def test_an_uncapped_step_reports_itself_as_uncapped(self):
        result = update_ability(AbilityState(theta=0.0, rd=0.6), difficulty=0.0, score=0.6)
        assert not result.was_capped
        assert result.delta_theta == pytest.approx(result.raw_delta_theta)

    def test_theta_stays_inside_the_declared_range(self):
        state = AbilityState(theta=0.0, rd=RD_MAX)
        for _ in range(60):
            state = update_ability(state, difficulty=3.0, score=1.0).after
        assert state.theta == pytest.approx(THETA_MAX)

        state = AbilityState(theta=0.0, rd=RD_MAX)
        for _ in range(60):
            state = update_ability(state, difficulty=-3.0, score=0.0).after
        assert state.theta == pytest.approx(THETA_MIN)

    def test_delta_theta_always_equals_the_observed_movement(self):
        """The reported delta must be what the log would need, cap and clamp included."""
        before = AbilityState(theta=2.95, rd=RD_MAX)
        result = update_ability(before, difficulty=-1.0, score=1.0)
        assert result.delta_theta == pytest.approx(result.after.theta - before.theta)
        assert result.after.theta <= THETA_MAX

    def test_a_wrong_early_answer_is_recovered_from_quickly(self):
        """plan section 8.7: high initial RD means one bad item does not sink a session."""
        strong = AbilityState(theta=0.0, rd=RD_MAX)
        after_failure = update_ability(strong, difficulty=0.0, score=0.0).after
        assert after_failure.theta < 0.0

        state = after_failure
        for _ in range(4):
            state = update_ability(state, difficulty=0.5, score=1.0).after
        assert state.theta > 0.0

    def test_the_state_is_not_mutated(self):
        before = AbilityState(theta=0.25, rd=0.8, n_observations=2)
        update_ability(before, difficulty=0.0, score=1.0)
        assert before.theta == 0.25
        assert before.rd == 0.8
        assert before.n_observations == 2

    @pytest.mark.parametrize("score", [0.0, 0.5, 1.0])
    def test_the_score_boundaries_are_accepted(self, score):
        result = update_ability(AbilityState(theta=0.0, rd=0.9), difficulty=0.0, score=score)
        assert math.isfinite(result.after.theta)
        assert result.after.rd > 0.0

    @pytest.mark.parametrize("bad", [-0.001, 1.001, 2.0, -1.0, math.nan])
    def test_a_score_outside_zero_one_is_rejected_not_clamped(self, bad):
        """A grader bug must surface here, not be absorbed into someone's ability."""
        with pytest.raises(ValueError, match=r"score must be in \[0, 1\]"):
            update_ability(AbilityState(theta=0.0, rd=0.9), difficulty=0.0, score=bad)


# ---------------------------------------------------------------------------
# 6. Precision-weighted aggregation
# ---------------------------------------------------------------------------


class TestAggregation:
    def test_the_precise_child_dominates(self):
        """plan section 9.2's whole argument, as one assertion."""
        precise = AbilityState(theta=1.0, rd=0.2)  # precision 25
        vague = AbilityState(theta=0.0, rd=1.0)  # precision 1
        parent = aggregate_ability([precise, vague])
        assert parent.theta == pytest.approx(25 / 26)
        assert parent.theta > 0.9  # far closer to A than a plain average's 0.5

    def test_equal_uncertainty_reduces_to_a_plain_average(self):
        children = [AbilityState(theta=t, rd=0.7) for t in (-1.0, 0.0, 2.0)]
        assert aggregate_ability(children).theta == pytest.approx(1 / 3)

    def test_a_very_uncertain_child_barely_moves_the_parent(self):
        confident = [AbilityState(theta=1.0, rd=0.3) for _ in range(3)]
        without = aggregate_ability(confident)
        with_noise = aggregate_ability([*confident, AbilityState(theta=-3.0, rd=1.3)])
        assert with_noise.theta == pytest.approx(without.theta, abs=0.08)

    def test_parent_rd_shrinks_as_evidence_is_added(self):
        one = aggregate_ability([AbilityState(theta=0.0, rd=0.5)])
        two = aggregate_ability([AbilityState(theta=0.0, rd=0.5)] * 2)
        four = aggregate_ability([AbilityState(theta=0.0, rd=0.5)] * 4)
        assert one.rd > two.rd > four.rd
        assert two.rd == pytest.approx(0.5 / math.sqrt(2))
        assert four.rd == pytest.approx(0.25)

    def test_a_single_child_is_passed_through_unchanged(self):
        child = AbilityState(theta=0.42, rd=0.77, n_observations=5)
        parent = aggregate_ability([child])
        assert parent.theta == pytest.approx(child.theta)
        assert parent.rd == pytest.approx(child.rd)
        assert parent.n_observations == 5

    def test_the_parent_rd_formula(self):
        children = [AbilityState(theta=0.0, rd=0.2), AbilityState(theta=0.0, rd=1.0)]
        assert aggregate_ability(children).rd == pytest.approx(1 / math.sqrt(26))

    def test_the_parent_is_not_floored_at_the_subtopic_rd_minimum(self):
        """A topic backed by five measured subtopics really is known better."""
        parent = aggregate_ability([AbilityState(theta=0.0, rd=RD_MIN)] * 5)
        assert parent.rd < RD_MIN

    def test_parent_theta_never_leaves_the_range_of_its_children(self):
        children = [
            AbilityState(theta=-2.0, rd=0.4),
            AbilityState(theta=0.5, rd=1.1),
            AbilityState(theta=1.8, rd=0.6),
        ]
        parent = aggregate_ability(children)
        assert -2.0 <= parent.theta <= 1.8

    def test_observation_counts_are_summed(self):
        children = [
            AbilityState(theta=0.0, rd=0.5, n_observations=2),
            AbilityState(theta=1.0, rd=0.5, n_observations=3),
        ]
        assert aggregate_ability(children).n_observations == 5

    def test_order_does_not_matter(self):
        children = [
            AbilityState(theta=-1.0, rd=0.3),
            AbilityState(theta=0.9, rd=1.2),
            AbilityState(theta=2.0, rd=0.55),
        ]
        forward = aggregate_ability(children)
        backward = aggregate_ability(list(reversed(children)))
        assert forward.theta == pytest.approx(backward.theta)
        assert forward.rd == pytest.approx(backward.rd)

    def test_a_mapping_and_a_list_agree(self):
        by_key = {
            "trees": AbilityState(theta=1.0, rd=0.4),
            "graphs": AbilityState(theta=0.0, rd=0.9),
        }
        assert aggregate_ability(by_key).theta == pytest.approx(
            aggregate_ability(list(by_key.values())).theta
        )

    def test_aggregating_nothing_is_an_error_not_a_zero(self):
        """A parent with no measured children has no ability, not an ability of 0."""
        with pytest.raises(ValueError, match="empty set of children"):
            aggregate_ability([])


# ---------------------------------------------------------------------------
# 7. The hierarchy: subtopic -> topic -> domain
# ---------------------------------------------------------------------------

# plan section 9.1's own example, trimmed to what one test needs.
SUBTOPIC_TO_TOPIC = {
    "trees": "dsa",
    "graphs": "dsa",
    "transactions": "databases",
    "indexing": "databases",
}
TOPIC_TO_DOMAIN = {
    "dsa": "cs_fundamentals",
    "databases": "backend",
}


class TestHierarchy:
    def test_subtopics_roll_up_into_topics(self):
        subtopics = {
            "trees": AbilityState(theta=1.0, rd=0.4, n_observations=4),
            "graphs": AbilityState(theta=-0.5, rd=0.8, n_observations=1),
            "transactions": AbilityState(theta=0.2, rd=0.6, n_observations=2),
            "indexing": AbilityState(theta=0.6, rd=0.6, n_observations=2),
        }
        topics = roll_up(subtopics, SUBTOPIC_TO_TOPIC)

        assert set(topics) == {"databases", "dsa"}
        assert topics["dsa"].theta == pytest.approx(
            aggregate_ability([subtopics["trees"], subtopics["graphs"]]).theta
        )
        # databases is two equally-certain subtopics, so a plain average.
        assert topics["databases"].theta == pytest.approx(0.4)
        assert topics["dsa"].n_observations == 5

    def test_subtopics_roll_up_through_topics_into_domains(self):
        subtopics = {
            "trees": AbilityState(theta=1.0, rd=0.4),
            "graphs": AbilityState(theta=-0.5, rd=0.8),
            "transactions": AbilityState(theta=0.2, rd=0.6),
            "indexing": AbilityState(theta=0.6, rd=0.6),
        }
        topics = roll_up(subtopics, SUBTOPIC_TO_TOPIC)
        domains = roll_up(topics, TOPIC_TO_DOMAIN)

        assert set(domains) == {"backend", "cs_fundamentals"}
        # Each domain here has one topic, so the topic passes straight through.
        assert domains["backend"].theta == pytest.approx(topics["databases"].theta)
        assert domains["cs_fundamentals"].rd == pytest.approx(topics["dsa"].rd)

    def test_parents_are_derived_so_changing_a_child_changes_them(self):
        """The no-second-source-of-truth property, asserted rather than asserted about."""
        subtopics = {
            "trees": AbilityState(theta=0.0, rd=0.5),
            "graphs": AbilityState(theta=0.0, rd=0.5),
        }
        before = roll_up(subtopics, SUBTOPIC_TO_TOPIC)["dsa"]

        subtopics["trees"] = update_ability(subtopics["trees"], difficulty=0.0, score=1.0).after
        after = roll_up(subtopics, SUBTOPIC_TO_TOPIC)["dsa"]

        assert after.theta > before.theta
        assert after.rd < before.rd

    def test_a_domain_aggregated_in_one_step_is_not_the_two_step_answer(self):
        """Documents the choice: rolling up level by level weights topics, not leaves.

        Both are defensible; the plan specifies level-by-level (section 9.2), and
        this pins which one the code does so a future change is deliberate.
        """
        subtopics = {
            "trees": AbilityState(theta=1.0, rd=0.4),
            "graphs": AbilityState(theta=1.0, rd=0.4),
            "transactions": AbilityState(theta=-1.0, rd=0.4),
            "indexing": AbilityState(theta=-1.0, rd=0.4),
        }
        one_domain = {key: "everything" for key in subtopics}
        flat = roll_up(subtopics, one_domain)["everything"]
        assert flat.theta == pytest.approx(0.0)
        assert flat.rd == pytest.approx(0.2)

        stepwise = roll_up(roll_up(subtopics, SUBTOPIC_TO_TOPIC), TOPIC_TO_DOMAIN)
        assert stepwise["cs_fundamentals"].theta == pytest.approx(1.0)
        assert stepwise["backend"].theta == pytest.approx(-1.0)

    def test_a_deeper_chain_is_just_another_call(self):
        leaves = {"a1": AbilityState(theta=1.0, rd=0.5), "a2": AbilityState(theta=3.0, rd=0.5)}
        level1 = roll_up(leaves, {"a1": "a", "a2": "a"})
        level2 = roll_up(level1, {"a": "root"})
        level3 = roll_up(level2, {"root": "everything"})
        assert level3["everything"].theta == pytest.approx(2.0)

    def test_an_unmapped_child_is_an_error_not_a_silent_drop(self):
        states = {
            "trees": AbilityState(theta=0.0, rd=0.5),
            "mystery": AbilityState(theta=9e9, rd=0.5),
        }
        with pytest.raises(ValueError, match="no parent mapped for: mystery"):
            roll_up(states, SUBTOPIC_TO_TOPIC)

    def test_the_result_is_ordered_by_parent_key(self):
        subtopics = {
            "transactions": AbilityState(theta=0.0, rd=0.5),
            "trees": AbilityState(theta=0.0, rd=0.5),
        }
        assert list(roll_up(subtopics, SUBTOPIC_TO_TOPIC)) == ["databases", "dsa"]

    def test_rolling_up_nothing_gives_nothing(self):
        assert roll_up({}, SUBTOPIC_TO_TOPIC) == {}


# ---------------------------------------------------------------------------
# 8. Properties, swept over a deterministic grid
# ---------------------------------------------------------------------------


class TestProbabilityProperties:
    def test_p_is_always_a_probability(self):
        for theta, b, a in itertools.product(THETAS, DIFFICULTIES, DISCRIMINATIONS):
            p = probability_correct(theta, b, a)
            assert 0.0 <= p <= 1.0
            assert math.isfinite(p)

    def test_p_never_decreases_as_ability_rises(self):
        for b, a in itertools.product(DIFFICULTIES, DISCRIMINATIONS):
            values = [probability_correct(theta, b, a) for theta in sorted(THETAS)]
            assert values == sorted(values)

    def test_p_never_increases_as_difficulty_rises(self):
        for theta, a in itertools.product(THETAS, DISCRIMINATIONS):
            values = [probability_correct(theta, b, a) for b in sorted(DIFFICULTIES)]
            assert values == sorted(values, reverse=True)


class TestUpdateProperties:
    def test_the_update_is_always_valid_and_in_range(self):
        for theta, b, a, rd, score in itertools.product(
            THETAS[1:-1], DIFFICULTIES, DISCRIMINATIONS, RDS, SCORES
        ):
            result = update_ability(
                AbilityState(theta=theta, rd=rd), difficulty=b, score=score, discrimination=a
            )
            assert math.isfinite(result.after.theta)
            assert THETA_MIN <= result.after.theta <= THETA_MAX
            assert math.isfinite(result.after.rd)
            assert RD_MIN <= result.after.rd <= RD_MAX
            assert abs(result.delta_theta) <= MAX_THETA_STEP + 1e-12

    def test_beating_the_prediction_never_lowers_theta(self):
        for theta, b, a, rd, score in itertools.product(
            THETAS[1:-1], DIFFICULTIES, DISCRIMINATIONS, RDS, SCORES
        ):
            state = AbilityState(theta=theta, rd=rd)
            result = update_ability(state, difficulty=b, score=score, discrimination=a)
            if score > result.p:
                assert result.after.theta >= state.theta
            elif score < result.p:
                assert result.after.theta <= state.theta

    def test_evidence_never_raises_uncertainty(self):
        for theta, b, a, rd in itertools.product(THETAS[1:-1], DIFFICULTIES, DISCRIMINATIONS, RDS):
            state = AbilityState(theta=theta, rd=rd)
            result = update_ability(state, difficulty=b, score=0.5, discrimination=a)
            assert result.after.rd <= state.rd + 1e-12
            assert result.after.rd > 0.0


class TestAggregationProperties:
    def test_the_parent_is_always_valid_and_bounded_by_its_children(self):
        pool = [AbilityState(theta=t, rd=rd) for t, rd in zip(THETAS[1:-1], RDS * 2, strict=False)]
        for size in range(1, len(pool) + 1):
            for group in itertools.combinations(pool, size):
                parent = aggregate_ability(list(group))
                assert math.isfinite(parent.theta)
                assert min(c.theta for c in group) - 1e-12 <= parent.theta
                assert parent.theta <= max(c.theta for c in group) + 1e-12
                assert 0.0 < parent.rd <= min(c.rd for c in group) + 1e-12
