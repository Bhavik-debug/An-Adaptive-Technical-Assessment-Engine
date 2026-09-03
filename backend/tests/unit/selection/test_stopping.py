"""The stopping rule: four conditions, each tested alone and then together.

The interesting cases are the boundaries and the reset. Three consecutive small
updates stop the interview; two do not; and one substantial update anywhere in
the run puts the counter back to zero.
"""

from __future__ import annotations

import math

import pytest

from app.ability import AbilityState
from app.selection.stopping import (
    RD_PRECISION_TARGET,
    SMALL_DELTA_THETA,
    SMALL_UPDATE_RUN,
    STOP_ITEM_BUDGET,
    STOP_NO_NEW_INFORMATION,
    STOP_PRECISION,
    STOP_TIME_BUDGET,
    consecutive_small_updates,
    precision_reached,
    should_stop,
)

PARENT_OF = {"caching": "systems", "sharding": "systems", "sorting": "algorithms"}

#: Two subtopics whose precision-weighted parent RD is comfortably above 0.40,
#: so no test accidentally satisfies the precision condition by mistake.
IMPRECISE = {
    "caching": AbilityState(theta=0.5, rd=1.0, n_observations=1),
    "sorting": AbilityState(theta=0.0, rd=1.0, n_observations=1),
}


def running(**overrides):
    """A mid-interview state that stops for no reason at all."""
    kwargs = {
        "ability": IMPRECISE,
        "parent_of": PARENT_OF,
        "required_topics": ["systems", "algorithms"],
        "items_asked": 4,
        "item_budget": 12,
        "time_elapsed_s": 300.0,
        "time_budget_s": 1800.0,
        "recent_theta_deltas": [0.4, 0.3],
    }
    kwargs.update(overrides)
    return should_stop(**kwargs)


class TestThresholds:
    def test_they_are_the_plans_numbers(self):
        assert RD_PRECISION_TARGET == 0.40
        assert SMALL_DELTA_THETA == 0.05
        assert SMALL_UPDATE_RUN == 3


class TestNothingFired:
    def test_a_healthy_mid_interview_state_does_not_stop(self):
        decision = running()
        assert decision.should_stop is False
        assert decision.reasons == ()
        assert not decision


# ---------------------------------------------------------------------------
# 1. sufficient precision
# ---------------------------------------------------------------------------


class TestPrecisionCondition:
    def test_every_required_topic_precise_enough_stops_the_interview(self):
        ability = {
            "caching": AbilityState(theta=1.0, rd=0.35, n_observations=4),
            "sorting": AbilityState(theta=0.5, rd=0.35, n_observations=4),
        }
        assert precision_reached(ability, PARENT_OF, ["systems", "algorithms"]) is True
        assert running(ability=ability).reasons == (STOP_PRECISION,)

    def test_one_imprecise_required_topic_is_enough_to_keep_going(self):
        ability = {
            "caching": AbilityState(theta=1.0, rd=0.35, n_observations=4),
            "sorting": AbilityState(theta=0.5, rd=1.1, n_observations=1),
        }
        assert precision_reached(ability, PARENT_OF, ["systems", "algorithms"]) is False

    def test_rd_exactly_at_the_target_is_not_precise_enough(self):
        """`RD < 0.40`, strictly."""
        ability = {"caching": AbilityState(theta=0.0, rd=0.40, n_observations=3)}
        assert precision_reached(ability, PARENT_OF, ["systems"]) is False

    def test_rd_just_under_the_target_is(self):
        ability = {"caching": AbilityState(theta=0.0, rd=0.3999, n_observations=3)}
        assert precision_reached(ability, PARENT_OF, ["systems"]) is True

    def test_a_required_topic_with_no_measured_subtopic_is_not_satisfied(self):
        ability = {"caching": AbilityState(theta=1.0, rd=0.35, n_observations=4)}
        assert precision_reached(ability, PARENT_OF, ["systems", "algorithms"]) is False

    def test_topics_outside_the_required_set_are_ignored(self):
        ability = {
            "caching": AbilityState(theta=1.0, rd=0.35, n_observations=4),
            "sorting": AbilityState(theta=0.5, rd=1.3, n_observations=0),
        }
        assert precision_reached(ability, PARENT_OF, ["systems"]) is True

    def test_topic_rd_is_the_precision_weighted_aggregate_not_a_subtopic_rd(self):
        """Two subtopics at RD 0.5 aggregate to 1/sqrt(4+4) = 0.354, which is
        under target even though neither subtopic is. That is Day 11's
        `roll_up`, reused rather than re-derived."""
        ability = {
            "caching": AbilityState(theta=1.0, rd=0.5, n_observations=2),
            "sharding": AbilityState(theta=0.8, rd=0.5, n_observations=2),
        }
        assert precision_reached(ability, PARENT_OF, ["systems"]) is True

    def test_no_measurements_at_all_is_not_precision(self):
        assert precision_reached({}, PARENT_OF, ["systems"]) is False

    def test_an_empty_required_set_is_not_a_vacuous_success(self):
        """Otherwise the rule would fire before the first question."""
        assert precision_reached(IMPRECISE, PARENT_OF, []) is False

    def test_an_unmapped_subtopic_raises_rather_than_being_dropped(self):
        ability = {"orphan": AbilityState(theta=0.0, rd=0.3)}
        with pytest.raises(ValueError, match="no parent mapped"):
            precision_reached(ability, PARENT_OF, ["systems"])

    def test_the_target_is_overridable(self):
        ability = {"caching": AbilityState(theta=0.0, rd=0.5, n_observations=2)}
        assert precision_reached(ability, PARENT_OF, ["systems"]) is False
        assert precision_reached(ability, PARENT_OF, ["systems"], rd_target=0.6) is True


# ---------------------------------------------------------------------------
# 2. the item budget
# ---------------------------------------------------------------------------


class TestItemBudget:
    def test_below_budget_keeps_going(self):
        assert running(items_asked=11, item_budget=12).reasons == ()

    def test_exactly_at_budget_stops(self):
        assert running(items_asked=12, item_budget=12).reasons == (STOP_ITEM_BUDGET,)

    def test_over_budget_stops(self):
        assert running(items_asked=13, item_budget=12).reasons == (STOP_ITEM_BUDGET,)

    def test_a_zero_budget_stops_immediately(self):
        assert STOP_ITEM_BUDGET in running(items_asked=0, item_budget=0).reasons


# ---------------------------------------------------------------------------
# 3. the time budget
# ---------------------------------------------------------------------------


class TestTimeBudget:
    def test_time_remaining_keeps_going(self):
        assert running(time_elapsed_s=1799.0, time_budget_s=1800.0).reasons == ()

    def test_exactly_at_the_budget_stops(self):
        assert running(time_elapsed_s=1800.0, time_budget_s=1800.0).reasons == (STOP_TIME_BUDGET,)

    def test_over_the_budget_stops(self):
        assert running(time_elapsed_s=1900.0, time_budget_s=1800.0).reasons == (STOP_TIME_BUDGET,)


# ---------------------------------------------------------------------------
# 4. three consecutive small updates
# ---------------------------------------------------------------------------


class TestConsecutiveSmallUpdates:
    def test_no_updates_yet_is_a_run_of_zero(self):
        assert consecutive_small_updates([]) == 0

    def test_it_counts_the_trailing_run(self):
        assert consecutive_small_updates([0.4, 0.01, 0.02]) == 2

    def test_a_substantial_update_resets_the_run(self):
        assert consecutive_small_updates([0.01, 0.01, 0.40, 0.01]) == 1

    def test_the_sign_of_the_step_does_not_matter(self):
        assert consecutive_small_updates([-0.01, 0.02, -0.03]) == 3

    def test_exactly_at_the_threshold_is_not_small(self):
        """`|dtheta| < 0.05`, strictly."""
        assert consecutive_small_updates([0.01, 0.01, 0.05]) == 0

    def test_just_under_the_threshold_is(self):
        assert consecutive_small_updates([0.01, 0.01, 0.0499]) == 3

    def test_exactly_three_small_updates_stop_the_interview(self):
        assert running(recent_theta_deltas=[0.4, 0.01, 0.02, 0.01]).reasons == (
            STOP_NO_NEW_INFORMATION,
        )

    def test_only_two_small_updates_do_not(self):
        assert running(recent_theta_deltas=[0.4, 0.01, 0.02]).reasons == ()

    def test_one_large_update_resets_the_count_and_the_interview_continues(self):
        assert running(recent_theta_deltas=[0.01, 0.01, 0.30, 0.01, 0.01]).reasons == ()

    def test_more_than_three_still_stops(self):
        assert STOP_NO_NEW_INFORMATION in running(recent_theta_deltas=[0.01] * 6).reasons

    def test_a_non_finite_delta_is_rejected(self):
        with pytest.raises(ValueError, match="delta must be finite"):
            consecutive_small_updates([math.nan])

    def test_the_threshold_and_run_length_are_overridable(self):
        assert consecutive_small_updates([0.2, 0.2], threshold=0.5) == 2
        assert (
            STOP_NO_NEW_INFORMATION
            in running(recent_theta_deltas=[0.01, 0.01], small_update_run=2).reasons
        )

    def test_a_run_length_below_one_is_rejected(self):
        with pytest.raises(ValueError, match="small_update_run must be at least 1"):
            running(small_update_run=0)


# ---------------------------------------------------------------------------
# 5. together
# ---------------------------------------------------------------------------


class TestAllConditionsTogether:
    def test_every_reason_that_applies_is_reported(self):
        precise = {
            "caching": AbilityState(theta=1.0, rd=0.30, n_observations=6),
            "sorting": AbilityState(theta=0.5, rd=0.30, n_observations=6),
        }
        decision = should_stop(
            ability=precise,
            parent_of=PARENT_OF,
            required_topics=["systems", "algorithms"],
            items_asked=12,
            item_budget=12,
            time_elapsed_s=1800.0,
            time_budget_s=1800.0,
            recent_theta_deltas=[0.01, 0.01, 0.01],
        )
        assert decision.should_stop is True
        assert set(decision.reasons) == {
            STOP_PRECISION,
            STOP_ITEM_BUDGET,
            STOP_TIME_BUDGET,
            STOP_NO_NEW_INFORMATION,
        }

    def test_any_single_condition_is_enough(self):
        for override in (
            {"items_asked": 12},
            {"time_elapsed_s": 1800.0},
            {"recent_theta_deltas": [0.01, 0.01, 0.01]},
        ):
            assert running(**override).should_stop is True

    def test_the_reasons_keep_the_plans_order(self):
        decision = should_stop(
            ability=IMPRECISE,
            parent_of=PARENT_OF,
            required_topics=["systems", "algorithms"],
            items_asked=12,
            item_budget=12,
            time_elapsed_s=1800.0,
            time_budget_s=1800.0,
            recent_theta_deltas=[0.01, 0.01, 0.01],
        )
        assert decision.reasons == (
            STOP_ITEM_BUDGET,
            STOP_TIME_BUDGET,
            STOP_NO_NEW_INFORMATION,
        )


class TestValidation:
    @pytest.mark.parametrize(
        "overrides",
        [{"items_asked": -1}, {"item_budget": -1}],
    )
    def test_negative_item_counts_are_rejected(self, overrides):
        with pytest.raises(ValueError, match="must not be negative"):
            running(**overrides)

    @pytest.mark.parametrize(
        "overrides",
        [{"time_elapsed_s": -1.0}, {"time_budget_s": -1.0}],
    )
    def test_negative_times_are_rejected(self, overrides):
        with pytest.raises(ValueError, match="must not be negative"):
            running(**overrides)

    @pytest.mark.parametrize("bad", [math.nan, math.inf])
    def test_non_finite_times_are_rejected(self, bad):
        with pytest.raises(ValueError, match="must be finite"):
            running(time_elapsed_s=bad)
