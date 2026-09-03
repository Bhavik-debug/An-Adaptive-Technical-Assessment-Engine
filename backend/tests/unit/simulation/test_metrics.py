"""The measurement functions, against hand-worked examples.

Pure arithmetic, so every value below is one somebody can check on paper. The
cases that matter most are the awkward ones: a censored convergence, a topic
nobody asked about, and a session that stopped before the budget.
"""

from __future__ import annotations

import math

import pytest

from app.ability import AbilityState
from app.simulation.metrics import (
    CensoredDistribution,
    absolute_errors,
    count_by,
    first_index_at_or_below,
    forward_filled,
    mae,
    mean_curve,
    rmse,
    summarise,
    summarise_censored,
    worst_topic_rd,
)

PARENT_OF = {"arrays": "algorithms", "graphs": "algorithms", "caching": "systems"}


class TestAbsoluteErrors:
    def test_it_is_the_per_subtopic_distance(self):
        errors = absolute_errors({"a": 1.0, "b": -0.5}, {"a": 0.4, "b": 0.5}, ["a", "b"])
        assert errors == pytest.approx([0.6, 1.0])

    def test_the_subtopic_list_decides_what_counts(self):
        """A policy that never asked about `b` still has an opinion about it -
        the prior - and a report would print it, so measuring only what was
        asked would flatter every policy that ignored half the blueprint."""
        estimated, truth = {"a": 1.0, "b": 0.0}, {"a": 1.0, "b": 2.0}
        assert absolute_errors(estimated, truth, ["a"]) == [0.0]
        assert absolute_errors(estimated, truth, ["a", "b"]) == [0.0, 2.0]

    def test_a_missing_subtopic_is_an_error_not_a_skip(self):
        with pytest.raises(KeyError, match="no value for subtopic"):
            absolute_errors({"a": 1.0}, {"a": 1.0, "b": 0.0}, ["a", "b"])

    def test_order_follows_the_requested_subtopics(self):
        errors = absolute_errors({"a": 1.0, "b": 3.0}, {"a": 0.0, "b": 0.0}, ["b", "a"])
        assert errors == pytest.approx([3.0, 1.0])


class TestMaeAndRmse:
    def test_worked_example(self):
        errors = [0.0, 1.0, 2.0]
        assert mae(errors) == pytest.approx(1.0)
        assert rmse(errors) == pytest.approx(math.sqrt(5.0 / 3.0))

    def test_they_agree_when_every_error_is_equal(self):
        assert mae([0.4] * 5) == pytest.approx(rmse([0.4] * 5))

    def test_rmse_exceeds_mae_when_errors_are_lopsided(self):
        """The disagreement is the information: it means one subtopic was never
        pinned down and the average is hiding it."""
        errors = [0.0, 0.0, 0.0, 3.0]
        assert rmse(errors) > mae(errors)

    def test_both_are_zero_for_a_perfect_estimate(self):
        assert mae([0.0, 0.0]) == 0.0
        assert rmse([0.0, 0.0]) == 0.0

    @pytest.mark.parametrize("fn", [mae, rmse])
    def test_an_empty_sample_is_rejected(self, fn):
        with pytest.raises(ValueError, match="empty"):
            fn([])


class TestWorstTopicRd:
    def test_it_aggregates_subtopics_with_day_elevens_roll_up(self):
        """Two subtopics at RD 0.5 give a topic RD of 1/sqrt(4+4) = 0.354 - the
        precision-weighted aggregate, not an average of 0.5."""
        ability = {
            "arrays": AbilityState(theta=0.0, rd=0.5),
            "graphs": AbilityState(theta=0.0, rd=0.5),
        }
        assert worst_topic_rd(ability, PARENT_OF, ["algorithms"]) == pytest.approx(0.3536, abs=1e-4)

    def test_it_takes_the_worst_topic_not_the_average(self):
        ability = {
            "arrays": AbilityState(theta=0.0, rd=0.3),
            "caching": AbilityState(theta=0.0, rd=1.2),
        }
        assert worst_topic_rd(ability, PARENT_OF, ["algorithms", "systems"]) == pytest.approx(1.2)

    def test_an_untouched_topic_is_infinite(self):
        """ "We have no idea" must never satisfy a precision threshold."""
        ability = {"arrays": AbilityState(theta=0.0, rd=0.3)}
        assert worst_topic_rd(ability, PARENT_OF, ["algorithms", "systems"]) == math.inf

    def test_nothing_measured_at_all_is_infinite(self):
        assert worst_topic_rd({}, PARENT_OF, ["algorithms"]) == math.inf

    def test_only_measured_subtopics_are_aggregated(self):
        """Including unmeasured subtopics at the prior would make a topic look
        more precise the more subtopics nobody asked about."""
        one = {"arrays": AbilityState(theta=0.0, rd=0.6)}
        assert worst_topic_rd(one, PARENT_OF, ["algorithms"]) == pytest.approx(0.6)

    def test_an_empty_topic_list_is_rejected(self):
        with pytest.raises(ValueError, match="empty topic list"):
            worst_topic_rd({}, PARENT_OF, [])


class TestConvergenceIndex:
    def test_it_finds_the_first_crossing(self):
        assert first_index_at_or_below([1.0, 0.8, 0.4, 0.9], 0.5) == 2

    def test_never_crossing_is_none_not_a_number(self):
        assert first_index_at_or_below([1.0, 0.9], 0.5) is None

    def test_already_inside_the_threshold_is_step_zero(self):
        """Real, and different from `None`: it happens to a candidate whose true
        ability is near zero everywhere, and counting it as fast convergence
        would be wrong."""
        assert first_index_at_or_below([0.2, 0.1], 0.5) == 0

    def test_exactly_at_the_threshold_counts(self):
        assert first_index_at_or_below([0.6, 0.5], 0.5) == 1

    def test_infinity_never_crosses(self):
        assert first_index_at_or_below([math.inf, math.inf], 0.35) is None


class TestTrajectories:
    def test_forward_fill_repeats_the_last_value(self):
        """A stopped session's estimate does not move, so "after 5 items" reads
        the same as "after 3" - and truncating instead would drop the
        early-stopping sessions out of the tail of every average."""
        assert forward_filled([1.0, 0.8, 0.6], 5) == [1.0, 0.8, 0.6, 0.6, 0.6]

    def test_it_truncates_a_longer_trajectory(self):
        assert forward_filled([1.0, 0.8, 0.6], 2) == [1.0, 0.8]

    def test_an_empty_trajectory_is_rejected(self):
        with pytest.raises(ValueError, match="empty trajectory"):
            forward_filled([], 3)

    def test_the_mean_curve_averages_across_sessions(self):
        curve = mean_curve([[1.0, 0.5], [3.0, 1.5]], 2)
        assert curve == pytest.approx([2.0, 1.0])

    def test_short_sessions_still_contribute_to_the_tail(self):
        curve = mean_curve([[1.0, 0.5], [1.0]], 2)
        assert curve == pytest.approx([1.0, 0.75])

    def test_no_trajectories_is_rejected(self):
        with pytest.raises(ValueError, match="empty set of trajectories"):
            mean_curve([], 3)


class TestAggregation:
    def test_the_descriptive_statistics(self):
        d = summarise([1.0, 2.0, 3.0, 4.0])
        assert d.n == 4
        assert d.mean == pytest.approx(2.5)
        assert d.median == pytest.approx(2.5)
        assert d.stdev == pytest.approx(1.2910, abs=1e-4)
        assert (d.minimum, d.maximum) == (1.0, 4.0)

    def test_a_single_value_has_no_spread(self):
        assert summarise([2.0]).stdev == 0.0

    def test_an_empty_sample_is_rejected(self):
        with pytest.raises(ValueError, match="empty sample"):
            summarise([])

    def test_censored_values_are_counted_not_dropped(self):
        """Averaging over only the sessions that converged is the classic way to
        make a policy that rarely converges look fast."""
        d = summarise_censored([3, None, 5, None, None])
        assert d.n_total == 5
        assert d.n_reached == 2
        assert d.n_censored == 3
        assert d.reached is not None and d.reached.mean == pytest.approx(4.0)
        assert d.reached_fraction == pytest.approx(0.4)

    def test_a_fully_censored_measurement_says_so(self):
        d = summarise_censored([None, None])
        assert d.reached is None
        assert "never reached" in str(d)
        assert d.reached_fraction == 0.0

    def test_an_empty_censored_sample_is_all_zero(self):
        assert CensoredDistribution(0, 0, None).reached_fraction == 0.0

    def test_counting_is_sorted_and_complete(self):
        assert count_by(["b", "a", "b", "c"]) == {"a": 1, "b": 2, "c": 1}
        assert count_by([]) == {}
