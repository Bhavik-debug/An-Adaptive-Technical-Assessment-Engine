"""Day 14's aggregation: curves, reach rates, stability, stopping, coverage.

Two kinds of test here, and the split is deliberate.

**Hand-worked examples** on tiny synthetic trajectories, where the expected
number can be computed on paper. These are what actually pin the arithmetic:
an aggregation asserted only against the simulation's own output would agree
with itself no matter what it computed.

**Agreement with Day 13**, on the real (tiny) experiment. Day 14 draws pictures
of numbers Day 13 already published, so wherever the two overlap they must be
the *same* number - a chart that disagrees with the table beside it is worse
than no chart. No test here asserts that any policy beats another.
"""

from __future__ import annotations

import dataclasses
import math

import pytest

from app.simulation.convergence import (
    Band,
    at_or_below_curve,
    band,
    coverage_band,
    coverage_vs_error,
    curve_length,
    mae_band,
    reach_curve,
    representative_candidates,
    rmse_band,
    session_for,
    stability,
    stopping_by_items,
    theta_series,
)
from app.simulation.environment import build_environment
from app.simulation.metrics import count_by
from app.simulation.runner import SessionResult, TrajectoryStep, run_experiment
from app.simulation.strategies import ADAPTIVE, FIXED, RANDOM, STRATEGY_ORDER
from tests.unit.simulation.conftest import TINY

ENV = build_environment(TINY)
RESULT = run_experiment(TINY, environment=ENV)
SUBTOPICS = TINY.subtopics


# ---------------------------------------------------------------------------
# hand-built sessions, so the expected numbers can be worked out on paper
# ---------------------------------------------------------------------------


def step(index: int, mae: float, *, delta: float | None = None, subtopic: str | None = None):
    """A minimal step carrying only what the function under test reads."""
    return TrajectoryStep(
        items_asked=index,
        item_id=None if subtopic is None else f"{subtopic}-{index:02d}",
        subtopic_key=subtopic,
        topic_key=None,
        score=None if subtopic is None else 0.5,
        delta_theta=delta,
        theta=dict.fromkeys(SUBTOPICS, 0.0),
        rd=dict.fromkeys(SUBTOPICS, 1.3),
        subtopics_measured=0 if subtopic is None else 1,
        mae=mae,
        rmse=mae,
        worst_topic_rd=math.inf,
    )


def session(maes, *, strategy=ADAPTIVE, candidate="c000", stop="item_budget_reached", **kwargs):
    """A session whose MAE trajectory is exactly ``maes``.

    ``kwargs`` describe the *answers*, so they are applied from index 1 onwards:
    index 0 is the cold start and has no answer to describe.
    """
    steps = tuple(
        step(index, value, **(kwargs if index else {})) for index, value in enumerate(maes)
    )
    return SessionResult(
        strategy=strategy,
        candidate_id=candidate,
        items_asked=len(maes) - 1,
        asked_ids=(),
        stop_reasons=(stop,),
        final_theta=dict.fromkeys(SUBTOPICS, 0.0),
        final_rd=dict.fromkeys(SUBTOPICS, 1.3),
        ground_truth_theta=dict.fromkeys(SUBTOPICS, 0.0),
        mae=maes[-1],
        rmse=maes[-1],
        mae_measured=maes[-1],
        subtopics_measured=1,
        items_to_mae=None,
        items_to_se=None,
        coverage_met=False,
        difficulty_gaps=(),
        trajectory=steps,
        time_used_s=0.0,
    )


class TestBand:
    def test_it_summarises_each_column_of_the_population(self):
        sessions = [
            session([1.0, 0.0], candidate="a"),
            session([2.0, 0.0], candidate="b"),
            session([3.0, 0.0], candidate="c"),
            session([4.0, 0.0], candidate="d"),
        ]
        result = band(sessions, lambda s: s.mae)
        assert result.n == 4
        assert result.length == 2
        assert result.mean[0] == pytest.approx(2.5)
        assert result.median[0] == pytest.approx(2.5)
        assert result.p25[0] == pytest.approx(1.75)
        assert result.p75[0] == pytest.approx(3.25)
        assert result.minimum[0] == 1.0
        assert result.maximum[0] == 4.0

    def test_the_quartiles_bracket_the_median(self):
        for entry in (mae_band(RESULT.sessions_for(name)) for name in STRATEGY_ORDER):
            for index in range(entry.length):
                assert entry.minimum[index] <= entry.p25[index] <= entry.median[index]
                assert entry.median[index] <= entry.p75[index] <= entry.maximum[index]

    def test_a_short_session_is_forward_filled_rather_than_dropped(self):
        """A stopped session's estimate does not move; truncating instead would
        bias the tail towards whichever policy runs longest."""
        sessions = [session([1.0, 0.9, 0.8], candidate="a"), session([1.0, 0.5], candidate="b")]
        result = band(sessions, lambda s: s.mae, length=3)
        assert result.mean[2] == pytest.approx((0.8 + 0.5) / 2)
        assert result.n == 2

    def test_it_can_be_extended_past_every_session(self):
        result = band([session([1.0, 0.4])], lambda s: s.mae, length=5)
        assert result.mean == (1.0, 0.4, 0.4, 0.4, 0.4)

    def test_a_non_finite_value_is_rejected_rather_than_averaged(self):
        with pytest.raises(ValueError, match="non-finite"):
            band(RESULT.sessions_for(ADAPTIVE), lambda s: s.worst_topic_rd)

    def test_an_empty_population_is_rejected(self):
        with pytest.raises(ValueError, match="empty set of sessions"):
            band([], lambda s: s.mae)
        with pytest.raises(ValueError, match="empty set of sessions"):
            curve_length([])

    def test_a_non_positive_length_is_rejected(self):
        with pytest.raises(ValueError, match="length must be positive"):
            band([session([1.0])], lambda s: s.mae, length=0)

    def test_at_clamps_to_the_end_of_the_curve(self):
        result = band([session([1.0, 0.4])], lambda s: s.mae)
        assert result.at(0) == 1.0
        assert result.at(99) == 0.4
        with pytest.raises(IndexError):
            Band(n=0, length=0, mean=(), median=(), p25=(), p75=(), minimum=(), maximum=()).at(0)


class TestAgreementWithDay13:
    """Wherever Day 14 redraws a Day 13 number, it must be the same number."""

    def test_the_mae_band_mean_is_day_13s_mae_curve(self):
        for name in STRATEGY_ORDER:
            summary = RESULT.summary(name)
            curve = mae_band(RESULT.sessions_for(name), length=len(summary.mae_curve)).mean
            assert curve == pytest.approx(summary.mae_curve)

    def test_the_final_mean_is_day_13s_headline_mae(self):
        for name in STRATEGY_ORDER:
            entry = mae_band(RESULT.sessions_for(name), length=curve_length(RESULT.sessions))
            assert entry.mean[-1] == pytest.approx(RESULT.summary(name).mae.mean)

    def test_the_rmse_band_ends_at_day_13s_headline_rmse(self):
        for name in STRATEGY_ORDER:
            entry = rmse_band(RESULT.sessions_for(name), length=curve_length(RESULT.sessions))
            assert entry.mean[-1] == pytest.approx(RESULT.summary(name).rmse.mean)

    def test_the_coverage_band_ends_at_day_13s_subtopics_measured(self):
        for name in STRATEGY_ORDER:
            entry = coverage_band(RESULT.sessions_for(name), length=curve_length(RESULT.sessions))
            assert entry.mean[-1] == pytest.approx(RESULT.summary(name).subtopics_measured.mean)

    def test_the_reach_curve_ends_at_day_13s_reached_count(self):
        for name in STRATEGY_ORDER:
            sessions = RESULT.sessions_for(name)
            curve = reach_curve(
                sessions,
                lambda s: s.mae,
                TINY.convergence_mae,
                length=curve_length(RESULT.sessions),
            )
            censored = RESULT.summary(name).items_to_mae
            assert curve[-1] == pytest.approx(censored.reached_fraction)


class TestReach:
    def test_it_is_cumulative_and_never_falls(self):
        for name in STRATEGY_ORDER:
            curve = reach_curve(RESULT.sessions_for(name), lambda s: s.mae, 0.6)
            assert all(a <= b for a, b in zip(curve, curve[1:], strict=False))

    def test_a_crossing_that_is_undone_still_counts_as_reached(self):
        """MAE is not monotone, so the two readings genuinely differ."""
        sessions = [session([1.0, 0.4, 0.9])]
        assert reach_curve(sessions, lambda s: s.mae, 0.5) == (0.0, 1.0, 1.0)
        assert at_or_below_curve(sessions, lambda s: s.mae, 0.5) == (0.0, 1.0, 0.0)

    def test_the_instantaneous_reading_never_exceeds_the_cumulative_one(self):
        for name in STRATEGY_ORDER:
            sessions = RESULT.sessions_for(name)
            ever = reach_curve(sessions, lambda s: s.mae, TINY.convergence_mae)
            still = at_or_below_curve(sessions, lambda s: s.mae, TINY.convergence_mae)
            assert all(a <= b for a, b in zip(still, ever, strict=True))

    def test_a_cold_start_already_inside_the_threshold_counts_at_zero(self):
        assert reach_curve([session([0.1, 0.1])], lambda s: s.mae, 0.5)[0] == 1.0

    def test_an_empty_population_is_rejected(self):
        with pytest.raises(ValueError, match="empty set of sessions"):
            reach_curve([], lambda s: s.mae, 0.5)
        with pytest.raises(ValueError, match="empty set of sessions"):
            at_or_below_curve([], lambda s: s.mae, 0.5)


class TestCoverage:
    def test_it_never_falls_and_never_exceeds_the_blueprint(self):
        for name in STRATEGY_ORDER:
            entry = coverage_band(RESULT.sessions_for(name))
            assert entry.mean[0] == 0.0
            assert all(a <= b for a, b in zip(entry.mean, entry.mean[1:], strict=False))
            assert entry.maximum[-1] <= len(SUBTOPICS)

    def test_the_bins_partition_the_population(self):
        for name in STRATEGY_ORDER:
            sessions = RESULT.sessions_for(name)
            bins = coverage_vs_error(sessions)
            assert sum(entry.n_sessions for entry in bins) == len(sessions)
            assert [entry.subtopics_measured for entry in bins] == sorted(
                {s.subtopics_measured for s in sessions}
            )

    def test_a_bin_averages_only_its_own_sessions(self):
        bins = coverage_vs_error([session([1.0, 0.25]), session([1.0, 0.75], candidate="b")])
        assert len(bins) == 1
        assert bins[0].n_sessions == 2
        assert bins[0].mean_mae == pytest.approx(0.5)

    def test_an_empty_population_is_rejected(self):
        with pytest.raises(ValueError, match="empty set of sessions"):
            coverage_vs_error([])


class TestStability:
    def test_the_reversal_rate_counts_sign_changes_within_a_subtopic(self):
        """+0.2, -0.2, -0.2 on one subtopic is one reversal out of two pairs."""
        maes = [1.0, 0.9, 0.8, 0.7]
        steps = (
            step(0, maes[0]),
            step(1, maes[1], delta=+0.2, subtopic="arrays"),
            step(2, maes[2], delta=-0.2, subtopic="arrays"),
            step(3, maes[3], delta=-0.2, subtopic="arrays"),
        )
        one = session(maes)
        one = dataclasses.replace(one, trajectory=steps)
        summary = stability([one])
        assert summary.reversal_rate == pytest.approx(0.5)
        assert summary.mean_abs_delta == pytest.approx(0.2)
        assert summary.mae_increase_rate == 0.0

    def test_updates_to_different_subtopics_are_not_compared(self):
        """Consecutive updates to unrelated subtopics have no reason to agree in
        sign; counting them would make every broad policy look unstable."""
        maes = [1.0, 0.9, 0.8]
        steps = (
            step(0, maes[0]),
            step(1, maes[1], delta=+0.3, subtopic="arrays"),
            step(2, maes[2], delta=-0.3, subtopic="graphs"),
        )
        one = session(maes)
        one = dataclasses.replace(one, trajectory=steps)
        assert stability([one]).reversal_rate == 0.0

    def test_a_rising_error_is_counted(self):
        assert stability([session([0.5, 0.6, 0.5])]).mae_increase_rate == pytest.approx(0.5)

    def test_the_cold_start_is_not_a_zero_sized_update(self):
        """Step 0 carries ``delta_theta = None``; averaging it in as 0.0 would
        drag every session's mean movement towards zero."""
        summary = stability([session([1.0, 0.5], delta=0.4, subtopic="arrays")])
        assert summary.mean_abs_delta == pytest.approx(0.4)
        assert summary.delta_band.mean[0] == 0.0

    def test_the_late_window_looks_only_at_the_end(self):
        maes = [1.0, 0.9, 0.8, 0.7]
        steps = (
            step(0, maes[0]),
            step(1, maes[1], delta=1.0, subtopic="arrays"),
            step(2, maes[2], delta=0.1, subtopic="arrays"),
            step(3, maes[3], delta=0.1, subtopic="arrays"),
        )
        one = session(maes)
        one = dataclasses.replace(one, trajectory=steps)
        summary = stability([one], late_window=2)
        assert summary.late_mean_abs_delta == pytest.approx(0.1)
        assert summary.mean_abs_delta == pytest.approx(0.4)

    def test_the_rates_stay_in_range_on_the_real_experiment(self):
        for name in STRATEGY_ORDER:
            summary = stability(RESULT.sessions_for(name))
            assert summary.n_sessions == TINY.candidate_count
            assert 0.0 <= summary.reversal_rate <= 1.0
            assert 0.0 <= summary.mae_increase_rate <= 1.0
            assert summary.mean_abs_delta >= 0.0

    def test_invalid_arguments_are_rejected(self):
        with pytest.raises(ValueError, match="empty set of sessions"):
            stability([])
        with pytest.raises(ValueError, match="late_window must be positive"):
            stability([session([1.0, 0.5])], late_window=0)


class TestStopping:
    def test_the_row_totals_are_day_13s_stopping_table(self):
        for name in STRATEGY_ORDER:
            sessions = RESULT.sessions_for(name)
            profile = stopping_by_items(sessions)
            expected = count_by([s.primary_stop_reason for s in sessions])
            assert {reason: profile.total_for(reason) for reason in profile.reasons} == expected
            assert sum(profile.counts.values()) == len(sessions)

    def test_it_crosses_the_reason_with_the_question_count(self):
        sessions = [
            session([1.0] * 4, candidate="a", stop="no_new_information"),
            session([1.0] * 4, candidate="b", stop="no_new_information"),
            session([1.0] * 7, candidate="c", stop="item_budget_reached"),
        ]
        profile = stopping_by_items(sessions)
        assert profile.item_counts == (3, 6)
        assert profile.count("no_new_information", 3) == 2
        assert profile.count("item_budget_reached", 6) == 1
        assert profile.count("item_budget_reached", 3) == 0

    def test_the_mean_length_is_weighted_by_how_many_sessions_ended_there(self):
        sessions = [
            session([1.0] * 3, candidate="a", stop="no_new_information"),
            session([1.0] * 3, candidate="b", stop="no_new_information"),
            session([1.0] * 9, candidate="c", stop="no_new_information"),
        ]
        profile = stopping_by_items(sessions)
        assert profile.mean_items_for("no_new_information") == pytest.approx((2 + 2 + 8) / 3)
        assert profile.mean_items_for("never_happened") is None

    def test_an_empty_population_is_rejected(self):
        with pytest.raises(ValueError, match="empty set of sessions"):
            stopping_by_items([])


class TestRepresentatives:
    def test_they_are_ordered_by_final_error(self):
        sessions = RESULT.sessions_for(ADAPTIVE)
        picked = representative_candidates(sessions, quantiles=(0.0, 0.5, 1.0))
        errors = [session_for(sessions, candidate).mae for candidate in picked]
        assert errors == sorted(errors)
        assert errors[0] == min(s.mae for s in sessions)
        assert errors[-1] == max(s.mae for s in sessions)

    def test_the_choice_is_deterministic(self):
        sessions = RESULT.sessions_for(ADAPTIVE)
        assert representative_candidates(sessions) == representative_candidates(sessions)
        assert representative_candidates(sessions) == representative_candidates(
            list(reversed(sessions))
        )

    def test_a_duplicate_pick_is_reported_once(self):
        """Two quantiles can land on the same candidate in a small population;
        the figure set must not then contain the same chart twice."""
        picked = representative_candidates(RESULT.sessions_for(ADAPTIVE), quantiles=(0.5, 0.5, 0.5))
        assert len(picked) == 1

    def test_invalid_arguments_are_rejected(self):
        with pytest.raises(ValueError, match="empty set of sessions"):
            representative_candidates([])
        with pytest.raises(ValueError, match="quantile must be in"):
            representative_candidates(RESULT.sessions_for(ADAPTIVE), quantiles=(1.5,))


class TestThetaSeries:
    def test_it_has_one_value_per_step_and_starts_at_the_prior(self):
        one = RESULT.sessions_for(ADAPTIVE)[0]
        for subtopic in SUBTOPICS:
            series = theta_series(one, subtopic)
            assert len(series) == len(one.trajectory)
            assert series[0] == 0.0
            assert series[-1] == pytest.approx(one.final_theta[subtopic])

    def test_a_subtopic_nobody_asked_about_never_moves(self):
        one = RESULT.sessions_for(ADAPTIVE)[0]
        asked = {item_id.rsplit("-", 1)[0] for item_id in one.asked_ids}
        for subtopic in set(SUBTOPICS) - asked:
            assert set(theta_series(one, subtopic)) == {0.0}

    def test_an_unknown_subtopic_is_an_error_rather_than_a_default(self):
        with pytest.raises(KeyError, match="no estimate for subtopic"):
            theta_series(RESULT.sessions_for(ADAPTIVE)[0], "nonexistent")

    def test_an_unknown_candidate_is_an_error(self):
        with pytest.raises(KeyError, match="no session for candidate"):
            session_for(RESULT.sessions_for(FIXED), "c999")


class TestReproducibility:
    def test_the_analysis_is_a_pure_function_of_the_experiment(self):
        again = run_experiment(TINY, environment=ENV)
        for name in (ADAPTIVE, RANDOM, FIXED):
            assert mae_band(again.sessions_for(name)) == mae_band(RESULT.sessions_for(name))
            assert stability(again.sessions_for(name)) == stability(RESULT.sessions_for(name))
            assert stopping_by_items(again.sessions_for(name)) == stopping_by_items(
                RESULT.sessions_for(name)
            )
            assert coverage_vs_error(again.sessions_for(name)) == coverage_vs_error(
                RESULT.sessions_for(name)
            )
