"""The grading-validity metrics - Day 19, plan sections 12.1 and 3's Day-19 row.

These are the arithmetic the project's central claim is made of, so they are
tested three ways: against values worked out by hand, against an *independent
reimplementation* by a different algebraic route, and on the properties every
agreement statistic must have.

**Why not sklearn.** The plan's Day-19 study task says to check QWK against
``sklearn.metrics.cohen_kappa_score(weights="quadratic")``. This project has no
scientific-Python dependency - no numpy, no scipy, no sklearn - and adding one
weighing tens of megabytes so that a test can confirm forty lines of arithmetic
would be a poor trade. The cross-check below is stronger than a library
comparison in one specific way: it is a different *formula*, not a different
implementation of the same one, so an algebra error in the matrix form cannot
hide behind it.
"""

from __future__ import annotations

import random

import pytest

from app.grading.validity import (
    MAX_BAND,
    N_BANDS,
    AgreementReport,
    ValidityError,
    agreement,
    confusion_matrix,
    exact_and_adjacent,
    mean_absolute_error,
    mean_signed_error,
    quadratic_weighted_kappa,
    retest_reliability,
    score_to_band,
    spearman_rho,
)


def qwk_by_identity(x: list[int], y: list[int]) -> float:
    """QWK by an exact identity that uses neither a matrix nor the marginals.

    For quadratic weights the kappa collapses to::

        QWK = 1 - n * sum_i (x_i - y_i)^2 / sum_i sum_j (x_i - y_j)^2

    because the ``(k-1)^2`` weight denominator cancels between the observed and
    expected terms. The double sum is the expected disagreement under
    independence, computed directly from every cross pair rather than from
    marginal counts.

    This is a genuinely independent route to the same number: no confusion
    matrix, no row or column totals, no band count. If the implementation under
    test has an indexing or marginal-ordering bug, this disagrees.
    """
    n = len(x)
    observed = sum((a - b) ** 2 for a, b in zip(x, y, strict=True))
    expected = sum((a - b) ** 2 for a in x for b in y)
    if expected == 0:
        return 0.0
    return 1.0 - n * observed / expected


class TestScoreToBand:
    """The [0,1] -> 0-4 mapping. It changes QWK, so it is a tested decision."""

    @pytest.mark.parametrize(
        ("score", "band"),
        [(0.0, 0), (0.1, 0), (0.2, 1), (0.3, 1), (0.5, 2), (0.7, 3), (0.9, 4), (1.0, 4)],
    )
    def test_equal_width_bins(self, score, band):
        assert score_to_band(score) == band

    def test_the_whole_unit_interval_maps_into_the_scale(self):
        for step in range(0, 1001):
            band = score_to_band(step / 1000)
            assert 0 <= band <= MAX_BAND

    def test_it_is_monotone(self):
        """A higher score can never produce a lower band."""
        bands = [score_to_band(step / 200) for step in range(201)]
        assert bands == sorted(bands)

    def test_a_score_outside_the_unit_interval_is_refused(self):
        with pytest.raises(ValidityError, match="outside"):
            score_to_band(1.5)
        with pytest.raises(ValidityError, match="outside"):
            score_to_band(-0.01)

    def test_midpoint_lands_in_the_middle_band(self):
        """0.5 is the middle of the grader's range and 2 is the middle rung."""
        assert score_to_band(0.5) == 2


class TestQuadraticWeightedKappa:
    def test_perfect_agreement_is_one(self):
        reference = [0, 1, 2, 3, 4, 2, 3, 1]
        assert quadratic_weighted_kappa(reference, list(reference)) == pytest.approx(1.0)

    def test_a_hand_worked_example(self):
        """Worked out on paper, then asserted.

        reference [0,0,1,1] against predicted [1,1,0,0]:
        observed weighted disagreement = 4/16 / 4 items = 0.0625 per item,
        expected = 0.03125, so QWK = 1 - 0.25/0.125 = -1.0. Every pair is one
        band out *and* the marginals are identical, which is the worst case
        quadratic weighting can express on two bands.
        """
        assert quadratic_weighted_kappa([0, 0, 1, 1], [1, 1, 0, 0]) == pytest.approx(-1.0)

    def test_it_matches_the_independent_identity_on_random_data(self):
        """The cross-check. Different formula, same answer, 300 random cases."""
        rng = random.Random(19)
        for _ in range(300):
            n = rng.randint(2, 60)
            x = [rng.randint(0, MAX_BAND) for _ in range(n)]
            y = [rng.randint(0, MAX_BAND) for _ in range(n)]
            assert quadratic_weighted_kappa(x, y) == pytest.approx(
                qwk_by_identity(x, y), abs=1e-12
            ), (x, y)

    def test_it_matches_the_identity_on_correlated_data_too(self):
        """Random pairs are mostly near chance; this covers the useful range."""
        rng = random.Random(1912)
        for _ in range(200):
            n = rng.randint(5, 80)
            x = [rng.randint(0, MAX_BAND) for _ in range(n)]
            # y agrees with x most of the time, so kappa lands well above 0.
            y = [
                v if rng.random() < 0.75 else max(0, min(MAX_BAND, v + rng.choice((-1, 1))))
                for v in x
            ]
            assert quadratic_weighted_kappa(x, y) == pytest.approx(qwk_by_identity(x, y), abs=1e-12)

    def test_both_raters_constant_returns_zero_rather_than_one(self):
        """The degenerate case, and the most dangerous one to get wrong.

        With every item in one band there is no chance baseline to improve on,
        so kappa is undefined. Returning 1.0 would report "perfect agreement"
        for a rater that said 3 to everything - which is exactly the failure a
        compressed dataset invites.
        """
        assert quadratic_weighted_kappa([3] * 10, [3] * 10) == 0.0
        assert quadratic_weighted_kappa([3] * 10, [2] * 10) == 0.0

    def test_one_rater_constant_is_zero(self):
        assert quadratic_weighted_kappa([3] * 8, [0, 1, 2, 3, 4, 3, 2, 1]) == 0.0

    def test_big_disagreements_are_penalised_more_than_small_ones(self):
        """The 'quadratic' in the name, as a behaviour rather than a formula."""
        near = quadratic_weighted_kappa([0, 1, 2, 3, 4], [1, 2, 3, 4, 3])
        far = quadratic_weighted_kappa([0, 1, 2, 3, 4], [4, 3, 2, 1, 0])
        assert near > far

    def test_it_is_symmetric(self):
        x, y = [0, 2, 4, 1, 3, 2], [1, 2, 3, 0, 4, 2]
        assert quadratic_weighted_kappa(x, y) == pytest.approx(quadratic_weighted_kappa(y, x))

    def test_mismatched_lengths_are_refused(self):
        with pytest.raises(ValidityError, match="same length"):
            quadratic_weighted_kappa([1, 2], [1])

    def test_empty_input_is_refused(self):
        with pytest.raises(ValidityError, match="no pairs"):
            quadratic_weighted_kappa([], [])

    def test_a_band_out_of_range_is_refused(self):
        with pytest.raises(ValidityError, match="out of range"):
            quadratic_weighted_kappa([0, 5], [0, 1])


class TestSpearmanRho:
    def test_a_perfect_monotone_relationship_is_one(self):
        assert spearman_rho([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)

    def test_a_perfectly_reversed_relationship_is_minus_one(self):
        assert spearman_rho([1, 2, 3, 4], [40, 30, 20, 10]) == pytest.approx(-1.0)

    def test_it_measures_rank_not_value(self):
        """Spearman's whole point: immune to the two scales being shaped differently."""
        linear = [1.0, 2.0, 3.0, 4.0]
        exponential = [1.0, 10.0, 1000.0, 100000.0]
        assert spearman_rho(linear, exponential) == pytest.approx(1.0)

    def test_ties_are_averaged_rather_than_broken_arbitrarily(self):
        """A 0-4 scale over 120 items is mostly ties, so this is the common case.

        With all of ``b`` tied there is no ordering, so rho is 0 - not a
        division by zero and not 1.0.
        """
        assert spearman_rho([1, 2, 3, 4], [5, 5, 5, 5]) == 0.0

    def test_tied_ranks_give_the_textbook_value(self):
        """Hand-checked: x=[1,2,2,3], y=[1,2,3,4].

        Average ranks for x are [1, 2.5, 2.5, 4] and for y [1, 2, 3, 4].
        Pearson on those is 0.9486832980505138.
        """
        assert spearman_rho([1, 2, 2, 3], [1, 2, 3, 4]) == pytest.approx(0.9486832980505138)

    def test_a_single_pair_is_refused(self):
        with pytest.raises(ValidityError, match="at least two"):
            spearman_rho([1], [1])


class TestErrorMetrics:
    def test_mae_is_the_average_absolute_gap(self):
        assert mean_absolute_error([1, 2, 3], [1, 3, 5]) == pytest.approx((0 + 1 + 2) / 3)

    def test_mae_is_symmetric_and_never_negative(self):
        assert mean_absolute_error([4, 0], [0, 4]) == mean_absolute_error([0, 4], [4, 0]) == 4.0

    def test_signed_error_keeps_the_direction(self):
        assert mean_signed_error([1, 1, 1], [2, 2, 2]) == pytest.approx(1.0)
        assert mean_signed_error([2, 2, 2], [1, 1, 1]) == pytest.approx(-1.0)

    def test_signed_error_cancels_where_mae_does_not(self):
        """The reason both are reported.

        A rater that is one point high half the time and one point low the other
        half has zero *bias* and a real *error*. One number cannot say both.
        """
        reference, predicted = [2, 2, 2, 2], [1, 3, 1, 3]
        assert mean_signed_error(reference, predicted) == pytest.approx(0.0)
        assert mean_absolute_error(reference, predicted) == pytest.approx(1.0)

    def test_exact_and_adjacent_counts(self):
        exact, adjacent = exact_and_adjacent([2, 2, 2, 2], [2, 3, 1, 4])
        assert exact == 1
        assert adjacent == 3


class TestConfusionMatrix:
    def test_rows_are_reference_and_columns_are_predicted(self):
        matrix = confusion_matrix([0, 0, 4], [1, 1, 4])
        assert matrix[0][1] == 2
        assert matrix[4][4] == 1
        assert sum(sum(row) for row in matrix) == 3

    def test_it_is_square_on_the_band_count(self):
        matrix = confusion_matrix([0], [0])
        assert len(matrix) == N_BANDS
        assert all(len(row) == N_BANDS for row in matrix)


class TestRetestReliability:
    def test_identical_runs_have_zero_sigma(self):
        report = retest_reliability([[0.5, 0.6], [0.5, 0.6], [0.5, 0.6]])
        assert report.mean_sigma == 0.0
        assert report.max_sigma == 0.0
        assert report.n_items_varying == 0
        assert report.fully_deterministic is True

    def test_it_reports_the_population_standard_deviation(self):
        """Two runs of 0.4 and 0.6: mean 0.5, population sigma 0.1.

        The sample (n-1) form would give 0.1414, and the plan's target of
        sigma < 0.04 is a descriptive spread over the runs actually performed.
        """
        report = retest_reliability([[0.4], [0.6]])
        assert report.mean_sigma == pytest.approx(0.1)
        assert report.max_range == pytest.approx(0.2)

    def test_the_worst_item_is_reported_not_just_the_mean(self):
        """A mean sigma of 0.01 with one item at 0.3 is not a stable grader."""
        report = retest_reliability([[0.5, 0.0], [0.5, 1.0], [0.5, 0.0]])
        assert report.mean_sigma < report.max_sigma
        assert report.max_sigma == pytest.approx(0.4714045207910317)
        assert report.n_items_varying == 1

    def test_shape_and_emptiness_are_refused(self):
        with pytest.raises(ValidityError, match="at least two runs"):
            retest_reliability([[0.5]])
        with pytest.raises(ValidityError, match="same number of items"):
            retest_reliability([[0.5, 0.5], [0.5]])
        with pytest.raises(ValidityError, match="no items"):
            retest_reliability([[], []])


class TestAgreementReport:
    def test_it_carries_n_and_both_distributions(self):
        """A statistic without its sample size and marginals is not a result."""
        report = agreement([0, 1, 2, 3, 4], [0, 1, 2, 3, 4], label="self")
        assert isinstance(report, AgreementReport)
        assert report.n == 5
        assert report.qwk == pytest.approx(1.0)
        assert report.exact == 5
        assert report.exact_rate == pytest.approx(1.0)
        assert report.adjacent_rate == pytest.approx(1.0)
        assert sum(report.reference_distribution.values()) == 5
        assert sum(report.predicted_distribution.values()) == 5

    def test_the_confusion_matrix_totals_the_sample(self):
        report = agreement([0, 1, 2], [1, 1, 1], label="x")
        assert sum(sum(row) for row in report.confusion) == report.n

    def test_every_statistic_is_computed_over_the_same_pairs(self):
        reference = [0, 1, 2, 3, 4, 1, 2]
        predicted = [1, 1, 2, 2, 4, 0, 3]
        report = agreement(reference, predicted, label="x")
        assert report.qwk == pytest.approx(quadratic_weighted_kappa(reference, predicted))
        assert report.spearman == pytest.approx(spearman_rho(reference, predicted))
        assert report.mae == pytest.approx(mean_absolute_error(reference, predicted))
        assert report.signed_error == pytest.approx(mean_signed_error(reference, predicted))


class TestPurity:
    def test_the_metrics_module_imports_nothing_from_the_grader(self):
        """A metric that could reach the thing it measures is not a metric."""
        import ast
        from pathlib import Path

        import app.grading.validity as validity

        tree = ast.parse(Path(validity.__file__).read_text(encoding="utf-8"))
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
        forbidden = {
            "app.grading.scoring",
            "app.grading.pipeline",
            "app.grading.grader",
            "app.grading.evidence",
            "app.grading.evalset",
            "app.grading.reflabel",
            "app.llm.client",
        }
        assert not (imported & forbidden), imported & forbidden
