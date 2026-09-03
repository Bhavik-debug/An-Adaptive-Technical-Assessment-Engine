"""Fisher information, and the normalised information the objective uses.

Pure arithmetic, so the property classes sweep a deterministic grid rather than
using Hypothesis - which is not a dependency of this project. Same trade as
``tests/unit/test_ability.py``: less coverage than a randomised search, but it
fails identically on every machine and in every CI run.
"""

from __future__ import annotations

import itertools
import math

import pytest

from app.ability import probability_correct
from app.selection.information import (
    MAX_INFORMATION,
    fisher_information,
    information_from_p,
    normalised_information,
    selection_probability,
)

THETAS = [-3.0, -1.5, -0.4, 0.0, 0.4, 1.5, 3.0]
DIFFICULTIES = [-3.0, -1.2, 0.0, 0.7, 2.0, 3.0]
DISCRIMINATIONS = [0.25, 0.5, 1.0, 1.6, 2.0]
PROBABILITIES = [0.0, 0.01, 0.25, 0.5, 0.75, 0.99, 1.0]


# ---------------------------------------------------------------------------
# 1. I(theta, b) = a^2 * p * (1-p)
# ---------------------------------------------------------------------------


class TestFisherInformation:
    def test_the_formula(self):
        """a^2 * p * (1-p), against p computed independently."""
        p = probability_correct(0.4, -0.3, 1.6)
        assert fisher_information(0.4, -0.3, 1.6) == pytest.approx(1.6**2 * p * (1 - p))

    def test_maximised_when_difficulty_matches_ability(self):
        """b == theta gives p == 0.5, so I == a^2/4 - the largest it can be."""
        assert fisher_information(1.7, 1.7) == pytest.approx(0.25)
        assert fisher_information(-2.0, -2.0, 2.0) == pytest.approx(1.0)

    @pytest.mark.parametrize("theta", THETAS)
    def test_the_match_point_beats_every_other_difficulty(self, theta):
        matched = fisher_information(theta, theta)
        for offset in (0.25, 0.5, 1.0, 1.5, 3.0):
            assert fisher_information(theta, theta + offset) < matched
            assert fisher_information(theta, theta - offset) < matched

    def test_information_falls_away_monotonically_from_the_match(self):
        distances = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
        values = [fisher_information(0.0, d) for d in distances]
        assert values == sorted(values, reverse=True)

    def test_symmetric_in_the_gap(self):
        """Too easy by 1.2 is exactly as uninformative as too hard by 1.2."""
        assert fisher_information(0.5, 1.7) == pytest.approx(fisher_information(0.5, -0.7))

    def test_the_plan_table(self):
        """plan section 5.10: p of 0.95/0.50/0.05 reads 0.0475/0.25/0.0475."""
        assert information_from_p(0.95) == pytest.approx(0.0475)
        assert information_from_p(0.50) == pytest.approx(0.2500)
        assert information_from_p(0.05) == pytest.approx(0.0475)

    def test_discrimination_scales_information_as_a_squared(self):
        """The `a^2` in the formula, isolated: at the match point p is 0.5 for
        every `a`, so any difference is the square term and nothing else."""
        base = fisher_information(0.0, 0.0, 1.0)
        assert fisher_information(0.0, 0.0, 2.0) == pytest.approx(4.0 * base)
        assert fisher_information(0.0, 0.0, 0.5) == pytest.approx(0.25 * base)

    def test_a_sharper_item_is_more_informative_at_the_match_point(self):
        assert fisher_information(0.0, 0.0, 2.0) > fisher_information(0.0, 0.0, 1.0)

    def test_rejects_a_non_positive_discrimination(self):
        """Delegated to `probability_correct`, and asserted so it stays that way."""
        with pytest.raises(ValueError, match="discrimination must be positive"):
            fisher_information(0.0, 0.0, 0.0)

    @pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
    def test_rejects_non_finite_inputs(self, bad):
        with pytest.raises(ValueError, match="must be finite"):
            fisher_information(bad, 0.0)
        with pytest.raises(ValueError, match="must be finite"):
            fisher_information(0.0, bad)


# ---------------------------------------------------------------------------
# 2. The normalised term the objective adds up
# ---------------------------------------------------------------------------


class TestNormalisedInformation:
    def test_a_coin_flip_is_exactly_one(self):
        assert normalised_information(0.5) == 1.0

    def test_certainty_is_zero(self):
        assert normalised_information(0.0) == 0.0
        assert normalised_information(1.0) == 0.0

    @pytest.mark.parametrize("p", [0.01, 0.99])
    def test_near_certainty_is_near_zero(self, p):
        assert normalised_information(p) == pytest.approx(0.0396, abs=1e-4)

    @pytest.mark.parametrize("p", PROBABILITIES)
    def test_always_within_zero_and_one(self, p):
        assert 0.0 <= normalised_information(p) <= 1.0

    def test_it_is_exactly_the_ratio_the_plan_writes(self):
        for p in PROBABILITIES:
            assert normalised_information(p) == pytest.approx(
                information_from_p(p) / MAX_INFORMATION
            )

    def test_max_information_is_the_value_it_normalises_by(self):
        assert MAX_INFORMATION == 0.25
        assert information_from_p(0.5) == MAX_INFORMATION

    @pytest.mark.parametrize("bad", [-0.01, 1.01])
    def test_rejects_a_p_outside_zero_and_one(self, bad):
        with pytest.raises(ValueError, match=r"p must be in \[0, 1\]"):
            normalised_information(bad)

    @pytest.mark.parametrize("bad", [math.nan, math.inf])
    def test_rejects_a_non_finite_p(self, bad):
        with pytest.raises(ValueError, match="p must be finite"):
            information_from_p(bad)


class TestSelectionProbability:
    def test_it_is_the_a_equals_one_sigmoid(self):
        """The plan's `sigmoid(state.theta - q.difficulty_b)` - no discrimination."""
        assert selection_probability(0.8, -0.2) == pytest.approx(probability_correct(0.8, -0.2))
        assert selection_probability(0.0, 0.0) == 0.5

    def test_it_ignores_the_items_discrimination_by_construction(self):
        """There is no argument for `a`; this pins that as intentional."""
        with pytest.raises(TypeError):
            selection_probability(0.0, 0.0, 2.0)  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# 3. Properties, over a deterministic grid
# ---------------------------------------------------------------------------


class TestInformationProperties:
    def test_fisher_information_is_never_negative_and_always_finite(self):
        for theta, b, a in itertools.product(THETAS, DIFFICULTIES, DISCRIMINATIONS):
            value = fisher_information(theta, b, a)
            assert math.isfinite(value)
            assert value >= 0.0

    def test_fisher_information_never_exceeds_a_squared_over_four(self):
        for theta, b, a in itertools.product(THETAS, DIFFICULTIES, DISCRIMINATIONS):
            assert fisher_information(theta, b, a) <= (a * a) / 4.0 + 1e-12

    def test_normalised_information_stays_in_the_unit_interval(self):
        for theta, b in itertools.product(THETAS, DIFFICULTIES):
            value = normalised_information(selection_probability(theta, b))
            assert 0.0 <= value <= 1.0

    def test_closer_difficulty_never_carries_less_information(self):
        for theta in THETAS:
            for near, far in itertools.combinations([0.0, 0.3, 0.9, 1.5, 2.4], 2):
                assert fisher_information(theta, theta + near) >= fisher_information(
                    theta, theta + far
                )
