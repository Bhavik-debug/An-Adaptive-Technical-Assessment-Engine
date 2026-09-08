"""Dividing an item budget across topics.

Pure arithmetic, so every expectation here is worked out by hand rather than
read back from the implementation.  The two properties that matter most are
that the quotas **sum to the budget** - a lost item is a coverage requirement
that silently fails - and that the result is **deterministic under ties**, which
is what Day 13's reproducibility guarantee runs through.
"""

from __future__ import annotations

import pytest

from app.blueprint.quotas import allocate_quotas
from app.blueprint.spec import BlueprintError


class TestPlainDivision:
    def test_it_divides_in_proportion(self):
        """Exact shares 8.57, 6.67, 4.76; the two spare items go to the two
        largest fractions, so the lightest topic gains one and the heaviest
        does not."""
        assert allocate_quotas(20, {"a": 0.9, "b": 0.7, "c": 0.5}) == {"a": 8, "b": 7, "c": 5}

    def test_the_quotas_sum_to_the_budget(self):
        for budget in range(0, 40):
            quotas = allocate_quotas(budget, {"a": 0.9, "b": 0.7, "c": 0.5})
            assert sum(quotas.values()) == budget

    def test_largest_remainder_rather_than_repeated_flooring(self):
        """Three equal topics and a budget of 20 must not lose two items."""
        quotas = allocate_quotas(20, {"a": 1.0, "b": 1.0, "c": 1.0})
        assert sum(quotas.values()) == 20
        assert sorted(quotas.values()) == [6, 7, 7]

    def test_a_heavier_topic_gets_at_least_its_floor_share(self):
        """Ordering by weight holds on the *integer part*, not on the total:
        largest remainder can hand a spare item to a lighter topic, which is
        the whole point of the method."""
        quotas = allocate_quotas(21, {"a": 0.9, "b": 0.7, "c": 0.5})
        assert quotas["a"] >= quotas["b"] >= quotas["c"]

    def test_it_is_deterministic_under_ties(self):
        weights = {"a": 1.0, "b": 1.0, "c": 1.0}
        assert allocate_quotas(20, weights) == allocate_quotas(20, weights)

    def test_the_result_does_not_depend_on_insertion_order(self):
        forward = allocate_quotas(20, {"a": 1.0, "b": 1.0, "c": 1.0})
        backward = allocate_quotas(20, {"c": 1.0, "b": 1.0, "a": 1.0})
        assert forward == backward

    def test_it_does_not_mutate_its_arguments(self):
        weights = {"a": 0.9, "b": 0.7}
        minimums = {"a": 1}
        maximums: dict[str, int | None] = {"b": 4}
        allocate_quotas(10, weights, minimums=minimums, maximums=maximums)
        assert weights == {"a": 0.9, "b": 0.7}
        assert minimums == {"a": 1}
        assert maximums == {"b": 4}

    def test_a_zero_budget_gives_every_topic_nothing(self):
        assert allocate_quotas(0, {"a": 1.0, "b": 1.0}) == {"a": 0, "b": 0}


class TestDay13Compatibility:
    """Day 13's ``split_budget_by_jd`` now delegates here; nothing may move.

    The reference implementation below is Day 13's own, copied verbatim from the
    commit that introduced it.  It is not the code under test - it is the
    contract the code under test inherited, and if the two ever disagree the
    committed simulation report has silently changed.
    """

    @staticmethod
    def day_13_reference(item_budget: int, jd_weights: dict[str, float]) -> dict[str, int]:
        total = sum(jd_weights.values())
        exact = {t: item_budget * w / total for t, w in jd_weights.items()}
        quotas = {t: int(v) for t, v in exact.items()}
        remaining = item_budget - sum(quotas.values())
        order = sorted(exact, key=lambda t: (-(exact[t] - quotas[t]), t))
        for topic in order[:remaining]:
            quotas[topic] += 1
        return quotas

    @pytest.mark.parametrize("budget", [0, 1, 3, 5, 8, 12, 20, 37, 60, 100])
    @pytest.mark.parametrize(
        "weights",
        [
            {"algorithms": 0.9, "databases": 0.7, "systems": 0.5},
            {"a": 1.0, "b": 1.0, "c": 1.0},
            {"a": 1.0},
            {"a": 0.1, "b": 0.9},
            {"a": 0.05, "b": 0.05, "c": 0.9, "d": 0.5, "e": 0.33},
        ],
    )
    def test_the_unconstrained_path_is_bit_identical_to_day_13(self, budget, weights):
        assert allocate_quotas(budget, weights) == self.day_13_reference(budget, dict(weights))

    def test_the_day_13_experiment_quotas_are_unchanged(self):
        """The exact mapping printed in the committed simulation report.

        Day 13's own docstring claimed 9/7/4 here, which no version of the code
        has produced; the report has said 8/7/5 since it was generated. Day 15
        corrected the docstring, not the arithmetic.
        """
        assert allocate_quotas(20, {"algorithms": 0.9, "databases": 0.7, "systems": 0.5}) == {
            "algorithms": 8,
            "databases": 7,
            "systems": 5,
        }


class TestMinimums:
    def test_a_floor_is_honoured_before_anything_is_divided(self):
        quotas = allocate_quotas(10, {"a": 0.9, "b": 0.1}, minimums={"b": 4})
        assert quotas["b"] >= 4
        assert sum(quotas.values()) == 10

    def test_a_floor_can_outrank_the_weights(self):
        """A topic the JD barely wants still gets its promised minimum."""
        quotas = allocate_quotas(6, {"a": 1.0, "b": 0.0}, minimums={"b": 3})
        assert quotas == {"a": 3, "b": 3}

    def test_floors_that_exceed_the_budget_are_rejected(self):
        with pytest.raises(BlueprintError, match="minimums total 12 items but the budget is 8"):
            allocate_quotas(8, {"a": 1.0, "b": 1.0}, minimums={"a": 6, "b": 6})

    def test_floors_that_exactly_fill_the_budget_are_fine(self):
        assert allocate_quotas(8, {"a": 1.0, "b": 1.0}, minimums={"a": 4, "b": 4}) == {
            "a": 4,
            "b": 4,
        }

    def test_a_negative_floor_is_rejected(self):
        with pytest.raises(BlueprintError, match="must not be negative"):
            allocate_quotas(10, {"a": 1.0}, minimums={"a": -1})


class TestMaximums:
    def test_a_cap_holds_and_the_overflow_goes_elsewhere(self):
        quotas = allocate_quotas(10, {"a": 0.9, "b": 0.1}, maximums={"a": 3})
        assert quotas["a"] == 3
        assert quotas["b"] == 7
        assert sum(quotas.values()) == 10

    def test_overflow_cascades_through_several_caps(self):
        quotas = allocate_quotas(12, {"a": 0.9, "b": 0.6, "c": 0.3}, maximums={"a": 2, "b": 3})
        assert quotas == {"a": 2, "b": 3, "c": 7}
        assert sum(quotas.values()) == 12

    def test_caps_below_the_budget_leave_a_shortfall_rather_than_looping(self):
        """The function reports what it could allocate; the blueprint decides
        whether the shortfall is an error, because only it knows."""
        quotas = allocate_quotas(20, {"a": 1.0, "b": 1.0}, maximums={"a": 3, "b": 4})
        assert quotas == {"a": 3, "b": 4}
        assert sum(quotas.values()) == 7

    def test_a_cap_below_its_floor_is_rejected(self):
        with pytest.raises(BlueprintError, match="maximum .* is below minimum"):
            allocate_quotas(10, {"a": 1.0}, minimums={"a": 5}, maximums={"a": 2})

    def test_a_zero_weight_topic_can_still_absorb_overflow(self):
        """Once the weighted topics are capped, the split falls back to name
        order rather than refusing to place the remaining items."""
        quotas = allocate_quotas(9, {"a": 1.0, "b": 0.0, "c": 0.0}, maximums={"a": 1})
        assert quotas["a"] == 1
        assert sum(quotas.values()) == 9
        assert quotas["b"] == quotas["c"] == 4


class TestRejections:
    def test_a_negative_budget_is_rejected(self):
        with pytest.raises(BlueprintError, match="must not be negative"):
            allocate_quotas(-1, {"a": 1.0})

    def test_no_topics_at_all_is_rejected(self):
        with pytest.raises(BlueprintError, match="across zero topics"):
            allocate_quotas(10, {})

    def test_weights_summing_to_zero_are_rejected(self):
        with pytest.raises(BlueprintError, match="must not sum to zero"):
            allocate_quotas(20, {"a": 0.0})

    def test_zero_weights_are_fine_when_floors_spend_the_whole_budget(self):
        assert allocate_quotas(4, {"a": 0.0, "b": 0.0}, minimums={"a": 2, "b": 2}) == {
            "a": 2,
            "b": 2,
        }

    def test_a_negative_weight_is_rejected(self):
        with pytest.raises(BlueprintError, match="finite and non-negative"):
            allocate_quotas(10, {"a": -0.5, "b": 1.0})

    def test_a_non_finite_weight_is_rejected(self):
        with pytest.raises(BlueprintError, match="finite and non-negative"):
            allocate_quotas(10, {"a": float("nan"), "b": 1.0})
