"""Dividing an item budget across topics.  One function, and it is arithmetic.

The rule is **largest remainder**, which is the same method used to allocate
parliamentary seats and for the same reason: repeated flooring silently starves
whichever topic has the smallest share, and repeated rounding overspends the
budget.  Largest remainder gives every topic its integer part and then hands the
leftover items to the topics with the biggest fractional claim, so the quotas sum
to *exactly* the budget.

Ties are broken by topic name so two runs of the same experiment agree.  That is
not cosmetic: Day 13's reproducibility guarantee runs through this function.

Floors and caps
---------------

Beyond the plain division, a blueprint can say "at least two questions on
transactions" and "no more than six on algorithms".  Those turn the one-shot
division into a small loop:

1. every topic starts at its floor;
2. what is left is divided by JD weight, largest remainder;
3. anything that would push a topic past its cap is taken back and redivided
   among the topics that still have room;
4. repeat until nothing is left or every topic is full.

The loop terminates because each pass either allocates at least one item or
removes at least one topic from the eligible set.  Both are asserted by test
rather than argued for here.

**This is where ``app.simulation.config.split_budget_by_jd`` went.**  Day 13
needed a quota mapping and there was nothing to produce one, so it wrote a local
division and a docstring saying it must not grow into the Day 15 builder.  Day 15
exists; the simulation now delegates here, and a test pins that the no-floor,
no-cap path returns exactly what Day 13's implementation returned.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

from app.blueprint.spec import BlueprintError


def _largest_remainder(budget: int, weights: Mapping[str, float]) -> dict[str, int]:
    """Divide ``budget`` in proportion to ``weights``.  Sums to exactly ``budget``.

    Assumes a positive weight total and a non-negative budget; the caller checks
    both, because the useful error message depends on why it was called.
    """
    total = math.fsum(weights.values())
    exact = {topic: budget * weight / total for topic, weight in weights.items()}
    quotas = {topic: int(value) for topic, value in exact.items()}
    remaining = budget - sum(quotas.values())
    order = sorted(exact, key=lambda topic: (-(exact[topic] - quotas[topic]), topic))
    for topic in order[:remaining]:
        quotas[topic] += 1
    return quotas


def _even_split(budget: int, topics: list[str]) -> dict[str, int]:
    """Hand out ``budget`` one at a time, in name order.

    The fallback for a round in which every remaining topic has weight zero -
    which happens when the weighted topics have all hit their caps.  Refusing to
    allocate would leave the budget short of what the blueprint promised, and
    picking arbitrarily would be non-deterministic; going round in name order is
    neither.
    """
    quotas = dict.fromkeys(topics, 0)
    for index in range(budget):
        quotas[topics[index % len(topics)]] += 1
    return quotas


def allocate_quotas(
    item_budget: int,
    weights: Mapping[str, float],
    *,
    minimums: Mapping[str, int] | None = None,
    maximums: Mapping[str, int | None] | None = None,
) -> dict[str, int]:
    """``budget + weights (+ floors, caps)`` -> ``topic -> quota``.

    The quotas always sum to ``item_budget`` unless every topic has hit its cap
    first, in which case they sum to the total of the caps and the shortfall is
    the caller's to notice - :class:`~app.blueprint.blueprint.Blueprint`
    validation is what turns that into an error, because only there is it known
    whether the shortfall matters.

    Raises :class:`BlueprintError` for the two states that cannot produce an
    answer at all: a budget too small for the floors, and a zero weight total
    with no floors to fall back on.
    """
    if item_budget < 0:
        raise BlueprintError(f"item_budget must not be negative, got {item_budget!r}")
    if not weights:
        raise BlueprintError("cannot divide an item budget across zero topics")

    floors = {topic: int(minimums.get(topic, 0)) if minimums else 0 for topic in weights}
    caps = {topic: (maximums or {}).get(topic) for topic in weights}
    for topic, floor in floors.items():
        if floor < 0:
            raise BlueprintError(f"minimum for {topic!r} must not be negative, got {floor!r}")
        cap = caps[topic]
        if cap is not None and cap < floor:
            raise BlueprintError(f"maximum ({cap}) is below minimum ({floor}) for topic {topic!r}")
    for topic, weight in weights.items():
        if not math.isfinite(weight) or weight < 0.0:
            raise BlueprintError(
                f"weight for {topic!r} must be finite and non-negative, got {weight!r}"
            )

    floor_total = sum(floors.values())
    if floor_total > item_budget:
        raise BlueprintError(
            f"the per-topic minimums total {floor_total} items but the budget is "
            f"{item_budget}; lengthen the assessment or lower the minimums"
        )

    quotas = dict(floors)
    remaining = item_budget - floor_total
    if math.fsum(weights.values()) <= 0.0 and remaining > 0:
        raise BlueprintError(
            f"weights must not sum to zero: there is no proportion to divide the "
            f"remaining {remaining} item(s) by; give at least one topic a positive "
            "weight, or state the whole budget as per-topic minimums"
        )

    # Each pass allocates to the topics that still have room; anything clipped by
    # a cap returns to `remaining` and is offered to the others on the next pass.
    while remaining > 0:
        eligible = [
            topic
            for topic in sorted(weights)
            if caps[topic] is None or quotas[topic] < int(caps[topic] or 0)
        ]
        if not eligible:
            break
        eligible_weights = {topic: weights[topic] for topic in eligible}
        if math.fsum(eligible_weights.values()) > 0.0:
            share = _largest_remainder(remaining, eligible_weights)
        else:
            share = _even_split(remaining, eligible)

        allocated = 0
        for topic, extra in share.items():
            cap = caps[topic]
            room = extra if cap is None else min(extra, int(cap) - quotas[topic])
            quotas[topic] += room
            allocated += room
        remaining -= allocated
        if allocated == 0:
            # Nothing moved and nothing is capped-out: only reachable if every
            # eligible topic received zero, which the largest-remainder method
            # cannot do while `remaining > 0`.
            break  # pragma: no cover

    return dict(sorted(quotas.items()))


__all__ = ["allocate_quotas"]
