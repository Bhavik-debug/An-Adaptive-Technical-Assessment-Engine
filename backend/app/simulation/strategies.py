"""The three policies being compared.  Plan section 8.6, step 3.

::

    (a) adaptive  - the real Day 12 selector, unmodified
    (b) random    - the same hard filter, then a uniform draw
    (c) fixed     - a predetermined sequence that never looks at theta

All three implement one interface and are handed exactly the same arguments::

    select(state, bank, rng) -> CandidateItem | None

Note what is *not* in that signature: the candidate.  A strategy has no
reference through which the simulator's ground truth could be read, which makes
"the policy cannot see the answer key" a structural property rather than a
convention someone might forget.

What differs, and what does not
-------------------------------

============  ================  ==========  ==========  ==========  ==========
strategy      no repeats        topic cap   time fits   |b-θ|≤1.5   ranking
============  ================  ==========  ==========  ==========  ==========
adaptive      yes               yes         yes         yes         Day 12 score + ε-greedy
random        yes               yes         yes         yes         uniform
fixed         yes               yes         yes         **no**      predetermined
============  ================  ==========  ==========  ==========  ==========

The one asymmetry is deliberate and is the reason there are three policies
rather than two:

* **Adaptive vs random** isolates the *objective*.  Both see the same eligible
  pool - including the adaptive difficulty window - so the only difference is
  whether the six-term score and ε-greedy choose within it, or a coin does.
* **Adaptive vs fixed** compares *adaptivity as a whole*, and is **not** a
  controlled comparison.  The difficulty window is a function of the running
  theta estimate, so a "fixed" policy that applied it would be adapting;
  excluding it is what makes the baseline a fixed test.  But that means the
  contrast bundles at least three changes together - the difficulty window, the
  ranking objective and epsilon-greedy - so it can say *whether* an adaptive test
  beats a non-adaptive one and **cannot attribute the difference to any single
  mechanism**.  Fixed still respects the three constraints that do **not** depend
  on theta - no repeats, the topic cap, and the clock - because those are
  properties of the session, not of the policy, and dropping them would compare
  different exams.

Reading these two the wrong way round is the likeliest misinterpretation of the
whole experiment: a large adaptive-vs-fixed gap is *not* evidence that the
weighted objective is earning its place, because most of that gap could come from
the difficulty window alone.  Only the adaptive-vs-random contrast speaks to the
objective.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol

from app.selection import (
    DEFAULT_WEIGHTS,
    DIFFICULTY_WINDOW,
    EPSILON,
    EXPLORATION_POOL_SIZE,
    CandidateItem,
    ObjectiveWeights,
    SelectionState,
    choose_next,
    filter_eligible,
)

ADAPTIVE = "adaptive"
RANDOM = "random"
FIXED = "fixed"

#: Report order: the policy under test first, then its two baselines.
STRATEGY_ORDER: tuple[str, ...] = (ADAPTIVE, RANDOM, FIXED)


class Strategy(Protocol):
    """Anything that can pick the next question.

    Deliberately narrow.  It receives the engine's *estimate* of the candidate
    and the bank, and returns one item or ``None`` when nothing is askable.
    """

    @property
    def name(self) -> str: ...

    def select(
        self,
        state: SelectionState,
        bank: Sequence[CandidateItem],
        rng: random.Random,
    ) -> CandidateItem | None: ...


# ---------------------------------------------------------------------------
# (a) the real thing
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AdaptiveStrategy:
    """Day 12's ``choose_next``, called directly.  No simplification.

    Every part of the production policy runs: the four hard constraints, all six
    scored terms - information, JD weight, resume affinity, coverage deficit,
    redundancy, time cost - the ranking with its id tie-break, and the ε-greedy
    draw at ε = 0.10 over the top five.

    ``weights`` is how the ablation switches one term off.  It defaults to the
    production objective, so the strategy used in the headline comparison is the
    shipped one, byte for byte.
    """

    weights: ObjectiveWeights = DEFAULT_WEIGHTS
    epsilon: float = EPSILON
    pool_size: int = EXPLORATION_POOL_SIZE
    difficulty_window: float = DIFFICULTY_WINDOW
    label: str = ADAPTIVE

    @property
    def name(self) -> str:
        return self.label

    def select(
        self,
        state: SelectionState,
        bank: Sequence[CandidateItem],
        rng: random.Random,
    ) -> CandidateItem | None:
        selection = choose_next(
            state,
            bank,
            rng=rng,
            epsilon=self.epsilon,
            pool_size=self.pool_size,
            difficulty_window=self.difficulty_window,
            weights=self.weights,
        )
        return None if selection is None else selection.item


# ---------------------------------------------------------------------------
# (b) the baseline that isolates the objective
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RandomStrategy:
    """Uniform over the *same* eligible pool the adaptive policy scores.

    It calls Day 12's own ``filter_eligible`` rather than reimplementing the
    constraints, so the pool really is the same pool, and then ignores every one
    of the six weighted terms.  The difference between this and
    :class:`AdaptiveStrategy` is therefore exactly the value of the objective
    plus ε-greedy - nothing else varies.

    Sorting the pool before drawing is not cosmetic: ``filter_eligible``
    preserves input order, and drawing from a differently-ordered list under the
    same seed would give a different item, which would make the run depend on
    the bank's construction order.
    """

    difficulty_window: float = DIFFICULTY_WINDOW
    label: str = RANDOM

    @property
    def name(self) -> str:
        return self.label

    def select(
        self,
        state: SelectionState,
        bank: Sequence[CandidateItem],
        rng: random.Random,
    ) -> CandidateItem | None:
        eligible = filter_eligible(bank, state, difficulty_window=self.difficulty_window)
        if not eligible:
            return None
        return rng.choice(sorted(eligible, key=lambda item: item.id))


# ---------------------------------------------------------------------------
# (c) the baseline that isolates adaptivity
# ---------------------------------------------------------------------------


def fixed_sequence(
    bank: Sequence[CandidateItem],
    quotas: Mapping[str, int],
) -> tuple[CandidateItem, ...]:
    """The predetermined exam paper: every bank item, in a theta-free order.

    Built once, from the bank and the blueprint only.  No ability estimate is
    consulted, here or later, which is what makes the policy non-adaptive.

    **Within a topic - an even spread of difficulty, easy first.**  The plan
    says "fixed sequence (easy -> hard)".  Read literally over a 192-item bank
    that means *the twenty easiest questions in existence*, all clustered near
    b = -2.5, which no real fixed test looks like and which would be a
    deliberately feeble baseline.  So the topic's ``quota`` items are taken at
    evenly spaced percentiles of its difficulty-sorted list - the easy, medium
    and hard questions a paper exam actually contains - and presented in
    increasing difficulty.  The rest of the topic's items follow, also easy to
    hard, as fallback for a turn where the front of the paper does not fit the
    clock.

    **Across topics - proportional interleaving.**  At each position the topic
    furthest behind its quota share goes next, ties broken by name.  So coverage
    accrues evenly through the session instead of finishing one topic before
    starting the next, and the paper satisfies the blueprint by construction -
    which means the topic cap never actually binds for this policy and cannot
    disadvantage it.
    """
    per_topic: dict[str, list[CandidateItem]] = {}
    for topic in sorted({item.topic_key for item in bank}):
        ordered = sorted(
            (item for item in bank if item.topic_key == topic),
            key=lambda item: (item.difficulty_b, item.id),
        )
        quota = max(0, quotas.get(topic, 0))
        if 0 < quota < len(ordered):
            stride = len(ordered) / quota
            chosen_at = sorted(
                {min(len(ordered) - 1, int((i + 0.5) * stride)) for i in range(quota)}
            )
            spread = [ordered[i] for i in chosen_at]
            rest = [item for index, item in enumerate(ordered) if index not in set(chosen_at)]
            per_topic[topic] = spread + rest
        else:
            per_topic[topic] = ordered

    total = sum(len(items) for items in per_topic.values())
    taken: dict[str, int] = dict.fromkeys(per_topic, 0)
    sequence: list[CandidateItem] = []
    for _ in range(total):
        available = [topic for topic, items in per_topic.items() if taken[topic] < len(items)]
        if not available:  # pragma: no cover - the loop count makes this unreachable
            break
        # Furthest behind its quota share first; an unquoted topic sorts last.
        topic = min(
            available,
            key=lambda t: (taken[t] / quotas[t] if quotas.get(t) else float("inf"), t),
        )
        sequence.append(per_topic[topic][taken[topic]])
        taken[topic] += 1
    return tuple(sequence)


@dataclass(frozen=True, slots=True)
class FixedStrategy:
    """Walk a predetermined sequence, skipping only what the session forbids.

    The three checks it applies - already asked, topic quota spent, does not fit
    the remaining time - are all functions of the *session*, not of theta, so
    the policy's choice is identical for a candidate who has answered everything
    perfectly and one who has answered nothing correctly.  A test asserts exactly
    that.

    ``sequence`` is supplied rather than derived, so the paper is built once per
    experiment and every candidate sits the same exam.
    """

    sequence: tuple[CandidateItem, ...] = field(default_factory=tuple)
    label: str = FIXED

    @property
    def name(self) -> str:
        return self.label

    def select(
        self,
        state: SelectionState,
        bank: Sequence[CandidateItem],
        rng: random.Random,
    ) -> CandidateItem | None:
        asked = state.asked_ids
        for item in self.sequence:
            if item.id in asked:
                continue
            if state.quota_remaining(item.topic_key) <= 0:
                continue
            if item.time_estimate_s > state.time_left_s:
                continue
            return item
        return None


def build_strategies(
    bank: Sequence[CandidateItem],
    quotas: Mapping[str, int],
) -> dict[str, Strategy]:
    """The three policies of the headline comparison, ready to run."""
    return {
        ADAPTIVE: AdaptiveStrategy(),
        RANDOM: RandomStrategy(),
        FIXED: FixedStrategy(sequence=fixed_sequence(bank, quotas)),
    }


__all__ = [
    "ADAPTIVE",
    "FIXED",
    "RANDOM",
    "STRATEGY_ORDER",
    "AdaptiveStrategy",
    "FixedStrategy",
    "RandomStrategy",
    "Strategy",
    "build_strategies",
    "fixed_sequence",
]
