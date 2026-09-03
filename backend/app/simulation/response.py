"""The Beta response model: what score a synthetic candidate gets on an item.

Plan section 8.6, step 2: *"for candidate i and item j, draw a score from a Beta
distribution centred on p(theta_i, b_j) - this simulates a graded answer with
realistic noise"*.

The two steps
-------------

**1. The expected score comes from Day 11, unchanged.**

::

    p = probability_correct(true_theta[subtopic], item.b, item.a)

That is ``app.ability.probability_correct`` - the production 2PL - called with
the candidate's **ground-truth** theta rather than the engine's estimate.  This
module is the only place in the package that reads ground truth, and ``p`` never
leaves it: what comes out is a score, which is exactly what a real grader would
hand back.

Note that the item's own discrimination ``a`` *is* used here.  It is not used in
Day 12's selection score (plan section 8.3 writes ``sigmoid(theta - b)``), so a
sharp item genuinely behaves differently from a flat one in the world while the
policy cannot see that difference - which is the correct asymmetry to simulate.

**2. The observed score is a Beta draw centred on it.**

::

    alpha = p * k
    beta  = (1 - p) * k
    s     = Beta(alpha, beta)          mean p, variance p(1-p)/(k+1)

Why Beta rather than a coin flip: Day 11's ``update_ability`` accepts a
**soft** score in ``[0, 1]``, because a graded answer is a rubric total and not
a right/wrong bit.  A Bernoulli draw would throw that away and would make the
simulation test a model the system does not use.  Beta is the natural
distribution on ``[0, 1]``, and parameterising it by mean and concentration is
what lets ``p`` stay exactly the expected score while ``k`` controls how noisy
the grader is.

``k = 10`` gives a standard deviation of 0.151 on a coin-flip item - about the
spread two competent human graders show on the same answer.  Larger ``k`` is a
more reliable grader; ``k -> infinity`` would be a noiseless one.

The extremes
------------

At ``p = 0`` exactly, ``alpha = 0`` and the Beta distribution is degenerate, so
``p`` is clamped into ``[0.01, 0.99]`` before it becomes parameters.  The clamp
moves the mean by at most 0.01 and only for items at the far end of the
difficulty range; without it the model is undefined precisely where the bank is
widest.  The draw is clamped to ``[0, 1]`` afterwards as well, because Day 11
*rejects* an out-of-range score rather than clamping it, and a floating-point
edge should not look like a scoring bug.
"""

from __future__ import annotations

import random

from app.ability import probability_correct
from app.selection import CandidateItem
from app.simulation.config import RESPONSE_CONCENTRATION, RESPONSE_P_EPSILON
from app.simulation.environment import SyntheticCandidate, derive_seed


def expected_score(candidate: SyntheticCandidate, item: CandidateItem) -> float:
    """``p(theta_true, b, a)`` - the mean of the response distribution.

    Day 11's own 2PL, called with ground truth.  Separated from the draw so a
    test can assert the model's *shape* (harder item, lower expectation)
    without touching randomness at all.
    """
    return probability_correct(
        candidate.theta(item.subtopic_key),
        item.difficulty_b,
        item.discrimination_a,
    )


def beta_parameters(p: float, concentration: float = RESPONSE_CONCENTRATION) -> tuple[float, float]:
    """``(alpha, beta)`` for a Beta with mean ``p`` and the given concentration.

    ``alpha + beta = k``, so ``k`` is the "sample size" the grader's opinion is
    worth: the variance is ``p(1-p)/(k+1)``.
    """
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"p must be in [0, 1], got {p!r}")
    if concentration <= 0.0:
        raise ValueError(f"concentration must be positive, got {concentration!r}")
    safe = min(max(p, RESPONSE_P_EPSILON), 1.0 - RESPONSE_P_EPSILON)
    return safe * concentration, (1.0 - safe) * concentration


def draw_score(
    p: float,
    rng: random.Random,
    *,
    concentration: float = RESPONSE_CONCENTRATION,
) -> float:
    """One graded answer: a Beta draw centred on ``p``, clamped to ``[0, 1]``."""
    alpha, beta = beta_parameters(p, concentration)
    score = rng.betavariate(alpha, beta)
    return 0.0 if score < 0.0 else 1.0 if score > 1.0 else score


def response_rng(seed: int, candidate: SyntheticCandidate, item: CandidateItem) -> random.Random:
    """The generator for *this* candidate answering *this* item.

    Keyed by the pair and nothing else - not by the strategy, not by the turn
    number, not by what was asked before.  So candidate ``c007`` answering
    ``arrays-19`` produces the same score under CAT, under random selection and
    under the fixed sequence, whether it was their first question or their
    twentieth.

    This is **common random numbers**, the standard variance-reduction technique
    for comparing policies on a shared environment.  Two consequences worth
    stating: any difference between the strategies' results is caused by *which
    questions they chose* and not by luckier grading, and the comparison needs
    far fewer candidates to be stable than it would with independent draws.
    """
    return random.Random(derive_seed(seed, "response", candidate.id, item.id))


def graded_score(
    seed: int,
    candidate: SyntheticCandidate,
    item: CandidateItem,
    *,
    concentration: float = RESPONSE_CONCENTRATION,
) -> float:
    """The score this candidate gets on this item.  Deterministic given ``seed``.

    The only function in :mod:`app.simulation` that reads ground truth, and the
    only channel by which ground truth influences anything downstream.
    """
    return draw_score(
        expected_score(candidate, item),
        response_rng(seed, candidate, item),
        concentration=concentration,
    )


__all__ = [
    "beta_parameters",
    "draw_score",
    "expected_score",
    "graded_score",
    "response_rng",
]
