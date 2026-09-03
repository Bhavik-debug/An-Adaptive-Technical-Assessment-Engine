"""Fisher information - how much one question can teach us about one candidate.

Plan section 5.10.  Pure arithmetic, exactly like :mod:`app.ability`: no
database, no clock, no global state.  Day 11 answered *"where is this candidate
now?"*; this module answers the question that comes immediately after, *"which
question would move that estimate the most?"*.

The formula
-----------

For the 2PL model of :func:`app.ability.probability_correct`::

    I(theta, b) = a^2 * p * (1 - p)        where p = 1 / (1 + exp(-a*(theta-b)))

``p * (1 - p)`` is the variance of a Bernoulli trial with success probability
``p``, and it is maximised at ``p = 0.5`` - which happens exactly when
``b = theta``.  So the most informative item in the bank is the one the model
genuinely cannot call.

    | situation             |    p | ``p*(1-p)``               |
    |-----------------------|-----:|---------------------------|
    | way too easy          | 0.95 | 0.0475 - almost nothing   |
    | perfectly matched     | 0.50 | 0.2500 - **maximum**      |
    | way too hard          | 0.05 | 0.0475 - almost nothing   |

**The intuition.**  If you are already confident someone will get a question
right, watching them get it right teaches you nothing.  Uncertainty in the
*prediction* is information in the *result*.  The viva-ready analogy is binary
search: you probe the middle of the array, not the ends, because the middle is
the probe that halves what you do not know.  Adaptive testing is binary search
over ability, and Fisher information is its continuous, probabilistic form.

Two quantities, deliberately kept apart
---------------------------------------

:func:`fisher_information`
    The real thing, ``a^2 * p * (1-p)``.  Its scale depends on ``a``: a sharp
    item (``a = 2``) carries four times the information of a flat one
    (``a = 1``) at the same ``p``.  This is what the RD update in
    :func:`app.ability.update_uncertainty` adds to the precision, and it is the
    quantity to quote when someone asks "what is Fisher information?".

:func:`normalised_information`
    ``p * (1-p) / 0.25``, in ``[0, 1]``.  This is what the *selection
    objective* (plan section 8.3) uses, because that objective is a weighted sum
    of six terms and a term whose range depends on the item's discrimination
    would silently re-weight the whole thing.  The plan's ``score_item`` writes
    ``p = sigmoid(state.theta[q.subtopic] - q.difficulty_b)`` with no ``a`` at
    all, so the selection ``p`` is the ``a = 1`` special case on purpose.

They are not interchangeable and no function here quietly converts one into the
other.  ``a`` is still doing its job - it shapes ``p`` in the ability model and
it drives the RD update - it is simply *not* a multiplier on the selection
score, because the plan does not make it one.
"""

from __future__ import annotations

import math

from app.ability import DEFAULT_DISCRIMINATION, probability_correct

#: ``max p*(1-p) = 0.25``, at ``p = 0.5``.  The normaliser, named rather than
#: written as a bare literal so that the one place it comes from is visible.
MAX_INFORMATION = 0.25


def information_from_p(p: float) -> float:
    """``p * (1 - p)`` - the un-normalised, discrimination-free information term.

    In ``[0, 0.25]`` for any valid probability, symmetric about ``p = 0.5``.
    Rejects a ``p`` outside ``[0, 1]`` rather than clamping: the only way to get
    one is a caller bug, and returning a negative "information" would quietly
    invert a ranking.
    """
    if not math.isfinite(p):
        raise ValueError(f"p must be finite, got {p!r}")
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"p must be in [0, 1], got {p!r}")
    return p * (1.0 - p)


def normalised_information(p: float) -> float:
    """``p * (1 - p) / 0.25`` - the ``[0, 1]`` information term the objective uses.

    1.0 when the item is a coin flip (``b == theta``), 0.0 when the outcome is
    certain.  The division is by a constant, so this is a pure rescaling: it
    changes nothing about *which* item is most informative, only the units the
    weighted sum in :mod:`app.selection.objective` adds it up in.
    """
    return information_from_p(p) / MAX_INFORMATION


def fisher_information(
    theta: float,
    difficulty: float,
    discrimination: float = DEFAULT_DISCRIMINATION,
) -> float:
    """``I(theta, b) = a^2 * p * (1 - p)`` for the 2PL model.

    Non-negative and finite for every valid input, because ``a^2 >= 0`` and
    ``p in [0, 1]``.  Maximised at ``theta == difficulty``, where ``p`` is
    exactly 0.5 and the value is ``a^2 / 4``.

    ``p`` comes from :func:`app.ability.probability_correct` - the same function
    the ability update uses - so the selection layer and the scoring layer can
    never disagree about what the model predicts.  Validation (finite inputs,
    positive ``a``) is that function's, and is not repeated here.
    """
    p = probability_correct(theta, difficulty, discrimination)
    return discrimination * discrimination * p * (1.0 - p)


def selection_probability(theta: float, difficulty: float) -> float:
    """``sigmoid(theta - b)`` - the ``p`` the plan's ``score_item`` uses.

    A named alias for ``probability_correct(theta, b)`` at the default ``a = 1``,
    so that a reader of the objective sees *which* probability is meant without
    having to remember that the discrimination argument was left out on purpose.
    """
    return probability_correct(theta, difficulty, DEFAULT_DISCRIMINATION)


__all__ = [
    "MAX_INFORMATION",
    "fisher_information",
    "information_from_p",
    "normalised_information",
    "selection_probability",
]
