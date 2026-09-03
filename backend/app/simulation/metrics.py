"""How a simulated session is scored, and how 200 of them are summarised.

Pure functions over numbers - no environment, no policy, no randomness - so
every metric here can be checked against a hand-worked example.  That
separation matters more than usual in an experiment: formatting and aggregation
are where the temptation to flatter a result lives, and they should be nowhere
near the code that produced it.

The three questions
-------------------

**1. How wrong is the estimate?**  ``theta_hat`` against the hidden
``theta_true``, per subtopic, aggregated as MAE and RMSE.

**2. How fast did it get there?**  The first item count at which an error or a
precision criterion is met, or ``None`` if it never was.  A ``None`` is
**censored**, not zero and not infinite, and is counted separately - averaging
convergence over only the sessions that converged is the classic way to make a
policy that rarely converges look fast.

**3. Was the improvement consistent?**  Mean, median, spread and range across
the population, plus the per-session values so a reader can see the tail.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from app.ability import AbilityState, aggregate_ability

# ---------------------------------------------------------------------------
# 1. estimation error
# ---------------------------------------------------------------------------


def absolute_errors(
    estimated: Mapping[str, float],
    truth: Mapping[str, float],
    subtopics: Sequence[str],
) -> list[float]:
    """``|theta_hat - theta_true|`` for each subtopic, in the given order.

    ``subtopics`` is passed explicitly rather than taken from either mapping,
    because *which* subtopics count is the whole question.  A policy that never
    asked about graphs still has an opinion about graphs - the cold-start prior -
    and a report would print it, so an error measured only over what was asked
    would flatter every policy that ignored half the blueprint.

    A subtopic missing from ``estimated`` is an error in the caller, not a
    licence to skip it: the session state always has an answer, even if that
    answer is the prior.
    """
    missing = [s for s in subtopics if s not in estimated or s not in truth]
    if missing:
        raise KeyError(f"no value for subtopic(s): {', '.join(sorted(missing))}")
    return [abs(estimated[s] - truth[s]) for s in subtopics]


def mae(errors: Sequence[float]) -> float:
    """Mean absolute error - the average distance from the truth, in theta units.

    Chosen as the headline because it is in the same units as theta and reads
    directly: "0.62" means the estimate is on average 0.62 of a difficulty band
    away from the truth.
    """
    if not errors:
        raise ValueError("cannot average an empty error list")
    return math.fsum(errors) / len(errors)


def rmse(errors: Sequence[float]) -> float:
    """Root mean squared error - MAE's pessimistic sibling.

    Squaring before averaging makes one badly-missed subtopic count for more
    than several slightly-missed ones.  Reported alongside MAE precisely because
    the two disagree when errors are lopsided, and that disagreement is
    information: RMSE much larger than MAE means the average is hiding a
    subtopic the policy never pinned down.
    """
    if not errors:
        raise ValueError("cannot average an empty error list")
    return math.sqrt(math.fsum(e * e for e in errors) / len(errors))


# ---------------------------------------------------------------------------
# 2. precision
# ---------------------------------------------------------------------------


def worst_topic_rd(
    ability: Mapping[str, AbilityState],
    parent_of: Mapping[str, str],
    topics: Sequence[str],
) -> float:
    """The largest per-topic RD across ``topics``; ``inf`` if any is unmeasured.

    Topic RD is the precision-weighted aggregate of its measured subtopics -
    Day 11's ``aggregate_ability``, the same function Day 12's stopping rule and
    any future report would use.  Deriving it here rather than storing it is
    what stops the number that ends an interview and the number in its report
    from drifting apart.

    **The worst topic, not the average.**  "The assessment is precise" has to
    mean *every* required topic is precise; averaging would let a
    well-measured topic hide one nobody asked about.  A topic with no measured
    subtopic returns ``inf``, so it can never satisfy a threshold - the honest
    reading of "we have no idea".

    Note that only *measured* subtopics are aggregated.  Including unmeasured
    ones at the prior would make a topic look more precise the more subtopics it
    has, because precisions add: six untouched priors aggregate to an RD of
    0.53, which would be a confident-looking number derived entirely from
    ignorance.
    """
    if not topics:
        raise ValueError("cannot evaluate precision over an empty topic list")
    worst = 0.0
    for topic in topics:
        children = [
            state for subtopic, state in ability.items() if parent_of.get(subtopic) == topic
        ]
        if not children:
            return math.inf
        worst = max(worst, aggregate_ability(children).rd)
    return worst


# ---------------------------------------------------------------------------
# 3. convergence
# ---------------------------------------------------------------------------


def first_index_at_or_below(values: Sequence[float], threshold: float) -> int | None:
    """The first index whose value is ``<= threshold``, or ``None``.

    ``values`` is a trajectory indexed by *items asked*, so ``values[0]`` is the
    state before any question.  An index of 0 is therefore a real answer -
    "already inside the threshold at the cold start" - and is not the same as
    ``None``.  It happens to a candidate whose true ability is near zero
    everywhere, and counting it as a fast convergence would be wrong, which is
    why the report shows the zero-step count separately.
    """
    for index, value in enumerate(values):
        if value <= threshold:
            return index
    return None


def forward_filled(trajectory: Sequence[float], length: int) -> list[float]:
    """Extend a trajectory to ``length`` by repeating its last value.

    A session that stopped after 11 items has no 12th estimate - and the correct
    reading of "what would the report say after 15 items?" is "the same as after
    11", because a stopped session's estimate does not move.  Truncating instead
    would silently drop the early-stopping sessions from the tail of every
    average, which biases the curve towards whichever policy runs longest.
    """
    if not trajectory:
        raise ValueError("cannot extend an empty trajectory")
    if length <= len(trajectory):
        return list(trajectory[:length])
    return list(trajectory) + [trajectory[-1]] * (length - len(trajectory))


def mean_curve(trajectories: Sequence[Sequence[float]], length: int) -> list[float]:
    """The population mean at each item count, over forward-filled trajectories."""
    if not trajectories:
        raise ValueError("cannot average an empty set of trajectories")
    filled = [forward_filled(t, length) for t in trajectories]
    return [math.fsum(t[i] for t in filled) / len(filled) for i in range(length)]


# ---------------------------------------------------------------------------
# 4. aggregation
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Distribution:
    """Mean, median, spread and range of one measurement over the population.

    Four numbers rather than one because "adaptive is better on average" and
    "adaptive is better for most candidates" are different claims, and only the
    median and the standard deviation can tell them apart.
    """

    n: int
    mean: float
    median: float
    stdev: float
    minimum: float
    maximum: float

    def __str__(self) -> str:
        return f"{self.mean:.3f} +/- {self.stdev:.3f} (median {self.median:.3f})"


def summarise(values: Sequence[float]) -> Distribution:
    """Descriptive statistics for one measurement.  Sample standard deviation."""
    if not values:
        raise ValueError("cannot summarise an empty sample")
    return Distribution(
        n=len(values),
        mean=statistics.fmean(values),
        median=statistics.median(values),
        stdev=statistics.stdev(values) if len(values) > 1 else 0.0,
        minimum=min(values),
        maximum=max(values),
    )


@dataclass(frozen=True, slots=True)
class CensoredDistribution:
    """A convergence measurement, keeping the sessions that never converged.

    ``reached`` describes only the sessions that met the criterion;
    ``n_censored`` counts those that did not.  Both must be read together: a
    mean of 6.0 items over 12 reached sessions out of 200 is not a fast policy,
    it is a policy that almost never converges, and a single-number summary
    would say the opposite.
    """

    n_total: int
    n_reached: int
    reached: Distribution | None

    @property
    def n_censored(self) -> int:
        return self.n_total - self.n_reached

    @property
    def reached_fraction(self) -> float:
        return self.n_reached / self.n_total if self.n_total else 0.0

    def __str__(self) -> str:
        if self.reached is None:
            return f"never reached (0/{self.n_total})"
        return (
            f"{self.reached.mean:.1f} items "
            f"(median {self.reached.median:.1f}), "
            f"reached by {self.n_reached}/{self.n_total}"
        )


def summarise_censored(values: Sequence[int | None]) -> CensoredDistribution:
    """Summarise convergence steps, counting ``None`` as censored rather than dropping it."""
    reached = [float(v) for v in values if v is not None]
    return CensoredDistribution(
        n_total=len(values),
        n_reached=len(reached),
        reached=summarise(reached) if reached else None,
    )


def count_by(labels: Sequence[str]) -> dict[str, int]:
    """A stable, sorted tally - for the stopping-reason distribution."""
    counts: dict[str, int] = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    return dict(sorted(counts.items()))


__all__ = [
    "CensoredDistribution",
    "Distribution",
    "absolute_errors",
    "count_by",
    "first_index_at_or_below",
    "forward_filled",
    "mae",
    "mean_curve",
    "rmse",
    "summarise",
    "summarise_censored",
    "worst_topic_rd",
]
