"""The grading-validity metrics - Day 19, plan sections 12.1 and 3's Day-19 row.

    | 19 | Grading validity harness: QWK, Spearman rho, MAE, test-retest sigma
    |    | over 5 runs. Bias probes: length, confidence, jargon, position.
    |    | Prompt iteration v1 -> v3 against the metric.

**Everything here is pure.** No LLM, no network, no clock, no filesystem. The
same discipline ``scoring.py`` follows, and for a sharper reason: these are the
numbers the project's central claim is made of. A metric you can only reproduce
by paying a provider is a metric nobody will audit, and an LLM asked to compute
a kappa will produce a plausible number that is wrong. Arithmetic belongs in
code.

Nothing in this module imports the grader, the pipeline or the scorer. It takes
two lists of numbers and returns statistics about them; it has no opinion about
where either list came from, which is what lets the same function compute
grader-vs-reference, pass-A-vs-pass-B and v1-vs-v3 without special-casing.

The band mapping, and why it needs stating
------------------------------------------

The grader emits a **continuous score in [0, 1]** (``scoring.py``). Reference
labels are **ordinal 0-4**. QWK is defined on matching ordinal scales, so the
grader's score has to be mapped into 0-4 bands before any kappa exists, and
**the choice of mapping changes the number**. It is therefore a named,
documented, tested function rather than an inline ``round(score * 4)``.

:func:`score_to_band` uses equal-width bins with midpoint rounding, which is the
mapping that makes no assumption about the score distribution. The alternative -
quantile binning, matched to the label distribution - would raise QWK on this
dataset by construction, because it forces the two marginals to agree. That
would be measuring the binning, so it is not used, and :func:`score_to_band` is
the only mapping the harness offers.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

#: The ordinal scale both sides of every comparison live on.
MIN_BAND = 0
MAX_BAND = 4
N_BANDS = MAX_BAND - MIN_BAND + 1


class ValidityError(ValueError):
    """The inputs cannot support the statistic being asked for."""


def score_to_band(score: float, *, max_band: int = MAX_BAND) -> int:
    """Map a continuous [0, 1] grader score onto the ordinal 0-4 label scale.

    Equal-width bins: ``band = round(score * max_band)``, so the cut points sit
    at 0.125, 0.375, 0.625 and 0.875 for a 0-4 scale. A score of exactly 0.5
    lands on band 2, which is the middle of the scale and the honest place for
    it.

    **Why equal width.** It assumes nothing about how either distribution is
    shaped. Quantile binning - choosing cut points so the band counts match the
    label counts - would inflate QWK on this dataset mechanically, because kappa
    rewards matching marginals and quantile binning imposes them. Equal width
    can look *worse* than the grader deserves if the grader's scores are
    squeezed into part of the range, and that is the right direction for a
    measurement to err.

    Rounding is Python's banker's rounding only at exact halves, which for a
    0-4 scale means scores at exactly 0.125/0.375/0.625/0.875. Those are
    measure-zero in practice and the behaviour is deterministic either way.
    """
    if not 0.0 <= score <= 1.0:
        raise ValidityError(f"score {score} is outside [0, 1]")
    if max_band < 1:
        raise ValidityError("max_band must be at least 1")
    return int(round(score * max_band))


def _checked_pairs(a: Sequence[float], b: Sequence[float]) -> None:
    if len(a) != len(b):
        raise ValidityError(f"paired inputs must be the same length, got {len(a)} and {len(b)}")
    if not a:
        raise ValidityError("no pairs to compare")


def confusion_matrix(
    reference: Sequence[int], predicted: Sequence[int], *, n_bands: int = N_BANDS
) -> list[list[int]]:
    """Counts of (reference, predicted) band pairs. Rows are reference bands.

    Printed in the report because a kappa is one number over a 5x5 table, and
    which cells are populated is usually more informative than the number: a
    grader that is uniformly one band low and one that is randomly scattered can
    produce the same kappa and need completely different fixes.
    """
    _checked_pairs(reference, predicted)
    out = [[0] * n_bands for _ in range(n_bands)]
    for ref, pred in zip(reference, predicted, strict=True):
        if not 0 <= ref < n_bands or not 0 <= pred < n_bands:
            raise ValidityError(f"band out of range: reference={ref}, predicted={pred}")
        out[ref][pred] += 1
    return out


def quadratic_weighted_kappa(
    reference: Sequence[int], predicted: Sequence[int], *, n_bands: int = N_BANDS
) -> float:
    """Quadratic Weighted Kappa. Plan section 12.1's primary metric.

    ::

        QWK = 1 - (weighted observed disagreement)
                  / (weighted expected-by-chance disagreement)

    with weights ``w[i][j] = (i - j)^2 / (n_bands - 1)^2``, so being three bands
    out is nine times as bad as being one band out, and agreement you would get
    by luck from the two marginal distributions is divided out.

    Returns 1.0 for perfect agreement, 0.0 for no better than chance, and a
    negative number for worse than chance.

    **The degenerate case is explicit rather than a ZeroDivisionError.** If
    expected disagreement is zero - which happens when both raters put every
    item in one band - kappa is undefined, because there is no chance baseline
    to improve on. This returns ``0.0`` and callers are expected to report the
    sample's distribution alongside the number so a reader can see it happened.
    Silently returning 1.0 there would turn "both raters are constant" into
    "perfect agreement", which is the single most misleading thing this function
    could do.

    **How this is verified, which is not what the plan's study task suggests.**
    That task says "implement it from the definition, then check against
    ``sklearn.metrics.cohen_kappa_score(weights='quadratic')``". This project has
    no numpy/scipy/sklearn dependency and one is not added for a test. Instead
    ``test_validity.py`` checks this against an exact algebraic identity for
    quadratic weights::

        QWK = 1 - n * sum_i (x_i - y_i)^2 / sum_i sum_j (x_i - y_j)^2

    over 500 random and correlated cases, plus values worked out by hand. That
    is a different *formula* rather than a different implementation of this one,
    so an indexing or marginal-ordering error here cannot hide behind it.
    """
    _checked_pairs(reference, predicted)
    observed = confusion_matrix(reference, predicted, n_bands=n_bands)
    total = len(reference)

    ref_counts = [sum(row) for row in observed]
    pred_counts = [sum(observed[i][j] for i in range(n_bands)) for j in range(n_bands)]

    denominator = (n_bands - 1) ** 2
    numerator_sum = 0.0
    expected_sum = 0.0
    for i in range(n_bands):
        for j in range(n_bands):
            weight = ((i - j) ** 2) / denominator
            numerator_sum += weight * observed[i][j]
            # Expected count under independence of the two marginals.
            expected_sum += weight * ref_counts[i] * pred_counts[j] / total

    if expected_sum == 0:
        return 0.0
    return 1.0 - (numerator_sum / expected_sum)


def _ranks(values: Sequence[float]) -> list[float]:
    """Ranks, averaging ties. Ties matter here: a 0-4 scale is almost all ties."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    position = 0
    while position < len(order):
        end = position
        while end + 1 < len(order) and values[order[end + 1]] == values[order[position]]:
            end += 1
        average = (position + end) / 2 + 1
        for index in range(position, end + 1):
            ranks[order[index]] = average
        position = end + 1
    return ranks


def spearman_rho(a: Sequence[float], b: Sequence[float]) -> float:
    """Spearman rank correlation. Plan section 12.1's secondary metric.

    *"Does it order candidates the same way?"* - Pearson correlation computed on
    ranks rather than values, so it measures monotone agreement and is immune to
    the two scales being differently shaped.

    Tie-corrected: computed as Pearson on average ranks rather than with the
    ``1 - 6*sum(d^2)/(n^3-n)`` shortcut, which is only valid without ties. A 0-4
    ordinal scale over 120 items is mostly ties, so the shortcut would be wrong
    here by a wide margin.

    Returns 0.0 when either input is constant: with no variance there is no
    ordering to agree about, and the correlation is undefined rather than zero.
    The caller reports n and the distributions.
    """
    _checked_pairs(a, b)
    if len(a) < 2:
        raise ValidityError("rank correlation needs at least two pairs")
    rank_a, rank_b = _ranks(a), _ranks(b)
    mean_a = sum(rank_a) / len(rank_a)
    mean_b = sum(rank_b) / len(rank_b)
    cov = sum((x - mean_a) * (y - mean_b) for x, y in zip(rank_a, rank_b, strict=True))
    var_a = sum((x - mean_a) ** 2 for x in rank_a)
    var_b = sum((y - mean_b) ** 2 for y in rank_b)
    if var_a == 0 or var_b == 0:
        return 0.0
    return cov / math.sqrt(var_a * var_b)


def mean_absolute_error(a: Sequence[float], b: Sequence[float]) -> float:
    """Average absolute difference. Plan section 12.1: "MAE in score units".

    The most directly interpretable of the three: "on average the grader is this
    many points away from the reference label". Unlike QWK it has no chance
    correction, so it is not comparable across datasets with different
    distributions - but it is the number a reader understands without a
    tutorial.
    """
    _checked_pairs(a, b)
    return sum(abs(x - y) for x, y in zip(a, b, strict=True)) / len(a)


def mean_signed_error(a: Sequence[float], b: Sequence[float]) -> float:
    """Average signed difference, ``b - a``. Positive means ``b`` scores higher.

    Reported next to the MAE because they answer different questions. A grader
    that is uniformly half a point generous has a correctable calibration bias;
    one that is randomly half a point out does not. The MAE is identical in both
    cases and this number tells them apart.
    """
    _checked_pairs(a, b)
    return sum(y - x for x, y in zip(a, b, strict=True)) / len(a)


def exact_and_adjacent(a: Sequence[int], b: Sequence[int]) -> tuple[int, int]:
    """How many pairs agree exactly, and how many within one band."""
    _checked_pairs(a, b)
    exact = sum(1 for x, y in zip(a, b, strict=True) if x == y)
    adjacent = sum(1 for x, y in zip(a, b, strict=True) if abs(x - y) <= 1)
    return exact, adjacent


# ---------------------------------------------------------------------------
# Reliability
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RetestReliability:
    """Test-retest over repeated grading runs. Plan section 12.1: target sigma < 0.04."""

    #: How many answers were graded repeatedly.
    n_items: int
    #: How many times each was graded.
    n_runs: int
    #: Mean of the per-item standard deviations, in [0, 1] score units. This is
    #: the number the plan's "report sigma. Target sigma < 0.04" refers to.
    mean_sigma: float
    #: The worst single item, because a mean sigma of 0.01 with one item at 0.3
    #: is not a stable grader.
    max_sigma: float
    #: Items whose repeated scores were not all identical.
    n_items_varying: int
    #: Largest spread (max - min) seen on any one item, in score units.
    max_range: float

    @property
    def fully_deterministic(self) -> bool:
        return self.n_items_varying == 0


def retest_reliability(runs: Sequence[Sequence[float]]) -> RetestReliability:
    """Per-item variability across repeated grading runs of the same answers.

    ``runs`` is one sequence per run, each holding the scores for the same items
    in the same order.

    **Population standard deviation, not sample.** These runs are the entire set
    of observations for that item, not a sample drawn from a larger pool of runs,
    and the plan's target is a descriptive spread. Using the sample (n-1) form
    would inflate sigma by about 12% at n=5 for no reason a reader could follow.
    """
    if len(runs) < 2:
        raise ValidityError("test-retest needs at least two runs")
    width = len(runs[0])
    if any(len(run) != width for run in runs):
        raise ValidityError("every run must score the same number of items")
    if width == 0:
        raise ValidityError("no items to measure")

    sigmas: list[float] = []
    ranges: list[float] = []
    varying = 0
    for index in range(width):
        values = [run[index] for run in runs]
        mean = sum(values) / len(values)
        sigma = math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))
        sigmas.append(sigma)
        ranges.append(max(values) - min(values))
        if max(values) != min(values):
            varying += 1

    return RetestReliability(
        n_items=width,
        n_runs=len(runs),
        mean_sigma=sum(sigmas) / len(sigmas),
        max_sigma=max(sigmas),
        n_items_varying=varying,
        max_range=max(ranges),
    )


# ---------------------------------------------------------------------------
# Agreement report
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AgreementReport:
    """One complete comparison of two scorings of the same items.

    Carries its own ``n`` and both marginal distributions, because every statistic
    in here is uninterpretable without them - a QWK of 0.2 over a distribution
    with four populated bands and a QWK of 0.2 over one with two mean different
    things, and a report that printed only the kappa would hide that.
    """

    label: str
    n: int
    qwk: float
    spearman: float
    mae: float
    signed_error: float
    exact: int
    adjacent: int
    reference_distribution: dict[int, int]
    predicted_distribution: dict[int, int]
    confusion: list[list[int]]

    @property
    def exact_rate(self) -> float:
        return self.exact / self.n if self.n else 0.0

    @property
    def adjacent_rate(self) -> float:
        return self.adjacent / self.n if self.n else 0.0


def _distribution(values: Sequence[int], *, n_bands: int = N_BANDS) -> dict[int, int]:
    counts = dict.fromkeys(range(n_bands), 0)
    for value in values:
        counts[value] += 1
    return counts


def agreement(
    reference: Sequence[int],
    predicted: Sequence[int],
    *,
    label: str,
    n_bands: int = N_BANDS,
) -> AgreementReport:
    """Every paired statistic at once, so a report cannot quote one and omit n."""
    _checked_pairs(reference, predicted)
    exact, adjacent = exact_and_adjacent(reference, predicted)
    return AgreementReport(
        label=label,
        n=len(reference),
        qwk=quadratic_weighted_kappa(reference, predicted, n_bands=n_bands),
        spearman=spearman_rho(reference, predicted),
        mae=mean_absolute_error(reference, predicted),
        signed_error=mean_signed_error(reference, predicted),
        exact=exact,
        adjacent=adjacent,
        reference_distribution=_distribution(reference, n_bands=n_bands),
        predicted_distribution=_distribution(predicted, n_bands=n_bands),
        confusion=confusion_matrix(reference, predicted, n_bands=n_bands),
    )


__all__ = [
    "MAX_BAND",
    "MIN_BAND",
    "N_BANDS",
    "AgreementReport",
    "RetestReliability",
    "ValidityError",
    "agreement",
    "confusion_matrix",
    "exact_and_adjacent",
    "mean_absolute_error",
    "mean_signed_error",
    "quadratic_weighted_kappa",
    "score_to_band",
    "spearman_rho",
    "retest_reliability",
]
