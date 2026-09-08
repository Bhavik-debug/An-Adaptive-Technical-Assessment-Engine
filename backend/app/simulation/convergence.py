"""Day 14: how fast, and how reliably, does the estimate reach the truth?

Plan section 8.6's deferred half.  Day 13 asked *where the error ends* - one MAE
per policy, plus eight sampled points of a curve.  This module asks *how it got
there*, which is a different question and occasionally has a different answer: a
policy can finish at the same error having taken a smooth path or a jagged one,
and only one of those is a measurement you would show a candidate.

Everything here is a **pure function over Day 13's ``SessionResult`` values**.
No environment, no policy, no randomness, no ground truth beyond what the
simulator already wrote into the record after each session ended.  That
separation is the same one ``metrics.py`` draws and for the same reason:
aggregation is where the temptation to flatter a result lives.

The six questions
-----------------

===========================  ===================================================
question                     what answers it
===========================  ===================================================
how wrong, turn by turn?     :func:`error_band` over ``TrajectoryStep.mae``
                             and ``.rmse``
how many candidates are      :func:`reach_curve` - and its non-cumulative
close enough yet?            sibling :func:`at_or_below_curve`
how much of the blueprint    :func:`coverage_band`
has been touched?
does the estimate settle,    :func:`stability`
or oscillate?
why do sessions end, and     :func:`stopping_by_items`
when?
does breadth cost accuracy?  :func:`coverage_vs_error`
===========================  ===================================================

Two definitions, reused rather than reinvented
----------------------------------------------

**Convergence** is Day 13's, unchanged: the first item count at which a
trajectory is at or below a threshold, via
:func:`app.simulation.metrics.first_index_at_or_below`, with the thresholds
``ExperimentConfig.convergence_mae`` (0.5) and ``convergence_se`` (0.35).  No
new threshold is introduced anywhere in this module.

**Forward-filling** is Day 13's too: a session that stopped after 11 items has
no 12th estimate, and the correct reading of "what would the report say after
15?" is "the same as after 11".  Truncating instead would quietly drop the
early-stopping sessions from the tail of every average - which biases every
curve towards whichever policy runs longest, and would matter here precisely
because the three policies do *not* run equally long (Day 13: 18.9, 19.0, 16.6
items).

The one place this module is deliberately redundant
---------------------------------------------------

:func:`reach_curve` is cumulative - "has this session *ever* been at or below
the threshold by item n" - because that is what
:attr:`SessionResult.items_to_mae` means, and a chart drawn from a different
definition than the table beside it is a trap.  But MAE is **not monotone**: an
estimate can cross 0.5 and cross back.  So :func:`at_or_below_curve` reports the
instantaneous fraction as well, and the gap between the two at the final item is
a direct measurement of how much convergence is being *undone*.  Reporting only
the cumulative curve would hide that; reporting only the instantaneous one would
disagree with Day 13.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from app.simulation.metrics import first_index_at_or_below, forward_filled
from app.simulation.runner import SessionResult, TrajectoryStep

#: A per-step scalar read off one observation record.
StepValue = Callable[[TrajectoryStep], float]


def _percentile(sorted_values: Sequence[float], q: float) -> float:
    """The ``q``-th percentile of an already-sorted sample, linearly interpolated.

    The "inclusive" definition (numpy's default, and
    ``statistics.quantiles(method="inclusive")``): position ``q * (n - 1)``,
    interpolating between the two neighbouring order statistics.  Written out
    rather than called from ``statistics`` because it is needed at three
    arbitrary probabilities per curve point and the stdlib helper computes a
    whole quantile vector to answer one of them.
    """
    if not sorted_values:
        raise ValueError("cannot take a percentile of an empty sample")
    if not 0.0 <= q <= 1.0:
        raise ValueError(f"q must be in [0, 1], got {q!r}")
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = q * (len(sorted_values) - 1)
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return sorted_values[low]
    weight = position - low
    return sorted_values[low] * (1.0 - weight) + sorted_values[high] * weight


# ---------------------------------------------------------------------------
# 1. curves with spread
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Band:
    """One measurement's population distribution at each question count.

    Six parallel tuples rather than one mean, because "the mean error falls" and
    "most candidates' error falls" are different claims and Day 13 already
    showed the spread across candidates (sd 0.21-0.25) to be several times the
    gap between the policies (0.07).  A chart of means alone over differences
    that small would be misleading by omission.

    Every tuple has ``length`` entries, indexed by items asked, so index 0 is
    the cold start.
    """

    n: int
    length: int
    mean: tuple[float, ...]
    median: tuple[float, ...]
    p25: tuple[float, ...]
    p75: tuple[float, ...]
    minimum: tuple[float, ...]
    maximum: tuple[float, ...]

    def at(self, index: int) -> float:
        """The mean at ``index`` items, clamped to the last recorded value."""
        if self.length == 0:
            raise IndexError("an empty band has no values")
        return self.mean[min(max(index, 0), self.length - 1)]


def curve_length(sessions: Sequence[SessionResult]) -> int:
    """The longest trajectory in the sample, in steps.

    All curves for one experiment are drawn to this length so that the policies'
    x-axes line up, and every shorter session is forward-filled into it.
    """
    if not sessions:
        raise ValueError("cannot measure an empty set of sessions")
    return max(len(session.trajectory) for session in sessions)


def band(
    sessions: Sequence[SessionResult],
    value: StepValue,
    *,
    length: int | None = None,
) -> Band:
    """Aggregate one per-step scalar across a population of sessions.

    ``value`` is applied to every :class:`TrajectoryStep`, the resulting
    trajectories are forward-filled to a common ``length`` (Day 13's rule), and
    each column is summarised.

    Non-finite values are rejected rather than skipped.  The one per-step
    quantity that is legitimately infinite - ``worst_topic_rd`` while a topic is
    untouched - has no meaningful mean, so a curve of it would be ``inf``
    everywhere it mattered and 0.4-ish everywhere it did not; see
    :func:`reach_curve`, which is how that quantity is reported instead.
    """
    if not sessions:
        raise ValueError("cannot build a band from an empty set of sessions")
    width = curve_length(sessions) if length is None else length
    if width <= 0:
        raise ValueError(f"length must be positive, got {width!r}")

    filled = [
        forward_filled([value(step) for step in session.trajectory], width) for session in sessions
    ]
    for trajectory in filled:
        for entry in trajectory:
            if not math.isfinite(entry):
                raise ValueError(f"cannot aggregate a non-finite trajectory value: {entry!r}")

    mean: list[float] = []
    median: list[float] = []
    p25: list[float] = []
    p75: list[float] = []
    minimum: list[float] = []
    maximum: list[float] = []
    for index in range(width):
        column = sorted(trajectory[index] for trajectory in filled)
        mean.append(statistics.fmean(column))
        median.append(_percentile(column, 0.50))
        p25.append(_percentile(column, 0.25))
        p75.append(_percentile(column, 0.75))
        minimum.append(column[0])
        maximum.append(column[-1])

    return Band(
        n=len(sessions),
        length=width,
        mean=tuple(mean),
        median=tuple(median),
        p25=tuple(p25),
        p75=tuple(p75),
        minimum=tuple(minimum),
        maximum=tuple(maximum),
    )


def mae_band(sessions: Sequence[SessionResult], *, length: int | None = None) -> Band:
    """MAE over every blueprint subtopic, turn by turn."""
    return band(sessions, lambda step: step.mae, length=length)


def rmse_band(sessions: Sequence[SessionResult], *, length: int | None = None) -> Band:
    """RMSE turn by turn.  Read beside :func:`mae_band`.

    RMSE pulling away from MAE as a session runs means the remaining error is
    concentrating in a few subtopics rather than spreading thin - which is
    exactly what a policy that stops exploring would produce.
    """
    return band(sessions, lambda step: step.rmse, length=length)


def coverage_band(sessions: Sequence[SessionResult], *, length: int | None = None) -> Band:
    """How many subtopics have been touched, turn by turn.

    Forward-filled like every other curve, so a session that stopped at 16 items
    keeps its 5 subtopics through item 20 rather than vanishing from the
    average - which is the honest reading, since a stopped session does not
    acquire coverage it never had.
    """
    return band(sessions, lambda step: float(step.subtopics_measured), length=length)


# ---------------------------------------------------------------------------
# 2. convergence
# ---------------------------------------------------------------------------


def reach_curve(
    sessions: Sequence[SessionResult],
    value: StepValue,
    threshold: float,
    *,
    length: int | None = None,
) -> tuple[float, ...]:
    """Fraction of sessions that have reached ``<= threshold`` by each item count.

    **Cumulative, and deliberately so.**  This is Day 13's
    :func:`~app.simulation.metrics.first_index_at_or_below` evaluated at every
    n instead of once, so the value at the final item is exactly the
    ``reached`` column of Day 13's convergence table.  A chart that disagreed
    with the table beside it would be worse than no chart.

    A session that stopped early keeps whatever it had reached: forward-filling
    a *cumulative* criterion cannot manufacture a crossing, because the filled
    values repeat the last real one.
    """
    if not sessions:
        raise ValueError("cannot build a reach curve from an empty set of sessions")
    width = curve_length(sessions) if length is None else length
    first = [
        first_index_at_or_below([value(step) for step in session.trajectory], threshold)
        for session in sessions
    ]
    total = len(sessions)
    return tuple(
        sum(1 for index in first if index is not None and index <= n) / total for n in range(width)
    )


def at_or_below_curve(
    sessions: Sequence[SessionResult],
    value: StepValue,
    threshold: float,
    *,
    length: int | None = None,
) -> tuple[float, ...]:
    """Fraction of sessions *currently* at or below ``threshold`` at each item count.

    The non-cumulative reading.  It exists because MAE is not monotone - an
    estimate can cross 0.5 and cross back - so a candidate counted as
    "converged" by :func:`reach_curve` may not be converged when the interview
    actually ends.  The gap between the two curves at the final item is how much
    convergence the session went on to undo, and it is a number this project
    would otherwise never have looked at.
    """
    if not sessions:
        raise ValueError("cannot build a reach curve from an empty set of sessions")
    width = curve_length(sessions) if length is None else length
    filled = [
        forward_filled([value(step) for step in session.trajectory], width) for session in sessions
    ]
    total = len(sessions)
    return tuple(
        sum(1 for trajectory in filled if trajectory[n] <= threshold) / total for n in range(width)
    )


# ---------------------------------------------------------------------------
# 3. stability
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class StabilitySummary:
    """Does the estimate settle, or does it keep jumping around?

    Four numbers and a curve, all read off the ``delta_theta`` and ``mae``
    already on the record.  They answer a question Day 13 could not: two
    policies that finish at the same MAE are not equally good if one arrived
    smoothly and the other is still swinging when the clock stops.

    ``reversal_rate`` is the direct oscillation measure.  It is computed
    **within a subtopic**, because consecutive updates to *different* subtopics
    have no reason to agree in sign and counting those would make every policy
    that alternates subtopics look unstable by construction.
    """

    n_sessions: int
    #: Mean ``|dtheta|`` per answered item, over every session.
    mean_abs_delta: float
    #: The same, restricted to the last :data:`LATE_WINDOW` answers of each
    #: session - "was it still moving when the interview ended?"
    late_mean_abs_delta: float
    #: Of consecutive same-subtopic updates, the fraction whose signs differ.
    #: 0.5 is a coin flip - what an estimate sitting on the truth and being
    #: pushed around by grading noise produces.  Well below 0.5 means the
    #: estimate is still travelling in one direction, i.e. has not arrived.
    reversal_rate: float
    #: Of all turns, the fraction where MAE went *up*.  Some is unavoidable:
    #: an answer that is informative about one subtopic is silent about five.
    mae_increase_rate: float
    #: Mean ``|dtheta|`` after 1, 2, ... items.  Index 0 is the cold start and
    #: is always 0.0 - no answer has been given - which is a padding value, not
    #: a measurement.
    delta_band: Band


#: How many of a session's final answers count as "late".  Chosen to match Day
#: 12's ``SMALL_UPDATE_RUN`` (3) rounded up to the nearest odd window that still
#: fits the shortest sessions Day 13 produces; it is a reporting window, not a
#: threshold, and nothing decides anything on it.
LATE_WINDOW = 5


def stability(
    sessions: Sequence[SessionResult],
    *,
    length: int | None = None,
    late_window: int = LATE_WINDOW,
) -> StabilitySummary:
    """Aggregate the movement of the estimate across a population."""
    if not sessions:
        raise ValueError("cannot measure stability over an empty set of sessions")
    if late_window <= 0:
        raise ValueError(f"late_window must be positive, got {late_window!r}")

    deltas: list[float] = []
    late: list[float] = []
    reversals = 0
    pairs = 0
    increases = 0
    turns = 0

    for session in sessions:
        answered = [step for step in session.trajectory if step.delta_theta is not None]
        session_deltas = [
            abs(step.delta_theta) for step in answered if step.delta_theta is not None
        ]
        deltas.extend(session_deltas)
        late.extend(session_deltas[-late_window:])

        by_subtopic: dict[str, list[float]] = {}
        for step in answered:
            if step.subtopic_key is None or step.delta_theta is None:  # pragma: no cover
                continue
            by_subtopic.setdefault(step.subtopic_key, []).append(step.delta_theta)
        for series in by_subtopic.values():
            for earlier, later in zip(series, series[1:], strict=False):
                # A step of exactly zero has no direction, so it is neither a
                # reversal nor a continuation and does not enter the ratio.
                if earlier == 0.0 or later == 0.0:
                    continue
                pairs += 1
                if (earlier > 0.0) != (later > 0.0):
                    reversals += 1

        maes = [step.mae for step in session.trajectory]
        for earlier_mae, later_mae in zip(maes, maes[1:], strict=False):
            turns += 1
            if later_mae > earlier_mae:
                increases += 1

    return StabilitySummary(
        n_sessions=len(sessions),
        mean_abs_delta=statistics.fmean(deltas) if deltas else 0.0,
        late_mean_abs_delta=statistics.fmean(late) if late else 0.0,
        reversal_rate=reversals / pairs if pairs else 0.0,
        mae_increase_rate=increases / turns if turns else 0.0,
        delta_band=band(
            sessions,
            lambda step: abs(step.delta_theta) if step.delta_theta is not None else 0.0,
            length=length,
        ),
    )


# ---------------------------------------------------------------------------
# 4. stopping behaviour
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class StoppingProfile:
    """Which reason ended a session, tabulated against *when* it ended.

    Day 13 reported the reasons as three totals per policy.  That hides the
    thing worth knowing: ``no_new_information`` firing at item 17 is a test that
    ran its course, and the same reason firing at item 6 is a test that gave up.
    """

    #: Every item count at which at least one session ended, ascending.
    item_counts: tuple[int, ...]
    #: Every reason seen, in a stable order.
    reasons: tuple[str, ...]
    #: ``(reason, items_asked) -> sessions``.
    counts: Mapping[tuple[str, int], int]
    n_sessions: int

    def count(self, reason: str, items_asked: int) -> int:
        return self.counts.get((reason, items_asked), 0)

    def total_for(self, reason: str) -> int:
        return sum(n for (label, _), n in self.counts.items() if label == reason)

    def mean_items_for(self, reason: str) -> float | None:
        """Mean session length among the sessions that ended for this reason."""
        weighted = [
            (items, n) for (label, items), n in sorted(self.counts.items()) if label == reason
        ]
        total = sum(n for _, n in weighted)
        if not total:
            return None
        return sum(items * n for items, n in weighted) / total


def stopping_by_items(sessions: Sequence[SessionResult]) -> StoppingProfile:
    """Cross-tabulate the primary stopping reason against items asked.

    ``primary_stop_reason`` is Day 13's own one-per-session choice - the first
    reason in the plan's order - so this table's row totals equal Day 13's
    stopping table exactly.
    """
    if not sessions:
        raise ValueError("cannot profile stopping over an empty set of sessions")
    counts: dict[tuple[str, int], int] = {}
    for session in sessions:
        key = (session.primary_stop_reason, session.items_asked)
        counts[key] = counts.get(key, 0) + 1
    return StoppingProfile(
        item_counts=tuple(sorted({items for _, items in counts})),
        reasons=tuple(sorted({reason for reason, _ in counts})),
        counts=dict(sorted(counts.items())),
        n_sessions=len(sessions),
    )


# ---------------------------------------------------------------------------
# 5. coverage against accuracy
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CoverageBin:
    """Final error among the sessions that reached a given breadth."""

    subtopics_measured: int
    n_sessions: int
    mean_mae: float
    mean_mae_measured: float | None


def coverage_vs_error(sessions: Sequence[SessionResult]) -> tuple[CoverageBin, ...]:
    """Group sessions by how many subtopics they touched, and average the error.

    This is the arithmetic behind Day 13's central trade-off, made explicit.
    Day 13 observed that adaptive measures fewer subtopics (4.71 of 6) far more
    precisely (MAE(seen) 0.333) and that random's broader 5.84 wins the
    all-subtopic MAE.  Binning shows whether that is a *mechanical* consequence
    of breadth - an untouched subtopic contributes its full prior error to MAE,
    so more coverage must help - or something about the policies.

    **The bins are not a controlled comparison.**  Sessions land in a bin by
    what the policy did, so a bin also selects on candidate difficulty, session
    length and luck.  It describes the association; it does not establish that
    adding a subtopic to a given session would lower its MAE.
    """
    if not sessions:
        raise ValueError("cannot bin an empty set of sessions")
    grouped: dict[int, list[SessionResult]] = {}
    for session in sessions:
        grouped.setdefault(session.subtopics_measured, []).append(session)

    bins: list[CoverageBin] = []
    for measured, group in sorted(grouped.items()):
        seen = [s.mae_measured for s in group if s.mae_measured is not None]
        bins.append(
            CoverageBin(
                subtopics_measured=measured,
                n_sessions=len(group),
                mean_mae=statistics.fmean([s.mae for s in group]),
                mean_mae_measured=statistics.fmean(seen) if seen else None,
            )
        )
    return tuple(bins)


# ---------------------------------------------------------------------------
# 6. picking a candidate to look at
# ---------------------------------------------------------------------------

#: Which candidates the trajectory figures show, as quantiles of final MAE.
#: Three rather than one because a single "typical" trajectory is a claim about
#: the whole population that one line cannot support, and the interesting
#: failures live in the tail.
REPRESENTATIVE_QUANTILES: tuple[float, ...] = (0.10, 0.50, 0.90)


def representative_candidates(
    sessions: Sequence[SessionResult],
    *,
    quantiles: Sequence[float] = REPRESENTATIVE_QUANTILES,
) -> tuple[str, ...]:
    """Candidate ids at given quantiles of final MAE, worst last.

    Deterministic by construction: sessions are ordered by ``(mae,
    candidate_id)`` so ties cannot depend on iteration order, and the quantile
    index is a floor rather than an interpolation because a candidate id cannot
    be interpolated.

    Picking on **one** policy's ordering and then showing all three policies for
    those candidates is deliberate.  Choosing a separate "median" candidate per
    policy would compare three different people, and the point of the figure is
    to hold the person fixed and vary the policy.  The consequence - that the
    candidates are median *for the policy used to rank them* - is stated in the
    figure caption rather than designed away.
    """
    if not sessions:
        raise ValueError("cannot pick a representative from an empty set of sessions")
    ordered = sorted(sessions, key=lambda session: (session.mae, session.candidate_id))
    picked: list[str] = []
    for quantile in quantiles:
        if not 0.0 <= quantile <= 1.0:
            raise ValueError(f"quantile must be in [0, 1], got {quantile!r}")
        index = min(len(ordered) - 1, int(quantile * len(ordered)))
        candidate_id = ordered[index].candidate_id
        if candidate_id not in picked:
            picked.append(candidate_id)
    return tuple(picked)


def theta_series(session: SessionResult, subtopic: str) -> tuple[float, ...]:
    """The engine's estimate of one subtopic after 0, 1, 2, ... items.

    Includes the turns where a *different* subtopic was asked, during which the
    value simply does not move - which is the shape that makes an adaptive
    session visibly different from a broad one on the chart.
    """
    try:
        return tuple(step.theta[subtopic] for step in session.trajectory)
    except KeyError as exc:  # pragma: no cover - a taxonomy/session mismatch
        raise KeyError(f"session recorded no estimate for subtopic {subtopic!r}") from exc


def session_for(sessions: Sequence[SessionResult], candidate_id: str) -> SessionResult:
    """The one session in ``sessions`` belonging to ``candidate_id``."""
    for session in sessions:
        if session.candidate_id == candidate_id:
            return session
    raise KeyError(f"no session for candidate {candidate_id!r}")


__all__ = [
    "LATE_WINDOW",
    "REPRESENTATIVE_QUANTILES",
    "Band",
    "CoverageBin",
    "StabilitySummary",
    "StepValue",
    "StoppingProfile",
    "at_or_below_curve",
    "band",
    "coverage_band",
    "coverage_vs_error",
    "curve_length",
    "mae_band",
    "reach_curve",
    "representative_candidates",
    "rmse_band",
    "session_for",
    "stability",
    "stopping_by_items",
    "theta_series",
]
