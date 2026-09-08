"""Day 14's figure set: which charts get drawn, and what each one is for.

One module between the analysis and the renderer, so that
:mod:`app.simulation.convergence` never knows a colour and
:mod:`app.simulation.charts` never knows what a policy is.  This is the only
file that decides *which* pictures Day 14 produces.

The set is deliberately short.  Seven figures plus one per representative
candidate, and every one answers a question the brief asked; a chart that only
restates a number already in a table was not added.  Two candidates for
inclusion were considered and dropped, which is worth recording because their
absence looks like an oversight otherwise:

* **A curve of worst-topic RD.**  It is ``inf`` for most of a short session -
  Day 13 established that no session reaches RD <= 0.40 within 20 items and only
  two do within 60 - so the curve would be an unplottable infinity followed by a
  flat line well above every threshold.  The fact is reported as a number.
* **A scatter of coverage against final MAE.**  Six hundred points in six
  vertical stripes (subtopics measured is an integer from 1 to 6) is a
  strip plot pretending to be a scatter.  The binned table in
  :func:`app.simulation.convergence.coverage_vs_error` says the same thing
  legibly.

Every figure carries its own numbers as a table.  That is not redundancy: it is
the accessible reading of a chart whose identity channel is colour, and it is
what makes a figure checkable rather than merely decorative.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from app.selection.stopping import SMALL_DELTA_THETA
from app.simulation.charts import (
    REFERENCE_SLOT,
    ColumnChart,
    ColumnPanel,
    LineChart,
    Panel,
    Series,
    Stack,
    render_column_chart,
    render_line_chart,
)
from app.simulation.convergence import (
    REPRESENTATIVE_QUANTILES,
    at_or_below_curve,
    coverage_band,
    curve_length,
    mae_band,
    reach_curve,
    representative_candidates,
    rmse_band,
    session_for,
    stability,
    stopping_by_items,
    theta_series,
)
from app.simulation.metrics import default_steps, forward_filled
from app.simulation.runner import ExperimentResult, SessionResult
from app.simulation.strategies import ADAPTIVE, STRATEGY_ORDER

#: ``policy -> categorical slot``.  Fixed for the whole project: adaptive is
#: always blue, random always orange, fixed always aqua, in every figure and
#: whichever subset a figure shows.  Colour follows the entity, never its rank.
POLICY_SLOTS: Mapping[str, int] = {name: index + 1 for index, name in enumerate(STRATEGY_ORDER)}

#: Stopping reasons get their own slots, starting past the policy hues so that a
#: reader flipping between figures never sees "adaptive blue" mean something
#: else.  Sorted by name so the assignment does not depend on which reasons a
#: particular run happened to produce.
STOP_REASON_SLOTS: Mapping[str, int] = {
    "item_budget_reached": 4,
    "no_new_information": 5,
    "precision_reached": 6,
    "time_budget_reached": 7,
    "pool_exhausted": 8,
}

#: The caveat that travels with every figure.  Kept short because it shares one
#: line with the run's parameters and a subtitle that overruns the chart is
#: worse than a terse one; the full statement is in `docs/simulation.md`.
SYNTHETIC_NOTE = "Synthetic environment — see docs/simulation.md."


@dataclass(frozen=True, slots=True)
class Figure:
    """One chart, its caption, and the numbers behind it.

    ``name`` is the file stem, so a figure's identity is stable across runs and
    a regenerated report updates in place rather than accumulating files.
    """

    name: str
    title: str
    caption: str
    svg: str
    #: Header row first, then data rows.  Rendered as the figure's table view.
    table: tuple[tuple[str, ...], ...]

    @property
    def filename(self) -> str:
        return f"{self.name}.svg"


def _slot(strategy: str) -> int:
    try:
        return POLICY_SLOTS[strategy]
    except KeyError as exc:  # pragma: no cover - a strategy outside STRATEGY_ORDER
        raise KeyError(f"no colour slot for strategy {strategy!r}") from exc


def _steps(length: int) -> tuple[int, ...]:
    """Which item counts the figure tables print, clamped to the curve."""
    return tuple(step for step in default_steps(length - 1) if step < length)


def _curve_table(
    curves: Mapping[str, Sequence[float]],
    steps: Sequence[int],
    fmt: str,
) -> tuple[tuple[str, ...], ...]:
    rows = [("policy", *(f"@{step}" for step in steps))]
    for strategy in STRATEGY_ORDER:
        if strategy not in curves:  # pragma: no cover - all three are always present
            continue
        values = curves[strategy]
        rows.append((strategy, *(fmt.format(values[step]) for step in steps)))
    return tuple(rows)


def _by_strategy(result: ExperimentResult) -> dict[str, list[SessionResult]]:
    return {strategy: result.sessions_for(strategy) for strategy in STRATEGY_ORDER}


# ---------------------------------------------------------------------------
# 1-2. error against questions asked
# ---------------------------------------------------------------------------


def error_figure(
    result: ExperimentResult,
    *,
    metric: str = "mae",
) -> Figure:
    """Mean error after N questions, all three policies on one set of axes.

    The headline convergence chart.  Means rather than medians here because the
    mean is the statistic Day 13's tables report and the two must agree; the
    spread lives in :func:`spread_figure`, which is the next figure precisely
    so the mean is never read alone.
    """
    if metric not in ("mae", "rmse"):
        raise ValueError(f"metric must be 'mae' or 'rmse', got {metric!r}")
    sessions = _by_strategy(result)
    length = curve_length(result.sessions)
    build = mae_band if metric == "mae" else rmse_band
    bands = {name: build(group, length=length) for name, group in sessions.items()}
    label = metric.upper()

    panel = Panel(
        x=tuple(float(n) for n in range(length)),
        series=tuple(
            Series(
                label=name,
                values=bands[name].mean,
                slot=_slot(name),
                end_label=f"{bands[name].mean[-1]:.3f}",
            )
            for name in STRATEGY_ORDER
        ),
    )
    reference = (
        (result.config.convergence_mae, f"{label} = {result.config.convergence_mae}")
        if metric == "mae"
        else None
    )
    chart = LineChart(
        title=f"Mean {label} against questions asked",
        subtitle=(
            f"{result.config.candidate_count} synthetic candidates, "
            f"{result.config.item_budget}-item budget. {SYNTHETIC_NOTE}"
        ),
        panels=(panel,),
        x_label="questions asked",
        y_label=f"{label} across all {len(result.config.subtopics)} subtopics (θ units)",
        y_min=0.0,
        reference=reference,
        x_tick_step=max(1, (length - 1) // 10),
    )
    caption = (
        f"Population-mean {label} over every blueprint subtopic, with unmeasured "
        "subtopics left at the cold-start prior θ = 0 — the number a report would "
        "actually print. Sessions that stopped early are forward-filled at their "
        "last estimate rather than dropped, so the tail is not biased towards "
        "whichever policy runs longest."
    )
    return Figure(
        name=f"{metric}-vs-questions",
        title=chart.title,
        caption=caption,
        svg=render_line_chart(chart),
        table=_curve_table(
            {name: band.mean for name, band in bands.items()},
            _steps(length),
            "{:.3f}",
        ),
    )


def spread_figure(result: ExperimentResult) -> Figure:
    """Median MAE with the interquartile range, one panel per policy.

    Small multiples rather than three shaded bands on one set of axes: with
    three overlapping washes the reader cannot tell which band an area belongs
    to, and the guidance is explicit that converging series get faceted rather
    than layered.  The panels share a scale, which is the only way the shapes
    are comparable.

    The band is where the **middle half of candidates** sit.  It is not a
    confidence interval on the mean and is much wider than one - Day 13 measured
    the between-candidate spread at three times the gap between the policies,
    which is the single most important thing to hold in mind when reading the
    figure before it.
    """
    sessions = _by_strategy(result)
    length = curve_length(result.sessions)
    bands = {name: mae_band(group, length=length) for name, group in sessions.items()}
    x = tuple(float(n) for n in range(length))

    panels = tuple(
        Panel(
            title=name,
            x=x,
            series=(
                Series(
                    label=name,
                    values=bands[name].median,
                    slot=_slot(name),
                    band=(bands[name].p25, bands[name].p75),
                ),
            ),
        )
        for name in STRATEGY_ORDER
    )
    chart = LineChart(
        title="MAE spread across candidates",
        subtitle=f"Median with the interquartile range. {SYNTHETIC_NOTE}",
        panels=panels,
        columns=3,
        panel_width=250.0,
        panel_height=230.0,
        x_label="questions asked",
        y_label="MAE (θ units)",
        y_min=0.0,
        reference=(result.config.convergence_mae, f"MAE = {result.config.convergence_mae}"),
        x_tick_step=max(1, (length - 1) // 4),
        legend=tuple(
            Series(label=name, values=(0.0,), slot=_slot(name)) for name in STRATEGY_ORDER
        ),
    )
    rows: list[tuple[str, ...]] = [("policy", "step", "p25", "median", "p75")]
    for name in STRATEGY_ORDER:
        band = bands[name]
        for step in _steps(length):
            rows.append(
                (
                    name,
                    str(step),
                    f"{band.p25[step]:.3f}",
                    f"{band.median[step]:.3f}",
                    f"{band.p75[step]:.3f}",
                )
            )
    return Figure(
        name="mae-spread",
        title=chart.title,
        caption=(
            "The shaded area holds the middle half of the 200 candidates. It is a "
            "population spread, not a confidence interval on the mean: it is several "
            "times wider than the differences between the policies, which is why "
            "none of those differences is called decisive."
        ),
        svg=render_line_chart(chart),
        table=tuple(rows),
    )


# ---------------------------------------------------------------------------
# 3. convergence
# ---------------------------------------------------------------------------


def convergence_figure(result: ExperimentResult) -> Figure:
    """Share of candidates that have reached MAE <= 0.5 by each question count.

    The threshold is Day 13's ``convergence_mae`` and the criterion is Day 13's
    ``first_index_at_or_below``, so the right-hand end of each line equals the
    ``reached`` column of Day 13's convergence table exactly.  No new threshold
    was invented for this figure.
    """
    sessions = _by_strategy(result)
    length = curve_length(result.sessions)
    threshold = result.config.convergence_mae
    reached = {
        name: reach_curve(group, lambda step: step.mae, threshold, length=length)
        for name, group in sessions.items()
    }
    current = {
        name: at_or_below_curve(group, lambda step: step.mae, threshold, length=length)
        for name, group in sessions.items()
    }

    panel = Panel(
        x=tuple(float(n) for n in range(length)),
        series=tuple(
            Series(
                label=name,
                values=reached[name],
                slot=_slot(name),
                end_label=f"{reached[name][-1]:.0%}",
            )
            for name in STRATEGY_ORDER
        ),
    )
    chart = LineChart(
        title=f"Candidates reaching MAE ≤ {threshold}",
        subtitle=f"Cumulative — “has ever been at or below”. {SYNTHETIC_NOTE}",
        panels=(panel,),
        x_label="questions asked",
        y_label="share of candidates",
        y_min=0.0,
        y_max=1.0,
        y_format="{:.0%}",
        x_tick_step=max(1, (length - 1) // 10),
    )
    rows: list[tuple[str, ...]] = [
        ("policy", "reached ever", "at or below at the end", "difference")
    ]
    for name in STRATEGY_ORDER:
        ever, still = reached[name][-1], current[name][-1]
        rows.append((name, f"{ever:.1%}", f"{still:.1%}", f"{still - ever:+.1%}"))
    return Figure(
        name="convergence-reach",
        title=chart.title,
        caption=(
            f"A candidate counts as reached once their MAE has touched {threshold}. "
            "MAE is not monotone, so some of them are back above it when the "
            "interview ends; the table gives both readings and their difference."
        ),
        svg=render_line_chart(chart),
        table=tuple(rows),
    )


# ---------------------------------------------------------------------------
# 4. coverage
# ---------------------------------------------------------------------------


def coverage_figure(result: ExperimentResult) -> Figure:
    """Subtopics touched after N questions.

    Read against the MAE figure: an untouched subtopic contributes its whole
    prior error to the all-subtopic MAE, so breadth and headline accuracy are
    coupled by arithmetic and not only by policy.
    """
    sessions = _by_strategy(result)
    length = curve_length(result.sessions)
    bands = {name: coverage_band(group, length=length) for name, group in sessions.items()}
    total = len(result.config.subtopics)

    panel = Panel(
        x=tuple(float(n) for n in range(length)),
        series=tuple(
            Series(
                label=name,
                values=bands[name].mean,
                slot=_slot(name),
                end_label=f"{bands[name].mean[-1]:.2f}",
            )
            for name in STRATEGY_ORDER
        ),
    )
    chart = LineChart(
        title="Subtopics measured against questions asked",
        subtitle=f"Blueprint has {total} subtopics. {SYNTHETIC_NOTE}",
        panels=(panel,),
        x_label="questions asked",
        y_label="subtopics with at least one item",
        y_min=0.0,
        y_max=float(total),
        y_format="{:.0f}",
        reference=(float(total), f"all {total}"),
        x_tick_step=max(1, (length - 1) // 10),
    )
    return Figure(
        name="coverage-vs-questions",
        title=chart.title,
        caption=(
            "Mean number of the blueprint's subtopics that have received at least "
            "one question. A policy that concentrates measures what it measures "
            "well and leaves the rest at the prior — which the all-subtopic MAE "
            "then charges it for."
        ),
        svg=render_line_chart(chart),
        table=_curve_table(
            {name: band.mean for name, band in bands.items()},
            _steps(length),
            "{:.2f}",
        ),
    )


# ---------------------------------------------------------------------------
# 5. stability
# ---------------------------------------------------------------------------


def stability_figure(result: ExperimentResult) -> Figure:
    """How far θ still moves on each successive answer.

    The reference line is Day 12's own ``SMALL_DELTA_THETA`` (0.05) - the
    threshold its "no new information" stopping rule uses - so the figure shows
    directly how close the average session is to the condition that ends it.
    Nothing here changes that constant; it is drawn, not tuned.
    """
    sessions = _by_strategy(result)
    length = curve_length(result.sessions)
    summaries = {name: stability(group, length=length) for name, group in sessions.items()}

    # Step 0 is the cold start, where no answer exists and the padded value is
    # zero.  Plotting it would draw a spurious spike from 0 up to the first real
    # step, so the curve starts at item 1.
    x = tuple(float(n) for n in range(1, length))
    panel = Panel(
        x=x,
        series=tuple(
            Series(
                label=name,
                values=summaries[name].delta_band.mean[1:],
                slot=_slot(name),
            )
            for name in STRATEGY_ORDER
        ),
    )
    chart = LineChart(
        title="Movement of θ per answer",
        subtitle=f"Mean |Δθ| on the answer at each position. {SYNTHETIC_NOTE}",
        panels=(panel,),
        x_label="question number",
        y_label="mean |Δθ| (θ units)",
        y_min=0.0,
        y_format="{:.2f}",
        reference=(SMALL_DELTA_THETA, f"Day 12 “no new information” = {SMALL_DELTA_THETA}"),
        x_tick_step=max(1, (length - 1) // 10),
    )
    rows: list[tuple[str, ...]] = [
        ("policy", "mean |Δθ|", "mean |Δθ|, last 5", "reversal rate", "MAE rose on")
    ]
    for name in STRATEGY_ORDER:
        summary = summaries[name]
        rows.append(
            (
                name,
                f"{summary.mean_abs_delta:.3f}",
                f"{summary.late_mean_abs_delta:.3f}",
                f"{summary.reversal_rate:.1%}",
                f"{summary.mae_increase_rate:.1%}",
            )
        )
    return Figure(
        name="stability",
        title=chart.title,
        caption=(
            "A falling curve means the estimate is settling. The reversal rate is "
            "the share of consecutive updates *to the same subtopic* whose signs "
            "disagree: 50% is what an estimate sitting on the truth and being "
            "pushed by grading noise produces, and well below 50% means the "
            "estimate is still travelling in one direction."
        ),
        svg=render_line_chart(chart),
        table=tuple(rows),
    )


# ---------------------------------------------------------------------------
# 6. stopping behaviour
# ---------------------------------------------------------------------------


def stopping_figure(result: ExperimentResult) -> Figure:
    """When sessions ended, and why.

    Day 13 reported the reasons as three totals per policy. The same reason at a
    different question number is a different event - ``no_new_information`` at
    item 17 is a test that ran its course; at item 6 it is a test that gave up -
    so the reason is crossed with the item count here.
    """
    sessions = _by_strategy(result)
    profiles = {name: stopping_by_items(group) for name, group in sessions.items()}
    reasons = sorted({reason for profile in profiles.values() for reason in profile.reasons})
    buckets = sorted({items for profile in profiles.values() for items in profile.item_counts})

    panels = tuple(
        ColumnPanel(
            title=name,
            categories=tuple(str(items) for items in buckets),
            stacks=tuple(
                Stack(
                    label=reason,
                    values=tuple(float(profiles[name].count(reason, items)) for items in buckets),
                    slot=STOP_REASON_SLOTS[reason],
                )
                for reason in reasons
            ),
        )
        for name in STRATEGY_ORDER
    )
    chart = ColumnChart(
        title="When sessions stopped, and why",
        subtitle=f"One column per session length. {SYNTHETIC_NOTE}",
        panels=panels,
        columns=1,
        panel_width=660.0,
        panel_height=150.0,
        x_label="questions asked when the session ended",
        y_label="sessions",
        label_every=max(1, len(buckets) // 12),
    )
    rows: list[tuple[str, ...]] = [("policy", "reason", "sessions", "mean questions")]
    for name in STRATEGY_ORDER:
        profile = profiles[name]
        for reason in reasons:
            total = profile.total_for(reason)
            if not total:
                continue
            mean = profile.mean_items_for(reason)
            rows.append((name, reason, str(total), "-" if mean is None else f"{mean:.1f}"))
    return Figure(
        name="stopping-by-question",
        title=chart.title,
        caption=(
            "The primary stopping reason — Day 12 reports every reason that fired "
            "and this counts the first in the plan's order, so the row totals equal "
            "Day 13's stopping table. Column totals per policy are the 200 sessions."
        ),
        svg=render_column_chart(chart),
        table=tuple(rows),
    )


# ---------------------------------------------------------------------------
# 7. one candidate, six subtopics
# ---------------------------------------------------------------------------


def trajectory_figure(
    result: ExperimentResult,
    candidate_id: str,
    *,
    quantile: float | None = None,
    rank_by: str = ADAPTIVE,
) -> Figure:
    """Estimated θ against the hidden truth, one panel per subtopic.

    The ground truth is a dashed line in muted ink, never a series colour, so it
    cannot be mistaken for one of the three things being compared.  A policy's
    line is flat across every turn that asked about a *different* subtopic,
    which is what makes a concentrated session and a broad one look different at
    a glance rather than only in a table.

    Held constant across the three lines of any one panel: the candidate, the
    ground truth, and - by common random numbers - the score any given question
    would have received. What varies is which questions were asked.

    ``quantile`` and ``rank_by`` only label the figure. They record *how* this
    candidate was chosen, because "a representative candidate" is a claim that
    means nothing without the ordering it was representative of.
    """
    sessions = _by_strategy(result)
    length = curve_length(result.sessions)
    picked = {name: session_for(group, candidate_id) for name, group in sessions.items()}
    truth = picked[ADAPTIVE].ground_truth_theta
    x = tuple(float(n) for n in range(length))
    provenance = (
        ""
        if quantile is None
        else (
            f" This candidate sits at the {quantile:.0%} point of the "
            f"{rank_by} policy's final-MAE ordering — representative of *that* "
            "ordering, which is why the same three candidates are shown for "
            "every policy rather than a different “median” candidate each time."
        )
    )

    panels = tuple(
        Panel(
            title=f"{subtopic} — true θ {truth[subtopic]:+.2f}",
            x=x,
            series=(
                *(
                    Series(
                        label=name,
                        values=tuple(forward_filled(theta_series(picked[name], subtopic), length)),
                        slot=_slot(name),
                    )
                    for name in STRATEGY_ORDER
                ),
                Series(
                    label="true θ (hidden from every policy)",
                    values=(truth[subtopic],) * length,
                    slot=REFERENCE_SLOT,
                    dashed=True,
                ),
            ),
        )
        for subtopic in result.config.subtopics
    )
    chart = LineChart(
        title=f"Ability estimate against ground truth — candidate {candidate_id}",
        subtitle=f"One panel per subtopic, shared scale. {SYNTHETIC_NOTE}",
        panels=panels,
        columns=3,
        panel_width=250.0,
        panel_height=180.0,
        x_label="questions asked",
        y_label="θ",
        y_format="{:.1f}",
        x_tick_step=max(1, (length - 1) // 4),
    )
    rows: list[tuple[str, ...]] = [
        ("subtopic", "true θ", *(f"{name} final θ̂" for name in STRATEGY_ORDER))
    ]
    for subtopic in result.config.subtopics:
        rows.append(
            (
                subtopic,
                f"{truth[subtopic]:+.3f}",
                *(f"{picked[name].final_theta[subtopic]:+.3f}" for name in STRATEGY_ORDER),
            )
        )
    return Figure(
        name=f"trajectory-{candidate_id}",
        title=chart.title,
        caption=(
            "The dashed line is the hidden ground truth, which no policy can read: "
            "it enters the simulation only through the graded score. A flat stretch "
            "is a run of turns spent on other subtopics. A panel where every line "
            "stays at 0 is a subtopic nobody asked about." + provenance
        ),
        svg=render_line_chart(chart),
        table=tuple(rows),
    )


# ---------------------------------------------------------------------------
# the set
# ---------------------------------------------------------------------------


def build_figures(
    result: ExperimentResult,
    *,
    rank_by: str = ADAPTIVE,
) -> tuple[Figure, ...]:
    """Every Day 14 figure, in report order.

    ``rank_by`` names the policy whose final MAE orders the candidates for the
    trajectory figures.  It defaults to the policy under test and is exposed so
    that the choice is visible rather than buried - the representative
    candidates are representative *of that ordering*, and the figure captions
    say so.
    """
    representatives = representative_candidates(result.sessions_for(rank_by))
    return (
        error_figure(result, metric="mae"),
        spread_figure(result),
        error_figure(result, metric="rmse"),
        convergence_figure(result),
        coverage_figure(result),
        stability_figure(result),
        stopping_figure(result),
        *(
            trajectory_figure(result, candidate, quantile=quantile, rank_by=rank_by)
            for candidate, quantile in zip(representatives, REPRESENTATIVE_QUANTILES, strict=False)
        ),
    )


def write_figures(figures: Sequence[Figure], directory: Path) -> tuple[Path, ...]:
    """Write each figure's SVG into ``directory``, returning the paths.

    UTF-8 with explicit ``\\n`` line endings: the files are committed, and a
    platform-dependent newline would make the same run produce a different file
    on Windows and on CI.
    """
    directory.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for figure in figures:
        path = directory / figure.filename
        path.write_text(figure.svg, encoding="utf-8", newline="\n")
        written.append(path)
    return tuple(written)


__all__ = [
    "POLICY_SLOTS",
    "STOP_REASON_SLOTS",
    "SYNTHETIC_NOTE",
    "Figure",
    "build_figures",
    "convergence_figure",
    "coverage_figure",
    "error_figure",
    "spread_figure",
    "stability_figure",
    "stopping_figure",
    "trajectory_figure",
    "write_figures",
]
