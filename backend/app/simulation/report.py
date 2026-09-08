"""Turning an experiment into tables a person reads.

Separate from ``runner.py`` for the same reason ``app/evaluation/report.py`` is
separate from its runner: formatting is where the temptation to flatter a result
lives, and it should be nowhere near the code that computed it.  Nothing here
rounds, filters or reorders in a way that changes what a table says - the
policies print in a fixed order, and a policy that lost prints exactly as it
lost.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from app.simulation.config import ExperimentConfig
from app.simulation.convergence import coverage_vs_error, stability
from app.simulation.figures import Figure
from app.simulation.metrics import default_steps
from app.simulation.runner import AblationResult, ExperimentResult, StrategySummary


def _table(header: Sequence[str], rows: Sequence[Sequence[str]], *, markdown: bool) -> str:
    if markdown:
        # A pipe inside a cell ends the cell, so `|dtheta|` would silently split
        # one column into three and misalign every row after it.  Escaping is
        # done here rather than at each call site because a formatter that
        # corrupts a table on some inputs is a formatter that will do it again.
        def cell(text: str) -> str:
            return text.replace("|", "\\|")

        lines = [
            "| " + " | ".join(cell(h) for h in header) + " |",
            "|" + "|".join(["---"] * len(header)) + "|",
        ]
        lines += ["| " + " | ".join(cell(value) for value in row) + " |" for row in rows]
        return "\n".join(lines)
    widths = [max(len(header[i]), *(len(r[i]) for r in rows)) for i in range(len(header))]
    out = ["  ".join(h.ljust(widths[i]) for i, h in enumerate(header))]
    out.append("  ".join("-" * w for w in widths))
    out += ["  ".join(r[i].ljust(widths[i]) for i in range(len(header))) for r in rows]
    return "\n".join(out)


def config_block(config: ExperimentConfig, *, markdown: bool = False) -> str:
    """Everything needed to reproduce the run, printed above its results."""
    rows = [[label, value] for label, value in config.describe()]
    return _table(["parameter", "value"], rows, markdown=markdown)


def comparison_table(result: ExperimentResult, *, markdown: bool = False) -> str:
    """The headline: one row per policy.

    ``MAE`` is over every blueprint subtopic, unmeasured ones at the cold-start
    prior - the number a report would actually print.  ``MAE(seen)`` is over
    only the subtopics the policy asked about, and must be read next to
    ``subtopics`` because they trade off against each other.
    """
    header = [
        "policy",
        "MAE",
        "RMSE",
        "MAE(seen)",
        "subtopics",
        "items",
        "abs(b-theta)",
        "coverage",
    ]
    rows = [
        [
            s.strategy,
            f"{s.mae.mean:.3f}",
            f"{s.rmse.mean:.3f}",
            f"{s.mae_measured.mean:.3f}",
            f"{s.subtopics_measured.mean:.2f}",
            f"{s.items_asked.mean:.1f}",
            f"{s.difficulty_gap.mean:.3f}",
            f"{s.coverage_compliance:.1%}",
        ]
        for s in result.summaries
    ]
    return _table(header, rows, markdown=markdown)


def spread_table(result: ExperimentResult, *, markdown: bool = False) -> str:
    """Is the difference consistent, or one lucky candidate?  Median and spread."""
    header = ["policy", "MAE mean", "MAE median", "MAE sd", "MAE min", "MAE max"]
    rows = [
        [
            s.strategy,
            f"{s.mae.mean:.3f}",
            f"{s.mae.median:.3f}",
            f"{s.mae.stdev:.3f}",
            f"{s.mae.minimum:.3f}",
            f"{s.mae.maximum:.3f}",
        ]
        for s in result.summaries
    ]
    return _table(header, rows, markdown=markdown)


def convergence_table(result: ExperimentResult, *, markdown: bool = False) -> str:
    """Items to reach each criterion, with the sessions that never did."""
    config = result.config
    header = [
        "policy",
        f"items to MAE<={config.convergence_mae}",
        "reached",
        f"items to SE<={config.convergence_se}",
        "reached",
    ]
    rows = []
    for s in result.summaries:
        mae_c, se_c = s.items_to_mae, s.items_to_se
        rows.append(
            [
                s.strategy,
                "-" if mae_c.reached is None else f"{mae_c.reached.mean:.1f}",
                f"{mae_c.n_reached}/{mae_c.n_total}",
                "-" if se_c.reached is None else f"{se_c.reached.mean:.1f}",
                f"{se_c.n_reached}/{se_c.n_total}",
            ]
        )
    return _table(header, rows, markdown=markdown)


def stopping_table(result: ExperimentResult, *, markdown: bool = False) -> str:
    """Why each policy's sessions ended.  One row per policy, one column per reason."""
    reasons = sorted({reason for s in result.summaries for reason in s.stop_reasons})
    header = ["policy", *reasons]
    rows = [
        [s.strategy, *(str(s.stop_reasons.get(reason, 0)) for reason in reasons)]
        for s in result.summaries
    ]
    return _table(header, rows, markdown=markdown)


def curve_table(
    result: ExperimentResult,
    *,
    steps: Sequence[int] | None = None,
    markdown: bool = False,
) -> str:
    """Population-mean MAE after N items, for a handful of N.

    The numbers behind Day 14's chart, and deliberately only the numbers: this
    is a text table of eight columns, not a plotting system.
    """
    if steps is None:
        steps = default_steps(result.config.item_budget)
    header = ["policy", *(f"@{n}" for n in steps)]
    rows = []
    for s in result.summaries:
        cells = []
        for n in steps:
            index = min(n, len(s.mae_curve) - 1)
            cells.append(f"{s.mae_curve[index]:.3f}")
        rows.append([s.strategy, *cells])
    return _table(header, rows, markdown=markdown)


def ablation_table(
    ablation: AblationResult,
    *,
    baseline: str = "full",
    markdown: bool = False,
) -> str:
    """One row per objective variant, with its delta against the full objective.

    A positive ``dMAE`` means removing that component made the estimate *worse*;
    a negative one means the component was costing accuracy in this environment.
    """
    reference: StrategySummary | None = None
    for entry in ablation.summaries:
        if entry.strategy == baseline:
            reference = entry
    header = [
        "objective",
        "MAE",
        "dMAE",
        "MAE(seen)",
        "subtopics",
        "items",
        "abs(b-theta)",
        "coverage",
    ]
    rows = []
    for s in ablation.summaries:
        delta = "-" if reference is None else f"{s.mae.mean - reference.mae.mean:+.4f}"
        rows.append(
            [
                s.strategy,
                f"{s.mae.mean:.4f}",
                delta,
                f"{s.mae_measured.mean:.4f}",
                f"{s.subtopics_measured.mean:.2f}",
                f"{s.items_asked.mean:.2f}",
                f"{s.difficulty_gap.mean:.3f}",
                f"{s.coverage_compliance:.1%}",
            ]
        )
    return _table(header, rows, markdown=markdown)


def text_report(result: ExperimentResult, ablation: AblationResult | None = None) -> str:
    """The whole thing, for a terminal."""
    blocks = [
        "CONFIGURATION",
        config_block(result.config),
        "",
        "POLICY COMPARISON",
        comparison_table(result),
        "",
        "SPREAD ACROSS CANDIDATES",
        spread_table(result),
        "",
        "CONVERGENCE",
        convergence_table(result),
        "",
        "MEAN MAE AFTER N ITEMS",
        curve_table(result),
        "",
        "WHY SESSIONS STOPPED",
        stopping_table(result),
    ]
    if ablation is not None:
        blocks += ["", "WEIGHT ABLATION (adaptive policy only)", ablation_table(ablation)]
    return "\n".join(blocks)


def markdown_report(
    result: ExperimentResult,
    ablation: AblationResult | None = None,
    *,
    extended: ExperimentResult | None = None,
) -> str:
    """The committed artefact in ``evals/reports/``.

    Numbers only, with the caveats that must travel with them.  Interpretation
    lives in ``docs/simulation.md``; this file is the measurement.
    """
    md = True
    lines = [
        "# Adaptive selection simulation — CAT vs random vs fixed",
        "",
        "*Generated by `python backend/scripts/run_selection_simulation.py --write-report`.*",
        "*Plan §8.6, Day 13. Regenerate rather than edit.*",
        "",
        "**These are results about a synthetic environment.** They say whether the",
        "implementation recovers known abilities and how the policies compare under",
        "the assumptions listed below. They say nothing about real candidates, real",
        "questions, real interviews or hiring validity — see `docs/simulation.md`,",
        '"What this cannot tell us".',
        "",
        "## Configuration",
        "",
        config_block(result.config, markdown=md),
        "",
        "## Policy comparison",
        "",
        comparison_table(result, markdown=md),
        "",
        "`MAE` is over every blueprint subtopic, with unmeasured subtopics left at the",
        "cold-start prior (θ = 0) — the number a report would actually print.",
        "`MAE(seen)` covers only the subtopics a policy asked about, and must be read",
        "beside `subtopics`: measuring two subtopics precisely and ignoring four is not",
        "a better assessment.",
        "",
        "**The two comparisons are not equally clean.** `random` sees the *same*",
        "eligible pool as `adaptive` — the same hard constraints, including the",
        "θ-dependent `|b − θ| ≤ 1.5` window — so that contrast isolates the ranking",
        "objective and nothing else. `fixed` cannot apply that window (it reads the",
        "running estimate, and a policy consulting it would not be fixed), so the",
        "adaptive-vs-fixed contrast bundles the window together with the objective and",
        "ε-greedy: it shows whether an adaptive test beats a non-adaptive one, but",
        "cannot attribute the gap to any single mechanism.",
        "",
        "No policy wins every column. Read the rows together rather than picking one.",
        "",
        "## Spread across candidates",
        "",
        spread_table(result, markdown=md),
        "",
        "## Convergence",
        "",
        convergence_table(result, markdown=md),
        "",
        "## Mean MAE after N items",
        "",
        curve_table(result, markdown=md),
        "",
        "## Why sessions stopped",
        "",
        stopping_table(result, markdown=md),
    ]
    if extended is not None:
        lines += [
            "",
            f"## Extended run — {extended.config.item_budget}-item budget",
            "",
            "The same environment run to a longer budget, because no RD-based",
            "criterion fires within the plan's 20 items.",
            "",
            config_block(extended.config, markdown=md),
            "",
            comparison_table(extended, markdown=md),
            "",
            convergence_table(extended, markdown=md),
            "",
            stopping_table(extended, markdown=md),
        ]
    if ablation is not None:
        lines += [
            "",
            "## Weight ablation",
            "",
            "Adaptive policy only, same environment, same seeds. One weight is set to",
            "zero per row; the others are left alone, because scaling all six by a",
            "constant cannot change an argmax and renormalisation would therefore be a",
            "no-op. `dMAE` is the change against the full objective: positive means",
            "removing the component made the estimate worse.",
            "",
            ablation_table(ablation, markdown=md),
        ]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Day 14 - convergence and its figures
# ---------------------------------------------------------------------------


def coverage_error_table(result: ExperimentResult, *, markdown: bool = False) -> str:
    """Final error binned by how many subtopics a session touched.

    The arithmetic behind Day 13's central trade-off.  **Not a controlled
    comparison**: sessions land in a bin by what the policy did, so a bin also
    selects on candidate, session length and luck.  It describes an
    association and establishes no cause.
    """
    header = ["policy", "subtopics", "sessions", "MAE", "MAE(seen)"]
    rows = []
    for summary in result.summaries:
        for entry in coverage_vs_error(result.sessions_for(summary.strategy)):
            rows.append(
                [
                    summary.strategy,
                    str(entry.subtopics_measured),
                    str(entry.n_sessions),
                    f"{entry.mean_mae:.3f}",
                    "-" if entry.mean_mae_measured is None else f"{entry.mean_mae_measured:.3f}",
                ]
            )
    return _table(header, rows, markdown=markdown)


def stability_table(result: ExperimentResult, *, markdown: bool = False) -> str:
    """Whether the estimate settles, and whether it oscillates while doing so."""
    header = [
        "policy",
        "mean abs(dtheta)",
        "last 5 answers",
        "reversal rate",
        "turns where MAE rose",
    ]
    rows = []
    for summary in result.summaries:
        entry = stability(result.sessions_for(summary.strategy))
        rows.append(
            [
                summary.strategy,
                f"{entry.mean_abs_delta:.3f}",
                f"{entry.late_mean_abs_delta:.3f}",
                f"{entry.reversal_rate:.1%}",
                f"{entry.mae_increase_rate:.1%}",
            ]
        )
    return _table(header, rows, markdown=markdown)


def figure_table(figure: Figure, *, markdown: bool = True) -> str:
    """A figure's own numbers - the table view of a colour-coded chart."""
    if not figure.table:  # pragma: no cover - every figure ships a table
        return ""
    header, *rows = figure.table
    return _table(list(header), [list(row) for row in rows], markdown=markdown)


def convergence_report(
    result: ExperimentResult,
    figures: Sequence[Figure],
    *,
    figure_dir: str = "figures",
) -> str:
    """Day 14's committed artefact: the figures, their numbers, and the caveats.

    Interpretation lives in ``docs/convergence.md``; this file is the
    measurement, regenerated rather than edited.  Every figure appears with the
    table it was drawn from, so nothing here is legible only in colour.
    """
    md = True
    lines = [
        "# Convergence of the ability estimate — adaptive vs random vs fixed",
        "",
        "*Generated by `python backend/scripts/run_convergence_analysis.py --write-report`.*",
        "*Plan §8.6, Day 14. Regenerate rather than edit.*",
        "",
        "**These are results about a synthetic environment.** They describe how the",
        "estimate approaches a ground truth this simulator generated, under the",
        "assumptions listed below. They say nothing about real candidates, real",
        "questions, real interviews or hiring validity — see `docs/simulation.md`,",
        '"What this cannot tell us".',
        "",
        "**No claim of statistical significance is made anywhere in this report.**",
        "No hypothesis test was run. The differences between the policies are small",
        "relative to the spread across candidates, and the spread figure is included",
        "so that this is visible rather than merely stated.",
        "",
        "## Configuration",
        "",
        config_block(result.config, markdown=md),
        "",
        "## Convergence criteria",
        "",
        "Both are Day 13's, unchanged, and no new threshold was introduced:",
        "",
        f"- **MAE ≤ {result.config.convergence_mae}** — a design choice, and an arbitrary",
        "  one; it is reported beside the full error curve so a reader who dislikes it",
        "  can ignore it.",
        f"- **worst-topic RD ≤ {result.config.convergence_se}** — plan §8.6's own criterion,",
        "  applied at topic level through Day 11's `roll_up`.",
        "",
        convergence_table(result, markdown=md),
        "",
    ]
    # A figure and the commentary its subject needs belong on the same screen,
    # so the extra tables are attached to the figure they explain rather than
    # collected in an appendix nobody reaches.
    follow_up: dict[str, list[str]] = {
        "coverage-vs-questions": [
            "### Coverage against final error",
            "",
            coverage_error_table(result, markdown=md),
            "",
            "An untouched subtopic contributes its whole prior error to the all-subtopic",
            "MAE, so breadth and headline accuracy are coupled by arithmetic before any",
            "policy is involved. The bins are **not** a controlled comparison: a session",
            "lands in a bin because of what its policy did, so the bin also selects on",
            "the candidate and on session length. It describes an association and",
            "establishes no cause.",
            "",
        ],
    }
    for index, figure in enumerate(figures, start=1):
        lines += [
            f"## Figure {index} — {figure.title}",
            "",
            f"![{figure.title}]({figure_dir}/{figure.filename})",
            "",
            figure.caption,
            "",
            figure_table(figure, markdown=md),
            "",
            *follow_up.get(figure.name, ()),
        ]
    return "\n".join(lines) + "\n"


def format_seconds(seconds: float) -> str:
    """``93.4 s`` or ``1 m 33 s`` - for the script's timing line."""
    if not math.isfinite(seconds) or seconds < 60.0:
        return f"{seconds:.1f} s"
    return f"{int(seconds // 60)} m {int(seconds % 60)} s"


__all__ = [
    "ablation_table",
    "comparison_table",
    "config_block",
    "convergence_report",
    "convergence_table",
    "coverage_error_table",
    "curve_table",
    "figure_table",
    "format_seconds",
    "markdown_report",
    "spread_table",
    "stability_table",
    "stopping_table",
    "text_report",
]
