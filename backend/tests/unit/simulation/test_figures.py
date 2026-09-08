"""The figure set, and the Day 14 report that carries it.

The question this file answers is not "does it draw" but **"does the picture
say what the table says"**. A figure whose caption, chart and numbers can drift
apart is worse than no figure, because it is trusted. So every figure's table is
checked against the analysis it was built from, and the report is checked for the
caveats that must travel with a synthetic result.

There is no test that adaptive beats anything. That remains an empirical
hypothesis about a synthetic world.
"""

from __future__ import annotations

import re

import pytest

from app.selection.stopping import SMALL_DELTA_THETA
from app.simulation.convergence import (
    coverage_band,
    coverage_vs_error,
    mae_band,
    reach_curve,
    representative_candidates,
    rmse_band,
    session_for,
    stability,
    stopping_by_items,
)
from app.simulation.environment import build_environment
from app.simulation.figures import (
    POLICY_SLOTS,
    STOP_REASON_SLOTS,
    Figure,
    build_figures,
    convergence_figure,
    coverage_figure,
    error_figure,
    spread_figure,
    stability_figure,
    stopping_figure,
    trajectory_figure,
    write_figures,
)
from app.simulation.report import (
    convergence_report,
    coverage_error_table,
    figure_table,
    stability_table,
)
from app.simulation.runner import run_experiment
from app.simulation.strategies import ADAPTIVE, FIXED, RANDOM, STRATEGY_ORDER
from tests.unit.simulation.conftest import TINY

ENV = build_environment(TINY)
RESULT = run_experiment(TINY, environment=ENV)
FIGURES = build_figures(RESULT)
BY_NAME = {figure.name: figure for figure in FIGURES}


def rows(figure: Figure) -> list[tuple[str, ...]]:
    """A figure's table without its header."""
    return list(figure.table[1:])


def cells(figure: Figure, first: str) -> tuple[str, ...]:
    for row in rows(figure):
        if row[0] == first:
            return row
    raise AssertionError(f"no row starting {first!r} in {figure.name}")


class TestTheSet:
    def test_every_required_figure_is_present(self):
        """The brief names five minimum charts; these are them, plus the three
        the analysis turned out to need."""
        required = {
            "mae-vs-questions",
            "rmse-vs-questions",
            "convergence-reach",
            "coverage-vs-questions",
            "mae-spread",
            "stability",
            "stopping-by-question",
        }
        assert required <= set(BY_NAME)
        assert sum(name.startswith("trajectory-") for name in BY_NAME) >= 1

    def test_every_figure_is_complete(self):
        for figure in FIGURES:
            assert figure.svg.startswith("<svg")
            assert figure.svg.endswith("</svg>\n")
            assert figure.title
            assert len(figure.caption) > 40
            assert len(figure.table) >= 2, figure.name
            assert figure.filename == f"{figure.name}.svg"

    def test_no_two_figures_share_a_filename(self):
        """A collision would silently overwrite one chart with another."""
        names = [figure.name for figure in FIGURES]
        assert len(names) == len(set(names))

    def test_every_figure_carries_the_synthetic_caveat(self):
        for figure in FIGURES:
            assert "Synthetic environment" in figure.svg

    def test_a_policy_keeps_its_colour_in_every_chart(self):
        assert POLICY_SLOTS == {ADAPTIVE: 1, RANDOM: 2, FIXED: 3}
        assert len(set(POLICY_SLOTS.values())) == len(POLICY_SLOTS)

    def test_stopping_reasons_never_borrow_a_policy_colour(self):
        """Otherwise "adaptive blue" would mean something else two figures on."""
        assert set(STOP_REASON_SLOTS.values()).isdisjoint(POLICY_SLOTS.values())
        assert len(set(STOP_REASON_SLOTS.values())) == len(STOP_REASON_SLOTS)

    def test_every_day_12_stopping_reason_has_a_colour(self):
        from app.selection.stopping import (
            STOP_ITEM_BUDGET,
            STOP_NO_NEW_INFORMATION,
            STOP_PRECISION,
            STOP_TIME_BUDGET,
        )
        from app.simulation.runner import STOP_POOL_EXHAUSTED

        for reason in (
            STOP_PRECISION,
            STOP_ITEM_BUDGET,
            STOP_TIME_BUDGET,
            STOP_NO_NEW_INFORMATION,
            STOP_POOL_EXHAUSTED,
        ):
            assert reason in STOP_REASON_SLOTS


class TestTablesMatchTheAnalysis:
    """The table under a chart must be the numbers the chart was drawn from."""

    def test_the_mae_table_is_the_mae_band(self):
        figure = BY_NAME["mae-vs-questions"]
        steps = [int(header.lstrip("@")) for header in figure.table[0][1:]]
        for name in STRATEGY_ORDER:
            curve = mae_band(RESULT.sessions_for(name), length=max(steps) + 1).mean
            assert cells(figure, name)[1:] == tuple(f"{curve[step]:.3f}" for step in steps)

    def test_the_rmse_table_is_the_rmse_band(self):
        figure = BY_NAME["rmse-vs-questions"]
        steps = [int(header.lstrip("@")) for header in figure.table[0][1:]]
        for name in STRATEGY_ORDER:
            curve = rmse_band(RESULT.sessions_for(name), length=max(steps) + 1).mean
            assert cells(figure, name)[1:] == tuple(f"{curve[step]:.3f}" for step in steps)

    def test_the_coverage_table_is_the_coverage_band(self):
        figure = BY_NAME["coverage-vs-questions"]
        steps = [int(header.lstrip("@")) for header in figure.table[0][1:]]
        for name in STRATEGY_ORDER:
            curve = coverage_band(RESULT.sessions_for(name), length=max(steps) + 1).mean
            assert cells(figure, name)[1:] == tuple(f"{curve[step]:.2f}" for step in steps)

    def test_the_reach_table_reports_both_readings_and_their_gap(self):
        figure = BY_NAME["convergence-reach"]
        for name in STRATEGY_ORDER:
            ever, still, difference = cells(figure, name)[1:]
            curve = reach_curve(RESULT.sessions_for(name), lambda s: s.mae, TINY.convergence_mae)
            assert ever == f"{curve[-1]:.1%}"
            # The gap is the share of candidates that crossed and crossed back.
            assert float(difference.rstrip("%")) == pytest.approx(
                float(still.rstrip("%")) - float(ever.rstrip("%")), abs=0.05
            )

    def test_the_stability_table_is_the_stability_summary(self):
        figure = BY_NAME["stability"]
        for name in STRATEGY_ORDER:
            summary = stability(
                RESULT.sessions_for(name), length=len(RESULT.summary(name).mae_curve)
            )
            row = cells(figure, name)
            assert row[1] == f"{summary.mean_abs_delta:.3f}"
            assert row[3] == f"{summary.reversal_rate:.1%}"

    def test_the_stopping_table_is_the_stopping_profile(self):
        figure = BY_NAME["stopping-by-question"]
        counted = {(row[0], row[1]): int(row[2]) for row in rows(figure)}
        for name in STRATEGY_ORDER:
            profile = stopping_by_items(RESULT.sessions_for(name))
            for reason in profile.reasons:
                assert counted[(name, reason)] == profile.total_for(reason)
        assert sum(counted.values()) == len(RESULT.sessions)

    def test_the_spread_table_reports_ordered_quartiles(self):
        for row in rows(BY_NAME["mae-spread"]):
            p25, median, p75 = (float(value) for value in row[2:])
            assert p25 <= median <= p75

    def test_a_trajectory_table_reports_the_hidden_truth_and_each_final_estimate(self):
        candidate = representative_candidates(RESULT.sessions_for(ADAPTIVE))[0]
        figure = trajectory_figure(RESULT, candidate)
        for row in rows(figure):
            subtopic = row[0]
            truth = session_for(RESULT.sessions_for(ADAPTIVE), candidate).ground_truth_theta
            assert float(row[1]) == pytest.approx(truth[subtopic], abs=5e-4)
            for index, name in enumerate(STRATEGY_ORDER):
                final = session_for(RESULT.sessions_for(name), candidate).final_theta[subtopic]
                assert float(row[2 + index]) == pytest.approx(final, abs=5e-4)


class TestFigureContent:
    def test_the_mae_chart_draws_day_12s_own_convergence_threshold(self):
        assert f"MAE = {TINY.convergence_mae}" in BY_NAME["mae-vs-questions"].svg

    def test_the_stability_chart_draws_day_12s_own_quiet_threshold(self):
        """It is drawn, not tuned - the figure shows how close a session gets to
        the condition that ends it."""
        assert str(SMALL_DELTA_THETA) in BY_NAME["stability"].svg

    def test_the_stability_curve_starts_at_the_first_answer(self):
        """Step 0 is the cold start, where the padded movement is zero; plotting
        it would draw a spurious spike out of the origin."""
        svg = BY_NAME["stability"].svg
        first_tick = re.search(r'text-anchor="middle">(\d+)</text>', svg)
        assert first_tick is not None and int(first_tick.group(1)) >= 1

    def test_the_trajectory_chart_marks_ground_truth_as_a_reference(self):
        figure = BY_NAME[next(name for name in BY_NAME if name.startswith("trajectory-"))]
        assert "hidden from every policy" in figure.svg
        assert 'stroke="var(--muted)"' in figure.svg

    def test_the_trajectory_chart_has_a_panel_per_subtopic(self):
        figure = BY_NAME[next(name for name in BY_NAME if name.startswith("trajectory-"))]
        for subtopic in TINY.subtopics:
            assert subtopic in figure.svg

    def test_a_representative_figure_says_how_it_was_chosen(self):
        """ "A representative candidate" means nothing without the ordering."""
        figure = trajectory_figure(RESULT, "c000", quantile=0.5, rank_by=ADAPTIVE)
        assert "50%" in figure.caption
        assert ADAPTIVE in figure.caption
        assert "quantile" not in trajectory_figure(RESULT, "c000").caption

    def test_the_coverage_chart_shows_the_whole_blueprint(self):
        assert f"all {len(TINY.subtopics)}" in BY_NAME["coverage-vs-questions"].svg

    def test_an_unknown_metric_is_rejected(self):
        with pytest.raises(ValueError, match="metric must be"):
            error_figure(RESULT, metric="mape")


class TestWriting:
    def test_it_writes_one_file_per_figure(self, tmp_path):
        written = write_figures(FIGURES, tmp_path / "out")
        assert len(written) == len(FIGURES)
        for path, figure in zip(written, FIGURES, strict=True):
            assert path.name == figure.filename
            assert path.read_text(encoding="utf-8") == figure.svg

    def test_it_writes_platform_independent_newlines(self, tmp_path):
        """The files are committed; a CRLF on Windows would make the same run
        produce a different artefact than CI does."""
        path = write_figures(FIGURES[:1], tmp_path)[0]
        assert b"\r\n" not in path.read_bytes()

    def test_a_rerun_replaces_rather_than_accumulates(self, tmp_path):
        write_figures(FIGURES, tmp_path)
        write_figures(FIGURES, tmp_path)
        assert len(list(tmp_path.glob("*.svg"))) == len(FIGURES)


class TestReport:
    REPORT = convergence_report(RESULT, FIGURES, figure_dir="convergence")

    def test_it_embeds_every_figure(self):
        for figure in FIGURES:
            assert f"](convergence/{figure.filename})" in self.REPORT
            assert figure.caption in self.REPORT

    def test_it_states_the_synthetic_caveat_before_any_number(self):
        caveat = self.REPORT.index("results about a synthetic environment")
        assert caveat < self.REPORT.index("| policy |")

    def test_it_disclaims_statistical_significance(self):
        """The brief forbids claiming significance without testing for it."""
        assert "No claim of statistical significance" in self.REPORT
        assert "No hypothesis test was run" in self.REPORT

    def test_it_records_what_would_reproduce_it(self):
        assert str(TINY.seed) in self.REPORT
        assert "run_convergence_analysis.py" in self.REPORT

    def test_it_says_the_convergence_criteria_are_day_13s(self):
        assert "no new threshold was introduced" in self.REPORT
        assert f"MAE ≤ {TINY.convergence_mae}" in self.REPORT
        assert f"RD ≤ {TINY.convergence_se}" in self.REPORT

    def test_the_coverage_commentary_sits_with_the_coverage_figure(self):
        coverage_figure_at = self.REPORT.index("coverage-vs-questions.svg")
        commentary_at = self.REPORT.index("Coverage against final error")
        stopping_at = self.REPORT.index("stopping-by-question.svg")
        assert coverage_figure_at < commentary_at < stopping_at

    def test_it_warns_that_the_coverage_bins_are_not_causal(self):
        assert "not** a controlled comparison" in self.REPORT
        assert "establishes no cause" in self.REPORT

    def test_every_markdown_table_row_has_the_same_column_count(self):
        """A stray pipe inside a cell silently splits a column."""
        widths: list[int] = []
        for line in self.REPORT.splitlines():
            stripped = line.strip()
            if not stripped.startswith("|"):
                widths = []
                continue
            count = len(re.split(r"(?<!\\)\|", stripped)) - 2
            widths.append(count)
            assert count == widths[0], line

    def test_a_pipe_inside_a_cell_is_escaped(self):
        assert r"mean \|Δθ\|" in self.REPORT

    def test_it_is_a_pure_function_of_the_experiment(self):
        again = build_figures(run_experiment(TINY, environment=ENV))
        assert convergence_report(RESULT, again, figure_dir="convergence") == self.REPORT
        assert [figure.svg for figure in again] == [figure.svg for figure in FIGURES]


class TestReportTables:
    def test_the_coverage_table_lists_every_bin_of_every_policy(self):
        table = coverage_error_table(RESULT, markdown=True)
        for name in STRATEGY_ORDER:
            for entry in coverage_vs_error(RESULT.sessions_for(name)):
                assert f"| {name} | {entry.subtopics_measured} | {entry.n_sessions} |" in table

    def test_the_stability_table_has_one_row_per_policy(self):
        table = stability_table(RESULT)
        for name in STRATEGY_ORDER:
            assert name in table

    def test_a_figure_table_renders_its_own_header(self):
        rendered = figure_table(BY_NAME["mae-vs-questions"])
        assert rendered.splitlines()[0].startswith("| policy |")


class TestBuiltFromScratch:
    """Each builder works standalone, so a caller can draw one chart."""

    @pytest.mark.parametrize(
        "builder",
        [convergence_figure, coverage_figure, spread_figure, stability_figure, stopping_figure],
    )
    def test_a_single_builder_produces_a_usable_figure(self, builder):
        figure = builder(RESULT)
        assert figure.svg.startswith("<svg")
        assert len(figure.table) >= 2
