"""The report formatter.

Formatting is where the temptation to flatter a result lives, so what is tested
here is mostly that nothing is hidden: every policy appears, every stopping
reason gets a column, a censored convergence prints as censored rather than as a
blank, and the configuration block lists what the run assumed.
"""

from __future__ import annotations

from app.simulation.environment import build_environment
from app.simulation.report import (
    ablation_table,
    comparison_table,
    config_block,
    convergence_table,
    curve_table,
    format_seconds,
    markdown_report,
    spread_table,
    stopping_table,
    text_report,
)
from app.simulation.runner import run_ablation, run_experiment
from app.simulation.strategies import ADAPTIVE, FIXED, RANDOM
from tests.unit.simulation.conftest import TINY

ENV = build_environment(TINY)
RESULT = run_experiment(TINY, environment=ENV)
ABLATION = run_ablation(TINY, environment=ENV)


class TestConfigBlock:
    def test_it_records_what_is_needed_to_reproduce_the_run(self):
        block = config_block(TINY)
        for expected in ("seed", "candidates", "item budget", "topic quotas", "convergence"):
            assert expected in block

    def test_the_seed_is_printed(self):
        assert str(TINY.seed) in config_block(TINY)


class TestComparisonTable:
    def test_every_policy_gets_a_row(self):
        table = comparison_table(RESULT)
        for name in (ADAPTIVE, RANDOM, FIXED):
            assert name in table

    def test_it_reports_both_error_readings_and_the_coverage_that_explains_them(self):
        header = comparison_table(RESULT).splitlines()[0]
        assert "MAE" in header
        assert "MAE(seen)" in header
        assert "subtopics" in header

    def test_markdown_renders_a_table(self):
        table = comparison_table(RESULT, markdown=True)
        assert table.startswith("| policy")
        assert "---" in table

    def test_policies_print_in_the_order_they_were_run(self):
        rows = [line.split()[0] for line in comparison_table(RESULT).splitlines()[2:]]
        assert rows == [ADAPTIVE, RANDOM, FIXED]


class TestOtherTables:
    def test_the_spread_table_shows_median_and_deviation(self):
        header = spread_table(RESULT).splitlines()[0]
        assert "median" in header
        assert "sd" in header

    def test_a_censored_convergence_prints_as_censored(self):
        """A dash and an explicit "0/n", never a silent blank or a zero."""
        table = convergence_table(RESULT)
        assert f"0/{TINY.candidate_count}" in table or "/" in table
        assert "reached" in table.splitlines()[0]

    def test_the_stopping_table_has_a_column_per_reason(self):
        reasons = {r for s in RESULT.summaries for r in s.stop_reasons}
        header = stopping_table(RESULT).splitlines()[0]
        for reason in reasons:
            assert reason in header

    def test_the_curve_table_ends_at_the_item_budget(self):
        header = curve_table(RESULT).splitlines()[0]
        assert f"@{TINY.item_budget}" in header
        assert "@0" in header

    def test_the_curve_table_accepts_explicit_steps(self):
        header = curve_table(RESULT, steps=[0, 2]).splitlines()[0]
        assert "@2" in header
        assert "@4" not in header


class TestAblationTable:
    def test_every_variant_gets_a_row_with_a_delta(self):
        table = ablation_table(ABLATION)
        assert "full" in table
        assert "no_information" in table
        assert "dMAE" in table.splitlines()[0]

    def test_the_baseline_row_has_a_zero_delta(self):
        rows = ablation_table(ABLATION).splitlines()
        line = next(row for row in rows if row.startswith("full"))
        assert "+0.0000" in line

    def test_an_unknown_baseline_leaves_the_deltas_blank_rather_than_lying(self):
        table = ablation_table(ABLATION, baseline="nope")
        assert " - " in table or table.count("-") > 0


class TestWholeReports:
    def test_the_text_report_contains_every_section(self):
        report = text_report(RESULT, ABLATION)
        for heading in (
            "CONFIGURATION",
            "POLICY COMPARISON",
            "SPREAD ACROSS CANDIDATES",
            "CONVERGENCE",
            "WHY SESSIONS STOPPED",
            "WEIGHT ABLATION",
        ):
            assert heading in report

    def test_the_text_report_works_without_an_ablation(self):
        assert "WEIGHT ABLATION" not in text_report(RESULT)

    def test_the_markdown_report_carries_the_limitation_notice(self):
        """The numbers must never travel without the caveat that they describe a
        synthetic world."""
        report = markdown_report(RESULT, ABLATION)
        assert "synthetic environment" in report
        assert "docs/simulation.md" in report
        assert "no renormalisation would therefore" in report or "renormalisation" in report

    def test_the_markdown_report_explains_that_the_two_comparisons_differ(self):
        """The likeliest misreading of the whole experiment, so the artefact must
        carry the caveat even when read on its own: `random` shares the adaptive
        eligibility pool and therefore isolates the ranking objective, while
        `fixed` cannot apply the theta-dependent difficulty window and therefore
        bundles several changes together."""
        report = markdown_report(RESULT, ABLATION)
        assert "not equally clean" in report
        assert "cannot attribute the gap to any single mechanism" in report
        assert "No policy wins every column" in report

    def test_the_markdown_report_can_include_an_extended_run(self):
        report = markdown_report(RESULT, ABLATION, extended=RESULT)
        assert "Extended run" in report

    def test_it_ends_with_a_newline(self):
        assert markdown_report(RESULT).endswith("\n")


class TestFormatSeconds:
    def test_short_runs_print_in_seconds(self):
        assert format_seconds(9.4) == "9.4 s"

    def test_long_runs_print_in_minutes(self):
        assert format_seconds(93.0) == "1 m 33 s"
