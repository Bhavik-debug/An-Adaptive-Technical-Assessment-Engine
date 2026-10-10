"""The grading-regression gate - Day 20, plan sections 12.3 and Phase-4 exit.

    CI fails the build if QWK drops more than 0.05 below the committed baseline.

Two kinds of test here.

The **mechanism** tests prove the gate does its job: it fails a deliberately
degraded result, passes one inside tolerance, and - the part that matters most -
**refuses rather than passes** when the comparison would be meaningless. A gate
that passes on missing data is not a gate.

The **committed artefact** tests run the real thing: the real baseline against
the real grading runs. That is the assertion CI actually makes on every PR.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.grading.evalset import gate_eligible, load_answers
from app.grading.reflabel import labels_by_answer, load_reference_labels
from app.grading.regression import (
    MAX_REGRESSION,
    MIN_ITEMS,
    BaselineError,
    GradingBaseline,
    evaluate_regression,
    load_baseline,
)
from app.grading.runs import load_runs, select
from app.grading.validity import quadratic_weighted_kappa, score_to_band

ROOT = Path(__file__).resolve().parents[3].parent
BASELINE_PATH = ROOT / "evals" / "baselines" / "grading_qwk.json"
RUNS_PATH = ROOT / "evals" / "reports" / "grading" / "runs.jsonl"
DATASET_DIR = ROOT / "evals" / "datasets" / "grading"
ANSWERS_PATH = DATASET_DIR / "answers.jsonl"
REFERENCE_PATH = DATASET_DIR / "reference_labels.jsonl"


def baseline(**overrides) -> GradingBaseline:
    payload: dict[str, object] = {
        "baseline_version": 1,
        "qwk": 0.50,
        "n_items": 120,
        "reference_artifact": "evals/datasets/grading/reference_labels.jsonl",
        "reference_pass": "blind",
        "reference_rubric_version": "labels-v2",
        "reference_label_source": "model",
        "prompt_version": "v1",
        "prompt_fingerprint": "abc123abc123",
        "grader_model": "some/model",
        "notes": "a baseline note long enough to satisfy the minimum length rule",
    }
    payload.update(overrides)
    return GradingBaseline.model_validate(payload)


def verdict(**overrides):
    kwargs: dict[str, object] = {
        "baseline": baseline(),
        "current_qwk": 0.50,
        "n_items": 120,
        "reference_label_source": "model",
        "reference_pass": "blind",
        "reference_rubric_version": "labels-v2",
    }
    kwargs.update(overrides)
    return evaluate_regression(**kwargs)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# The threshold
# ---------------------------------------------------------------------------


class TestTheGateFiresAtTheRightPlace:
    def test_the_threshold_is_the_plans_threshold(self):
        """Transcribed from the plan, not chosen here."""
        assert MAX_REGRESSION == 0.05

    def test_an_unchanged_score_passes(self):
        assert verdict(current_qwk=0.50).passed is True

    def test_an_improvement_passes(self):
        """One-sided on purpose: the gate stops quality falling, not rising."""
        result = verdict(current_qwk=0.80)
        assert result.passed is True
        assert result.regression == pytest.approx(-0.30)
        assert "above the committed baseline" in result.message

    def test_a_regression_inside_tolerance_passes(self):
        result = verdict(current_qwk=0.46)
        assert result.passed is True
        assert result.regression == pytest.approx(0.04)

    def test_a_regression_exactly_at_the_threshold_passes(self):
        """`<=` not `<`: "drops more than 0.05" is the plan's wording."""
        result = verdict(current_qwk=0.45)
        assert result.regression == pytest.approx(0.05)
        assert result.passed is True

    def test_a_regression_past_the_threshold_fails(self):
        result = verdict(current_qwk=0.44)
        assert result.passed is False
        assert result.regression == pytest.approx(0.06)

    def test_a_large_regression_fails(self):
        """The case the gate exists for: a prompt edit that breaks grading."""
        result = verdict(current_qwk=0.10)
        assert result.passed is False
        assert result.regression == pytest.approx(0.40)


class TestTheFailureMessage:
    """A gate whose failure nobody can act on gets disabled, not investigated."""

    def test_it_names_the_baseline_the_current_score_and_the_threshold(self):
        message = verdict(current_qwk=0.30).message
        assert "0.3000" in message, "the current score"
        assert "0.5000" in message, "the baseline"
        assert "0.2000" in message, "the regression"
        assert "0.05" in message, "the threshold"
        assert "FAIL" in message

    def test_it_names_what_produced_the_baseline(self):
        message = verdict(current_qwk=0.30).message
        assert "baseline v1" in message
        assert "v1/abc123abc123" in message
        assert "some/model" in message

    def test_it_names_the_reference_and_its_provenance(self):
        message = verdict(current_qwk=0.30).message
        assert "reference_labels.jsonl" in message
        assert "label_source=model" in message

    def test_it_tells_the_reader_not_to_relax_the_threshold(self):
        """The one instruction that keeps the gate honest a year from now."""
        message = verdict(current_qwk=0.30).message
        assert "move the baseline in a commit of its own" in message
        assert "do not relax the threshold" in message

    def test_a_provisional_baseline_says_so_on_every_verdict(self):
        for current in (0.50, 0.10):
            message = verdict(current_qwk=current).message
            assert "MODEL-GENERATED provisional" in message
            assert "NOT evidence for the Phase-4 acceptance gate" in message

    def test_a_human_baseline_does_not_carry_the_provisional_note(self):
        message = verdict(
            baseline=baseline(reference_label_source="human"),
            reference_label_source="human",
        ).message
        assert "provisional" not in message


# ---------------------------------------------------------------------------
# Refusals: the ways a gate could silently pass, and does not
# ---------------------------------------------------------------------------


class TestItRefusesRatherThanPasses:
    def test_too_few_items_fails(self):
        """An under-powered QWK is not evidence that nothing regressed."""
        result = verdict(current_qwk=0.50, n_items=MIN_ITEMS - 1)
        assert result.passed is False
        assert "REFUSED" in result.message
        assert result.regression is None

    def test_too_few_items_fails_even_when_the_score_improved(self):
        result = verdict(current_qwk=0.99, n_items=3)
        assert result.passed is False

    def test_a_different_reference_pass_fails(self):
        result = verdict(reference_pass="concept_aware")
        assert result.passed is False
        assert "REFUSED" in result.message
        assert "would not be a regression test" in result.message

    def test_a_different_rubric_version_fails(self):
        result = verdict(reference_rubric_version="labels-v1")
        assert result.passed is False
        assert "REFUSED" in result.message

    def test_a_provenance_change_fails_rather_than_comparing_across_it(self):
        """Model-referenced and human-referenced QWK are different measurements.

        Diffing them would let a human baseline be "met" by a model-referenced
        run, which is the exact confusion the whole provenance apparatus exists
        to prevent.
        """
        result = verdict(
            baseline=baseline(reference_label_source="human"),
            reference_label_source="model",
            current_qwk=0.99,
        )
        assert result.passed is False
        assert "label provenance changed" in result.message

    def test_a_missing_baseline_raises_rather_than_defaulting(self, tmp_path):
        with pytest.raises(BaselineError, match="no grading baseline"):
            load_baseline(tmp_path / "absent.json")

    def test_malformed_json_raises(self, tmp_path):
        path = tmp_path / "baseline.json"
        path.write_text("{not json", encoding="utf-8")
        with pytest.raises(BaselineError, match="not valid JSON"):
            load_baseline(path)

    def test_a_baseline_missing_a_field_raises(self, tmp_path):
        payload = baseline().model_dump(mode="json")
        del payload["reference_label_source"]
        path = tmp_path / "baseline.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(BaselineError, match="not a valid baseline"):
            load_baseline(path)

    def test_a_baseline_with_an_unknown_field_raises(self, tmp_path):
        """This file is hand-edited; a misspelt key must not silently default."""
        payload = baseline().model_dump(mode="json")
        payload["qwk_threshold"] = 0.9
        path = tmp_path / "baseline.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(BaselineError, match="not a valid baseline"):
            load_baseline(path)

    def test_a_baseline_without_notes_is_refused(self):
        with pytest.raises(ValueError, match="notes"):
            baseline(notes="too short")

    def test_an_out_of_range_qwk_is_refused(self):
        with pytest.raises(ValueError):
            baseline(qwk=1.5)


class TestAcceptanceEvidenceIsSeparateFromRegression:
    """The distinction the whole module exists to keep."""

    def test_a_provisional_baseline_is_never_acceptance_evidence(self):
        assert verdict(current_qwk=0.99).is_acceptance_evidence is False

    def test_only_a_human_baseline_can_be_acceptance_evidence(self):
        result = verdict(
            baseline=baseline(reference_label_source="human"),
            reference_label_source="human",
        )
        assert result.is_acceptance_evidence is True

    def test_the_committed_baseline_is_provisional(self):
        """It is measured against model labels, and says so."""
        committed = load_baseline(BASELINE_PATH)
        assert committed.reference_label_source == "model"
        assert committed.is_provisional is True
        assert "PROVISIONAL" in committed.notes

    def test_passing_this_gate_does_not_make_the_dataset_gate_eligible(self):
        """The two must never be confusable, so it is asserted directly."""
        answers = load_answers(ANSWERS_PATH)
        assert gate_eligible(answers) is False


# ---------------------------------------------------------------------------
# The real thing: what CI asserts on every PR
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not RUNS_PATH.exists(), reason="no grading runs in this checkout")
@pytest.mark.skipif(not BASELINE_PATH.exists(), reason="no baseline in this checkout")
class TestTheCommittedGradingRunsDoNotRegress:
    """The gate itself, on the committed artefacts. Deterministic, no provider."""

    @staticmethod
    def _measure() -> tuple[float, int]:
        labels = labels_by_answer(load_reference_labels(REFERENCE_PATH), pass_name="blind")
        runs = select(load_runs(RUNS_PATH), prompt_version="v1", variant="base", run_index=0)
        paired = sorted(set(labels) & set(runs))
        return (
            quadratic_weighted_kappa(
                [labels[aid].score for aid in paired],
                [score_to_band(runs[aid].score) for aid in paired],
            ),
            len(paired),
        )

    def test_the_gate_passes_on_the_committed_runs(self):
        committed = load_baseline(BASELINE_PATH)
        current, n_items = self._measure()
        result = evaluate_regression(
            baseline=committed,
            current_qwk=current,
            n_items=n_items,
            reference_label_source="model",
            reference_pass=committed.reference_pass,
            reference_rubric_version=committed.reference_rubric_version,
        )
        assert result.passed, result.message

    def test_the_baseline_matches_what_the_committed_runs_measure(self):
        """A baseline that has drifted from its own artefacts gates nothing.

        Exact equality: both sides are the same pure arithmetic over the same
        committed files, so any difference means the baseline was computed from
        something that is no longer in the repository.
        """
        committed = load_baseline(BASELINE_PATH)
        current, n_items = self._measure()
        assert current == pytest.approx(committed.qwk, abs=1e-12)
        assert n_items == committed.n_items

    def test_a_degraded_result_against_the_real_baseline_fails(self):
        """The negative control, against the committed baseline.

        Without this, "the gate passed" could mean the gate never fails.
        """
        committed = load_baseline(BASELINE_PATH)
        _, n_items = self._measure()
        result = evaluate_regression(
            baseline=committed,
            current_qwk=committed.qwk - 0.2,
            n_items=n_items,
            reference_label_source="model",
            reference_pass=committed.reference_pass,
            reference_rubric_version=committed.reference_rubric_version,
        )
        assert result.passed is False
        assert "FAIL" in result.message

    def test_the_grader_model_and_prompt_still_match_the_baseline(self):
        """A baseline for a different prompt is not this prompt's baseline."""
        committed = load_baseline(BASELINE_PATH)
        runs = select(load_runs(RUNS_PATH), prompt_version="v1", variant="base", run_index=0)
        assert {run.model for run in runs.values()} == {committed.grader_model}
        assert {run.prompt_fingerprint for run in runs.values()} == {committed.prompt_fingerprint}
