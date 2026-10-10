"""The persisted grading-run artefact, and the probes that read it - Day 19.

``runs.jsonl`` is where every number in the Day-19 report comes from, so the
tests that matter here are the ones about *re-derivability*: a stored row must
contain enough to recompute its own score, or "the report is recomputable from
the persisted rows" is not a true statement.
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from app.bank.schema import ExpectedConcept
from app.grading.probes import position_invariance, shuffled_concepts
from app.grading.runs import (
    GradingFailure,
    GradingRun,
    dump_failures,
    dump_runs,
    load_failures,
    load_runs,
    runs_index,
    select,
)
from app.grading.schema import ConceptJudgment, GradeJudgment, RubricScores
from app.grading.scoring import score_answer

ANSWER = (
    "A lost update happens when two transactions read the same row and both write "
    "back, so the second overwrites the first. Use select for update to lock it."
)


def concepts(*keys: str) -> list[ExpectedConcept]:
    return [
        ExpectedConcept(key=key, weight=(index % 3) + 1, hint="a hint that is long enough")
        for index, key in enumerate(keys)
    ]


def judgment(*, labels: dict[str, str] | None = None, **overrides) -> GradeJudgment:
    # GradeJudgment requires at least three concept entries, matching the
    # minimum any real bank item carries.
    labels = labels or {
        "lost_update": "covered",
        "pessimistic_locking": "partial",
        "atomic_write_operation": "absent",
    }
    payload: dict[str, object] = {
        "concepts": [
            ConceptJudgment(
                key=key,
                label=label,  # type: ignore[arg-type]
                evidence=(None if label == "absent" else "A lost update happens"),
            )
            for key, label in labels.items()
        ],
        "errors": [],
        "misconceptions_hit": [],
        "rubric": RubricScores(structure=3, tradeoff_awareness=2, specificity=3, communication=3),
        "grader_confidence": 0.8,
    }
    payload.update(overrides)
    return GradeJudgment.model_validate(payload)


def run(**overrides) -> GradingRun:
    payload: dict[str, object] = {
        "answer_id": "eval-0001",
        "question_id": "db-conc-001",
        "score": 0.62,
        "concept_score": 0.7,
        "rubric_score": 0.6875,
        "penalty": 0.0,
        "grader_confidence": 0.8,
        "n_concepts": 3,
        "prompt_version": "v1",
        "prompt_fingerprint": "abc123abc123",
        "model": "some/model",
        "temperature": 0.0,
        "n_calls": 1,
    }
    payload.update(overrides)
    return GradingRun.model_validate(payload)


class TestTheRunRecord:
    def test_a_score_outside_the_unit_interval_is_refused(self):
        with pytest.raises(ValidationError):
            run(score=1.4)

    def test_fewer_than_one_call_is_refused(self):
        """A run that cost no calls did not happen."""
        with pytest.raises(ValidationError):
            run(n_calls=0)

    def test_an_unknown_variant_is_refused(self):
        with pytest.raises(ValidationError):
            run(variant="paraphrased")

    def test_an_unexpected_field_is_refused(self):
        payload = run().model_dump(mode="json")
        payload["note"] = "hand-edited"
        with pytest.raises(ValidationError):
            GradingRun.model_validate(payload)

    def test_the_key_identifies_a_run_uniquely(self):
        assert run().key == ("eval-0001", "v1", "base", 0)
        assert run(variant="filler").key != run().key
        assert run(run_index=1).key != run().key
        assert run(prompt_version="v2").key != run().key

    def test_a_round_trip_preserves_the_stored_judgment(self, tmp_path):
        path = tmp_path / "runs.jsonl"
        rows = [run(judgment=judgment())]
        assert dump_runs(rows, path) == 1
        back = load_runs(path)
        assert back[0].judgment is not None
        assert back[0].judgment.model_dump() == rows[0].judgment.model_dump()

    def test_a_malformed_row_fails_the_whole_read(self, tmp_path):
        path = tmp_path / "runs.jsonl"
        path.write_text(json.dumps(run().model_dump(mode="json")) + "\n{nope}\n", encoding="utf-8")
        with pytest.raises(ValueError, match="line 2"):
            load_runs(path)

    def test_duplicate_keys_are_refused_by_the_index(self):
        """Two rows for one key would make every paired statistic ambiguous."""
        with pytest.raises(ValueError, match="duplicate grading run"):
            runs_index([run(), run()])

    def test_select_slices_by_version_variant_and_index(self):
        rows = [
            run(),
            run(answer_id="eval-0002"),
            run(answer_id="eval-0003", variant="filler"),
            run(answer_id="eval-0004", prompt_version="v2"),
            run(answer_id="eval-0005", run_index=3),
        ]
        assert set(select(rows, prompt_version="v1")) == {"eval-0001", "eval-0002"}
        assert set(select(rows, prompt_version="v1", variant="filler")) == {"eval-0003"}
        assert set(select(rows, prompt_version="v2")) == {"eval-0004"}
        assert set(select(rows, prompt_version="v1", run_index=3)) == {"eval-0005"}


class TestTheStoredJudgmentIsEnoughToRederiveTheScore:
    """The property that makes the artefact auditable rather than merely saved."""

    def test_rescoring_a_stored_judgment_reproduces_its_score(self):
        expected = concepts("lost_update", "pessimistic_locking", "atomic_write_operation")
        stored = judgment()
        breakdown = score_answer(stored, expected_concepts=expected)
        row = run(
            judgment=stored,
            score=breakdown.score,
            concept_score=breakdown.concept_score,
            rubric_score=breakdown.rubric_score,
            penalty=breakdown.penalty,
        )
        again = score_answer(row.judgment, expected_concepts=expected)  # type: ignore[arg-type]
        assert again.score == row.score
        assert again.concept_score == row.concept_score
        assert again.rubric_score == row.rubric_score


class TestPositionProbeOverRealScoring:
    """Plan section 12.1's position probe: "changes the score by **exactly 0**".

    Asserted against the real scorer, because the claim is about
    ``scoring.py``'s arithmetic and not about the probe helper.
    """

    def test_shuffling_the_concept_list_changes_the_score_by_exactly_zero(self):
        expected = concepts("a_concept", "b_concept", "c_concept", "d_concept")
        stored = judgment(
            labels={
                "a_concept": "covered",
                "b_concept": "partial",
                "c_concept": "absent",
                "d_concept": "covered",
            }
        )
        worst = position_invariance(stored, expected, score_fn=score_answer, trials=25)
        assert worst == 0.0, f"scoring depends on concept order: max delta {worst!r}"

    def test_it_holds_for_every_combination_of_labels(self):
        """Exhaustive over three concepts: 27 label assignments, all order-free."""
        import itertools

        expected = concepts("one", "two", "three")
        for combination in itertools.product(("covered", "partial", "absent"), repeat=3):
            stored = judgment(labels=dict(zip(("one", "two", "three"), combination, strict=True)))
            worst = position_invariance(stored, expected, score_fn=score_answer, trials=6)
            assert worst == 0.0, combination

    def test_it_holds_with_unequal_weights(self):
        """Weights are what could make a float sum order-dependent."""
        expected = [
            ExpectedConcept(key="w1", weight=1, hint="a hint that is long enough"),
            ExpectedConcept(key="w2", weight=2, hint="a hint that is long enough"),
            ExpectedConcept(key="w3", weight=3, hint="a hint that is long enough"),
        ]
        stored = judgment(labels={"w1": "partial", "w2": "covered", "w3": "partial"})
        assert position_invariance(stored, expected, score_fn=score_answer, trials=25) == 0.0

    def test_the_shuffle_really_reorders_so_the_probe_is_not_vacuous(self):
        expected = concepts("a_concept", "b_concept", "c_concept", "d_concept")
        permuted = shuffled_concepts(expected, seed=1219)
        assert [c.key for c in permuted] != [c.key for c in expected]
        assert sorted(c.key for c in permuted) == sorted(c.key for c in expected)


class TestGradingFailures:
    """A grading attempt that produced no score is a datum, not a gap.

    Plan section 12.1's fourth suite sets a target on schema/contract violation
    rate, which cannot be reported from a number that only existed in a
    terminal - so failures are persisted with the same discipline as scores.
    """

    @staticmethod
    def _failure(answer_id: str = "eval-0001", **overrides) -> GradingFailure:
        payload: dict[str, object] = {
            "answer_id": answer_id,
            "question_id": "db-conc-001",
            "prompt_version": "v1",
            "error_type": "GradingContractError",
            "message": "invented ['ping']; did not classify ['exchange_argument']",
        }
        payload.update(overrides)
        return GradingFailure.model_validate(payload)

    def test_a_round_trip_preserves_the_failure(self, tmp_path):
        path = tmp_path / "failures.jsonl"
        rows = [self._failure()]
        assert dump_failures(rows, path) == 1
        assert [r.model_dump() for r in load_failures(path)] == [r.model_dump() for r in rows]

    def test_a_missing_file_reads_as_empty(self, tmp_path):
        """No failures yet is the normal state, not an error."""
        assert load_failures(tmp_path / "nothing.jsonl") == []

    def test_repeated_attempts_at_the_same_row_are_deduplicated(self, tmp_path):
        """The pass is resumable, so an ungradeable answer fails on every run.

        Three identical rows would make the violation rate look three times
        worse than it is.
        """
        path = tmp_path / "failures.jsonl"
        same = [self._failure(), self._failure(), self._failure()]
        assert dump_failures(same, path) == 1

    def test_the_latest_attempt_wins(self, tmp_path):
        path = tmp_path / "failures.jsonl"
        dump_failures(
            [self._failure(message="first attempt"), self._failure(message="second attempt")],
            path,
        )
        assert load_failures(path)[0].message == "second attempt"

    def test_failures_at_different_keys_are_kept_apart(self, tmp_path):
        path = tmp_path / "failures.jsonl"
        rows = [
            self._failure("eval-0001"),
            self._failure("eval-0001", prompt_version="v2"),
            self._failure("eval-0001", variant="filler"),
            self._failure("eval-0002"),
        ]
        assert dump_failures(rows, path) == 4

    def test_an_unexpected_field_is_refused(self):
        payload = self._failure().model_dump(mode="json")
        payload["retryable"] = True
        with pytest.raises(ValidationError):
            GradingFailure.model_validate(payload)

    def test_an_empty_dump_writes_an_empty_file_rather_than_a_blank_line(self, tmp_path):
        path = tmp_path / "failures.jsonl"
        assert dump_failures([], path) == 0
        assert path.read_text(encoding="utf-8") == ""
        assert load_failures(path) == []
