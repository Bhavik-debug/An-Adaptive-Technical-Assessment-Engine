"""The persisted record of one grading run - Day 19.

Every grader invocation the validity harness makes is written here as a row, and
every number in the Day-19 report is computed from this file rather than from a
live call. That is what makes the report re-derivable: the metrics can be
recomputed, the binning changed, or a subgroup re-analysed without paying the
provider again, and a reader can check the arithmetic against the same rows the
report was built from.

One table serves the whole harness, keyed by ``(answer_id, prompt_version,
variant, run_index)``:

* the **main pass** is ``variant="base"``, ``run_index=0``;
* **test-retest** is ``variant="base"``, ``run_index`` 0-4;
* the **bias probes** are ``variant`` in ``filler`` / ``confidence`` /
  ``jargon`` / ``concepts_shuffled``, paired against the ``base`` row for the
  same answer;
* the **prompt comparison** varies ``prompt_version``.

Keeping them in one shape rather than four means the same metric code reads all
of them, and a probe cannot accidentally be compared against a baseline graded
under a different prompt.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.grading.schema import GradeJudgment

#: Which perturbation (if any) was applied to the input before grading.
#: ``base`` is the unmodified answer with the bank's concept order.
RunVariant = Literal["base", "filler", "confidence", "jargon", "concepts_shuffled"]


class GradingRun(BaseModel):
    """One grader invocation, with everything needed to interpret its score."""

    model_config = ConfigDict(extra="forbid")

    answer_id: str = Field(pattern=r"^eval-\d{4}$")
    question_id: str
    variant: RunVariant = "base"
    #: 0 for a single pass; 0..n-1 for the repeated test-retest runs.
    run_index: int = Field(default=0, ge=0)

    #: The final code-side score in [0, 1]. This is what the harness measures.
    score: float = Field(ge=0.0, le=1.0)
    #: The three components, so a disagreement can be attributed to the concept
    #: classifier, the rubric, or the error penalty rather than guessed at.
    concept_score: float = Field(ge=0.0, le=1.0)
    rubric_score: float = Field(ge=0.0, le=1.0)
    penalty: float = Field(ge=0.0)
    grader_confidence: float = Field(ge=0.0, le=1.0)

    #: Whether the plan section 7.3 confidence gate fired, and why. A gated
    #: score is the median of three temperature-0.7 re-grades, so it is
    #: *expected* to vary between runs - which makes this field the difference
    #: between "the grader is unstable" and "the gate did its job".
    gate_fired: bool = False
    gate_reason: str | None = None
    #: How many 0-4 concept labels the model returned, and how many of its
    #: evidence spans failed substring verification and were downgraded.
    n_concepts: int = Field(ge=0)
    n_evidence_downgrades: int = Field(default=0, ge=0)
    n_errors_major: int = Field(default=0, ge=0)
    n_errors_minor: int = Field(default=0, ge=0)
    #: Day 20: flagged errors whose claim was not a quotation from the answer,
    #: and the penalty they would have applied. Counted apart from
    #: ``n_evidence_downgrades`` because the two move the score in opposite
    #: directions - a fabricated span removes credit, a fabricated claim
    #: removes a penalty - and one combined counter would net them off.
    n_errors_dropped: int = Field(default=0, ge=0)
    dropped_penalty: float = Field(default=0.0, ge=0.0)

    #: Provenance.
    prompt_version: str
    prompt_fingerprint: str
    model: str
    temperature: float
    #: Chokepoint calls this grade cost: 1, or 4 when the gate fired.
    n_calls: int = Field(ge=1)
    graded_at: str | None = None

    #: Only set on a ``jargon`` row: how many terms were actually replaced. A
    #: variant with zero substitutions is not a jargon variant and the probe
    #: excludes it rather than counting it as a pass.
    n_substitutions: int | None = None

    #: The evidence-corrected judgment the score was computed from.
    #:
    #: Stored, at the cost of a larger artefact, because it is what makes
    #: "every number in the report is recomputable from these rows" literally
    #: true rather than nearly true. With it, ``score_answer`` can be re-run
    #: offline: the concept/rubric weighting can be swept (plan section 7.2
    #: asks for exactly that), and the **position-bias probe becomes a
    #: deterministic re-score of a real judgment with the concept list
    #: shuffled** - no second model call, which matters because re-asking the
    #: model would measure its nondeterminism instead of the code's order
    #: dependence.
    #:
    #: It is the corrected judgment, not the model's raw output: the raw one
    #: would re-introduce evidence spans that verification already downgraded,
    #: and re-scoring it would not reproduce this row's score.
    judgment: GradeJudgment | None = None

    @property
    def key(self) -> tuple[str, str, str, int]:
        return (self.answer_id, self.prompt_version, self.variant, self.run_index)


def load_runs(path: Path) -> list[GradingRun]:
    """Read the artefact. Every row validates or the whole read fails."""
    out: list[GradingRun] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            out.append(GradingRun.model_validate_json(line))
        except ValueError as exc:
            raise ValueError(f"{path.name} line {number}: {exc}") from exc
    return out


def dump_runs(runs: Iterable[GradingRun], path: Path) -> int:
    """Write the artefact, sorted by key so a git diff stays readable."""
    ordered = sorted(runs, key=lambda run: run.key)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            json.dumps(run.model_dump(mode="json"), sort_keys=True, ensure_ascii=False)
            for run in ordered
        )
        + "\n",
        encoding="utf-8",
    )
    return len(ordered)


def runs_index(runs: Sequence[GradingRun]) -> dict[tuple[str, str, str, int], GradingRun]:
    """Keyed by ``(answer_id, prompt_version, variant, run_index)``. No duplicates."""
    out: dict[tuple[str, str, str, int], GradingRun] = {}
    for run in runs:
        if run.key in out:
            raise ValueError(f"duplicate grading run for {run.key}")
        out[run.key] = run
    return out


def select(
    runs: Sequence[GradingRun],
    *,
    prompt_version: str,
    variant: RunVariant = "base",
    run_index: int = 0,
) -> dict[str, GradingRun]:
    """One slice of the table, keyed by answer id."""
    return {
        run.answer_id: run
        for run in runs
        if run.prompt_version == prompt_version
        and run.variant == variant
        and run.run_index == run_index
    }


__all__ = [
    "GradingFailure",
    "GradingRun",
    "RunVariant",
    "dump_failures",
    "dump_runs",
    "load_failures",
    "load_runs",
    "runs_index",
    "select",
]


class GradingFailure(BaseModel):
    """A grading attempt that produced no score, and why.

    Persisted beside the successful runs rather than only printed, for two
    reasons. Plan section 12.1's fourth suite sets a target on **schema- and
    contract-violation rate** ("< 0.5% after retry"), which cannot be reported
    from a number that only ever existed in a terminal. And an answer missing
    from a 120-row pass changes every denominator in the report, so *why* it is
    missing has to be as durable as the rows that succeeded.

    ``GradingContractError`` is the interesting kind: the model returned valid
    JSON that invented concept keys or failed to classify the ones it was given.
    That is not a schema failure - the shape was right - and it is caught by
    ``grader.check_judgment`` rather than by pydantic.
    """

    model_config = ConfigDict(extra="forbid")

    answer_id: str = Field(pattern=r"^eval-\d{4}$")
    question_id: str
    variant: RunVariant = "base"
    run_index: int = Field(default=0, ge=0)
    prompt_version: str
    error_type: str
    message: str = Field(max_length=2000)
    failed_at: str | None = None

    @property
    def key(self) -> tuple[str, str, str, int]:
        return (self.answer_id, self.prompt_version, self.variant, self.run_index)


def load_failures(path: Path) -> list[GradingFailure]:
    out: list[GradingFailure] = []
    if not path.exists():
        return out
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            out.append(GradingFailure.model_validate_json(line))
        except ValueError as exc:
            raise ValueError(f"{path.name} line {number}: {exc}") from exc
    return out


def dump_failures(failures: Iterable[GradingFailure], path: Path) -> int:
    """Write the failure log, deduplicated by key, newest entry winning.

    Deduplicated because the pass is resumable and re-run: an answer the model
    cannot grade fails again on every attempt, and three identical rows would
    make the violation rate look three times worse than it is.
    """
    latest: dict[tuple[str, str, str, int], GradingFailure] = {}
    for failure in failures:
        latest[failure.key] = failure
    ordered = sorted(latest.values(), key=lambda failure: failure.key)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            json.dumps(failure.model_dump(mode="json"), sort_keys=True, ensure_ascii=False)
            for failure in ordered
        )
        + ("\n" if ordered else ""),
        encoding="utf-8",
    )
    return len(ordered)
