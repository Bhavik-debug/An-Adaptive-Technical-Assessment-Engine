"""The grading-regression gate - Day 20, plan sections 12.3 and Phase-4 exit.

    - name: Grading validity
      run: pytest evals/suites/test_grading_validity.py
      # fails if QWK drops more than 0.05 below the committed baseline

Plan section 12.3's operational definition of AI engineering: *"treat prompts as
code under test. A prompt change is a PR; the eval suite is its test; a
regression fails CI."* This module is the comparison that makes that true.

**Pure.** No LLM, no network, no clock. It reads a committed baseline and a
committed set of grading runs and returns a verdict. That is what lets it run in
CI in milliseconds with no provider credential - plan section 12.3's "runs free
in ~40 s per PR" - and it is why the baseline is a file in the repository rather
than a number in someone's notes.

What this gate is, and the one thing it is not
----------------------------------------------

It is a **regression** gate: it answers "did this change make grading agreement
worse than the last committed measurement?" It is **not** the Phase-4
acceptance gate, which is *"QWK >= 0.65 against human labels"*.

The difference is the provenance of the reference labels, and it is enforced
rather than described. Every baseline records the ``label_source`` of the
reference it was measured against. ``gate_eligible()`` in ``evalset.py`` still
decides acceptance, still requires 120 human labels, and still returns
``False``. A baseline measured against provisional model labels is marked
``provisional`` and :attr:`RegressionVerdict.is_acceptance_evidence` is
``False`` for it, permanently. Protecting a provisional number from regression
is useful; it is not the same as having passed.

Why a regression gate over provisional labels is worth having anyway: it catches
the failure it is designed to catch. If someone edits the grading prompt and
agreement falls 0.2, that is a real regression whether or not the labels it is
measured against are human, and it should fail the build on the PR that caused
it rather than be discovered in Phase 9.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

#: Plan's Phase-4 exit criterion: "CI fails the build if QWK drops more than
#: 0.05". Transcribed, not chosen here.
MAX_REGRESSION = 0.05

#: Below this many paired items a QWK is too unstable to gate on. Day 19
#: measured the reference distribution at 81.7% in one band, which already
#: makes the statistic fragile; gating on a handful of items on top of that
#: would fail builds at random. The gate **refuses** rather than passes when it
#: has less than this - see :class:`RegressionVerdict`.
MIN_ITEMS = 100


class BaselineError(ValueError):
    """The baseline file is missing, malformed, or describes something else."""


class GradingBaseline(BaseModel):
    """A committed, versioned grading measurement to compare against.

    ``extra="forbid"`` because this file is hand-edited when a baseline is
    deliberately moved, and a misspelt field that silently defaulted would be a
    baseline nobody could trust.
    """

    model_config = ConfigDict(extra="forbid")

    #: Bumped by hand whenever the baseline value is deliberately moved. The
    #: reason goes in ``notes``; a baseline that moved without a note is a
    #: number nobody can defend.
    baseline_version: int = Field(ge=1)
    qwk: float = Field(ge=-1.0, le=1.0)
    #: How many paired items the baseline QWK was computed over.
    n_items: int = Field(ge=1)

    #: The reference labels it was measured against, in full, because a QWK is
    #: meaningless without them.
    reference_artifact: str
    reference_pass: str
    reference_rubric_version: str
    #: ``"model"`` or ``"human"``. Decides whether this baseline can ever be
    #: acceptance evidence; see the module docstring.
    reference_label_source: Literal["human", "model"]

    #: What produced the scores.
    prompt_version: str
    prompt_fingerprint: str
    grader_model: str

    #: Free text. What this baseline is, and why it is where it is.
    notes: str = Field(min_length=20)

    @property
    def is_provisional(self) -> bool:
        return self.reference_label_source != "human"


@dataclass(frozen=True, slots=True)
class RegressionVerdict:
    """The gate's answer, and everything needed to act on it."""

    passed: bool
    #: ``None`` when the gate could not be evaluated at all - a missing
    #: baseline, too few items, a provenance mismatch. A verdict that cannot be
    #: evaluated is **not** a pass; see :attr:`passed`.
    current_qwk: float | None
    baseline_qwk: float | None
    regression: float | None
    n_items: int
    threshold: float
    message: str
    #: ``True`` only when the comparison was made against a **human**-labelled
    #: reference. Always ``False`` for a provisional baseline, by construction.
    is_acceptance_evidence: bool = False


def load_baseline(path: Path) -> GradingBaseline:
    """Read and validate the committed baseline, or fail loudly.

    Every failure mode here would otherwise become a silent pass, which is the
    one behaviour a gate must never have: a missing file, unparseable JSON, or
    a field the schema does not recognise all mean "nobody knows what the
    baseline is", and that is a failure, not a licence to proceed.
    """
    if not path.exists():
        raise BaselineError(
            f"no grading baseline at {path}. The regression gate cannot pass without "
            "one: a missing baseline is an unknown baseline, not an absent "
            "requirement. Commit a baseline produced by "
            "`scripts/update_grading_baseline.py`."
        )
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise BaselineError(f"{path.name} is not valid JSON: {exc}") from exc
    try:
        return GradingBaseline.model_validate(raw)
    except ValueError as exc:
        raise BaselineError(f"{path.name} is not a valid baseline: {exc}") from exc


def evaluate_regression(
    *,
    baseline: GradingBaseline,
    current_qwk: float,
    n_items: int,
    reference_label_source: str,
    reference_pass: str,
    reference_rubric_version: str,
    threshold: float = MAX_REGRESSION,
    min_items: int = MIN_ITEMS,
) -> RegressionVerdict:
    """Compare a fresh measurement with the baseline. Pure.

    Refuses, rather than passes, when the comparison would be meaningless:

    * **too few items** - a QWK over a handful of rows is noise, and a gate that
      passes on noise is worse than no gate;
    * **a different reference** - comparing a QWK measured against one label set
      with a baseline measured against another is not a regression test, it is
      two unrelated numbers subtracted;
    * **a provenance upgrade** - if the baseline was measured against model
      labels and the current run claims human ones, the right action is to
      commit a new human baseline, not to diff across the boundary.

    An *improvement* always passes. The gate is one-sided on purpose: it exists
    to stop quality falling, not to stop it rising.
    """
    if n_items < min_items:
        return RegressionVerdict(
            passed=False,
            current_qwk=current_qwk,
            baseline_qwk=baseline.qwk,
            regression=None,
            n_items=n_items,
            threshold=threshold,
            message=(
                f"REFUSED: the current measurement covers {n_items} paired items, below "
                f"the minimum of {min_items}. A QWK over fewer items than that is too "
                "unstable to gate on, so this is reported as a failure rather than a "
                "pass - an under-powered measurement is not evidence that nothing "
                "regressed."
            ),
        )

    if (
        reference_pass != baseline.reference_pass
        or reference_rubric_version != baseline.reference_rubric_version
    ):
        return RegressionVerdict(
            passed=False,
            current_qwk=current_qwk,
            baseline_qwk=baseline.qwk,
            regression=None,
            n_items=n_items,
            threshold=threshold,
            message=(
                "REFUSED: the current run was measured against reference "
                f"{reference_pass!r}/{reference_rubric_version!r} but the baseline was "
                f"measured against {baseline.reference_pass!r}/"
                f"{baseline.reference_rubric_version!r}. Subtracting those two numbers "
                "would not be a regression test. Re-measure against the baseline's "
                "reference, or commit a new baseline for the new one."
            ),
        )

    if reference_label_source != baseline.reference_label_source:
        return RegressionVerdict(
            passed=False,
            current_qwk=current_qwk,
            baseline_qwk=baseline.qwk,
            regression=None,
            n_items=n_items,
            threshold=threshold,
            message=(
                f"REFUSED: label provenance changed from "
                f"{baseline.reference_label_source!r} (baseline) to "
                f"{reference_label_source!r} (current). A model-referenced QWK and a "
                "human-referenced QWK are different measurements and the difference "
                "between them is not a regression. Commit a new baseline for the new "
                "provenance."
            ),
        )

    regression = baseline.qwk - current_qwk
    passed = regression <= threshold
    direction = "below" if regression > 0 else "above"
    verdict = "PASS" if passed else "FAIL"
    message = (
        f"{verdict}: grading QWK is {current_qwk:+.4f}, which is "
        f"{abs(regression):.4f} {direction} the committed baseline of "
        f"{baseline.qwk:+.4f} (baseline v{baseline.baseline_version}, "
        f"{baseline.prompt_version}/{baseline.prompt_fingerprint}, "
        f"{baseline.grader_model}). The gate fails at a regression of more than "
        f"{threshold}. Measured over {n_items} paired items against "
        f"{baseline.reference_artifact} ({baseline.reference_pass}, "
        f"label_source={baseline.reference_label_source})."
    )
    if not passed:
        message += (
            " If this change is an intended trade-off, move the baseline in a commit "
            "of its own with the reason in `notes` - do not relax the threshold."
        )
    if baseline.is_provisional:
        message += (
            " NOTE: this baseline is measured against MODEL-GENERATED provisional "
            "reference labels. It protects against regression and is NOT evidence "
            "for the Phase-4 acceptance gate, which requires human labels."
        )

    return RegressionVerdict(
        passed=passed,
        current_qwk=current_qwk,
        baseline_qwk=baseline.qwk,
        regression=regression,
        n_items=n_items,
        threshold=threshold,
        message=message,
        is_acceptance_evidence=not baseline.is_provisional,
    )


__all__ = [
    "MAX_REGRESSION",
    "MIN_ITEMS",
    "BaselineError",
    "GradingBaseline",
    "RegressionVerdict",
    "evaluate_regression",
    "load_baseline",
]
