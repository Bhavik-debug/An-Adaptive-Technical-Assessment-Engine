"""Provisional reference labels for the grading eval set - Day 19, Part 2.

Plan section 12.2 step 3 asks for 120 **human** 0-4 labels. There are none, and
this module does not pretend otherwise. What it produces is a third pass of
**autonomous, model-generated provisional reference labels**, recorded as such,
for *development* evaluation of the Day-19 validity harness.

**Read this before using a number computed from these labels.**

Day 18 produced two label passes, both by models of the same family
(``claude-opus-5`` and ``claude-opus-5[1m]``), the second of which had read the
dataset README and so knew the first pass's aggregate distribution and six
individual rows. Neither is human. Neither is independent of the other. This
module adds a third pass whose only real advantage is **isolation**: it runs as
a structured LLM call whose prompt contains the question, the answer and the
rubric, and *physically cannot* contain ``quality_intent``, the concept split,
or any previous label, because those strings are never put into the inputs.

That is a genuine improvement in procedure and **not** a promotion in evidential
status. A model label is not a human label, :func:`gate_eligible` does not look
at this field, and the Phase-4 QWK gate stays unmet.

The passes, and why there are two
---------------------------------

``blind``
    Question + answer + rubric. Nothing else. This is the **primary** reference
    label, and it is deliberately the one with the *least* information, because
    the grader being measured sees the expected-concept checklist. A reference
    label that also saw the checklist would share an information set with the
    thing it is scoring, and some of the resulting agreement would come from
    shared inputs rather than shared judgement.

``concept_aware``
    Question + answer + rubric + **the question's full expected-concept list**.
    A second evaluator, used to investigate disagreement and to measure how much
    seeing the checklist moves a label.

    Showing that list leaks nothing about the construction intent, and this is
    load-bearing rather than hopeful: every one of a question's five answers
    carries the *same* full concept set (``concepts_included |
    concepts_omitted == question.concept_keys``, asserted by
    ``validate_dataset`` and re-checked by a test here). The per-row
    *split* - which concepts were planted and which were suppressed - is what
    would leak ``quality_intent``, and it is never shown to either pass.

``adjudication``
    A third call, on rows where the two passes disagree materially. It sees both
    rationales and is asked to decide between them. It is **not** an independent
    third opinion - it is the same model family reading two of its own outputs -
    and it is recorded as an adjudication, not as a vote.

Rubric versioning
-----------------

:data:`LABEL_ANCHORS_V2` keeps all five of v1's rung descriptors **verbatim** and
adds one clause v1 left undefined. See :data:`ERROR_PRECEDENCE_V2`. Anchor text
that changed would make three passes incomparable; an additive tie-break only
changes rows where v1 was genuinely ambiguous.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.grading.evalset import LABEL_ANCHORS, MAX_LABEL, MIN_LABEL

#: Bumped whenever the labelling instrument changes in a way that could move a
#: score. Recorded on every row, because a label whose rubric version is unknown
#: cannot be compared with anything.
REFERENCE_RUBRIC_VERSION = "labels-v2"

#: The version the two Day-18 passes were collected under. Their rows do not
#: carry it - the field did not exist - so it is written down here instead.
LEGACY_RUBRIC_VERSION = "labels-v1"

#: Which information set a pass was given. Part of the label's provenance: two
#: passes that saw different inputs are not two readings of the same task.
ReferencePass = Literal["blind", "concept_aware", "adjudication"]

#: How correct the answer is, as a coarse judgement independent of the 0-4
#: score. Recorded separately because "partially correct" and "scored 2" are
#: different claims, and a row where they disagree is worth looking at.
CorrectnessAssessment = Literal[
    "correct",
    "mostly_correct",
    "partially_correct",
    "mostly_incorrect",
    "incorrect",
]

#: The clause v1 did not have, and the only substantive change in v2.
#:
#: **The ambiguity, as found by applying v1 to all 120 answers.** v1's anchor 0
#: says *"A confident answer that is wrong belongs here, not at 1 - confidence is
#: not partial credit"*; its anchor 2 says *"Partially correct. Some real content
#: and at least one substantive point, with obvious gaps or vagueness. Would need
#: follow-up before you trusted it."* Neither covers the case that actually
#: recurs in this dataset: an answer that is **detailed, specific and
#: substantially correct, and confidently wrong about one of the three things the
#: question asked**. Anchor 0 reads as too harsh (two thirds of the answer is
#: correct and useful), anchor 2 as too kind (a reader acting on the false claim
#: would build the wrong thing), and anchor 1 does not apply at all because it
#: describes *emptiness*, not partial wrongness.
#:
#: Three of 120 rows sat in that gap under v1 and were scored by judgement rather
#: than by the rubric, which is precisely the drift anchored scoring exists to
#: prevent.
#:
#: **Note the direction of this change.** Capping a detailed, half-correct answer
#: at 2 moves labels *down* and *away* from the 3-4 mass where 84% of this
#: dataset already sits. It makes the distribution slightly wider and the
#: grader's job harder, not easier. A rubric edit chosen to flatter a grader
#: would have gone the other way.
ERROR_PRECEDENCE_V2 = (
    "ERROR PRECEDENCE. Decide the rung first, then apply this cap.\n"
    "A substantive factual error is a claim that is false and that would lead "
    "someone who believed it to build the wrong thing or reach the wrong "
    "conclusion. Vagueness, omission, informality and redundancy are NOT "
    "substantive errors.\n"
    "- No substantive error: score on the rungs (1-4).\n"
    "- A substantive error in the claim the question principally asks for, or "
    "an answer that is wrong throughout: score 0. This holds however fluent, "
    "confident, long or well organised the answer is.\n"
    "- A substantive error confined to one part of a multi-part question, where "
    "the remaining parts are correct and substantive: cap the score at 2.\n"
    "- Omissions and vagueness with no false claims: score on the rungs; do not "
    "apply a cap."
)

#: v1's five rungs, unchanged, plus the precedence clause. The rung text is read
#: from :data:`LABEL_ANCHORS` rather than retyped, so the two versions cannot
#: drift apart by a copy-paste edit - a test asserts they are identical.
LABEL_ANCHORS_V2: dict[int, str] = dict(LABEL_ANCHORS)


def render_anchors_v2() -> str:
    """The v2 instrument as the text a labeller is actually shown."""
    lines = ["The 0-4 scale:", ""]
    for score in range(MIN_LABEL, MAX_LABEL + 1):
        lines.append(f"{score}. {LABEL_ANCHORS_V2[score]}")
    lines.extend(["", ERROR_PRECEDENCE_V2])
    return "\n".join(lines)


class ReferenceJudgment(BaseModel):
    """What the labelling model is asked to return. One answer, one judgement.

    Deliberately not just a score. The rationale and the error/omission lists
    are what make a label auditable: a bare 0-4 from a model is unreviewable,
    and the whole complaint about the Day-18 passes is that nobody can check
    them. These fields are also how :func:`needs_adjudication` finds rows where
    two passes disagree *about the reasons* and not only about the number.
    """

    model_config = ConfigDict(extra="forbid")

    score: int = Field(
        ge=MIN_LABEL,
        le=MAX_LABEL,
        description="The 0-4 rung, after applying the error-precedence cap.",
    )
    rationale: str = Field(
        min_length=20,
        max_length=1200,
        description="Why this score, citing what the answer does and does not do.",
    )
    correctness: CorrectnessAssessment
    factual_errors: list[str] = Field(
        default_factory=list,
        max_length=10,
        description="Claims in the answer judged false. Empty if none.",
    )
    omissions: list[str] = Field(
        default_factory=list,
        max_length=10,
        description="Things the question asked for that the answer does not supply.",
    )
    unsupported_claims: list[str] = Field(
        default_factory=list,
        max_length=10,
        description="Assertions made without the mechanism or evidence to back them.",
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="How sure the labeller is of this score.",
    )


class ReferenceLabel(BaseModel):
    """One persisted provisional reference label, with its full provenance.

    Stored in its own artefact (``reference_labels.jsonl``) rather than as new
    columns on ``answers.jsonl``, for two reasons. The dataset file is the
    *construction* record and has been stable since Day 18 - appending a third
    scoring pass to it would make its diff unreadable. And a label that carries
    a rationale, three evidence lists and a confidence does not fit a column.

    ``label_source`` is fixed to ``"model"`` and there is no way to set it to
    anything else. That is not an oversight.
    """

    model_config = ConfigDict(extra="forbid")

    answer_id: str = Field(pattern=r"^eval-\d{4}$")
    question_id: str

    judgment: ReferenceJudgment

    #: Which information set produced this label. See :data:`ReferencePass`.
    pass_name: ReferencePass
    #: The model that produced it, exactly as the provider reported it.
    evaluator_model: str
    #: The task the call was routed as, and the prompt template's version and
    #: content fingerprint. A prompt edited without a version bump shows up as
    #: the same version with a different fingerprint.
    task: str
    prompt_version: str
    prompt_fingerprint: str
    #: The labelling instrument. See :data:`REFERENCE_RUBRIC_VERSION`.
    rubric_version: str = REFERENCE_RUBRIC_VERSION
    #: Sampling settings that could move the output.
    temperature: float
    #: UTC ISO-8601, when the call returned. ``None`` only if the clock failed.
    labelled_at: str | None = None

    #: **Always ``"model"``.** These labels are autonomous and model-generated.
    #: Nothing here was annotated or verified by a person, and this field exists
    #: so that no later reader has to take that on trust from prose.
    label_source: Literal["model"] = "model"

    #: Set on a ``blind`` row when adjudication ran for it.
    adjudication: Literal["not_needed", "agreed", "revised"] | None = None
    #: What the adjudicator decided, when it ran.
    adjudicated_score: int | None = Field(default=None, ge=MIN_LABEL, le=MAX_LABEL)

    #: Retries the call needed, and the last failure if any. Recorded because a
    #: label that took four attempts is a different kind of datum from one that
    #: took none.
    attempts: int = 1
    last_error: str | None = None

    @property
    def score(self) -> int:
        """The usable score for this row: the adjudicated one when it exists."""
        return self.adjudicated_score if self.adjudicated_score is not None else self.judgment.score


# ---------------------------------------------------------------------------
# Reading and writing
# ---------------------------------------------------------------------------


def load_reference_labels(path: Path) -> list[ReferenceLabel]:
    """Read the artefact. Every row validates or the whole read fails."""
    out: list[ReferenceLabel] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            out.append(ReferenceLabel.model_validate_json(line))
        except ValueError as exc:
            raise ValueError(f"{path.name} line {number}: {exc}") from exc
    return out


def dump_reference_labels(labels: Iterable[ReferenceLabel], path: Path) -> int:
    """Write the artefact, sorted by (pass, answer id) so diffs stay readable."""
    ordered = sorted(labels, key=lambda label: (label.pass_name, label.answer_id))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            json.dumps(label.model_dump(mode="json"), sort_keys=True, ensure_ascii=False)
            for label in ordered
        )
        + "\n",
        encoding="utf-8",
    )
    return len(ordered)


def labels_by_answer(
    labels: Sequence[ReferenceLabel], *, pass_name: ReferencePass
) -> dict[str, ReferenceLabel]:
    """One pass's labels, keyed by answer id. Refuses duplicates."""
    out: dict[str, ReferenceLabel] = {}
    for label in labels:
        if label.pass_name != pass_name:
            continue
        if label.answer_id in out:
            raise ValueError(f"{label.answer_id} has two {pass_name!r} labels")
        out[label.answer_id] = label
    return out


# ---------------------------------------------------------------------------
# Quality control
# ---------------------------------------------------------------------------

#: Two passes differing by this much are worth a third look. One point of
#: disagreement on a 0-4 ordinal scale is ordinary rater noise; two is a
#: different reading of the answer.
ADJUDICATION_GAP = 2


def needs_adjudication(
    blind: dict[str, ReferenceLabel],
    concept_aware: dict[str, ReferenceLabel],
    *,
    gap: int = ADJUDICATION_GAP,
) -> list[str]:
    """Answer ids where the two passes disagree materially, sorted.

    Bounded on purpose: adjudicating all 120 would cost a third full pass to
    re-read 90 rows the two passes already agree on.
    """
    return sorted(
        answer_id
        for answer_id, label in blind.items()
        if answer_id in concept_aware
        and abs(label.judgment.score - concept_aware[answer_id].judgment.score) >= gap
    )


def reference_label_problems(
    labels: Sequence[ReferenceLabel],
    *,
    expected_answers: Sequence[str],
    pass_name: ReferencePass = "blind",
) -> list[str]:
    """Every way the label set could be unusable. Returns problems, never raises.

    Returned rather than raised so a report can print all of them at once; the
    script turns a non-empty list into a non-zero exit.
    """
    problems: list[str] = []
    wanted = set(expected_answers)

    seen: dict[str, int] = {}
    for label in labels:
        if label.pass_name != pass_name:
            continue
        seen[label.answer_id] = seen.get(label.answer_id, 0) + 1

    duplicates = sorted(aid for aid, count in seen.items() if count > 1)
    if duplicates:
        problems.append(f"{pass_name}: duplicate labels for {duplicates}")

    missing = sorted(wanted - set(seen))
    if missing:
        problems.append(f"{pass_name}: {len(missing)} answers have no label ({missing[:5]})")

    unknown = sorted(set(seen) - wanted)
    if unknown:
        problems.append(f"{pass_name}: labels for unknown answers {unknown[:5]}")

    for label in labels:
        if label.label_source != "model":
            problems.append(f"{label.answer_id}: label_source is not 'model'")
        if not MIN_LABEL <= label.score <= MAX_LABEL:
            problems.append(f"{label.answer_id}: score {label.score} is off the 0-4 scale")
        if label.rubric_version != REFERENCE_RUBRIC_VERSION:
            problems.append(
                f"{label.answer_id}: rubric_version {label.rubric_version!r} is not "
                f"{REFERENCE_RUBRIC_VERSION!r}; labels from two instruments are not comparable"
            )
    return problems


def score_distribution(labels: Sequence[ReferenceLabel]) -> dict[int, int]:
    """How many labels sit at each 0-4 score."""
    counts = dict.fromkeys(range(MIN_LABEL, MAX_LABEL + 1), 0)
    for label in labels:
        counts[label.score] += 1
    return counts


__all__ = [
    "ADJUDICATION_GAP",
    "ERROR_PRECEDENCE_V2",
    "LABEL_ANCHORS_V2",
    "LEGACY_RUBRIC_VERSION",
    "REFERENCE_RUBRIC_VERSION",
    "CorrectnessAssessment",
    "ReferenceJudgment",
    "ReferenceLabel",
    "ReferencePass",
    "by_intent",
    "dump_reference_labels",
    "labels_by_answer",
    "load_reference_labels",
    "needs_adjudication",
    "reference_label_problems",
    "render_anchors_v2",
    "central_tendency",
    "score_distribution",
    "uncertain_rows",
]


# ---------------------------------------------------------------------------
# Cross-pass quality control
# ---------------------------------------------------------------------------


def by_intent(
    labels: Sequence[ReferenceLabel],
    intents: dict[str, str],
) -> dict[str, dict[str, object]]:
    """Per ``quality_intent``: n, mean, median and the band histogram.

    ``intents`` maps answer id to its construction intent and is passed in
    rather than read from the dataset, because this module must not be able to
    see ``quality_intent`` on the path that *produces* a label. It is a
    reporting input, supplied after the fact.

    **This is a diagnostic, not a target.** A level whose labels sit on top of
    its neighbour's is not a distinct experimental condition, and the Day-18
    record already shows that for ``wrong_confident``. Nothing here should ever
    be used to adjust a label toward its intent.
    """
    out: dict[str, dict[str, object]] = {}
    grouped: dict[str, list[int]] = {}
    for row in labels:
        intent = intents.get(row.answer_id)
        if intent is None:
            continue
        grouped.setdefault(intent, []).append(row.score)
    for intent, scores in grouped.items():
        ordered = sorted(scores)
        out[intent] = {
            "n": len(ordered),
            "mean": sum(ordered) / len(ordered),
            "median": ordered[len(ordered) // 2],
            "min": ordered[0],
            "max": ordered[-1],
            "distribution": {
                band: sum(1 for value in ordered if value == band)
                for band in range(MIN_LABEL, MAX_LABEL + 1)
            },
        }
    return out


def uncertain_rows(
    labels: Sequence[ReferenceLabel], *, below: float = 0.7
) -> list[tuple[str, float, int]]:
    """Rows the labeller itself was unsure about, worst first.

    Reported because a label carrying a confidence of 0.55 and a label carrying
    0.95 are not equally good evidence, and an aggregate that treats them alike
    hides which rows a reader should go and check by hand.
    """
    return sorted(
        (
            (row.answer_id, row.judgment.confidence, row.score)
            for row in labels
            if row.judgment.confidence < below
        ),
        key=lambda item: item[1],
    )


def central_tendency(labels: Sequence[ReferenceLabel]) -> dict[str, object]:
    """How concentrated the labels are, and on which band.

    A labeller that puts most items in one band has very little to say about
    the ordering of those items, and a quadratically weighted kappa computed
    against it is close to degenerate however good the thing being measured is.
    That failure is invisible in the kappa itself, so it is measured here and
    reported next to it.
    """
    distribution = score_distribution(labels)
    total = sum(distribution.values())
    if not total:
        return {"n": 0}
    modal_band, modal_count = max(distribution.items(), key=lambda item: item[1])
    populated = [band for band, count in distribution.items() if count]
    return {
        "n": total,
        "distribution": distribution,
        "modal_band": modal_band,
        "modal_share": modal_count / total,
        "populated_bands": len(populated),
        "effective_range": (min(populated), max(populated)),
    }
