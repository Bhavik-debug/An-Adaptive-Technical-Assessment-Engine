"""Pass C - the arithmetic, in code. Plan section 7.2.

This module is the answer to the question the whole project is built around:
*who computes the score?* Code does. Always. Every time. A model's output
reaches this file as **evidence**, and leaves it as a number the model never saw
and could not have influenced arithmetically.

**This module is pure.** No database, no Redis, no LLM, no clock, no filesystem,
no global mutable state - the same discipline ``app/ability.py`` follows, and for
the same reason: the two deterministic cores of this system are the parts an
examiner will actually interrogate, and a core you can only exercise through a
live provider is a core nobody will exercise.

The formula, exactly as plan section 7.2 writes it::

    COVER = {"covered": 1.0, "partial": 0.5, "absent": 0.0}

    concept_score = sum(w[c] * COVER[label[c]] for c in concepts) / sum(w.values())
    rubric_score  = sum(rubric.values()) / (4 * len(rubric))
    penalty       = min(0.30, 0.15 * n_major + 0.05 * n_minor)
    raw           = 0.65 * concept_score + 0.35 * rubric_score - penalty
    score         = clamp(raw, 0.0, 1.0)

Not a formula *like* that one, and not one improved on: the constants below are
transcribed from the plan and none was chosen here. Plan section 7.2 is explicit
that 0.65/0.35 is a judgment call to be **tuned against the labelled set** and
reported as having been tuned - which is Day 19's job, against a metric, and not
something to guess at now. They are named constants so that sweep is a parameter
change rather than an edit.

**Why the weights are read from the bank and never from the judgment.** The
model is never shown a concept's weight (see ``render_expected_concepts``), and
the judgment has nowhere to put one. Weights arrive here from
``ExpectedConcept``, which is reviewed, committed and diffable. The model says
*what it saw*; this file decides *what that was worth*.

**Order independence, concretely.** Everything below indexes by concept **key**,
never by list position. ``labels_by_key`` is a mapping and ``sum()`` over a
mapping is order-free in exact arithmetic - and, because every product is a
weight times one of three exact binary fractions (1.0, 0.5, 0.0), it is order-free
in IEEE-754 floating point too, which is the stronger claim the Phase-4
position-bias gate actually needs.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from app.bank.schema import ExpectedConcept
from app.grading.rubric import MAX_LEVEL
from app.grading.schema import ConceptLabel, GradeJudgment

#: Plan section 7.2. What each classification is worth, before weighting.
#: "partial" is exactly half: the honest middle, and an exact binary fraction,
#: so no ordering of the sum can change the total by a float ulp.
COVER: Mapping[ConceptLabel, float] = {
    "covered": 1.0,
    "partial": 0.5,
    "absent": 0.0,
}

#: Plan section 7.2. Concept coverage is the objective signal and takes the
#: majority share; the rubric catches what a checklist cannot see. The split is
#: a judgment call the plan says to tune against the labelled set (Day 19) and
#: to report as tuned - so it is a constant with a name, not a literal in an
#: expression.
CONCEPT_WEIGHT = 0.65
RUBRIC_WEIGHT = 0.35

#: Plan section 7.2's penalty term. Two magnitudes because the contract offers
#: two severities, and a ceiling so that a long list of small mistakes cannot
#: drive an otherwise-good answer to zero - the concept checklist, not the
#: penalty, is meant to be what decides a score.
MAJOR_ERROR_PENALTY = 0.15
MINOR_ERROR_PENALTY = 0.05
MAX_PENALTY = 0.30

#: The score's own bounds. It lands on the ``score`` argument of
#: ``app.ability.update_ability``, which rejects anything outside [0, 1].
MIN_SCORE = 0.0
MAX_SCORE = 1.0


class ScoringError(ValueError):
    """The judgment and the question do not describe the same item.

    ``check_judgment`` normally catches this one call earlier. It is re-checked
    here because :func:`score_answer` is a public, independently callable
    function, and a scorer that silently ignored a concept it had no weight for
    would return a number that looked fine and meant nothing.
    """


@dataclass(frozen=True, slots=True)
class ScoreBreakdown:
    """The score, and every intermediate that produced it.

    Returned instead of a bare float for three reasons, in increasing order of
    importance: the Day-25 report needs the components; the Day-19 weight sweep
    needs ``concept_score`` and ``rubric_score`` separately; and **the confidence
    gate's trigger is defined on the two sub-scores** (plan section 7.3), so a
    scorer that returned only the total would force the gate to recompute them
    and there would immediately be two implementations of one formula.

    It is also the audit trail. "Why did this answer score 0.50?" is answerable
    from this object alone, without re-running anything.
    """

    concept_score: float
    rubric_score: float
    penalty: float
    #: Before clamping. Kept because ``raw != score`` is exactly the signal that
    #: a bound did something, and a clamp that leaves no trace is a clamp nobody
    #: notices firing on 30% of answers.
    raw: float
    score: float
    major_errors: int
    minor_errors: int
    #: Denominator of ``concept_score``. Makes the arithmetic reproducible from
    #: the breakdown alone.
    total_concept_weight: int

    @property
    def was_clamped(self) -> bool:
        return self.raw != self.score

    @property
    def signal_disagreement(self) -> float:
        """``|concept_score - rubric_score|`` - the gate's first trigger."""
        return abs(self.concept_score - self.rubric_score)


def concept_coverage(
    judgment: GradeJudgment, expected_concepts: Sequence[ExpectedConcept]
) -> tuple[float, int]:
    """Weighted concept coverage in [0, 1], and the total weight it divided by.

    Plan section 7.2::

        concept_score = sum(w[c] * COVER[label[c]]) / sum(w.values())

    Indexed by key throughout. The judgment's concept set must equal the
    question's exactly - not a subset, not a superset - because a concept the
    model did not classify is not a zero, it is a missing observation, and
    scoring it as absent would silently reward a model for skipping the concepts
    it found hardest.
    """
    weights = {concept.key: concept.weight for concept in expected_concepts}
    if len(weights) != len(expected_concepts):
        raise ScoringError("expected_concepts contains duplicate keys")

    labels = judgment.labels_by_key
    if set(labels) != set(weights):
        missing = sorted(set(weights) - set(labels))
        unexpected = sorted(set(labels) - set(weights))
        raise ScoringError(
            "the judgment does not classify exactly the question's concepts: "
            f"missing {missing}, unexpected {unexpected}"
        )

    total_weight = sum(weights.values())
    if total_weight <= 0:  # pragma: no cover - ExpectedConcept bounds weight to 1-3
        raise ScoringError("total concept weight is zero; nothing could be scored")

    earned = sum(weights[key] * COVER[label] for key, label in labels.items())
    return earned / total_weight, total_weight


def rubric_blend(judgment: GradeJudgment) -> float:
    """The four anchored levels, normalised to [0, 1].

    Plan section 7.2: ``sum(rubric.values()) / (4 * len(rubric))``. The 4 is
    ``MAX_LEVEL`` - the top of the anchored scale - and it is imported rather
    than written as a literal so that adding a fifth *level* and adding a fifth
    *dimension* stay two different edits, only one of which touches this line.
    """
    levels = judgment.rubric.as_mapping()
    return sum(levels.values()) / (MAX_LEVEL * len(levels))


def error_penalty(judgment: GradeJudgment) -> tuple[float, int, int]:
    """The penalty, and the counts behind it.

    Plan section 7.2: ``min(0.30, 0.15 * n_major + 0.05 * n_minor)``.

    An omission is not an error - a missing concept is already priced as
    ``absent`` in ``concept_score``, and charging for it twice would make a
    silent answer score below a wrong one for no defensible reason.
    """
    major = sum(1 for error in judgment.errors if error.severity == "major")
    minor = len(judgment.errors) - major
    uncapped = MAJOR_ERROR_PENALTY * major + MINOR_ERROR_PENALTY * minor
    return min(MAX_PENALTY, uncapped), major, minor


def clamp(value: float, low: float = MIN_SCORE, high: float = MAX_SCORE) -> float:
    """Bound a value. Named, because ``score = clamp(raw, 0, 1)`` is in the plan."""
    return max(low, min(high, value))


def score_answer(
    judgment: GradeJudgment,
    *,
    expected_concepts: Sequence[ExpectedConcept],
) -> ScoreBreakdown:
    """Turn a validated judgment into a number. Plan section 7.2, Pass C.

    Pure and total: given the same judgment and the same concepts it returns the
    same breakdown, on any machine, with no provider, no network and no clock.
    Raises :class:`ScoringError` only when the two arguments describe different
    questions.

    **It does not verify evidence spans.** That is
    :func:`app.grading.evidence.verify_evidence`, deliberately a separate
    function: verification *corrects* a judgment and scoring *reads* one, and
    keeping them apart is what lets a test score a hand-built judgment without
    also owning an answer to check it against. The pipeline runs verification
    first, so the judgment that reaches here is already corrected.
    """
    concept_score, total_weight = concept_coverage(judgment, expected_concepts)
    rubric_score = rubric_blend(judgment)
    penalty, major, minor = error_penalty(judgment)

    raw = CONCEPT_WEIGHT * concept_score + RUBRIC_WEIGHT * rubric_score - penalty
    return ScoreBreakdown(
        concept_score=concept_score,
        rubric_score=rubric_score,
        penalty=penalty,
        raw=raw,
        score=clamp(raw),
        major_errors=major,
        minor_errors=minor,
        total_concept_weight=total_weight,
    )


__all__ = [
    "CONCEPT_WEIGHT",
    "COVER",
    "MAJOR_ERROR_PENALTY",
    "MAX_PENALTY",
    "MAX_SCORE",
    "MINOR_ERROR_PENALTY",
    "MIN_SCORE",
    "RUBRIC_WEIGHT",
    "ScoreBreakdown",
    "ScoringError",
    "clamp",
    "concept_coverage",
    "error_penalty",
    "rubric_blend",
    "score_answer",
]
