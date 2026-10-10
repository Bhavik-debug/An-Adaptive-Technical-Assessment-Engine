"""Evidence-span verification - plan section 7.2, and section 7.4 technique 2.

Plan section 7.2 calls this the highest-leverage line in the whole prompt, and
then says what to do about it:

    you verify in code that the span is actually a substring of the candidate's
    answer. A fabricated span fails the check, the concept is downgraded to
    ``absent``, and the incident is logged. That is a free, deterministic,
    zero-cost hallucination detector.

Three words in that sentence decide the whole design.

**"in code"** - no model is asked whether a quotation looks genuine. The check
is ``in``. It cannot be talked out of a verdict, it costs nothing, and it
returns the same answer on every machine forever.

**"downgraded"** - not rejected. This is the opposite policy to the Day-16
contract, and deliberately so. A judgment that claims ``covered`` with a null
span has *broken the contract*, and the chokepoint's repair retry is the right
response because the model can fix it. A judgment that claims ``covered`` and
quotes text that was never in the answer has *followed* the contract and lied
inside it; retrying invites a better-disguised lie, so the credit is removed and
the rest of the judgment is kept. Every other concept in that judgment may be
perfectly sound, and throwing them away would make a hallucination detector into
a hallucination amplifier.

**"logged"** - because the fabrication rate is an operational metric. A grader
that quietly downgrades 30% of its own spans is a broken grader, and the only
way anyone finds that out is if each incident is written down. The log line
carries the concept key and lengths, **never the span or the answer**: plan
section 14.1 puts candidate text out of the logs, and a fabricated span is still
a string the model built out of a candidate's words.

What is *not* verified here, because the plan does not ask for it: the ``claim``
strings in ``errors[]`` are the model's own paraphrase of a mistake rather than a
quotation, and ``misconceptions_hit`` is already constrained to a supplied list
by ``check_judgment``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import StrEnum

from app.grading.schema import (
    AnswerError,
    ConceptJudgment,
    ConceptLabel,
    ErrorSeverity,
    GradeJudgment,
)

log = logging.getLogger(__name__)

#: Where a fabricated concept lands. Plan section 7.2 says ``absent``, not
#: ``partial``: an unevidenced claim is no evidence at all, and half credit for
#: a fabrication would leave a model strictly better off inventing a span than
#: admitting a gap - which is precisely the incentive this check exists to
#: remove.
DOWNGRADE_TO: ConceptLabel = "absent"


class EvidenceMatch(StrEnum):
    """How a concept's span related to the answer it claimed to quote."""

    #: The span is a literal substring. The ordinary case.
    EXACT = "exact"
    #: The span matches once runs of whitespace are collapsed on both sides.
    #: See :func:`verify_span` for why this is allowed and why it is counted
    #: separately rather than folded into ``EXACT``.
    WHITESPACE_NORMALISED = "whitespace_normalised"
    #: The span does not appear in the answer. Credit is removed.
    FABRICATED = "fabricated"
    #: The concept was ``absent``, so the contract required no span. Not a pass
    #: and not a failure - there was nothing to check.
    NOT_REQUIRED = "not_required"

    @property
    def is_credited(self) -> bool:
        """True when the concept keeps the label the model gave it."""
        return self in (
            EvidenceMatch.EXACT,
            EvidenceMatch.WHITESPACE_NORMALISED,
            EvidenceMatch.NOT_REQUIRED,
        )


@dataclass(frozen=True, slots=True)
class EvidenceCheck:
    """What happened to one concept's span."""

    key: str
    claimed: ConceptLabel
    final: ConceptLabel
    match: EvidenceMatch

    @property
    def was_downgraded(self) -> bool:
        return self.claimed != self.final


@dataclass(frozen=True, slots=True)
class ErrorCheck:
    """What happened to one flagged error's claim - Day 20.

    The same treatment ``EvidenceCheck`` gives a concept span, for the same
    reason. ``AnswerError.claim`` is documented as "the incorrect claim, **as
    the candidate made it**", so it is a quotation, and a quotation that is not
    in the answer is a fabrication.

    This matters arithmetically rather than cosmetically: a ``major`` error
    removes 0.15 from the score and a ``minor`` 0.05 (plan section 7.2), so an
    invented claim is a real penalty applied for something the candidate never
    said.
    """

    claim_chars: int
    severity: ErrorSeverity
    match: EvidenceMatch

    @property
    def was_dropped(self) -> bool:
        return not self.match.is_credited


@dataclass(frozen=True, slots=True)
class VerifiedJudgment:
    """A judgment after verification, with the original kept beside it.

    Both are returned because they answer different questions. ``judgment`` is
    what gets scored. ``original`` is what the model actually said, and a report
    that showed only the corrected version would make a fabrication invisible to
    the one person - the candidate - who could dispute it.
    """

    judgment: GradeJudgment
    original: GradeJudgment
    checks: tuple[EvidenceCheck, ...]
    #: One per flagged error, in the order the model returned them. Empty when
    #: the model flagged no errors, which is the common case.
    error_checks: tuple[ErrorCheck, ...] = ()

    @property
    def downgraded_keys(self) -> tuple[str, ...]:
        return tuple(check.key for check in self.checks if check.was_downgraded)

    @property
    def fabrication_count(self) -> int:
        return sum(1 for check in self.checks if check.match is EvidenceMatch.FABRICATED)

    @property
    def any_fabricated(self) -> bool:
        return self.fabrication_count > 0

    @property
    def dropped_error_count(self) -> int:
        """Flagged errors whose claim was not in the answer - Day 20.

        Counted separately from :attr:`fabrication_count` rather than folded
        into it, because the two have opposite signs for the candidate: a
        fabricated *span* removes credit the model tried to give, a fabricated
        *claim* removes a penalty the model tried to apply. A single "the model
        made something up" counter would net them off and hide both.
        """
        return sum(1 for check in self.error_checks if check.was_dropped)

    @property
    def dropped_penalty(self) -> float:
        """How much score the dropped claims would have cost. Reporting only.

        Reads the same two constants the scorer uses, so a report cannot quote a
        penalty magnitude that disagrees with the arithmetic that was applied.
        """
        from app.grading.scoring import MAJOR_ERROR_PENALTY, MINOR_ERROR_PENALTY

        return sum(
            MAJOR_ERROR_PENALTY if check.severity == "major" else MINOR_ERROR_PENALTY
            for check in self.error_checks
            if check.was_dropped
        )


def _collapse_whitespace(text: str) -> str:
    return " ".join(text.split())


def verify_span(span: str | None, answer: str) -> EvidenceMatch:
    """Classify one span against the answer. Pure, and case-sensitive.

    The rule is ``span in answer``. Two deliberate refinements, and the reasoning
    for each matters more than the code:

    **Whitespace is normalised as a documented second attempt.** The answer
    reaches the model inside a fenced block and comes back re-quoted; a model
    that re-wraps a line, or collapses a newline to a space, has not fabricated
    anything. Collapsing runs of whitespace on *both* sides cannot invent a
    match - it preserves token order and token content exactly, and only forgives
    the spacing between them. It is reported as
    :attr:`EvidenceMatch.WHITESPACE_NORMALISED` rather than as ``EXACT`` so that
    the rate is measurable: if Day 19 finds the fallback firing often, that is a
    fact about the prompt worth knowing, not something to have buried.

    **Case is not folded, and punctuation is not stripped.** Both would be
    unsafe in the way that matters: they widen what counts as a match without
    any argument that the widened set is still honest quotation. "redis is acid
    compliant" is not the same sentence as "Redis is ACID compliant" in a
    technical answer, and a span that needs its punctuation removed to match is a
    paraphrase. The plan says substring; this stays as close to that as line
    wrapping allows.
    """
    if span is None or not span.strip():
        # Unreachable through the contract - `ConceptJudgment` refuses a blank
        # span on a credited label - but this function is public and pure, and a
        # blank quotation is not evidence under any reading.
        return EvidenceMatch.FABRICATED
    if span in answer:
        return EvidenceMatch.EXACT
    if _collapse_whitespace(span) in _collapse_whitespace(answer):
        return EvidenceMatch.WHITESPACE_NORMALISED
    return EvidenceMatch.FABRICATED


def verify_evidence(judgment: GradeJudgment, *, answer_text: str) -> VerifiedJudgment:
    """Check every credited span, downgrade the fabrications, log them.

    ``answer_text`` must be the string the model was actually shown - that is,
    :func:`app.grading.grader.graded_answer_text`, not the caller's raw input.
    The pipeline passes the right one; a caller passing the raw answer would get
    spurious fabrications on any answer that contained a fence sentinel.

    Pure apart from the log line. Returns a new object; the input judgment is
    frozen and is never mutated.
    """
    checks: list[EvidenceCheck] = []
    concepts: list[ConceptJudgment] = []

    for concept in judgment.concepts:
        if concept.label == "absent":
            checks.append(
                EvidenceCheck(
                    key=concept.key,
                    claimed=concept.label,
                    final=concept.label,
                    match=EvidenceMatch.NOT_REQUIRED,
                )
            )
            concepts.append(concept)
            continue

        match = verify_span(concept.evidence, answer_text)
        if match.is_credited:
            checks.append(
                EvidenceCheck(
                    key=concept.key, claimed=concept.label, final=concept.label, match=match
                )
            )
            concepts.append(concept)
            continue

        # Fabricated. Drop the label to `absent`, which the contract requires
        # to carry a null span - so the corrected judgment is still a valid one
        # and can be re-validated, stored or re-scored like any other.
        log.warning(
            "grading evidence span not found in the answer; downgrading concept to %s",
            DOWNGRADE_TO,
            extra={
                "grading.concept_key": concept.key,
                "grading.claimed_label": concept.label,
                # Lengths, never text: plan section 14.1 keeps candidate content
                # out of the logs, and a fabricated span is still built from it.
                "grading.span_chars": len(concept.evidence or ""),
                "grading.answer_chars": len(answer_text),
            },
        )
        checks.append(
            EvidenceCheck(
                key=concept.key,
                claimed=concept.label,
                final=DOWNGRADE_TO,
                match=EvidenceMatch.FABRICATED,
            )
        )
        concepts.append(ConceptJudgment(key=concept.key, label=DOWNGRADE_TO, evidence=None))

    # Day 20: the same substring check, applied to the error list.
    #
    # **Why this was missing and why it mattered.** Day 16 built the error list
    # and Day 17 built the penalty that reads it; nothing ever checked that a
    # flagged claim was in the answer. Day 19's length-bias probe is what
    # surfaced it: appending 207 words of contentless filler made the grader
    # flag *major* errors on two answers whose substantive text had not
    # changed, costing 0.15 each. Both claims were quotations of sentences the
    # candidate never wrote - "wrapping the two statements in a transaction is
    # by itself sufficient" and "equal counts of openers and closers means the
    # string is balanced". Measured over the probe's 24 pairs, 5 of 8 flagged
    # claims were not in the text they were flagged against, including both
    # majors.
    #
    # A dropped error is *not* a judgement that the answer is correct. It is a
    # refusal to penalise a candidate for a sentence that cannot be shown to
    # them, which is the same standard ``verify_span`` already applies to
    # credit. Credit and penalty are now symmetric: both require a quotation
    # that exists.
    error_checks: list[ErrorCheck] = []
    errors: list[AnswerError] = []
    for error in judgment.errors:
        match = verify_span(error.claim, answer_text)
        error_checks.append(
            ErrorCheck(claim_chars=len(error.claim), severity=error.severity, match=match)
        )
        if match.is_credited:
            errors.append(error)
            continue
        log.warning(
            "grading error claim not found in the answer; dropping the penalty",
            extra={
                # Lengths and severity, never the claim text: plan section 14.1
                # keeps candidate content out of the logs, and a fabricated
                # claim is still assembled from it.
                "grading.error_severity": error.severity,
                "grading.claim_chars": len(error.claim),
                "grading.answer_chars": len(answer_text),
            },
        )

    corrected = judgment.model_copy(update={"concepts": concepts, "errors": errors})
    return VerifiedJudgment(
        judgment=corrected,
        original=judgment,
        checks=tuple(checks),
        error_checks=tuple(error_checks),
    )


__all__ = [
    "DOWNGRADE_TO",
    "ErrorCheck",
    "EvidenceCheck",
    "EvidenceMatch",
    "VerifiedJudgment",
    "verify_evidence",
    "verify_span",
]
