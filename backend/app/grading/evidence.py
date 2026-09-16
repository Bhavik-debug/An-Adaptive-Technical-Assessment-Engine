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

from app.grading.schema import ConceptJudgment, ConceptLabel, GradeJudgment

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

    @property
    def downgraded_keys(self) -> tuple[str, ...]:
        return tuple(check.key for check in self.checks if check.was_downgraded)

    @property
    def fabrication_count(self) -> int:
        return sum(1 for check in self.checks if check.match is EvidenceMatch.FABRICATED)

    @property
    def any_fabricated(self) -> bool:
        return self.fabrication_count > 0


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

    corrected = judgment.model_copy(update={"concepts": concepts})
    return VerifiedJudgment(judgment=corrected, original=judgment, checks=tuple(checks))


__all__ = [
    "DOWNGRADE_TO",
    "EvidenceCheck",
    "EvidenceMatch",
    "VerifiedJudgment",
    "verify_evidence",
    "verify_span",
]
