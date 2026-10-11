"""The follow-up policy - plan section 8.5: decision in code, wording in the model.

::

    if item.followups_used >= 2:        no
    if state.followups_used >= 6:       no
    any absent concept with weight >= 3 -> concept_absent   (target: that concept)
    any major factual error            -> factual_error
    0.45 <= score <= 0.65              -> ambiguous_band
    RD of the item's subtopic > 0.7    -> high_uncertainty

The first rule that fires is the reason, in exactly that order (the plan's
order).  **Every input is computed by code** - labels after Day-17 evidence
verification, error claims after Day-20 error verification, the code-side
score, RD after Day-11's update - so nothing a model writes as free text can
decide whether a follow-up happens, and nothing the candidate writes can either:
the answer text is not an input to this function at all.

Thresholds: every number here is the plan's (section 8.5), not tuned:

* ``weight >= 3`` - the top of the bank's 1-3 concept weight scale
  (``app.bank.schema.ExpectedConcept``): only the concepts an item calls
  essential are worth a probe of their own.
* ``0.45 <= score <= 0.65`` - the middle of the score range, "the score we're
  least sure about": an answer there is consistent with both "knows it" and
  "doesn't", so one more observation is worth the most.  Inclusive at both ends,
  as written.
* ``RD > 0.7`` - strict.  Note the measured consequence, pinned by a test: with
  ``RD_MAX = 1.3`` the Day-11 update leaves RD above 0.7 for the first five
  observations of a subtopic (1.09, 0.957, 0.863, 0.793, 0.737 at p = 0.5;
  0.691 at the sixth), so this rule fires on nearly every early item and
  the caps, not the rule, decide how many follow-ups happen early in an
  interview.  That is the plan's rule taken at its word; retuning it is a
  calibration question for Phase 9, not something to change silently here.

**Missing information is never evidence.**  Each rule reports one of FIRED,
NOT_FIRED or UNAVAILABLE.  A concept with no label, a weight outside 1-3, a
score or RD that is missing or non-finite, a judgment that names a concept the
item does not have - each makes the *relevant* rule UNAVAILABLE, which can
never fire.  So a broken or partial grade produces "no follow-up, insufficient
evidence" rather than a probe that implies a poor answer.

The decision carries a trace - every rule's status and the inputs it saw, as
keys and numbers only, never the answer - which the worker stores on the turn.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from app.interview.state import MAX_FOLLOW_UPS_PER_ITEM, MAX_FOLLOW_UPS_PER_SESSION

RULE_CONCEPT_ABSENT = "concept_absent"
RULE_FACTUAL_ERROR = "factual_error"
RULE_AMBIGUOUS_BAND = "ambiguous_band"
RULE_HIGH_UNCERTAINTY = "high_uncertainty"
#: The plan's order.  The first rule that fires is the reason.
RULES = (RULE_CONCEPT_ABSENT, RULE_FACTUAL_ERROR, RULE_AMBIGUOUS_BAND, RULE_HIGH_UNCERTAINTY)

CAP_ITEM = "item_cap_reached"
CAP_SESSION = "session_cap_reached"

HIGH_WEIGHT = 3
AMBIGUOUS_LOW = 0.45
AMBIGUOUS_HIGH = 0.65
RD_FOLLOW_UP_THRESHOLD = 0.7

_VALID_LABELS = frozenset({"covered", "partial", "absent"})
_VALID_SEVERITIES = frozenset({"major", "minor"})


class RuleStatus(StrEnum):
    FIRED = "fired"
    NOT_FIRED = "not_fired"
    #: The input this rule needs is missing or invalid.  Never fires.
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True, slots=True)
class ConceptEvidence:
    """One expected concept: its bank weight and the grader's verified label.

    ``label is None`` means the grade said nothing about this concept, which is
    *not* the same as "absent".
    """

    key: str
    weight: int | None
    label: str | None


@dataclass(frozen=True, slots=True)
class PolicyInput:
    concepts: tuple[ConceptEvidence, ...] | None
    #: Severities of the *verified* error claims; ``None`` = no judgment at all.
    error_severities: tuple[str, ...] | None
    score: float | None
    #: RD of the item's subtopic *after* this grade's update.
    rd: float | None
    item_follow_ups: int
    session_follow_ups: int
    #: Concept keys the judgment labelled that the item does not have.  Present
    #: only to be reported: a contradictory grade is evidence of nothing.
    unknown_concepts: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RuleOutcome:
    rule: str
    status: RuleStatus
    detail: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class FollowUpDecision:
    follow_up: bool
    reason: str | None
    #: The concept the probe should give the candidate a chance to show.
    concept_key: str | None
    outcomes: tuple[RuleOutcome, ...]
    #: Set when a cap, not the rules, said no.
    capped_by: str | None = None

    def trace(self) -> dict[str, Any]:
        """JSON-ready.  Keys and numbers only - safe to store and to log."""
        return {
            "follow_up": self.follow_up,
            "reason": self.reason,
            "concept_key": self.concept_key,
            "capped_by": self.capped_by,
            "rules": [
                {"rule": o.rule, "status": o.status.value, **dict(o.detail)} for o in self.outcomes
            ],
        }


# ---------------------------------------------------------------------------
# Building the input
# ---------------------------------------------------------------------------


def policy_input(
    *,
    judgment: Mapping[str, Any] | None,
    expected: Sequence[tuple[str, int]],
    score: float | None,
    rd: float | None,
    item_follow_ups: int,
    session_follow_ups: int,
) -> PolicyInput:
    """Join a stored judgment (``GradeJudgment.model_dump()``) to the item's
    concept weights, tolerating anything malformed by marking it unavailable.

    ``expected`` is ``(key, weight)`` in the bank's order - which is also the
    tie-break order when two absent concepts share the top weight.
    """
    if not isinstance(judgment, Mapping):
        return PolicyInput(
            concepts=None,
            error_severities=None,
            score=score,
            rd=rd,
            item_follow_ups=item_follow_ups,
            session_follow_ups=session_follow_ups,
        )

    labels: dict[str, str | None] = {}
    raw_concepts = judgment.get("concepts")
    if isinstance(raw_concepts, Sequence) and not isinstance(raw_concepts, str):
        for entry in raw_concepts:
            if isinstance(entry, Mapping) and isinstance(entry.get("key"), str):
                label = entry.get("label")
                labels[entry["key"]] = label if label in _VALID_LABELS else None

    expected_keys = {key for key, _ in expected}
    concepts = tuple(
        ConceptEvidence(
            key=key,
            weight=weight if isinstance(weight, int) and 1 <= weight <= 3 else None,
            label=labels.get(key),
        )
        for key, weight in expected
    )

    severities: tuple[str, ...] | None
    raw_errors = judgment.get("errors")
    if isinstance(raw_errors, Sequence) and not isinstance(raw_errors, str):
        severities = tuple(
            e["severity"]
            for e in raw_errors
            if isinstance(e, Mapping) and e.get("severity") in _VALID_SEVERITIES
        )
    else:
        severities = None

    return PolicyInput(
        concepts=concepts,
        error_severities=severities,
        score=score,
        rd=rd,
        item_follow_ups=item_follow_ups,
        session_follow_ups=session_follow_ups,
        unknown_concepts=tuple(sorted(set(labels) - expected_keys)),
    )


# ---------------------------------------------------------------------------
# The rules
# ---------------------------------------------------------------------------


def _concept_absent(inp: PolicyInput) -> tuple[RuleOutcome, str | None]:
    if inp.concepts is None or inp.unknown_concepts:
        detail = {"unknown_concepts": list(inp.unknown_concepts)} if inp.unknown_concepts else {}
        return RuleOutcome(RULE_CONCEPT_ABSENT, RuleStatus.UNAVAILABLE, detail), None
    high = [c for c in inp.concepts if c.weight is not None and c.weight >= HIGH_WEIGHT]
    if not high or all(c.label is None for c in high):
        return RuleOutcome(
            RULE_CONCEPT_ABSENT, RuleStatus.UNAVAILABLE, {"high_weight_labelled": 0}
        ), None
    absent = [c for c in high if c.label == "absent"]
    if absent:
        target = absent[0]  # bank order breaks ties; all have the top weight
        return RuleOutcome(
            RULE_CONCEPT_ABSENT, RuleStatus.FIRED, {"concept_key": target.key}
        ), target.key
    return RuleOutcome(RULE_CONCEPT_ABSENT, RuleStatus.NOT_FIRED), None


def _factual_error(inp: PolicyInput) -> RuleOutcome:
    if inp.error_severities is None:
        return RuleOutcome(RULE_FACTUAL_ERROR, RuleStatus.UNAVAILABLE)
    majors = sum(1 for s in inp.error_severities if s == "major")
    status = RuleStatus.FIRED if majors else RuleStatus.NOT_FIRED
    return RuleOutcome(RULE_FACTUAL_ERROR, status, {"major_errors": majors})


def _ambiguous_band(inp: PolicyInput) -> RuleOutcome:
    s = inp.score
    if s is None or not isinstance(s, int | float) or not math.isfinite(s) or not 0 <= s <= 1:
        return RuleOutcome(RULE_AMBIGUOUS_BAND, RuleStatus.UNAVAILABLE)
    fired = AMBIGUOUS_LOW <= s <= AMBIGUOUS_HIGH
    status = RuleStatus.FIRED if fired else RuleStatus.NOT_FIRED
    return RuleOutcome(RULE_AMBIGUOUS_BAND, status, {"score": round(float(s), 4)})


def _high_uncertainty(inp: PolicyInput) -> RuleOutcome:
    rd = inp.rd
    if rd is None or not isinstance(rd, int | float) or not math.isfinite(rd) or rd <= 0:
        return RuleOutcome(RULE_HIGH_UNCERTAINTY, RuleStatus.UNAVAILABLE)
    status = RuleStatus.FIRED if rd > RD_FOLLOW_UP_THRESHOLD else RuleStatus.NOT_FIRED
    return RuleOutcome(RULE_HIGH_UNCERTAINTY, status, {"rd": round(float(rd), 4)})


def _weakest_concept(inp: PolicyInput) -> str | None:
    """For a probe whose rule names no concept: the highest-weight concept the
    grader labelled absent, else partial.  ``None`` if all were covered (the
    probe then asks for depth on the question as a whole)."""
    if inp.concepts is None:
        return None
    for wanted in ("absent", "partial"):
        candidates = [c for c in inp.concepts if c.label == wanted and c.weight is not None]
        if candidates:
            return max(candidates, key=lambda c: c.weight or 0).key
    return None


def decide(inp: PolicyInput) -> FollowUpDecision:
    """The decision.  Pure and total: same input, same output, never raises."""
    concept, concept_target = _concept_absent(inp)
    outcomes = (concept, _factual_error(inp), _ambiguous_band(inp), _high_uncertainty(inp))

    capped_by = None
    if inp.item_follow_ups >= MAX_FOLLOW_UPS_PER_ITEM:
        capped_by = CAP_ITEM
    elif inp.session_follow_ups >= MAX_FOLLOW_UPS_PER_SESSION:
        capped_by = CAP_SESSION
    if capped_by is not None:
        return FollowUpDecision(False, None, None, outcomes, capped_by)

    for outcome in outcomes:
        if outcome.status is RuleStatus.FIRED:
            target = concept_target if outcome.rule == RULE_CONCEPT_ABSENT else None
            return FollowUpDecision(True, outcome.rule, target or _weakest_concept(inp), outcomes)
    return FollowUpDecision(False, None, None, outcomes)


__all__ = [
    "AMBIGUOUS_HIGH",
    "AMBIGUOUS_LOW",
    "CAP_ITEM",
    "CAP_SESSION",
    "HIGH_WEIGHT",
    "RD_FOLLOW_UP_THRESHOLD",
    "RULES",
    "RULE_AMBIGUOUS_BAND",
    "RULE_CONCEPT_ABSENT",
    "RULE_FACTUAL_ERROR",
    "RULE_HIGH_UNCERTAINTY",
    "ConceptEvidence",
    "FollowUpDecision",
    "PolicyInput",
    "RuleOutcome",
    "RuleStatus",
    "decide",
    "policy_input",
]
