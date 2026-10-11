"""What each event in ``interview_events`` is allowed to carry.

One pydantic model per :class:`~app.interview.fsm.EventType`, every one with
``extra="forbid"``.  That setting is doing two jobs:

1. **Corruption detection.**  A payload with a field this build does not know
   is not silently ignored on replay; it fails validation and the log is
   reported as unreadable rather than half-understood.
2. **Data minimisation.**  The log is permanent and is read by replay, reports
   and evals.  An answer's text, a resume, an email address have no field here,
   so they *cannot* be written into an event by accident - they live in
   ``turns`` and (later) the resume tables, behind their own access rules.  An
   event refers to a turn by its ``turn_id`` and to a question by its bank id.

Every payload is validated **before** it is written and **again** when it is
read back.  The second check is what makes ``fold`` refuse a log that was
edited by hand, rather than trusting it because it came from our own table.
"""

from __future__ import annotations

import math
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from app.interview.fsm import EventType

#: A short machine-readable reason: the ``STOP_*`` constants of
#: :mod:`app.selection.stopping` fit, and so do Day 23's follow-up triggers.
#: Free text does not - a reason is a key, not a sentence.
Reason = Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")]

#: Bank question ids are short slugs (``app.bank.schema``); 128 is generous.
QuestionId = Annotated[str, Field(min_length=1, max_length=128)]

#: Concept keys (``app.bank.schema.ExpectedConcept.key``).
ConceptKey = Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]{1,63}$")]

#: The wording actually put to the candidate (Day 23).  Bounded: a question is a
#: few sentences, and the log is permanent.
MAX_QUESTION_TEXT = 2000


class _Payload(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=False)


class SessionCreated(_Payload):
    """seq 0 of every session.  The blueprint itself is on the session row."""


class PriorIn(_Payload):
    """One subtopic's starting ability (Day 24, ``app.interview.priors``)."""

    theta: float = Field(ge=-3.0, le=3.0)
    #: Never below 1.0: a resume claim may not make the engine more certain
    #: than the plan's tightest prior (section 9.4).  Enforced by the log itself.
    rd: float = Field(ge=1.0, le=1.3)

    @field_validator("theta", "rd")
    @classmethod
    def _finite(cls, v: float) -> float:
        if not math.isfinite(v):
            raise ValueError("must be finite")
        return v


class PreparationStarted(_Payload):
    """The session is being prepared (Day 24).

    ``priors`` are the resume-seeded starting abilities, computed **once**, at
    ``/start``, from the candidate-confirmed profile, and recorded here - so a
    later profile edit, re-upload or deletion cannot change a session already
    under way, and replay never re-reads the resume.  Empty means "no resume
    evidence": every subtopic starts at the blueprint's default prior.
    ``prior_source`` says why, for the report and for debugging.
    """

    priors: dict[Annotated[str, Field(min_length=1, max_length=64)], PriorIn] = Field(
        default_factory=dict, max_length=200
    )
    prior_source: Literal["none", "no_profile", "resume_profile"] = "none"


class QuestionAsked(_Payload):
    """A question was put to the candidate.

    ``question_id`` is the bank item for a fresh question and ``None`` for a
    follow-up, whose wording is generated (Day 23) and has no bank id.  Which of
    the two is legal depends on the state it is asked from - that rule lives in
    :mod:`app.interview.state`, next to the other cross-event invariants.
    """

    question_id: QuestionId | None = None
    is_follow_up: bool = False
    #: Day 23: the exact wording shown - the bank text for a fresh question
    #: (rendered, once rendering has context), the generated probe for a
    #: follow-up.  Optional so that pre-Day-23 logs still fold.  Not candidate
    #: data: it is what *we* said.
    text: Annotated[str, Field(min_length=1, max_length=MAX_QUESTION_TEXT)] | None = None

    @field_validator("question_id")
    @classmethod
    def _strip(cls, v: str | None) -> str | None:
        if v is not None and v != v.strip():
            raise ValueError("question_id must not have surrounding whitespace")
        return v


class AnswerSubmitted(_Payload):
    """The candidate answered.  Only the turn's id - the text is in ``turns``."""

    turn_id: uuid.UUID


class AbilitySnapshot(_Payload):
    """One subtopic's ability *after* the update, as computed by
    :func:`app.ability.update_ability`.

    The result is recorded, not the inputs, so replay never re-runs the update
    arithmetic: a later change to the K-factor must not rewrite what a past
    session concluded.  Re-grading under new arithmetic is an explicit job over
    the log (plan section 11.2), not a side effect of reading it.
    """

    subtopic_key: Annotated[str, Field(min_length=1, max_length=128)]
    theta: float
    rd: float = Field(gt=0.0)
    n_observations: int = Field(ge=0)

    @field_validator("theta", "rd")
    @classmethod
    def _finite(cls, v: float) -> float:
        if not math.isfinite(v):
            raise ValueError("must be finite")
        return v


class AnswerGraded(_Payload):
    """The grade for the current turn.

    ``score`` is the normalised code-side score in ``[0, 1]`` (plan section 7,
    :func:`app.grading.scoring.score_answer`).  ``ability`` is absent when the
    grade moved no estimate - for example a degraded grade.
    """

    turn_id: uuid.UUID
    score: float = Field(ge=0.0, le=1.0)
    ability: AbilitySnapshot | None = None
    #: Day 23: the theta step this grade applied (``AbilityUpdate.delta_theta``),
    #: so the stopping rule's "no new information" condition can be evaluated
    #: from the log alone.  Optional: older events and grades that moved no
    #: estimate carry none.
    delta_theta: float | None = None

    @field_validator("delta_theta")
    @classmethod
    def _finite_delta(cls, v: float | None) -> float | None:
        if v is not None and not math.isfinite(v):
            raise ValueError("must be finite")
        return v

    @field_validator("score")
    @classmethod
    def _finite(cls, v: float) -> float:
        if not math.isfinite(v):
            raise ValueError("must be finite")
        return v


class GradingFailed(_Payload):
    """No valid grade could be produced for the current turn (Day 22).

    ``error_code`` is a short machine key from
    :func:`app.interview.grading.classify_error` - never a provider message,
    which may echo the prompt and therefore the answer.  ``retries_exhausted``
    says whether the failure was transient-but-persistent (True) or permanent
    from the first attempt (False).  Recording this event changes no state and
    no ability: it is the absence of a grade, written down.
    """

    turn_id: uuid.UUID
    error_code: Reason
    attempts: int = Field(ge=1)
    retries_exhausted: bool


class FollowUpStarted(_Payload):
    """A follow-up on the current item was decided by the deterministic policy
    (:mod:`app.interview.followup`).  ``reasons`` holds the rule that fired;
    ``concept_key`` the concept the probe targets, when the rule names one.
    Neither is ever sent to the candidate (it would hand over the answer)."""

    reasons: tuple[Reason, ...] = ()
    concept_key: ConceptKey | None = None


class FollowUpAbandoned(_Payload):
    """The decided follow-up could not be worded (Day 23).

    Same shape as :class:`GradingFailed` and for the same reason: a short code,
    never a provider message.  No question is fabricated in its place; the
    session moves to ``SELECTING`` and continues normally.
    """

    error_code: Reason
    attempts: int = Field(ge=0)
    retries_exhausted: bool


class SelectionStarted(_Payload):
    """A new bank item is being chosen.  Days 12-15 own *which*."""


class ReportingStarted(_Payload):
    """The interview stopped asking.  At least one reason is mandatory - an
    interview that ended for no recorded reason is undebuggable."""

    reasons: tuple[Reason, ...] = Field(min_length=1)

    @field_validator("reasons")
    @classmethod
    def _unique(cls, v: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(v)) != len(v):
            raise ValueError("reasons must not repeat")
        return v


class SessionCompleted(_Payload):
    """The report exists; the session is over.  Terminal.

    Day 25: ``report_id`` links the terminal event to the persisted report,
    which is committed in the same transaction.  Optional so older logs fold.
    """

    report_id: uuid.UUID | None = None


PAYLOAD_MODELS: Mapping[EventType, type[_Payload]] = {
    EventType.SESSION_CREATED: SessionCreated,
    EventType.PREPARATION_STARTED: PreparationStarted,
    EventType.QUESTION_ASKED: QuestionAsked,
    EventType.ANSWER_SUBMITTED: AnswerSubmitted,
    EventType.ANSWER_GRADED: AnswerGraded,
    EventType.GRADING_FAILED: GradingFailed,
    EventType.FOLLOW_UP_STARTED: FollowUpStarted,
    EventType.FOLLOW_UP_ABANDONED: FollowUpAbandoned,
    EventType.SELECTION_STARTED: SelectionStarted,
    EventType.REPORTING_STARTED: ReportingStarted,
    EventType.SESSION_COMPLETED: SessionCompleted,
}


class InvalidEventPayload(ValueError):
    """A payload that does not match its event type's contract."""

    def __init__(self, event_type: str, detail: str) -> None:
        self.event_type = event_type
        super().__init__(f"invalid {event_type} payload: {detail}")


def validate_payload(event_type: EventType, payload: Mapping[str, Any]) -> _Payload:
    """Parse ``payload`` against ``event_type``'s model, or raise
    :class:`InvalidEventPayload`."""
    if not isinstance(payload, Mapping):
        raise InvalidEventPayload(event_type.value, "payload must be a JSON object")
    try:
        return PAYLOAD_MODELS[event_type].model_validate(dict(payload))
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in err['loc']) or '<root>'}: {err['msg']}"
            for err in exc.errors()
        )
        raise InvalidEventPayload(event_type.value, problems) from None


def canonical_payload(model: _Payload) -> dict[str, Any]:
    """The JSON object stored in ``interview_events.payload``.

    ``mode="json"`` turns UUIDs and tuples into JSON types, so the stored object
    and the re-validated one round-trip exactly.  ``exclude_none`` (Day 23): an
    optional field left unset is not written, so adding one (``text``,
    ``delta_theta``, ``concept_key``) does not put a ``null`` into every later
    event; every such field defaults to ``None``, so the round trip is unchanged.
    """
    return model.model_dump(mode="json", exclude_none=True)


@dataclass(frozen=True, slots=True)
class StoredEvent:
    """One row of the log as :func:`app.interview.state.fold` sees it.

    Deliberately not the ORM object: ``fold`` must be callable on events from a
    test, a fixture file or a production replay without a database session.
    ``type`` is a plain string because a corrupted row may hold one this build
    does not know, and that has to reach ``fold`` to be reported.
    """

    seq: int
    type: str
    payload: Mapping[str, Any]


__all__ = [
    "PAYLOAD_MODELS",
    "AbilitySnapshot",
    "AnswerGraded",
    "AnswerSubmitted",
    "FollowUpAbandoned",
    "FollowUpStarted",
    "GradingFailed",
    "InvalidEventPayload",
    "PreparationStarted",
    "QuestionAsked",
    "ReportingStarted",
    "SelectionStarted",
    "SessionCompleted",
    "SessionCreated",
    "StoredEvent",
    "canonical_payload",
    "validate_payload",
]
