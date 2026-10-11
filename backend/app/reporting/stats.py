"""Deterministic report aggregation - plan section 3, Day 25: "deterministic
aggregation in SQL/Python".

**Input:** the committed event log, the session's blueprint, and each asked
question's topic/subtopic.  **Output:** a JSON-ready ``stats`` dict.  Pure: the
same log and blueprint produce byte-identical stats, every time.

**Every number here is computed in code, from the log.**  The narrative model
(``app.reporting.narrative``) reads these numbers; it never produces one.

Where each number comes from
----------------------------

* **Scores** - ``ANSWER_GRADED.score``, one per turn.  The FSM allows exactly one
  successful grade per turn, so a retried, replayed or duplicate submission
  cannot be counted twice: there is nothing in the log to double-count.
* **Ability** - the fold's ``ability`` (observed subtopics only), which is
  ``ANSWER_GRADED.ability`` as computed by Day 11's update.
* **Display and interval** - plan section 9.3, verbatim::

      display = round(100 / (1 + exp(-1.1 * theta)))
      ci      = (display(theta - 1.96 * rd), display(theta + 1.96 * rd))

  ``theta +/- 1.96 * RD`` is the 95% interval ``app.ability`` documents RD as
  (RD is a standard error).  No new uncertainty formula is introduced.
* **"Insufficient evidence"** - plan section 9.3: ``RD > 0.8``.
* **Topic level** - ``app.ability.aggregate_ability`` (precision weighting) over
  the topic's *observed* subtopics.  Topics with none observed have no number.
* **Gaps / strengths / needs-more-assessment** - plan section 9.6: a gap is
  "low theta **and** low RD (confidently weak)"; "low theta, high RD means we
  don't know yet - put it in 'needs more assessment'".  Low/high theta is
  relative to the level's ``target_theta`` (theta 0 = "at the target level").

**Priors are reported apart from observations.**  A subtopic's starting point
(resume claim or default) is listed under ``prior``; its measurement under
``observed``.  A subtopic that was never asked about has ``observed: null`` -
its prior is not presented as a result.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from app.ability import AbilityState, aggregate_ability
from app.blueprint import Blueprint
from app.interview.events import StoredEvent
from app.interview.fsm import EventType, InterviewState
from app.interview.state import fold

REPORT_SCHEMA_VERSION = 1
INSUFFICIENT_RD = 0.8
Z95 = 1.96

#: Fixed methodology notes, attached to every report so a reader cannot mistake
#: what the numbers are.  Code-authored; never model-written.
METHODOLOGY_NOTES = (
    "Scores and ability estimates are computed in code from the recorded grades; "
    "the written summary only describes them.",
    "Intervals are theta +/- 1.96 x RD (a 95% interval). Subtopics with RD above "
    "0.8 are marked as insufficient evidence rather than given a number.",
    "Resume-based starting points are claims, not evidence; they are listed "
    "separately from observed results and are capped.",
    "Follow-up answers are graded against the original question's full concept "
    "checklist at half weight; a correct but narrow follow-up answer can score low.",
)


class ReportNotReady(ValueError):
    """The log is not in a state a report may be built from."""


@dataclass(frozen=True, slots=True)
class QuestionInfo:
    topic_key: str
    subtopic_key: str


def display(theta: float) -> int:
    """Plan section 9.3's 0-100 display scale."""
    return round(100 / (1 + math.exp(-1.1 * theta)))


def _ability_view(a: AbilityState) -> dict[str, Any]:
    return {
        "theta": round(a.theta, 4),
        "rd": round(a.rd, 4),
        "n_observations": a.n_observations,
        "display": display(a.theta),
        "interval": [display(a.theta - Z95 * a.rd), display(a.theta + Z95 * a.rd)],
        "insufficient_evidence": a.rd > INSUFFICIENT_RD,
    }


def aggregate(
    *,
    events: Sequence[StoredEvent],
    session_id: Any,
    blueprint: Blueprint,
    questions: Mapping[str, QuestionInfo],
) -> dict[str, Any]:
    """The report's statistics.  Raises :class:`ReportNotReady` unless the log
    has reached ``REPORTING`` (or ``COMPLETED``, for a rebuild)."""
    state = fold(session_id, events)
    if state.state not in (InterviewState.REPORTING, InterviewState.COMPLETED):
        raise ReportNotReady(f"session is {state.state}, not REPORTING")

    items: list[dict[str, Any]] = []
    by_turn: dict[str, dict[str, Any]] = {}
    #: What the next ANSWER_SUBMITTED answers: the item, or its latest follow-up.
    current: dict[str, Any] | None = None
    follow_ups_started = follow_ups_abandoned = grading_failures = 0
    prior_source = "none"
    for event in events:
        p = event.payload
        if event.type == EventType.PREPARATION_STARTED:
            prior_source = str(p.get("prior_source", "none"))
        elif event.type == EventType.QUESTION_ASKED:
            if p.get("is_follow_up"):
                current = {"score": None}
                items[-1]["follow_ups"].append(current)
            else:
                qid = str(p["question_id"])
                info = questions.get(qid)
                items.append(
                    {
                        "order": len(items) + 1,
                        "question_id": qid,
                        "topic_key": info.topic_key if info else None,
                        "subtopic_key": info.subtopic_key if info else None,
                        "score": None,
                        "follow_ups": [],
                    }
                )
                current = items[-1]
        elif event.type == EventType.ANSWER_SUBMITTED and current is not None:
            by_turn[str(p["turn_id"])] = current
        elif event.type == EventType.ANSWER_GRADED and str(p["turn_id"]) in by_turn:
            by_turn[str(p["turn_id"])]["score"] = round(float(p["score"]), 4)
        elif event.type == EventType.GRADING_FAILED:
            grading_failures += 1
        elif event.type == EventType.FOLLOW_UP_STARTED:
            follow_ups_started += 1
        elif event.type == EventType.FOLLOW_UP_ABANDONED:
            follow_ups_abandoned += 1

    parent_of = blueprint.parent_of
    target = blueprint.target_theta
    subtopic_keys = sorted(set(blueprint.subtopic_keys) | set(state.ability))
    subtopics: list[dict[str, Any]] = []
    for key in subtopic_keys:
        observed = state.ability.get(key)
        prior = state.priors.get(key)
        start = prior or blueprint.prior_ability()
        subtopics.append(
            {
                "subtopic_key": key,
                "topic_key": parent_of.get(key),
                "observed": _ability_view(observed) if observed else None,
                "prior": {
                    "theta": round(start.theta, 4),
                    "rd": round(start.rd, 4),
                    "source": "resume_profile" if prior else "default",
                },
            }
        )

    topics: list[dict[str, Any]] = []
    for plan in blueprint.topics:
        seen = {k: a for k, a in state.ability.items() if parent_of.get(k) == plan.topic_key}
        topics.append(
            {
                "topic_key": plan.topic_key,
                "jd_weight": plan.jd_weight,
                "required": plan.required,
                "quota": plan.quota,
                "questions_asked": sum(1 for i in items if i["topic_key"] == plan.topic_key),
                "observed": _ability_view(aggregate_ability(seen)) if seen else None,
            }
        )

    confident = [
        s for s in subtopics if s["observed"] and not s["observed"]["insufficient_evidence"]
    ]
    gaps = sorted(
        (s for s in confident if s["observed"]["theta"] < target),
        key=lambda s: (s["observed"]["theta"], s["subtopic_key"]),
    )
    strengths = sorted(
        (s for s in confident if s["observed"]["theta"] > target),
        key=lambda s: (-s["observed"]["theta"], s["subtopic_key"]),
    )
    needs_more = sorted(
        s["subtopic_key"]
        for s in subtopics
        if s["observed"] is None or s["observed"]["insufficient_evidence"]
    )
    follow_ups_asked = sum(len(i["follow_ups"]) for i in items)
    return {
        "version": REPORT_SCHEMA_VERSION,
        "role": blueprint.role,
        "level": blueprint.level,
        "target_theta": target,
        "stop_reasons": list(state.stop_reasons),
        "counts": {
            "primary_questions": len(items),
            "follow_ups_asked": follow_ups_asked,
            "follow_ups_decided": follow_ups_started,
            "follow_ups_abandoned": follow_ups_abandoned,
            "answers_graded": sum(1 for e in events if e.type == EventType.ANSWER_GRADED),
            "grading_failures": grading_failures,
            "item_budget": blueprint.item_budget,
        },
        "questions": items,
        "topics": topics,
        "subtopics": subtopics,
        "gaps": [s["subtopic_key"] for s in gaps],
        "strengths": [s["subtopic_key"] for s in strengths],
        "needs_more_assessment": needs_more,
        "priors": {
            "source": prior_source,
            "seeded_subtopics": sorted(state.priors),
            "note": "Starting points only. Not evidence of skill.",
        },
        "methodology_notes": list(METHODOLOGY_NOTES),
        "log_seq": state.seq,
    }


__all__ = [
    "INSUFFICIENT_RD",
    "METHODOLOGY_NOTES",
    "REPORT_SCHEMA_VERSION",
    "QuestionInfo",
    "ReportNotReady",
    "aggregate",
    "display",
]
