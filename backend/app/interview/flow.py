"""What happens after a grade, and how the next bank question is chosen.

Plan section 13.4::

    GRADING -> { PROBING -> ASKING | SELECTING -> ASKING }
                         stop rule -> REPORTING

Every function here **stages** events inside the caller's transaction (the
grade's, the follow-up job's, or ``/start``'s) and commits nothing, so a grade
and the decision it leads to are durable together or not at all.  There is no
window in which a grade is committed but nobody has decided what comes next.

Nothing here is a new algorithm:

* the follow-up decision is :func:`app.interview.followup.decide` (plan 8.5);
* the stop rule is Day 12's ``should_stop`` with the blueprint's own arguments
  (``Blueprint.stopping_arguments``);
* the next item is Day 12's ``select_next_item`` over the blueprint's
  ``selection_state``;
* the time budget is consumed the way Day 13's simulation consumes it - by each
  asked item's ``time_estimate_s`` (a budgeted clock, not the wall clock, so a
  replay of the log reaches the same decision).

The epsilon-greedy draw is seeded from ``(session_id, seq)``: reproducible for a
given position in a given session, different across sessions.  The draw's
result is recorded in the log anyway (``QUESTION_ASKED.question_id``), so replay
never re-runs it.
"""

from __future__ import annotations

import random
import re
import uuid
from collections.abc import Mapping, Sequence
from typing import Any

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.blueprint import Blueprint
from app.interview.followup import FollowUpDecision, decide, policy_input
from app.interview.fsm import EventType, InterviewState
from app.interview.state import SessionState
from app.interview.store import stage_append
from app.models.interview import FOLLOW_UP_PENDING, FollowUp
from app.models.question import Question
from app.models.reporting import REPORT_PENDING, Report
from app.selection import CandidateItem, select_next_item, should_stop
from app.wording.render import render_question

#: The reason recorded when no eligible item is left (plan section 8.7).  The
#: same string Day 13's simulation records.
STOP_POOL_EXHAUSTED = "pool_exhausted"

_CONCEPT_KEY = re.compile(r"^[a-z][a-z0-9_]{1,63}$")


class NoEligibleQuestion(RuntimeError):
    """``/start`` found nothing to ask: the bank has no item this blueprint allows."""


async def _append(
    db: AsyncSession,
    state: SessionState,
    *,
    user_id: uuid.UUID,
    event_type: EventType,
    payload: Mapping[str, Any] | None = None,
) -> SessionState:
    return await stage_append(
        db,
        session_id=state.session_id,
        user_id=user_id,
        event_type=event_type,
        payload=payload or {},
        expected_seq=state.seq,
    )


async def _stage_reporting(
    db: AsyncSession, state: SessionState, *, user_id: uuid.UUID, reasons: list[str]
) -> SessionState:
    """``REPORTING_STARTED`` plus its report job, in the caller's transaction
    (Day 25): a session never enters REPORTING without a durable job to finish it."""
    staged = await _append(
        db,
        state,
        user_id=user_id,
        event_type=EventType.REPORTING_STARTED,
        payload={"reasons": reasons},
    )
    await db.execute(
        pg_insert(Report)
        .values(session_id=state.session_id, status=REPORT_PENDING)
        .on_conflict_do_nothing(constraint="uq_reports_session_id")
    )
    return staged


async def load_asked_items(db: AsyncSession, ids: Sequence[str]) -> list[CandidateItem]:
    """The asked bank items as Day 12 ``CandidateItem``s, in asked order."""
    if not ids:
        return []
    rows = (
        await db.execute(
            sa.select(
                Question.id,
                Question.topic_key,
                Question.subtopic_key,
                Question.difficulty_b,
                Question.time_estimate_s,
                Question.discrimination_a,
                Question.embedding,
            ).where(Question.id.in_(list(ids)))
        )
    ).all()
    by_id = {
        row.id: CandidateItem(
            id=row.id,
            topic_key=row.topic_key,
            subtopic_key=row.subtopic_key,
            difficulty_b=float(row.difficulty_b),
            time_estimate_s=int(row.time_estimate_s),
            discrimination_a=float(row.discrimination_a),
            embedding=None if row.embedding is None else tuple(float(v) for v in row.embedding),
        )
        for row in rows
    }
    # An asked item deleted from the bank since is simply not counted.
    return [by_id[i] for i in ids if i in by_id]


async def stage_next_bank_question(
    db: AsyncSession,
    state: SessionState,
    *,
    user_id: uuid.UUID,
    blueprint: Blueprint,
) -> SessionState:
    """From ``GRADING`` (graded), ``SELECTING`` or ``PREPARING``: apply the stop
    rule (not at the start), then select, then ask - or stop.

    Raises :class:`NoEligibleQuestion` only from ``PREPARING``; anywhere else
    an empty pool is the normal end of an interview (``REPORTING``).
    """
    asked = await load_asked_items(db, state.asked_question_ids)
    used_s = float(sum(item.time_estimate_s for item in asked))

    if state.state is InterviewState.GRADING:
        stop = should_stop(
            **blueprint.stopping_arguments(
                ability=state.ability,
                items_asked=len(state.asked_question_ids),
                time_elapsed_s=used_s,
                recent_theta_deltas=state.theta_deltas,
            )
        )
        if stop.should_stop:
            return await _stage_reporting(db, state, user_id=user_id, reasons=list(stop.reasons))
        state = await _append(db, state, user_id=user_id, event_type=EventType.SELECTION_STARTED)

    # Day 24: resume priors are where unobserved subtopics *start* for
    # selection. The stop rule above deliberately sees only observed ability:
    # a claim must never count as measured precision.
    selection_state = blueprint.selection_state(
        ability={**state.priors, **state.ability},
        asked=asked,
        time_left_s=max(0.0, blueprint.time_budget_s - used_s),
    )
    selection = await select_next_item(
        db,
        selection_state,
        rng=random.Random(f"{state.session_id}:{state.seq}"),  # noqa: S311 - not crypto
        difficulty_window=blueprint.difficulty_window,
    )
    if selection is None:
        if state.state is InterviewState.PREPARING:
            raise NoEligibleQuestion("no question in the bank fits this session's blueprint")
        return await _stage_reporting(db, state, user_id=user_id, reasons=[STOP_POOL_EXHAUSTED])

    item_id = selection.chosen.item.id
    question = await db.get(Question, item_id)
    if question is None:  # pragma: no cover - selected from this table just now
        raise RuntimeError(f"selected question {item_id!r} vanished")
    # Day 23 has no candidate context to bind to (that is Day 24's resume/JD),
    # so this returns the canonical text without calling a model.
    rendered = await render_question(question.text, anchor_terms=question.anchor_terms or ())
    return await _append(
        db,
        state,
        user_id=user_id,
        event_type=EventType.QUESTION_ASKED,
        payload={"question_id": item_id, "text": rendered.text},
    )


async def stage_after_grade(
    db: AsyncSession,
    state: SessionState,
    *,
    user_id: uuid.UUID,
    blueprint: Blueprint,
    judgment: Mapping[str, Any] | None,
    expected: Sequence[tuple[str, int]],
    score: float | None,
    subtopic_key: str | None,
    source_turn_id: uuid.UUID,
) -> tuple[SessionState, FollowUpDecision]:
    """The decision after ``ANSWER_GRADED``, staged.  ``state`` must be the
    state *including* that grade."""
    ability = state.ability.get(subtopic_key) if subtopic_key else None
    decision = decide(
        policy_input(
            judgment=judgment,
            expected=expected,
            score=score,
            rd=ability.rd if ability is not None else None,
            item_follow_ups=state.follow_ups_for(state.current_item_id),
            session_follow_ups=state.follow_ups_asked,
        )
    )
    if decision.follow_up and state.follow_up_room:
        concept = decision.concept_key
        payload: dict[str, Any] = {"reasons": [decision.reason]}
        if concept and _CONCEPT_KEY.match(concept):
            payload["concept_key"] = concept
        else:
            concept = None
        item = state.current_item_id
        staged = await _append(
            db, state, user_id=user_id, event_type=EventType.FOLLOW_UP_STARTED, payload=payload
        )
        # The durable job for the wording. The uniques make a second follow-up
        # for one grade (or a third for one item) impossible; the grading fence
        # makes it unreachable. If it ever conflicts anyway, refuse to commit a
        # session that would sit in PROBING with no job to finish it.
        inserted = await db.execute(
            pg_insert(FollowUp)
            .values(
                session_id=state.session_id,
                source_turn_id=source_turn_id,
                item_question_id=item,
                ordinal=state.follow_ups_for(item) + 1,
                reason=decision.reason,
                concept_key=concept,
                status=FOLLOW_UP_PENDING,
            )
            .on_conflict_do_nothing()
            .returning(FollowUp.id)
        )
        if inserted.scalar_one_or_none() is None:
            raise RuntimeError("follow-up job already exists for this grade or item ordinal")
        return staged, decision
    staged = await stage_next_bank_question(db, state, user_id=user_id, blueprint=blueprint)
    return staged, decision


__all__ = [
    "STOP_POOL_EXHAUSTED",
    "NoEligibleQuestion",
    "load_asked_items",
    "stage_after_grade",
    "stage_next_bank_question",
]
