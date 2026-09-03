"""Stage 1 - the hard constraints, as SQL, before anything is scored.

Plan section 8.3, step 1::

    WHERE id NOT IN (asked_ids)                    -- never repeat
      AND topic_key   IN (topics_with_quota_left)  -- respect the blueprint
      AND ABS(difficulty_b - :theta_for_topic) <= 1.5
      AND time_estimate_s <= :time_remaining

**Why these are filters and not penalties.**  A penalty is a number added to a
score, and a large enough bonus elsewhere will always outweigh it - so "never
repeat a question" implemented as ``-0.5 * already_asked`` is not "never", it is
"usually".  These four rules are policy, not preference: an item that fails any
of them must be *impossible* to select, whatever the other five terms say.  The
only way to get that guarantee is to remove it from the pool.

Three more reasons the plan puts them in SQL rather than in Python:

* The database is where the questions are.  Filtering there ships back the
  survivors instead of the bank; scoring then touches tens of rows, not
  hundreds, and never grows into a full-table load as the bank does.
* Both indexes that exist for this already help:
  ``ix_questions_subtopic_key_difficulty_b`` is precisely the
  ``(subtopic, difficulty)`` shape the difficulty window asks for.
* It keeps the scoring function total.  ``score_item`` divides by
  ``state.time_left`` and reads ``state.theta[...]``; it never has to ask "but
  is this item even allowed?", because by then the question cannot arise.

**Two expressions of one rule.**  :func:`ineligibility_reason` is the
specification, in Python, exercised exhaustively offline; the SQL below is the
fast path over the real table.  Two implementations of one rule can drift, so
``tests/integration/test_selection_sql.py`` asserts they agree item-for-item on
the same fixture bank, and ``tests/unit/selection/test_constraints.py`` asserts
the compiled statement still contains all four clauses with every value bound
rather than interpolated.

Not implemented here, and deliberately: the *relaxation ladder* of plan section
8.7 (widen the window to 2.0, then drop the topic constraint, then end the topic
early and log why).  An empty pool is returned as an empty pool; deciding what
to do about it belongs to the session orchestrator, which does not exist yet.
DEFERRED.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.question import Question
from app.selection.state import CandidateItem, SelectionState

#: ``|b - theta| <= 1.5`` (plan section 8.3).  It does double duty: items this
#: far from ability carry almost no Fisher information (section 5.10), *and*
#: they bore or demoralise the candidate.  Measurement and user experience agree
#: here, which is rare and worth saying out loud.
DIFFICULTY_WINDOW = 1.5

#: How many survivors to bring back for scoring.  Not a policy limit - it is a
#: guard against a future bank of thousands making one selection load them all.
#: Well above any plausible pool for a 150-item bank, so it is inert today.
DEFAULT_CANDIDATE_LIMIT = 200

# The reasons an item can fail, in the order the plan lists the constraints.
REASON_ALREADY_ASKED = "already_asked"
REASON_TOPIC_QUOTA = "topic_quota_exhausted"
REASON_DIFFICULTY_WINDOW = "outside_difficulty_window"
REASON_TIME_REMAINING = "exceeds_time_remaining"


# ---------------------------------------------------------------------------
# the specification, in Python
# ---------------------------------------------------------------------------


def ineligibility_reason(
    item: CandidateItem,
    state: SelectionState,
    *,
    difficulty_window: float = DIFFICULTY_WINDOW,
) -> str | None:
    """Why this item cannot be asked next, or ``None`` if it can.

    A reason rather than a bare bool because "the pool was empty" is a question
    somebody will have to answer at 2am, and ``Counter(reasons)`` over the bank
    answers it immediately: *38 outside the difficulty window, 12 already
    asked*.  :func:`is_eligible` is this function's truthiness.

    Checks run in the plan's own order, so the reported reason is the first
    constraint the item fails, not an arbitrary one.

    ``theta`` is taken at **subtopic** level.  The plan's SQL writes
    ``:theta_for_topic``; the canonical state is per subtopic (section 9.1) and
    ``score_item`` reads ``state.theta[q.subtopic]``, so using the subtopic
    estimate is what keeps the filter and the score talking about the same
    number.  An unmeasured subtopic falls back to ``state.prior``.
    """
    if item.id in state.asked_ids:
        return REASON_ALREADY_ASKED
    if state.quota_remaining(item.topic_key) <= 0:
        return REASON_TOPIC_QUOTA
    if abs(item.difficulty_b - state.theta_for(item.subtopic_key)) > difficulty_window:
        return REASON_DIFFICULTY_WINDOW
    if item.time_estimate_s > state.time_left_s:
        return REASON_TIME_REMAINING
    return None


def is_eligible(
    item: CandidateItem,
    state: SelectionState,
    *,
    difficulty_window: float = DIFFICULTY_WINDOW,
) -> bool:
    """True when the item survives all four hard constraints."""
    return ineligibility_reason(item, state, difficulty_window=difficulty_window) is None


def filter_eligible(
    items: Sequence[CandidateItem],
    state: SelectionState,
    *,
    difficulty_window: float = DIFFICULTY_WINDOW,
) -> list[CandidateItem]:
    """The in-memory stage 1, for callers that already hold their candidates.

    Same rules, same order as the SQL.  Used by the offline tests and by any
    caller working from a pre-fetched list (a retrieval result, a simulation);
    the database path below is what a live session uses.
    """
    return [item for item in items if is_eligible(item, state, difficulty_window=difficulty_window)]


# ---------------------------------------------------------------------------
# the same rules, in SQL
# ---------------------------------------------------------------------------


def _theta_expression(state: SelectionState) -> sa.ColumnElement[float]:
    """``CASE questions.subtopic_key WHEN :k1 THEN :t1 ... ELSE :prior END``.

    A CASE rather than a join against a temporary table because it needs no
    server-side object and stays one statement, and rather than string
    formatting because every key and every theta below is a **bound parameter**
    - which is the only acceptable way to get a caller's values into SQL.

    ``ELSE`` is the cold-start prior, so a question whose subtopic has never
    been measured is filtered against theta 0 instead of being silently dropped
    by an inner join.
    """
    prior = sa.literal(state.prior.theta, type_=sa.Float)
    if not state.ability:
        return prior
    whens = {key: sa.literal(value.theta, type_=sa.Float) for key, value in state.ability.items()}
    return sa.case(whens, value=Question.subtopic_key, else_=prior)


def eligibility_clauses(
    state: SelectionState,
    *,
    difficulty_window: float = DIFFICULTY_WINDOW,
) -> list[sa.ColumnElement[bool]]:
    """The four WHERE clauses of plan section 8.3, in the plan's order.

    Returned as a list rather than baked into a query so that a caller who
    already has a ``select(Question)`` of their own - a retrieval result being
    narrowed, a debugging script - can apply exactly the same policy without
    reimplementing it.
    """
    clauses: list[sa.ColumnElement[bool]] = []

    asked = sorted(state.asked_ids)
    if asked:
        # An empty NOT IN is rendered differently by different SQLAlchemy
        # versions and means nothing anyway, so it is simply not emitted.
        clauses.append(Question.id.not_in(asked))

    clauses.append(Question.topic_key.in_(state.topics_with_quota_left))
    clauses.append(
        sa.func.abs(Question.difficulty_b - _theta_expression(state)) <= difficulty_window
    )
    clauses.append(Question.time_estimate_s <= state.time_left_s)
    return clauses


def eligible_items_statement(
    state: SelectionState,
    *,
    difficulty_window: float = DIFFICULTY_WINDOW,
    limit: int = DEFAULT_CANDIDATE_LIMIT,
) -> sa.Select[Any]:
    """``SELECT`` the survivors, and only the columns selection reads.

    Ordered by id: the ranking that matters is applied in stage 2, and a stable
    row order here makes the whole pipeline reproducible under a seeded RNG.
    """
    return (
        sa.select(
            Question.id,
            Question.topic_key,
            Question.subtopic_key,
            Question.difficulty_b,
            Question.time_estimate_s,
            Question.discrimination_a,
            Question.embedding,
        )
        .where(*eligibility_clauses(state, difficulty_window=difficulty_window))
        .order_by(Question.id)
        .limit(limit)
    )


async def eligible_items(
    session: AsyncSession,
    state: SelectionState,
    *,
    difficulty_window: float = DIFFICULTY_WINDOW,
    limit: int = DEFAULT_CANDIDATE_LIMIT,
) -> list[CandidateItem]:
    """Stage 1 against the real table: the items this session may ask next.

    Two states short-circuit without a round trip, because their answer is
    known and both are reachable in an ordinary session:

    * **no time left** - nothing with a positive time estimate can fit;
    * **no topic with quota left** - the blueprint is served, or empty.

    An empty result is a legitimate outcome (plan section 8.7's bank
    exhaustion), not an error; the caller decides whether to relax something or
    to end the interview.
    """
    if state.time_left_s <= 0.0 or not state.topics_with_quota_left:
        return []

    rows = (
        await session.execute(
            eligible_items_statement(state, difficulty_window=difficulty_window, limit=limit)
        )
    ).all()
    return [
        CandidateItem(
            id=row.id,
            topic_key=row.topic_key,
            subtopic_key=row.subtopic_key,
            difficulty_b=float(row.difficulty_b),
            time_estimate_s=int(row.time_estimate_s),
            discrimination_a=float(row.discrimination_a),
            embedding=None if row.embedding is None else tuple(float(v) for v in row.embedding),
        )
        for row in rows
    ]


__all__ = [
    "DEFAULT_CANDIDATE_LIMIT",
    "DIFFICULTY_WINDOW",
    "REASON_ALREADY_ASKED",
    "REASON_DIFFICULTY_WINDOW",
    "REASON_TIME_REMAINING",
    "REASON_TOPIC_QUOTA",
    "eligibility_clauses",
    "eligible_items",
    "eligible_items_statement",
    "filter_eligible",
    "ineligibility_reason",
    "is_eligible",
]
