"""When to stop asking - the four conditions of plan section 8.4.

::

    STOP when:  all required topics have RD < 0.40      # sufficient precision
            or  items_asked >= item_budget              # out of questions
            or  time_elapsed >= time_budget             # out of time
            or  3 consecutive items with |dtheta| < 0.05   # nothing new arriving

**Any** of them, not all: three are limits and one is an achievement, and an
interview should end at whichever comes first.

Why four, and why the fourth is the interesting one
---------------------------------------------------

The middle two are budgets - the interview cannot run forever and the candidate
was promised a length.  The first is the point of an adaptive test at all: once
every topic the blueprint required is measured to within +/-0.40, more questions
buy precision nobody needs, and asking them is time taken from a person for no
information.  That is the condition that lets a 20-item fixed test become an
11-item adaptive one.

The fourth - three consecutive items that moved theta by less than 0.05 - is the
"we have learned everything this bank can tell us about this candidate"
detector.  It catches the case the RD condition misses: the estimate has settled
even though RD has not yet crossed its threshold, because the remaining items
are not informative enough to move it.  Three in a row rather than one, because
a single small step happens routinely when a prediction happens to be right; a
run of three is a pattern.

The thresholds - 0.40, the item and time budgets, 0.05, and the run length of 3
- come from the plan and are keyword arguments here so a caller or a test can
vary them without any of them being quietly redefined.

This module is pure arithmetic over Day 11's state, like everything else in
:mod:`app.selection`.  In particular the precision condition is answered by
:func:`app.ability.roll_up` - the *same* precision-weighted aggregation a report
would use - rather than by a second opinion about what a topic's RD is.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from app.ability import AbilityState, roll_up

#: "Sufficient precision" for a required topic (plan section 8.4).  RD is a
#: standard error, so 0.40 is a 95% interval of about +/-0.78 on a scale that
#: runs -3 to 3 - tight enough to place someone in a band and to defend it.
RD_PRECISION_TARGET = 0.40

#: A theta step this small carried no news (plan section 8.4).  Strict: a step
#: of exactly 0.05 is *not* small.
SMALL_DELTA_THETA = 0.05

#: How many such steps in a row before we believe it.
SMALL_UPDATE_RUN = 3

STOP_PRECISION = "precision_reached"
STOP_ITEM_BUDGET = "item_budget_reached"
STOP_TIME_BUDGET = "time_budget_reached"
STOP_NO_NEW_INFORMATION = "no_new_information"


@dataclass(frozen=True, slots=True)
class StopDecision:
    """Whether to stop, and every reason that applied.

    All of them, not the first one: two conditions firing together is a
    genuinely different situation from one firing alone - "we ran out of time
    *and* we had already reached precision" is a good interview, "we ran out of
    time and had not" is a truncated one - and a report that can tell them apart
    is worth the tuple.
    """

    reasons: tuple[str, ...]

    @property
    def should_stop(self) -> bool:
        return bool(self.reasons)

    def __bool__(self) -> bool:
        return self.should_stop


def consecutive_small_updates(
    deltas: Sequence[float],
    *,
    threshold: float = SMALL_DELTA_THETA,
) -> int:
    """How many of the **most recent** updates in a row had ``|dtheta| < threshold``.

    Counting the trailing run - rather than "how many small ones are there" - is
    what gives the reset for free: one substantial step anywhere ends the run,
    so ``[0.01, 0.01, 0.40, 0.01]`` counts 1, not 3.  That is the intended
    reading of "3 consecutive items".

    ``deltas`` is oldest-first, which is the order an event log produces.
    """
    if threshold <= 0.0:
        raise ValueError(f"threshold must be positive, got {threshold!r}")
    run = 0
    for delta in reversed(deltas):
        if not math.isfinite(delta):
            raise ValueError(f"delta must be finite, got {delta!r}")
        if abs(delta) < threshold:
            run += 1
        else:
            break
    return run


def precision_reached(
    ability: Mapping[str, AbilityState],
    parent_of: Mapping[str, str],
    required_topics: Sequence[str],
    *,
    rd_target: float = RD_PRECISION_TARGET,
) -> bool:
    """True when every required **topic** is measured to ``RD < rd_target``.

    Topic RD is not stored - it is derived from the subtopic states by
    :func:`app.ability.roll_up`, the precision-weighted aggregation of plan
    section 9.2.  Reusing it here rather than writing a second rule matters:
    otherwise the number that ends the interview and the number that appears in
    the report could disagree, and only one of them would be visible.

    Note that aggregation *shrinks* RD - a topic backed by three measured
    subtopics is known better than any one of them - so this condition can be
    satisfied without every individual subtopic reaching 0.40.  That is correct:
    the blueprint's quotas are stated at topic level, and so is this.

    A required topic with no measured subtopic is **not** satisfied: we cannot
    claim precision about something never asked.

    An empty ``required_topics`` returns ``False``, not a vacuous ``True``.  A
    stopping rule that fires before the first question is a bug, not an
    achievement, and "no topics were required" is a broken blueprint rather than
    a completed one.

    ``parent_of`` must cover every subtopic in ``ability`` - ``roll_up`` raises
    otherwise, which is deliberate: silently dropping a subtopic would change a
    topic's RD with nothing to show for it.
    """
    if rd_target <= 0.0:
        raise ValueError(f"rd_target must be positive, got {rd_target!r}")
    if not required_topics:
        return False
    if not ability:
        return False

    topics = roll_up(ability, parent_of)
    return all(topic in topics and topics[topic].rd < rd_target for topic in required_topics)


def should_stop(
    *,
    ability: Mapping[str, AbilityState],
    parent_of: Mapping[str, str],
    required_topics: Sequence[str],
    items_asked: int,
    item_budget: int,
    time_elapsed_s: float,
    time_budget_s: float,
    recent_theta_deltas: Sequence[float] = (),
    rd_target: float = RD_PRECISION_TARGET,
    small_delta: float = SMALL_DELTA_THETA,
    small_update_run: int = SMALL_UPDATE_RUN,
) -> StopDecision:
    """Evaluate all four conditions and report every one that fired.

    Keyword-only throughout: eight of these arguments are numbers, and a
    positional call site that swapped ``items_asked`` and ``item_budget`` would
    end every interview at question zero while type-checking perfectly.

    Both budget comparisons are ``>=``: an interview that has asked its 12th of
    12 items is finished, not one item short.
    """
    if items_asked < 0 or item_budget < 0:
        raise ValueError(
            f"items_asked and item_budget must not be negative, "
            f"got {items_asked!r} and {item_budget!r}"
        )
    if not math.isfinite(time_elapsed_s) or not math.isfinite(time_budget_s):
        raise ValueError(
            f"time_elapsed_s and time_budget_s must be finite, "
            f"got {time_elapsed_s!r} and {time_budget_s!r}"
        )
    if time_elapsed_s < 0.0 or time_budget_s < 0.0:
        raise ValueError(
            f"time_elapsed_s and time_budget_s must not be negative, "
            f"got {time_elapsed_s!r} and {time_budget_s!r}"
        )
    if small_update_run < 1:
        raise ValueError(f"small_update_run must be at least 1, got {small_update_run!r}")

    reasons: list[str] = []
    if precision_reached(ability, parent_of, required_topics, rd_target=rd_target):
        reasons.append(STOP_PRECISION)
    if items_asked >= item_budget:
        reasons.append(STOP_ITEM_BUDGET)
    if time_elapsed_s >= time_budget_s:
        reasons.append(STOP_TIME_BUDGET)
    if consecutive_small_updates(recent_theta_deltas, threshold=small_delta) >= small_update_run:
        reasons.append(STOP_NO_NEW_INFORMATION)

    return StopDecision(reasons=tuple(reasons))


__all__ = [
    "RD_PRECISION_TARGET",
    "SMALL_DELTA_THETA",
    "SMALL_UPDATE_RUN",
    "STOP_ITEM_BUDGET",
    "STOP_NO_NEW_INFORMATION",
    "STOP_PRECISION",
    "STOP_TIME_BUDGET",
    "StopDecision",
    "consecutive_small_updates",
    "precision_reached",
    "should_stop",
]
