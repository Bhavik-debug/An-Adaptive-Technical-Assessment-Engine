"""Adaptive item selection: which question to ask next, and when to stop.

Plan sections 5.10 (Fisher information), 8.3 (coverage-constrained CAT) and 8.4
(the stopping rule).  Day 11 built the *estimate*; this package builds the
*decision* on top of it.

::

    Day 11  app.ability          theta + RD per subtopic
                    |
                    v
    Day 12  app.selection        which question next?
              information.py     how much would this item teach us?
              state.py           what the policy reads
              constraints.py     stage 1 - hard filters, in SQL
              objective.py       stage 2 - the six-term weighted score
              policy.py          stage 3 - rank, then epsilon-greedy
              stopping.py        when to stop asking
                    |
                    v
              the next question

The pipeline, in one line each
------------------------------

1. **Hard constraints** (``constraints.py``).  Never repeat an item; only topics
   with quota left; ``|b - theta| <= 1.5``; it must fit in the time remaining.
   Applied as SQL ``WHERE`` clauses **before** anything is scored, because these
   are policy rather than preference - a high score must never be able to buy
   its way past one.

2. **The weighted objective** (``objective.py``).  Six terms:
   ``0.40`` information ``+ 0.25`` JD alignment ``+ 0.15`` resume affinity
   ``+ 0.15`` coverage deficit ``- 0.10`` redundancy ``- 0.05`` time cost.
   Information gets the largest share because measurement is the point, and
   *not* the whole thing because a pure information-maximiser asks twelve
   questions about one subtopic and ignores the job description.
   **These six weights are design choices, not measured optima.**  Day 13's
   ablation (``app.simulation``, ``docs/simulation.md``) has now measured them
   in a synthetic environment: the information term is the only one whose
   removal clearly *worsens* the estimate, and the resume term costs accuracy
   there by concentrating items on the subtopics a resume mentions.  Nothing was
   retuned on the strength of one experiment - see that document's §9.

3. **Epsilon-greedy choice** (``policy.py``).  90% take the argmax; 10% sample
   uniformly from the top 5.  Exploration keeps the question order from being
   memorisable and generates the off-policy data difficulty recalibration will
   need.

Then ``stopping.py`` answers the other question a session has to ask each turn:
should there *be* a next item at all?

What is deliberately not here
-----------------------------

* **The blueprint builder** (topic quotas, item budget, time budget) - plan
  section 3, Day 15.  This package *consumes* a quota mapping; it does not build
  one.  DEFERRED.
* **The simulation harness** - 200 synthetic candidates, the response model and
  the CAT-vs-random-vs-fixed comparison (plan section 8.6) - now built, in
  ``app.simulation`` (Day 13).  It *imports* this package and never modifies it.
  The convergence chart is Day 14 and is still DEFERRED.
* **Difficulty recalibration** (plan section 5.11), which is what the
  exploration data is *for*.  DEFERRED.
* **The relaxation ladder** for an exhausted pool (plan section 8.7: widen the
  window, then relax the topic constraint, then end the topic early).  An empty
  pool is reported as an empty pool.  DEFERRED.
* **Follow-up policy** (plan section 8.5), grading, the event log, and any
  session persistence.  Nothing in this package performs I/O except
  ``constraints.eligible_items`` and ``policy.select_next_item``, which take a
  session and read one table.
* **Resume parsing and resume-seeded priors** (plan section 9.4).
  ``state.ResumeProfile`` is the minimal interface the objective needs, not an
  implementation of the thing that will fill it.  DEFERRED.
"""

from __future__ import annotations

from app.selection.constraints import (
    DEFAULT_CANDIDATE_LIMIT,
    DIFFICULTY_WINDOW,
    eligibility_clauses,
    eligible_items,
    eligible_items_statement,
    filter_eligible,
    ineligibility_reason,
    is_eligible,
)
from app.selection.information import (
    MAX_INFORMATION,
    fisher_information,
    information_from_p,
    normalised_information,
    selection_probability,
)
from app.selection.objective import (
    COMPONENT_NAMES,
    COVERAGE_WEIGHT,
    DEFAULT_WEIGHTS,
    INFORMATION_WEIGHT,
    JD_WEIGHT,
    REDUNDANCY_PENALTY,
    RESUME_WEIGHT,
    TIME_PENALTY,
    ObjectiveWeights,
    ScoreBreakdown,
    cosine_similarity,
    coverage_deficit,
    redundancy,
    resume_affinity,
    score_item,
    score_items,
    time_cost,
)
from app.selection.policy import (
    EPSILON,
    EXPLORATION_POOL_SIZE,
    Selection,
    choose_next,
    epsilon_greedy_select,
    rank_items,
    select_next_item,
)
from app.selection.state import (
    PRIOR_ABILITY,
    CandidateItem,
    ResumeProfile,
    SelectionState,
)
from app.selection.stopping import (
    RD_PRECISION_TARGET,
    SMALL_DELTA_THETA,
    SMALL_UPDATE_RUN,
    STOP_ITEM_BUDGET,
    STOP_NO_NEW_INFORMATION,
    STOP_PRECISION,
    STOP_TIME_BUDGET,
    StopDecision,
    consecutive_small_updates,
    precision_reached,
    should_stop,
)

__all__ = [
    "COMPONENT_NAMES",
    "COVERAGE_WEIGHT",
    "DEFAULT_CANDIDATE_LIMIT",
    "DEFAULT_WEIGHTS",
    "DIFFICULTY_WINDOW",
    "EPSILON",
    "EXPLORATION_POOL_SIZE",
    "INFORMATION_WEIGHT",
    "JD_WEIGHT",
    "MAX_INFORMATION",
    "PRIOR_ABILITY",
    "RD_PRECISION_TARGET",
    "REDUNDANCY_PENALTY",
    "RESUME_WEIGHT",
    "SMALL_DELTA_THETA",
    "SMALL_UPDATE_RUN",
    "STOP_ITEM_BUDGET",
    "STOP_NO_NEW_INFORMATION",
    "STOP_PRECISION",
    "STOP_TIME_BUDGET",
    "TIME_PENALTY",
    "CandidateItem",
    "ObjectiveWeights",
    "ResumeProfile",
    "ScoreBreakdown",
    "Selection",
    "SelectionState",
    "StopDecision",
    "choose_next",
    "consecutive_small_updates",
    "cosine_similarity",
    "coverage_deficit",
    "eligibility_clauses",
    "eligible_items",
    "eligible_items_statement",
    "epsilon_greedy_select",
    "filter_eligible",
    "fisher_information",
    "information_from_p",
    "ineligibility_reason",
    "is_eligible",
    "normalised_information",
    "precision_reached",
    "rank_items",
    "redundancy",
    "resume_affinity",
    "score_item",
    "score_items",
    "select_next_item",
    "selection_probability",
    "should_stop",
    "time_cost",
]
