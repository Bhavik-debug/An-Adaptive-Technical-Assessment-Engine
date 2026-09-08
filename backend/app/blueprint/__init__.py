"""Day 15: what assessment do we need?  The contract Day 12 executes.

Plan section 3, Day 15: *"Blueprint builder: role + level + duration + JD weights
-> topic quotas, item budget, time budget."*

Days 11-14 built an engine that could measure a candidate and an experiment that
said how well.  All four of them took their requirements from somewhere else -
hand-built test fixtures, ``ExperimentConfig``, a literal in a docstring.
``SelectionState`` has said so since Day 12::

    coverage_targets
        ``topic_key -> how many items this topic must get``.  Consumed, never
        built: the blueprint builder that produces it is Day 15, and is
        DEFERRED.

This package is that producer.

::

    AssessmentSpec              role, level, duration, mode, JD weights
            |                   "what is wanted" - may be under-specified
            |
    build_blueprint             divide the budget; check it is possible
            |
            v
    Blueprint                   quotas, budgets, required topics, cold start
            |                   "what must happen" - fully determined
            |
            +--> selection_state(...)     -> SelectionState        (Day 12)
            +--> stopping_arguments(...)  -> should_stop(**...)    (Day 12)
            +--> to_dict()                -> interview_sessions.blueprint

The boundary, and why it is worth the package
---------------------------------------------

======================  ===========================================
Blueprint               "what assessment do we need?"
Selector    (Day 12)    "which question should we ask next?"
Estimator   (Day 11)    "what do we currently believe?"
Stopping    (Day 12)    "should the assessment end?"
======================  ===========================================

A blueprint never looks at a question, never scores one, and never touches an
ability estimate.  Everything it produces is an argument some Day 11 or Day 12
function already declared, so integration is a constructor call rather than a
refactor - and ``app.selection`` does not import this package, which keeps the
dependency arrow pointing one way.

Two numbers that are calibrated rather than chosen
--------------------------------------------------

``MODE_SECONDS_PER_ITEM`` comes from the quartiles of ``time_estimate_s`` across
the 60 committed bank items (p25 292.5, p50 330, p75 390), rounded to the
30-second granularity the authors used.  A 45-minute standard assessment is
therefore **8 questions**, which is a smaller and more honest number than the
20-item budget Day 13's synthetic bank supported.  See ``docs/blueprint.md``.

``LEVEL_TARGET_THETA`` maps seniority to a cold-start theta, and ``mid`` is
exactly 0.0 - so a default blueprint reproduces Day 12's ``PRIOR_ABILITY``
unchanged and nothing about the selector's behaviour moves unless a caller picks
another level.

Deliberately not here
---------------------

* **Question selection.**  Day 12's, entirely.
* **JD parsing and resume parsing.**  A spec takes JD weights as numbers; deriving
  them from a job description is Phase 5.  ``ResumeProfile`` remains Day 12's
  minimal interface.  DEFERRED.
* **Persistence.**  ``to_dict``/``from_dict`` produce and consume the JSONB shape;
  writing the row belongs to the session layer, which does not exist yet.
  DEFERRED.
* **Validating a blueprint against the *bank*.**  ``check_against_taxonomy``
  verifies the keys exist in the controlled vocabulary; whether enough questions
  exist at the right difficulty to satisfy a quota is a database question and is
  plan section 8.7's relaxation ladder.  DEFERRED.
* **Simulating whether a level-seeded prior helps.**  ``LEVEL_TARGET_THETA`` is
  reasoned, not measured.  DEFERRED.
"""

from __future__ import annotations

from app.blueprint.blueprint import (
    BLUEPRINT_VERSION,
    Blueprint,
    TopicPlan,
)
from app.blueprint.builder import (
    build_blueprint,
    check_against_taxonomy,
    derive_item_budget,
    resolve_subtopics,
)
from app.blueprint.quotas import allocate_quotas
from app.blueprint.spec import (
    DEFAULT_LEVEL,
    DEFAULT_MODE,
    LEVEL_TARGET_THETA,
    LEVELS,
    MAX_DURATION_MINUTES,
    MAX_ITEM_BUDGET,
    MIN_DURATION_MINUTES,
    MIN_ITEM_BUDGET,
    MODE_SECONDS_PER_ITEM,
    MODES,
    AssessmentSpec,
    BlueprintError,
    TopicRequest,
)

__all__ = [
    "BLUEPRINT_VERSION",
    "DEFAULT_LEVEL",
    "DEFAULT_MODE",
    "LEVELS",
    "LEVEL_TARGET_THETA",
    "MAX_DURATION_MINUTES",
    "MAX_ITEM_BUDGET",
    "MIN_DURATION_MINUTES",
    "MIN_ITEM_BUDGET",
    "MODES",
    "MODE_SECONDS_PER_ITEM",
    "AssessmentSpec",
    "Blueprint",
    "BlueprintError",
    "TopicPlan",
    "TopicRequest",
    "allocate_quotas",
    "build_blueprint",
    "check_against_taxonomy",
    "derive_item_budget",
    "resolve_subtopics",
]
