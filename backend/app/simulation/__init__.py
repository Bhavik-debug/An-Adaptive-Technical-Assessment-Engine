"""Day 13: does the adaptive selector actually beat the obvious alternatives?

Plan section 8.6.  Days 11 and 12 built an ability model and a selection policy.
Neither of them is evidence that the policy *works* - and until this package
existed, the project had asserted its design and measured none of it.

::

    ExperimentConfig                one seed, every parameter
            |
    build_environment               192 synthetic questions
            |                       200 candidates with hidden ground-truth theta
            v
    for each candidate, for each policy:
            |
            +--> should_stop?           Day 12
            +--> select an item         adaptive (Day 12) | random | fixed
            +--> grade it               Beta draw centred on Day 11's 2PL
            +--> update_ability         Day 11
            |
            v
    SessionResult  ->  StrategySummary  ->  tables

What is measured
----------------

Estimation error against the hidden truth (MAE and RMSE), convergence speed,
questions used, why sessions stopped, coverage compliance, difficulty
appropriateness - and a seven-way ablation of the six selection weights, which
Day 12 deferred to here precisely because it had no evidence for them.

What is *not* measured, and cannot be
--------------------------------------

Everything real.  A synthetic candidate answers according to the same 2PL curve
the engine assumes, so this experiment can show that the implementation recovers
abilities it was given and that one policy beats another under those
assumptions.  It says nothing about real candidates, real questions, real
interviews, or hiring validity.  ``docs/simulation.md`` states this at length
because it is the easiest claim in the project to overstate.

Ground truth, and who may see it
--------------------------------

``SyntheticCandidate.true_theta`` is read by exactly one function -
``response.graded_score`` - and its only output is a score in ``[0, 1]``, which
is what a real grader would return.  A strategy is handed a ``SelectionState``
and the bank; neither holds a reference to a candidate, so no policy can reach
the answer key even by accident.

Deferred, deliberately
----------------------

* **Day 14** - the convergence *chart*, error bands, and the reporting system
  around them.  This package computes the curve and prints eight numbers of it;
  it does not plot.
* **Day 15** - the blueprint builder.  ``config.split_budget_by_jd`` divides an
  already-given budget by already-given JD weights, and must not grow.
* Difficulty calibration from observed responses (plan section 5.11), grading,
  follow-ups, session persistence, and anything served over HTTP.
"""

from __future__ import annotations

from app.simulation.config import (
    EXTENDED_CONFIG,
    MAIN_CONFIG,
    ExperimentConfig,
    split_budget_by_jd,
)
from app.simulation.environment import (
    SyntheticCandidate,
    SyntheticEnvironment,
    build_environment,
    derive_seed,
)
from app.simulation.metrics import (
    CensoredDistribution,
    Distribution,
    absolute_errors,
    mae,
    rmse,
    summarise,
    summarise_censored,
    worst_topic_rd,
)
from app.simulation.response import expected_score, graded_score
from app.simulation.runner import (
    AblationResult,
    ExperimentResult,
    SessionResult,
    StrategySummary,
    ablation_variants,
    run_ablation,
    run_experiment,
    run_session,
)
from app.simulation.strategies import (
    ADAPTIVE,
    FIXED,
    RANDOM,
    STRATEGY_ORDER,
    AdaptiveStrategy,
    FixedStrategy,
    RandomStrategy,
    Strategy,
    build_strategies,
)

__all__ = [
    "ADAPTIVE",
    "EXTENDED_CONFIG",
    "FIXED",
    "MAIN_CONFIG",
    "RANDOM",
    "STRATEGY_ORDER",
    "AblationResult",
    "AdaptiveStrategy",
    "CensoredDistribution",
    "Distribution",
    "ExperimentConfig",
    "ExperimentResult",
    "FixedStrategy",
    "RandomStrategy",
    "SessionResult",
    "Strategy",
    "StrategySummary",
    "SyntheticCandidate",
    "SyntheticEnvironment",
    "ablation_variants",
    "absolute_errors",
    "build_environment",
    "build_strategies",
    "derive_seed",
    "expected_score",
    "graded_score",
    "mae",
    "rmse",
    "run_ablation",
    "run_experiment",
    "run_session",
    "split_budget_by_jd",
    "summarise",
    "summarise_censored",
    "worst_topic_rd",
]
