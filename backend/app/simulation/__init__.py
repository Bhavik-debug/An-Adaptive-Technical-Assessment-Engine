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

Day 14 - the convergence analysis
---------------------------------

Day 13 answered *where the error ends*.  Day 14 answers *how it got there*,
from the same runs::

    run_experiment          Day 13, unchanged
            |
            |  SessionResult.trajectory - one TrajectoryStep per turn
            v
    convergence.py          bands, reach curves, stability, stopping profile
            |
    figures.py              which charts, and what each is for
            |
    charts.py               deterministic SVG - no plotting library
            |
            v
    evals/reports/convergence.md  +  evals/reports/convergence/*.svg

The only change Day 14 made to the Day 13 loop was to record what the engine
believed after each turn.  That record consumes no randomness and is read by no
policy, and Day 13's committed report regenerates byte-identically - which is
the check that the analysis did not perturb the thing it analyses.  Day 14
introduced **no new convergence threshold**: both criteria are Day 13's.

Deferred, deliberately
----------------------

* **Day 15** - the blueprint builder, now built as :mod:`app.blueprint`.
  ``config.split_budget_by_jd`` keeps its name and its one-division contract and
  delegates the arithmetic to ``app.blueprint.allocate_quotas``, so there is a
  single largest-remainder implementation.  The Day 13 report regenerates
  byte-identically across that change.
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
from app.simulation.convergence import (
    Band,
    CoverageBin,
    StabilitySummary,
    StoppingProfile,
    at_or_below_curve,
    coverage_band,
    coverage_vs_error,
    mae_band,
    reach_curve,
    representative_candidates,
    rmse_band,
    stability,
    stopping_by_items,
    theta_series,
)
from app.simulation.environment import (
    SyntheticCandidate,
    SyntheticEnvironment,
    build_environment,
    derive_seed,
)
from app.simulation.figures import (
    Figure,
    build_figures,
    write_figures,
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
    TrajectoryStep,
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
    "AblationResult",
    "AdaptiveStrategy",
    "Band",
    "CensoredDistribution",
    "CoverageBin",
    "Distribution",
    "EXTENDED_CONFIG",
    "ExperimentConfig",
    "ExperimentResult",
    "FIXED",
    "Figure",
    "FixedStrategy",
    "MAIN_CONFIG",
    "RANDOM",
    "RandomStrategy",
    "STRATEGY_ORDER",
    "SessionResult",
    "StabilitySummary",
    "StoppingProfile",
    "Strategy",
    "StrategySummary",
    "SyntheticCandidate",
    "SyntheticEnvironment",
    "TrajectoryStep",
    "ablation_variants",
    "absolute_errors",
    "at_or_below_curve",
    "build_environment",
    "build_figures",
    "build_strategies",
    "coverage_band",
    "coverage_vs_error",
    "derive_seed",
    "expected_score",
    "graded_score",
    "mae",
    "mae_band",
    "reach_curve",
    "representative_candidates",
    "rmse",
    "rmse_band",
    "run_ablation",
    "run_experiment",
    "run_session",
    "split_budget_by_jd",
    "stability",
    "stopping_by_items",
    "summarise",
    "summarise_censored",
    "theta_series",
    "worst_topic_rd",
    "write_figures",
]
