"""Running the experiment: one session, then 200 of them, then the ablation.

The session loop is the whole of Day 13's contact with the production system,
and it is deliberately short enough to read in one screen::

    while True:
        stop?                       app.selection.should_stop        (Day 12)
        pick an item                the strategy under test          (Day 12)
        grade the answer            app.simulation.response          (Day 13)
        fold it in                  app.ability.update_ability       (Day 11)
        advance the state

Nothing in that loop is a re-implementation.  The stopping rule, the selector
and the ability update are imported from the packages that ship them; this
module supplies the clock, the synthetic grader, and the bookkeeping that turns
a run into numbers.

Where the state comes from, and where it does not
-------------------------------------------------

``SelectionState.ability`` starts **empty** and gains a subtopic only when that
subtopic has actually been answered.  Two consequences, both wanted:

* an unmeasured subtopic reads back as Day 12's cold-start prior (theta 0,
  RD 1.30) through ``theta_for``, so error metrics see the number a report would
  print rather than a blank;
* ``roll_up`` aggregates only real evidence, so a topic cannot look precise
  because it has many subtopics nobody asked about.

Seeding
-------

Two streams, and neither is keyed by the strategy::

    derive_seed(seed, "response", candidate, item)   the grade
    derive_seed(seed, "policy", candidate)           the policy's own randomness

Keying the response stream by *(candidate, item)* is what makes the three
policies sit the same exam: the same question always gets the same answer from
the same person.  Keying the policy stream by candidate alone means no strategy
- and no ablation variant - gets a luckier sequence of exploration draws than
another.  Both are forms of common random numbers.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from app.ability import AbilityState, update_ability
from app.selection import (
    DEFAULT_WEIGHTS,
    CandidateItem,
    ObjectiveWeights,
    SelectionState,
    should_stop,
)
from app.simulation.config import ExperimentConfig
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
    count_by,
    first_index_at_or_below,
    mae,
    mean_curve,
    rmse,
    summarise,
    summarise_censored,
    worst_topic_rd,
)
from app.simulation.response import graded_score
from app.simulation.strategies import (
    ADAPTIVE,
    STRATEGY_ORDER,
    AdaptiveStrategy,
    Strategy,
    build_strategies,
    fixed_sequence,
)
from app.simulation.strategies import (
    FixedStrategy as _FixedStrategy,
)

#: Not one of Day 12's four stopping reasons.  It records the *simulation*
#: outcome "no item satisfied the hard constraints", which Day 12 reports by
#: returning ``None``; deciding what to do about it is the relaxation ladder of
#: plan section 8.7 and is deferred, so the session simply ends and says so.
STOP_POOL_EXHAUSTED = "pool_exhausted"


# ---------------------------------------------------------------------------
# one session
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SessionResult:
    """Everything one simulated interview produced.

    Ground truth is carried here because this object is the *simulator's*
    record, written after the session ended.  It was never inside the loop's
    ``SelectionState`` and no strategy could read it.
    """

    strategy: str
    candidate_id: str
    items_asked: int
    asked_ids: tuple[str, ...]
    #: Every Day 12 stopping reason that fired, in the plan's order; or
    #: ``(STOP_POOL_EXHAUSTED,)``.
    stop_reasons: tuple[str, ...]
    final_theta: Mapping[str, float]
    final_rd: Mapping[str, float]
    ground_truth_theta: Mapping[str, float]
    #: MAE over **every blueprint subtopic**, unmeasured ones at the prior.
    mae: float
    rmse: float
    #: MAE over only the subtopics that were actually asked about; ``None`` if
    #: none were.  The optimistic reading, kept beside the honest one - and read
    #: it together with ``subtopics_measured``, because a policy that measured
    #: two subtopics beautifully and ignored four is not a better assessment.
    mae_measured: float | None
    #: How many of the blueprint's subtopics received at least one item.  The
    #: variable that explains most of the gap between ``mae`` and
    #: ``mae_measured``, so it is measured rather than inferred.
    subtopics_measured: int
    items_to_mae: int | None
    items_to_se: int | None
    coverage_met: bool
    #: ``|b - theta|`` at the moment each item was chosen - plan section 8.6's
    #: "difficulty appropriateness" measurement.
    difficulty_gaps: tuple[float, ...]
    #: MAE after 0, 1, 2, ... items.  Index 0 is the cold start.
    mae_trajectory: tuple[float, ...]
    #: Worst per-topic RD after 0, 1, 2, ... items.  ``inf`` while any required
    #: topic is still untouched.
    rd_trajectory: tuple[float, ...]
    time_used_s: float

    @property
    def primary_stop_reason(self) -> str:
        """The first reason that fired, for the distribution table.

        Day 12 reports every applicable reason; a one-per-session tally needs
        one, and "the first in the plan's order" is the least arbitrary choice.
        The full tuple stays on the record for anyone who wants it.
        """
        return self.stop_reasons[0] if self.stop_reasons else STOP_POOL_EXHAUSTED


def _theta_map(state: SelectionState, subtopics: Sequence[str]) -> dict[str, float]:
    """What the engine currently believes, for every blueprint subtopic."""
    return {subtopic: state.theta_for(subtopic) for subtopic in subtopics}


def run_session(
    environment: SyntheticEnvironment,
    candidate: SyntheticCandidate,
    strategy: Strategy,
) -> SessionResult:
    """Simulate one candidate taking one interview under one policy."""
    config = environment.config
    subtopics = config.subtopics
    topics = config.topics
    parent_of = config.parent_of
    quotas = config.quotas

    rng = random.Random(derive_seed(config.seed, "policy", candidate.id))
    ability: dict[str, AbilityState] = {}
    asked: tuple[CandidateItem, ...] = ()
    time_left = config.time_budget_s
    deltas: list[float] = []
    gaps: list[float] = []

    def current_state() -> SelectionState:
        return SelectionState(
            ability=ability,
            coverage_targets=quotas,
            jd_weights=config.jd_weights,
            time_left_s=time_left,
            asked=asked,
            resume=candidate.resume,
        )

    state = current_state()
    mae_trajectory = [
        mae(absolute_errors(_theta_map(state, subtopics), candidate.true_theta, subtopics))
    ]
    rd_trajectory = [worst_topic_rd(ability, parent_of, topics)]
    stop_reasons: tuple[str, ...] = ()

    while True:
        decision = should_stop(
            ability=ability,
            parent_of=parent_of,
            required_topics=topics,
            items_asked=len(asked),
            item_budget=config.item_budget,
            time_elapsed_s=config.time_budget_s - time_left,
            time_budget_s=config.time_budget_s,
            recent_theta_deltas=deltas,
        )
        if decision.should_stop:
            stop_reasons = decision.reasons
            break

        item = strategy.select(state, environment.bank, rng)
        if item is None:
            stop_reasons = (STOP_POOL_EXHAUSTED,)
            break

        gaps.append(abs(item.difficulty_b - state.theta_for(item.subtopic_key)))
        score = graded_score(
            config.seed, candidate, item, concentration=config.response_concentration
        )
        update = update_ability(
            state.ability_for(item.subtopic_key),
            difficulty=item.difficulty_b,
            score=score,
            discrimination=item.discrimination_a,
        )

        ability = {**ability, item.subtopic_key: update.after}
        asked = (*asked, item)
        time_left -= item.time_estimate_s
        deltas.append(update.delta_theta)

        state = current_state()
        mae_trajectory.append(
            mae(absolute_errors(_theta_map(state, subtopics), candidate.true_theta, subtopics))
        )
        rd_trajectory.append(worst_topic_rd(ability, parent_of, topics))

    errors = absolute_errors(_theta_map(state, subtopics), candidate.true_theta, subtopics)
    measured = sorted(ability)
    return SessionResult(
        strategy=strategy.name,
        candidate_id=candidate.id,
        items_asked=len(asked),
        asked_ids=tuple(item.id for item in asked),
        stop_reasons=stop_reasons,
        final_theta=_theta_map(state, subtopics),
        final_rd={s: state.rd_for(s) for s in subtopics},
        ground_truth_theta=dict(candidate.true_theta),
        mae=mae(errors),
        rmse=rmse(errors),
        mae_measured=(
            mae(absolute_errors(_theta_map(state, measured), candidate.true_theta, measured))
            if measured
            else None
        ),
        subtopics_measured=len(measured),
        items_to_mae=first_index_at_or_below(mae_trajectory, config.convergence_mae),
        items_to_se=first_index_at_or_below(rd_trajectory, config.convergence_se),
        coverage_met=all(state.asked_count(topic) >= quota for topic, quota in quotas.items()),
        difficulty_gaps=tuple(gaps),
        mae_trajectory=tuple(mae_trajectory),
        rd_trajectory=tuple(rd_trajectory),
        time_used_s=config.time_budget_s - time_left,
    )


# ---------------------------------------------------------------------------
# a population
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class StrategySummary:
    """One policy's results over the whole population."""

    strategy: str
    n_sessions: int
    mae: Distribution
    rmse: Distribution
    mae_measured: Distribution
    subtopics_measured: Distribution
    items_asked: Distribution
    difficulty_gap: Distribution
    items_to_mae: CensoredDistribution
    items_to_se: CensoredDistribution
    #: Fraction of sessions that met every topic quota (plan section 8.6's
    #: "coverage compliance").
    coverage_compliance: float
    stop_reasons: Mapping[str, int]
    #: Population-mean MAE after 0, 1, 2, ... items.  The data behind Day 14's
    #: chart; Day 13 stops at the numbers.
    mae_curve: tuple[float, ...]


def summarise_sessions(strategy: str, sessions: Sequence[SessionResult]) -> StrategySummary:
    """Aggregate one policy's sessions.  Pure; no session is dropped."""
    if not sessions:
        raise ValueError(f"no sessions to summarise for strategy {strategy!r}")
    measured = [s.mae_measured for s in sessions if s.mae_measured is not None]
    gaps = [gap for s in sessions for gap in s.difficulty_gaps]
    curve_length = max(len(s.mae_trajectory) for s in sessions)
    return StrategySummary(
        strategy=strategy,
        n_sessions=len(sessions),
        mae=summarise([s.mae for s in sessions]),
        rmse=summarise([s.rmse for s in sessions]),
        mae_measured=summarise(measured if measured else [0.0]),
        subtopics_measured=summarise([float(s.subtopics_measured) for s in sessions]),
        items_asked=summarise([float(s.items_asked) for s in sessions]),
        difficulty_gap=summarise(gaps if gaps else [0.0]),
        items_to_mae=summarise_censored([s.items_to_mae for s in sessions]),
        items_to_se=summarise_censored([s.items_to_se for s in sessions]),
        coverage_compliance=sum(1 for s in sessions if s.coverage_met) / len(sessions),
        stop_reasons=count_by([s.primary_stop_reason for s in sessions]),
        mae_curve=tuple(mean_curve([s.mae_trajectory for s in sessions], curve_length)),
    )


@dataclass(frozen=True, slots=True)
class ExperimentResult:
    """The headline comparison: every session, and one summary per policy."""

    config: ExperimentConfig
    summaries: tuple[StrategySummary, ...]
    sessions: tuple[SessionResult, ...] = field(default=(), repr=False)

    def summary(self, strategy: str) -> StrategySummary:
        for entry in self.summaries:
            if entry.strategy == strategy:
                return entry
        raise KeyError(f"no summary for strategy {strategy!r}")

    def sessions_for(self, strategy: str) -> list[SessionResult]:
        return [s for s in self.sessions if s.strategy == strategy]


def run_experiment(
    config: ExperimentConfig,
    *,
    environment: SyntheticEnvironment | None = None,
    strategies: Mapping[str, Strategy] | None = None,
    order: Sequence[str] = STRATEGY_ORDER,
) -> ExperimentResult:
    """Run every candidate through every policy on one shared environment.

    The environment is built once and passed to all of them, which is the
    mechanical form of "same bank, same candidates, same difficulties, same
    discriminations, same topic distribution, same budgets".
    """
    env = environment if environment is not None else build_environment(config)
    policies = strategies if strategies is not None else build_strategies(env.bank, config.quotas)

    sessions: list[SessionResult] = []
    for name in order:
        policy = policies[name]
        for candidate in env.candidates:
            sessions.append(run_session(env, candidate, policy))

    return ExperimentResult(
        config=config,
        summaries=tuple(
            summarise_sessions(name, [s for s in sessions if s.strategy == name]) for name in order
        ),
        sessions=tuple(sessions),
    )


# ---------------------------------------------------------------------------
# the ablation
# ---------------------------------------------------------------------------

#: The label the unmodified production objective runs under.
ABLATION_FULL = "full"


def ablation_variants(
    components: Sequence[str],
    base: ObjectiveWeights = DEFAULT_WEIGHTS,
) -> dict[str, ObjectiveWeights]:
    """``{"full": base, "no_information": base.without("information"), ...}``.

    **Zeroing one weight is the whole operation, and no renormalisation
    follows.**  Within a session every item is scored with the same weights, so
    multiplying all six by a constant multiplies every score and leaves the
    argmax and the top five untouched.  Rescaling the survivors would therefore
    change nothing about the decisions while making the ablation harder to
    describe - so "remove a component" means exactly "its weight is 0".
    """
    variants = {ABLATION_FULL: base}
    for component in components:
        variants[f"no_{component}"] = base.without(component)
    return variants


@dataclass(frozen=True, slots=True)
class AblationResult:
    """One summary per objective variant, all on the same environment."""

    config: ExperimentConfig
    summaries: tuple[StrategySummary, ...]

    def summary(self, variant: str) -> StrategySummary:
        for entry in self.summaries:
            if entry.strategy == variant:
                return entry
        raise KeyError(f"no summary for variant {variant!r}")


def run_ablation(
    config: ExperimentConfig,
    *,
    environment: SyntheticEnvironment | None = None,
    variants: Mapping[str, ObjectiveWeights] | None = None,
) -> AblationResult:
    """Run the adaptive policy once per objective variant, everything else fixed.

    Only the weights change between variants: the same environment, the same
    candidates, the same response stream and the same policy stream.  Any
    difference in the numbers is caused by the objective and nothing else.
    """
    from app.selection import COMPONENT_NAMES

    env = environment if environment is not None else build_environment(config)
    configurations = variants if variants is not None else ablation_variants(COMPONENT_NAMES)

    summaries: list[StrategySummary] = []
    for label, weights in configurations.items():
        policy = AdaptiveStrategy(weights=weights, label=label)
        sessions = [run_session(env, candidate, policy) for candidate in env.candidates]
        summaries.append(summarise_sessions(label, sessions))
    return AblationResult(config=config, summaries=tuple(summaries))


def build_fixed_strategy(environment: SyntheticEnvironment) -> _FixedStrategy:
    """The fixed policy for an environment - exposed for tests and scripts."""
    return _FixedStrategy(sequence=fixed_sequence(environment.bank, environment.config.quotas))


__all__ = [
    "ABLATION_FULL",
    "ADAPTIVE",
    "STOP_POOL_EXHAUSTED",
    "AblationResult",
    "ExperimentResult",
    "SessionResult",
    "StrategySummary",
    "ablation_variants",
    "build_fixed_strategy",
    "run_ablation",
    "run_experiment",
    "run_session",
    "summarise_sessions",
]
