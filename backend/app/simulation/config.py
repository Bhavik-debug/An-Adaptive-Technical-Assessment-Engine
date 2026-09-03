"""Every number the Day 13 experiment depends on, in one place.

Plan section 8.6.  If a constant appears anywhere else in
:mod:`app.simulation`, that is a bug: an experiment whose parameters are
scattered across the code that runs it cannot be re-run with a different seed,
and cannot be reported honestly because nobody can list what it assumed.

Why the environment is synthetic
--------------------------------

The committed 60-item bank is a fine *dataset* and a poor *experimental
environment*, and the difference is worth stating because it is the reason this
module exists at all.  Measured on the real bank:

===========================  ==========================================
committed bank               why it cannot support this experiment
===========================  ==========================================
60 items                     a 20-item session consumes a third of it, so
                             the policies would mostly be choosing between
                             the same handful of remaining questions
35 subtopics, 1-3 items each a subtopic can be asked at most three times,
                             so theta there can never converge
b in [-1.2, +1.6]            a candidate at theta = -2.5 has *nothing*
                             inside the |b - theta| <= 1.5 window
every a = 1.0                discrimination has no variance, so the one
                             place the response model uses it is inert
150-420 s per item           twenty items is 110 minutes; a realistic time
                             budget would stop every session at item five
===========================  ==========================================

So Day 13 builds a controlled synthetic environment instead, and every
conclusion it produces is a conclusion **about that environment**.  See
``docs/simulation.md``, "What this cannot tell us".

The two readings of "200"
-------------------------

Plan section 8.6 says *"Generate 200 synthetic candidates with known
ground-truth theta vectors"*, so 200 is the **candidate** count.  The bank
happens to land at 192 items as well, which satisfies the looser reading too;
that is a coincidence of 6 subtopics x 32 items, not a second requirement being
smuggled in.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from app.ability import RD_MAX, THETA_MAX, THETA_MIN

# ---------------------------------------------------------------------------
# the taxonomy
# ---------------------------------------------------------------------------

#: ``topic -> its subtopics``.  Three topics of two subtopics each.
#:
#: **Why this size.**  It is a trade-off, and the losing side is worth naming.
#: Every observation lands on exactly one subtopic, so with a 20-item budget a
#: taxonomy of 16 subtopics gives 1.25 observations each and nothing converges;
#: a taxonomy of two gives ten each and the "coverage" half of the objective has
#: nothing to do.  Six subtopics puts ~3.3 observations on each, which is enough
#: for theta to move a long way from its prior and not enough for RD to reach
#: Day 12's 0.40 precision threshold - a fact this experiment reports rather
#: than designs around.
TAXONOMY: Mapping[str, tuple[str, ...]] = {
    "algorithms": ("arrays", "graphs"),
    "databases": ("indexing", "transactions"),
    "systems": ("caching", "queues"),
}

#: ``subtopic -> topic``.  The ``parent_of`` argument Day 11's ``roll_up`` and
#: Day 12's ``precision_reached`` both take.
PARENT_OF: Mapping[str, str] = {
    subtopic: topic for topic, subtopics in TAXONOMY.items() for subtopic in subtopics
}

#: Every subtopic, in a fixed order, so that error vectors are comparable.
ALL_SUBTOPICS: tuple[str, ...] = tuple(
    subtopic for subtopics in TAXONOMY.values() for subtopic in subtopics
)

ALL_TOPICS: tuple[str, ...] = tuple(TAXONOMY)

#: The role this simulated interview is for, as a per-topic alignment weight.
#:
#: **A stand-in, not a JD parser.**  Phase 5 derives these from a real job
#: description; here they are three fixed numbers, chosen only to be unequal so
#: that the JD term in the objective has something to discriminate between.
#: They are a property of the *role* and are identical for all 200 candidates -
#: which is what a job description is.
JD_WEIGHTS: Mapping[str, float] = {
    "algorithms": 0.9,
    "databases": 0.7,
    "systems": 0.5,
}


# ---------------------------------------------------------------------------
# the bank
# ---------------------------------------------------------------------------

#: Items per subtopic.  6 x 32 = 192 items, nearly ten times the 20-item budget,
#: so a policy is genuinely choosing rather than running out.
ITEMS_PER_SUBTOPIC = 32

#: Difficulties are spread evenly across this range within each subtopic, so
#: every subtopic has something inside the |b - theta| <= 1.5 window for any
#: candidate the population generator can produce.
BANK_DIFFICULTY_MIN = -2.5
BANK_DIFFICULTY_MAX = 2.5

#: A small deterministic wobble on top of the even spread, so the bank is not a
#: perfect lattice (which would make exact score ties the common case and hand
#: the id tie-break more influence than it should have).
BANK_DIFFICULTY_JITTER = 0.15

#: Discrimination range.  The real bank has a = 1.0 everywhere; giving it
#: variance here is what makes the ``a`` in the response model and in the RD
#: update do observable work.  It is deliberately *not* in the selection score -
#: that is Day 12's design (plan section 8.3 writes ``sigmoid(theta - b)``).
BANK_DISCRIMINATION_MIN = 0.7
BANK_DISCRIMINATION_MAX = 1.6

#: Per-item time estimates, in seconds.  Drawn uniformly from this tuple.
BANK_TIME_ESTIMATES_S: tuple[int, ...] = (60, 90, 120, 150, 180)

#: Width of the synthetic embedding space.  Small on purpose: the redundancy
#: term is O(pool x asked) cosines and dominates the run time, and 16 dimensions
#: separate six subtopics perfectly well.  Nothing here claims to be a language
#: model - it is a geometry with the one property redundancy needs, namely that
#: two items from the same subtopic are close and two from different subtopics
#: are not.
EMBEDDING_DIM = 16

#: How far an item's vector is allowed to drift from its subtopic's axis.
#:
#: Calibrated, not guessed.  The noise has norm ~ ``spread * sqrt(dim)``, so a
#: spread of 0.35 over 16 dimensions swamps the unit axis entirely and two items
#: from *different* subtopics can end up closer than two from the same one -
#: which would silently give the redundancy term a geometry with no subtopic
#: structure in it at all.  At 0.10, measured over every pair in the 192-item
#: bank:
#:
#:     same subtopic       mean 0.871, range 0.631 to 0.985
#:     different subtopic  mean -0.005, range -0.518 to 0.489
#:
#: so every same-subtopic pair is closer than every cross-subtopic pair, which
#: is the shape the real bge embedder produces on paraphrases versus unrelated
#: questions.  Asserted by test rather than assumed.
EMBEDDING_SPREAD = 0.10


# ---------------------------------------------------------------------------
# the candidate population
# ---------------------------------------------------------------------------

#: Plan section 8.6: "Generate 200 synthetic candidates".
CANDIDATE_COUNT = 200

#: Ground-truth theta is drawn hierarchically::
#:
#:     overall_i    ~ Normal(0, CANDIDATE_OVERALL_SD)
#:     theta_i,s    ~ Normal(overall_i, CANDIDATE_SUBTOPIC_SD),  clamped
#:
#: The two-level draw matters.  Drawing every subtopic independently would make
#: a candidate who is strong at arrays no more likely to be strong at graphs,
#: which is not how people are, and would remove the correlation that makes an
#: early answer informative about the *next* subtopic.  Drawing one number per
#: candidate would remove the per-subtopic variation the engine exists to find.
CANDIDATE_OVERALL_SD = 0.8
CANDIDATE_SUBTOPIC_SD = 0.6

#: Ground truth is clamped inside the bank's difficulty range rather than to
#: Day 11's [-3, 3], so that every candidate is in principle measurable by the
#: items that exist.  A candidate at theta = 3.0 with a hardest item of b = 2.5
#: could never be pinned down, and the resulting error would be a fact about the
#: bank, not about the policy.
CANDIDATE_THETA_MIN = max(THETA_MIN, BANK_DIFFICULTY_MIN)
CANDIDATE_THETA_MAX = min(THETA_MAX, BANK_DIFFICULTY_MAX)

#: How many subtopics a synthetic resume mentions, and how strong a mention is.
#:
#: **These affinities are drawn from a stream that never sees ground truth.**
#: That is a deliberate experimental choice, not an oversight: letting the
#: resume correlate with true ability would hand the adaptive policy an
#: information channel the baselines do not use, and "CAT wins" would then be
#: unfalsifiable.  The cost is that the ablation can only measure what the
#: resume term *costs* in measurement accuracy, never what it delivers in
#: perceived relevance - which this simulation cannot measure at all.
RESUME_SUBTOPICS = 2
RESUME_AFFINITY_MIN = 0.5
RESUME_AFFINITY_MAX = 1.0


# ---------------------------------------------------------------------------
# the response model
# ---------------------------------------------------------------------------

#: Beta concentration.  ``alpha = p*k``, ``beta = (1-p)*k``, so the mean is
#: exactly ``p`` and the variance is ``p(1-p)/(k+1)``.  At k = 10 a coin-flip
#: item is graded with a standard deviation of 0.151, which is about the spread
#: two competent human graders show on the same answer.  Larger k is a more
#: reliable grader, smaller k a noisier one.
RESPONSE_CONCENTRATION = 10.0

#: ``p`` is clamped into ``[eps, 1-eps]`` before it becomes Beta parameters. At
#: p = 0 exactly, ``alpha = 0`` and the distribution is degenerate; the clamp
#: keeps it defined at the extremes of the difficulty range.
RESPONSE_P_EPSILON = 0.01


# ---------------------------------------------------------------------------
# the session
# ---------------------------------------------------------------------------

#: Plan section 8.6: "Run three policies to 20 items each".
DEFAULT_ITEM_BUDGET = 20

#: 45 minutes.  Twenty items at the bank's mean 120 s is 40 minutes, so this
#: budget binds for sessions that pick expensive items and not for others -
#: which is what makes the stopping-reason distribution informative rather than
#: a single constant.  Identical for every strategy.
DEFAULT_TIME_BUDGET_S = 2700.0

#: The cold start every session begins from, for every strategy: no measured
#: subtopic at all, so ``SelectionState`` falls back to Day 12's
#: ``PRIOR_ABILITY`` (theta 0, RD 1.30).  Stated here as data so the report can
#: print what the run assumed.
INITIAL_THETA = 0.0
INITIAL_RD = RD_MAX

#: MAE at or below this counts as "close enough to the truth to be useful".
#:
#: **A design choice, and an arbitrary one.**  0.5 is one sixth of the [-3, 3]
#: scale and about the width of one difficulty band, so an estimate this close
#: puts a candidate in the right band.  It is reported alongside the full error
#: curve precisely so that a reader who dislikes the threshold can ignore it.
CONVERGENCE_MAE = 0.5

#: Plan section 8.6: "Report items-to-reach-SE-0.35".  Applied at **topic**
#: level, via Day 11's ``roll_up`` - the same aggregation Day 12's stopping rule
#: uses, one notch tighter than its 0.40.  Subtopic level would need roughly 23
#: observations *per subtopic* to reach and is unreachable at any budget this
#: experiment runs.
CONVERGENCE_SE = 0.35


# ---------------------------------------------------------------------------
# the experiment
# ---------------------------------------------------------------------------


def split_budget_by_jd(
    item_budget: int,
    jd_weights: Mapping[str, float],
) -> dict[str, int]:
    """Divide an item budget across topics in proportion to their JD weight.

    ``{algorithms: 0.9, databases: 0.7, systems: 0.5}`` and a budget of 20 give
    ``{algorithms: 9, databases: 7, systems: 4}``.  Largest-remainder rounding,
    so the quotas sum to exactly the budget and no topic is silently starved by
    repeated flooring.

    **This is not the Day 15 blueprint builder and must not grow into one.**
    Day 15 turns *role + level + duration* into JD weights, an item budget and a
    time budget; this takes all three as given and does one division, because
    Day 12's hard topic cap needs a quota mapping and there is nothing yet to
    produce one.  Ties are broken by topic name so the result is deterministic.
    """
    if item_budget < 0:
        raise ValueError(f"item_budget must not be negative, got {item_budget!r}")
    total = sum(jd_weights.values())
    if total <= 0.0:
        raise ValueError("jd_weights must not sum to zero")

    exact = {topic: item_budget * weight / total for topic, weight in jd_weights.items()}
    quotas = {topic: int(value) for topic, value in exact.items()}
    remaining = item_budget - sum(quotas.values())
    # Largest remainder first; topic name breaks ties so two runs agree.
    order = sorted(exact, key=lambda topic: (-(exact[topic] - quotas[topic]), topic))
    for topic in order[:remaining]:
        quotas[topic] += 1
    return quotas


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """One complete, re-runnable experiment.

    Every stochastic component downstream derives its seed from ``seed``, so two
    ``ExperimentConfig`` values that compare equal produce byte-identical
    results - which is asserted by test, not hoped for.
    """

    seed: int = 20260101
    candidate_count: int = CANDIDATE_COUNT
    items_per_subtopic: int = ITEMS_PER_SUBTOPIC
    item_budget: int = DEFAULT_ITEM_BUDGET
    time_budget_s: float = DEFAULT_TIME_BUDGET_S
    response_concentration: float = RESPONSE_CONCENTRATION
    convergence_mae: float = CONVERGENCE_MAE
    convergence_se: float = CONVERGENCE_SE
    taxonomy: Mapping[str, tuple[str, ...]] = field(default_factory=lambda: dict(TAXONOMY))
    jd_weights: Mapping[str, float] = field(default_factory=lambda: dict(JD_WEIGHTS))
    #: Set on the extended run so the report can name it.  Purely a label.
    label: str = "main"

    def __post_init__(self) -> None:
        if self.candidate_count <= 0:
            raise ValueError(f"candidate_count must be positive, got {self.candidate_count!r}")
        if self.items_per_subtopic <= 0:
            raise ValueError(
                f"items_per_subtopic must be positive, got {self.items_per_subtopic!r}"
            )
        if self.item_budget <= 0:
            raise ValueError(f"item_budget must be positive, got {self.item_budget!r}")
        if self.time_budget_s <= 0.0:
            raise ValueError(f"time_budget_s must be positive, got {self.time_budget_s!r}")
        if self.response_concentration <= 0.0:
            raise ValueError(
                f"response_concentration must be positive, got {self.response_concentration!r}"
            )
        if set(self.taxonomy) != set(self.jd_weights):
            raise ValueError("taxonomy and jd_weights must cover the same topics")

    @property
    def subtopics(self) -> tuple[str, ...]:
        return tuple(s for subtopics in self.taxonomy.values() for s in subtopics)

    @property
    def topics(self) -> tuple[str, ...]:
        return tuple(self.taxonomy)

    @property
    def parent_of(self) -> dict[str, str]:
        return {s: topic for topic, subs in self.taxonomy.items() for s in subs}

    @property
    def quotas(self) -> dict[str, int]:
        """The blueprint's per-topic caps, derived rather than stored."""
        return split_budget_by_jd(self.item_budget, self.jd_weights)

    @property
    def bank_size(self) -> int:
        return len(self.subtopics) * self.items_per_subtopic

    def describe(self) -> Sequence[tuple[str, str]]:
        """Label/value pairs for the report's configuration block."""
        return (
            ("label", self.label),
            ("seed", str(self.seed)),
            ("candidates", str(self.candidate_count)),
            ("bank size", f"{self.bank_size} items"),
            ("taxonomy", f"{len(self.topics)} topics x {len(self.subtopics) // len(self.topics)}"),
            ("item budget", str(self.item_budget)),
            ("time budget", f"{self.time_budget_s:.0f} s"),
            ("topic quotas", ", ".join(f"{t}={q}" for t, q in sorted(self.quotas.items()))),
            ("jd weights", ", ".join(f"{t}={w}" for t, w in sorted(self.jd_weights.items()))),
            ("initial theta / RD", f"{INITIAL_THETA} / {INITIAL_RD}"),
            ("beta concentration", str(self.response_concentration)),
            ("convergence MAE", str(self.convergence_mae)),
            ("convergence SE", str(self.convergence_se)),
        )


#: The plan's experiment: 200 candidates, 20 items each.
MAIN_CONFIG = ExperimentConfig()

#: The same environment run three times longer.  It exists for one reason: at a
#: 20-item budget no RD-based convergence criterion ever fires, so
#: ``items_to_se`` is censored for every session and reports nothing.  Sixty
#: items is where that metric starts to have values in it.  Clearly labelled as
#: secondary; the plan's experiment is ``MAIN_CONFIG``.
EXTENDED_CONFIG = ExperimentConfig(
    item_budget=60,
    time_budget_s=8100.0,
    label="extended",
)


__all__ = [
    "ALL_SUBTOPICS",
    "ALL_TOPICS",
    "BANK_DIFFICULTY_JITTER",
    "BANK_DIFFICULTY_MAX",
    "BANK_DIFFICULTY_MIN",
    "BANK_DISCRIMINATION_MAX",
    "BANK_DISCRIMINATION_MIN",
    "BANK_TIME_ESTIMATES_S",
    "CANDIDATE_COUNT",
    "CANDIDATE_OVERALL_SD",
    "CANDIDATE_SUBTOPIC_SD",
    "CANDIDATE_THETA_MAX",
    "CANDIDATE_THETA_MIN",
    "CONVERGENCE_MAE",
    "CONVERGENCE_SE",
    "DEFAULT_ITEM_BUDGET",
    "DEFAULT_TIME_BUDGET_S",
    "EMBEDDING_DIM",
    "EMBEDDING_SPREAD",
    "EXTENDED_CONFIG",
    "INITIAL_RD",
    "INITIAL_THETA",
    "ITEMS_PER_SUBTOPIC",
    "JD_WEIGHTS",
    "MAIN_CONFIG",
    "PARENT_OF",
    "RESPONSE_CONCENTRATION",
    "RESPONSE_P_EPSILON",
    "RESUME_AFFINITY_MAX",
    "RESUME_AFFINITY_MIN",
    "RESUME_SUBTOPICS",
    "TAXONOMY",
    "ExperimentConfig",
    "split_budget_by_jd",
]
