"""The candidate ability model: theta, RD, and what one answer does to them.

Plan sections 5.9 (the update), 8.1 (the state), 9.1 and 9.2 (the hierarchy and
its aggregation).  **This module is pure arithmetic.**  No database, no Redis,
no LLM, no clock, no filesystem, no global mutable state - every function takes
its inputs explicitly and returns a new value.  That is deliberate: the adaptive
engine is the intellectual core of the project, and a core you can only exercise
through a live Postgres is a core nobody will exercise.

The vocabulary
--------------

``theta`` (ability)
    A *latent variable* - never observed, only inferred from answers.  Roughly
    -3 to +3, where 0 is "at the target level for the role".  It lives on the
    **same scale as question difficulty** (see the sigmoid note below), which is
    the single most useful property of the whole model: abilities and
    difficulties can be subtracted.

``RD`` (rating deviation)
    How unsure we are about ``theta``, in the same units.  Roughly 0.3 (measured
    several times) to 1.3 (never measured).  It is a standard error, so
    ``theta +/- 1.96 * RD`` is the 95% interval a report must show.  RD is *not*
    a second ability score; a low theta with a high RD means "we do not know
    yet", and a low theta with a low RD means "confidently weak".  Conflating
    those two is what makes most skill reports useless.

``p(theta, b)``
    The model's prediction, before the answer is seen: the probability that this
    candidate produces a good answer to *this* question.  It is the thing the
    observed score is compared against.

Where the state lives
---------------------

Canonically at **subtopic level only** (plan section 9.1).  Topic- and
domain-level numbers are computed on read by :func:`aggregate_ability` and
:func:`roll_up`.  Never store a number you can derive - two sources of truth
drift, and reconciling them is a bug generator.

The three equations
-------------------

**1. The prediction - the two-parameter logistic model (2PL).**

    p(theta, b) = 1 / (1 + exp(-a * (theta - b)))

``a`` is *discrimination*: how sharply the item separates strong candidates from
weak ones.  A high-``a`` item is a cliff, a low-``a`` item a gentle slope that
teaches you little.  Why a sigmoid: it maps any real number to a probability,
it is monotonic (more ability never lowers your chance), and it is the inverse
of the log-odds - so ``theta - b`` *is* the log-odds of success, which is why
the two quantities share a scale at all.

**2. The ability update - Elo, which is one SGD step on log loss.**

    delta_theta = K * (score - p)
    theta_new   = theta + delta_theta

This is not a heuristic dressed up.  For log loss ``L = -[s*log p + (1-s)*log(1-p)]``
and the sigmoid's ``dp/dtheta = a*p*(1-p)``, the derivative is ``dL/dtheta = -a*(s-p)``,
so a gradient step ``theta - eta * dL/dtheta`` is exactly ``theta + eta*a*(s-p)``.
**K is the learning rate**, which is why it is decomposed rather than tuned as
one magic number::

    K = K0 * f_conf * f_rd * f_stake

Four factors, and deliberately not a fifth: the ``a`` that appears in the
derivation above is absorbed into ``K0`` rather than carried per item, so that K
depends only on how much the *evidence* is worth.  Discrimination is still used
where the plan specifies it - inside ``p`` and inside the RD update below.  See
:class:`KFactor`.

Ability moves *because the answer disagreed with the prediction*.  If the
candidate scores exactly what the model expected, there is nothing to learn and
``theta`` does not move, however hard or easy the question was.

**3. The uncertainty update - Glicko's precision accumulation.**

    RD_new = 1 / sqrt( 1/RD^2  +  a^2 * p * (1-p) )

Read it as precisions (``1/RD^2``) adding up.  The added term is what this
observation told us, and it is largest at ``p = 0.5`` - if you were already sure
how someone would do, watching them do it teaches you nothing.  RD therefore
**decreases after evidence and can never increase here**, because a
non-negative amount of precision was added.  (Uncertainty going back *up*
between sessions is a separate mechanism; see "Deferred" below.)

Aggregation - precision weighting, not averaging
------------------------------------------------

    precision_i  = 1 / RD_i^2
    theta_parent = sum(precision_i * theta_i) / sum(precision_i)
    RD_parent    = 1 / sqrt(sum(precision_i))

Precision is the inverse of variance, so an estimate we trust dominates one we
do not.  A plain average would let a subtopic measured once (RD 1.2) drag down
one measured five times (RD 0.4) equally, which is statistically wrong.  Three
thermometers, one known to be flaky: you do not average them equally.
(Formally it is the maximum-likelihood combination of independent normal
estimates.)  Note that ``RD_parent`` *shrinks* as subtopics are added - five
measured subtopics do tell you more about a topic than one does.

Deferred - explicitly not Day 11
--------------------------------

* **Cross-session RD inflation** (plan section 9.5,
  ``RD = min(1.3, sqrt(RD^2 + c^2*days))``).  It needs a clock and a
  ``last_tested_at``, so it belongs with session persistence, not with the pure
  model.
* **Fisher information and item selection** (plan sections 5.10 and 8.3), the
  stopping rule (8.4), and the simulation harness - Day 12 onwards.  The
  ``a^2 * p * (1-p)`` term inside :func:`update_uncertainty` is the same
  quantity selection will want, but exposing a selection API is not this day's
  job.
* **Display mapping and confidence intervals** (plan section 9.3) - a reporting
  concern that reads this state rather than part of it.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

# --------------------------------------------------------------------------
# Constants.  Every one of these is a modelling choice from the plan, named so
# that it can be argued with; each function takes it as a keyword argument so
# that a caller (and a test) can vary it without a settings object.  These are
# NOT configuration: they are the model, and changing one changes the meaning
# of every stored theta, so they live next to the mathematics rather than in
# app/config.py.
# --------------------------------------------------------------------------

#: Plausible range for theta and difficulty b (plan section 5.9's table).
#: theta is clamped to it so that a pathological run of scores cannot walk the
#: estimate off to a value no question in the bank could ever probe.
THETA_MIN = -3.0
THETA_MAX = 3.0

#: Plausible range for RD (plan section 8.1).  The ceiling is "we know nothing";
#: the floor is a deliberate refusal to ever claim more certainty than about
#: +/-0.3 from a handful of noisily-graded answers.
RD_MIN = 0.30
RD_MAX = 1.30

#: Cap on |delta_theta| per answer (plan section 8.7, "oscillating difficulty").
#: Without it one generous grade can swing the next question a full band and the
#: interview feels random to the candidate.
MAX_THETA_STEP = 0.5

#: K0, the base learning rate (plan section 5.9).
BASE_LEARNING_RATE = 0.6

#: f_rd = min(F_RD_CAP, RD / RD_REFERENCE).  A learning-rate schedule: big steps
#: while we know little, small steps once we are close.
RD_REFERENCE = 0.6
F_RD_CAP = 1.6

#: f_stake: how much ground truth the item carries.  A reviewed bank item is
#: strong evidence; a question the LLM invented mid-session is weaker, so it
#: gets a smaller step.
STAKE_BANK_ITEM = 1.0
STAKE_GENERATED_ITEM = 0.5

#: Default discrimination when an item does not declare one - the 1PL/Rasch
#: special case, and the value used in the plan's worked example.
DEFAULT_DISCRIMINATION = 1.0


def _check_finite(name: str, value: float) -> float:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return value


def _clamp(value: float, low: float, high: float) -> float:
    return low if value < low else high if value > high else value


# --------------------------------------------------------------------------
# 1. The prediction
# --------------------------------------------------------------------------


def probability_correct(
    theta: float,
    difficulty: float,
    discrimination: float = DEFAULT_DISCRIMINATION,
) -> float:
    """The 2PL probability of a good answer: ``1 / (1 + exp(-a*(theta - b)))``.

    Monotonically increasing in ``theta`` and decreasing in ``difficulty``; at
    ``theta == difficulty`` it is exactly 0.5 for any positive ``a``, because
    the item is then a genuine coin flip.

    The two-branch form below is the standard numerically stable sigmoid: for a
    large positive exponent ``exp(-z)`` underflows harmlessly to 0, but
    ``exp(z)`` would overflow to ``inf``, so each branch is evaluated only where
    its exponential is bounded by 1.  ``z = +-800`` therefore returns 1.0 / 0.0
    rather than raising ``OverflowError``.

    ``discrimination`` must be positive.  An item with ``a <= 0`` is one weak
    candidates do *better* on - a broken item (bad wording, wrong answer key),
    and calibration's job to find, not something to quietly model.
    """
    _check_finite("theta", theta)
    _check_finite("difficulty", difficulty)
    _check_finite("discrimination", discrimination)
    if discrimination <= 0.0:
        raise ValueError(f"discrimination must be positive, got {discrimination!r}")

    z = discrimination * (theta - difficulty)
    if z >= 0.0:
        return 1.0 / (1.0 + math.exp(-z))
    exp_z = math.exp(z)
    return exp_z / (1.0 + exp_z)


# --------------------------------------------------------------------------
# 2. The state
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AbilityState:
    """Ability at one subtopic: the canonical unit of the skill model.

    Frozen, because an update returns a new state rather than mutating one -
    that is what makes the update trivially replayable from an event log.

    The persisted form is ``app.models.skill.SkillState`` (one row per user per
    subtopic).  Mapping between the two is the job of whatever owns the session,
    not of this module; ``last_tested_at`` deliberately does not appear here
    because a pure function has no clock.
    """

    theta: float
    rd: float
    n_observations: int = 0

    def __post_init__(self) -> None:
        _check_finite("theta", self.theta)
        _check_finite("rd", self.rd)
        if self.rd <= 0.0:
            raise ValueError(f"rd must be positive, got {self.rd!r}")
        if self.n_observations < 0:
            raise ValueError(f"n_observations must not be negative, got {self.n_observations!r}")

    @property
    def precision(self) -> float:
        """``1 / RD^2`` - the inverse of variance, and the aggregation weight."""
        return 1.0 / (self.rd * self.rd)


# --------------------------------------------------------------------------
# 3. The K-factor
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class KFactor:
    """K, kept as its factors rather than collapsed into one tuned number.

    K is the learning rate of the gradient step in the module docstring, so
    every factor answers the same question: *how much should this particular
    piece of evidence be allowed to move the estimate?*  Keeping them apart is
    what lets a log line say "the step was small because the grader was unsure",
    which is the difference between an explainable engine and a magic constant.

    ``base``
        K0 = 0.6.  Sets the scale: with everything else neutral, an answer the
        model rated a coin flip and the candidate aced moves theta by about a
        quarter of a difficulty band.  Big enough to converge in ~10 items,
        small enough that no single item decides the interview.
    ``quality``
        ``f_conf``, the grader's own confidence in the score, in [0, 1].  An
        uncertain grade is weak evidence and must move theta less; at 0 it moves
        it not at all.
    ``uncertainty``
        ``f_rd = min(1.6, RD / 0.6)``.  Large while RD is large, so early
        answers move theta a lot and it converges fast; small once RD is small,
        so a settled estimate is stable.  This is learning-rate decay, exactly
        as in training a network - and the cap stops a cold-start RD of 1.3
        producing a wild first step.
    ``stake``
        ``f_stake``, the evidence strength of the *item*: 1.0 for a reviewed
        bank question, 0.5 for one the model generated mid-session.  Ground
        truth you reviewed deserves a bigger step than ground truth you
        improvised.

    All four multiply, so each is an independent dial on ``delta_theta`` and
    doubling any one of them doubles the step.

    **Discrimination is deliberately absent.**  The gradient of log loss is
    ``-a*(s-p)``, so an ``a`` could defensibly be carried here as a fifth
    factor; the plan does not carry one, and this follows the plan.  The
    practical reading is that ``K0 = 0.6`` is a learning rate calibrated for the
    bank's typical item, with per-item sharpness left out of the *step size* on
    purpose - it makes K depend only on how much the *evidence* is worth, not on
    which item produced it, which is the property that makes a log line like
    "small step because the grader was unsure" mean exactly what it says.
    ``a`` still does its two jobs elsewhere and is not ignored: it shapes the
    prediction in :func:`probability_correct` (so it changes ``score - p``, and
    therefore ``delta_theta``, indirectly) and it drives the information term in
    :func:`update_uncertainty`.
    """

    base: float
    quality: float
    uncertainty: float
    stake: float

    @property
    def value(self) -> float:
        """The scalar K that multiplies ``(score - p)``."""
        return self.base * self.quality * self.uncertainty * self.stake


def k_factor(
    rd: float,
    *,
    grader_confidence: float = 1.0,
    stake: float = STAKE_BANK_ITEM,
    base_learning_rate: float = BASE_LEARNING_RATE,
    rd_reference: float = RD_REFERENCE,
    f_rd_cap: float = F_RD_CAP,
) -> KFactor:
    """Decompose K for one observation: ``K0 * f_conf * f_rd * f_stake``.

    Takes no ``discrimination`` argument by design - see :class:`KFactor`.
    """
    _check_finite("rd", rd)
    if rd <= 0.0:
        raise ValueError(f"rd must be positive, got {rd!r}")
    if not 0.0 <= grader_confidence <= 1.0:
        raise ValueError(f"grader_confidence must be in [0, 1], got {grader_confidence!r}")
    if stake <= 0.0:
        raise ValueError(f"stake must be positive, got {stake!r}")
    if base_learning_rate <= 0.0:
        raise ValueError(f"base_learning_rate must be positive, got {base_learning_rate!r}")
    if rd_reference <= 0.0:
        raise ValueError(f"rd_reference must be positive, got {rd_reference!r}")
    if f_rd_cap <= 0.0:
        raise ValueError(f"f_rd_cap must be positive, got {f_rd_cap!r}")

    return KFactor(
        base=base_learning_rate,
        quality=grader_confidence,
        uncertainty=min(f_rd_cap, rd / rd_reference),
        stake=stake,
    )


# --------------------------------------------------------------------------
# 4. The uncertainty update
# --------------------------------------------------------------------------


def update_uncertainty(
    rd: float,
    *,
    p: float,
    discrimination: float = DEFAULT_DISCRIMINATION,
    rd_min: float = RD_MIN,
    rd_max: float = RD_MAX,
) -> float:
    """``RD_new = 1 / sqrt(1/RD^2 + a^2 * p * (1-p))``, clamped to a sane range.

    Precisions add: the prior contributes ``1/RD^2`` and the observation
    contributes ``a^2 * p * (1-p)``, which is how much this particular question
    could have told us.  That term is at its largest when ``p = 0.5`` and
    vanishes as ``p`` approaches 0 or 1 - watching someone do what you were
    already certain they would do is not evidence.

    Consequences worth stating, because the tests assert them: the added term is
    never negative, so **RD never increases here**; the sum is always strictly
    positive because ``1/RD^2`` is, so the square root is real and the result is
    finite and positive for every valid input.  A ``p`` of exactly 0.0 or 1.0
    (reachable through float underflow at extreme ``|theta - b|``) leaves RD
    unchanged, which is the correct reading of a maximally uninformative item.
    """
    _check_finite("rd", rd)
    if rd <= 0.0:
        raise ValueError(f"rd must be positive, got {rd!r}")
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"p must be in [0, 1], got {p!r}")
    if discrimination <= 0.0:
        raise ValueError(f"discrimination must be positive, got {discrimination!r}")
    if not 0.0 < rd_min <= rd_max:
        raise ValueError(f"need 0 < rd_min <= rd_max, got {rd_min!r} and {rd_max!r}")

    information = discrimination * discrimination * p * (1.0 - p)
    new_rd = 1.0 / math.sqrt(1.0 / (rd * rd) + information)
    return _clamp(new_rd, rd_min, rd_max)


# --------------------------------------------------------------------------
# 5. The ability update
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AbilityUpdate:
    """One observation's effect, with the arithmetic that produced it.

    Returned whole rather than as a bare new state because the plan's event log
    wants to record *why* theta moved (``{subtopic: caching, delta: +0.25,
    RD: 0.90 -> 0.83}``), and because an update you cannot explain is one you
    cannot debug when it misbehaves on a real candidate.
    """

    before: AbilityState
    after: AbilityState
    #: The prediction made *before* the answer was seen.
    p: float
    #: The observed score in [0, 1] that was compared against ``p``.
    score: float
    k: KFactor
    #: ``K * (score - p)`` before the per-turn cap.
    raw_delta_theta: float
    #: What was actually applied: ``raw_delta_theta`` capped, then whatever the
    #: theta range clamp allowed through.  Always ``after.theta - before.theta``.
    delta_theta: float

    @property
    def was_capped(self) -> bool:
        """True when the per-turn cap or the theta range shortened the step."""
        return abs(self.delta_theta - self.raw_delta_theta) > 1e-12


def update_ability(
    state: AbilityState,
    *,
    difficulty: float,
    score: float,
    discrimination: float = DEFAULT_DISCRIMINATION,
    grader_confidence: float = 1.0,
    stake: float = STAKE_BANK_ITEM,
    base_learning_rate: float = BASE_LEARNING_RATE,
    max_theta_step: float = MAX_THETA_STEP,
    theta_min: float = THETA_MIN,
    theta_max: float = THETA_MAX,
    rd_min: float = RD_MIN,
    rd_max: float = RD_MAX,
) -> AbilityUpdate:
    """Fold one graded answer into a subtopic's ability estimate.

    ``score`` is the **normalised** grade, ``score in [0, 1]``: 0 is nothing
    right, 1 is a model answer, and the values between are the code-side scoring
    of plan section 7.  A score outside that interval is rejected rather than
    clamped - the arithmetic below would still produce a number, but it would be
    a number derived from a caller's bug, and silently absorbing it is how a
    scoring regression reaches production disguised as a strong candidate.

    The whole update, in order::

        p            = 2PL prediction from (theta, difficulty, a)
        K            = K0 * f_conf * f_rd * f_stake        (no `a` - see KFactor)
        delta_theta  = clip(K * (score - p), +-max_theta_step)
        theta_new    = clamp(theta + delta_theta, theta range)
        rd_new       = 1 / sqrt(1/rd^2 + a^2*p*(1-p))

    ``discrimination`` therefore reaches ``delta_theta`` only through ``p``, and
    reaches ``rd_new`` directly - which is exactly the plan's specification.

    ``n_observations`` increments by one.  Note that ``rd`` falls even when
    ``score == p`` and theta does not move: those are different questions.
    "Did we learn where they are?" is answered by ``delta_theta``; "did we learn
    *anything*?" is answered by RD, and a confirmed prediction is still
    information.
    """
    if not 0.0 <= score <= 1.0:
        raise ValueError(f"score must be in [0, 1], got {score!r}")
    if max_theta_step <= 0.0:
        raise ValueError(f"max_theta_step must be positive, got {max_theta_step!r}")
    if theta_min > theta_max:
        raise ValueError(f"need theta_min <= theta_max, got {theta_min!r} and {theta_max!r}")

    p = probability_correct(state.theta, difficulty, discrimination)
    k = k_factor(
        state.rd,
        grader_confidence=grader_confidence,
        stake=stake,
        base_learning_rate=base_learning_rate,
    )

    raw_delta = k.value * (score - p)
    capped = _clamp(raw_delta, -max_theta_step, max_theta_step)
    new_theta = _clamp(state.theta + capped, theta_min, theta_max)
    new_rd = update_uncertainty(
        state.rd, p=p, discrimination=discrimination, rd_min=rd_min, rd_max=rd_max
    )

    after = AbilityState(theta=new_theta, rd=new_rd, n_observations=state.n_observations + 1)
    return AbilityUpdate(
        before=state,
        after=after,
        p=p,
        score=score,
        k=k,
        raw_delta_theta=raw_delta,
        delta_theta=after.theta - state.theta,
    )


# --------------------------------------------------------------------------
# 6. Aggregation up the hierarchy
# --------------------------------------------------------------------------


def aggregate_ability(
    children: Mapping[str, AbilityState] | Sequence[AbilityState],
) -> AbilityState:
    """Combine child estimates into their parent by precision weighting.

    ::

        theta_parent = sum(theta_i / RD_i^2) / sum(1 / RD_i^2)
        RD_parent    = 1 / sqrt(sum(1 / RD_i^2))

    Accepts either a mapping (the shape state is usually held in) or a plain
    sequence; only the values matter, and the result does not depend on their
    order.

    The result is deliberately **not** floored at ``RD_MIN``.  That floor exists
    to stop a single subtopic over-claiming certainty from a few noisy grades; a
    topic backed by five independently measured subtopics really is known more
    precisely than any one of them, and flooring it would throw that away.

    Raises on an empty input: the ability of a parent with no measured children
    is not zero, it is undefined, and returning a confident 0.0 would put a
    fabricated number in a report.
    """
    values = list(children.values()) if isinstance(children, Mapping) else list(children)
    if not values:
        raise ValueError("cannot aggregate an empty set of children")

    total_precision = math.fsum(child.precision for child in values)
    weighted_theta = math.fsum(child.precision * child.theta for child in values)
    if not math.isfinite(total_precision) or total_precision <= 0.0:
        raise ValueError(f"child precisions did not sum to a usable value: {total_precision!r}")

    return AbilityState(
        theta=weighted_theta / total_precision,
        rd=1.0 / math.sqrt(total_precision),
        n_observations=sum(child.n_observations for child in values),
    )


def roll_up(
    states: Mapping[str, AbilityState],
    parent_of: Mapping[str, str],
) -> dict[str, AbilityState]:
    """Aggregate one level of the hierarchy: keyed children -> keyed parents.

    ``parent_of`` maps every key in ``states`` to its parent key; it is the
    ``topics.parent_key`` column (``app.models.taxonomy.Topic``) passed in as
    data, which is what keeps this function free of the database that column
    lives in.

    The taxonomy is three levels (domain -> topic -> subtopic, plan section
    9.1), and applying this twice walks all of it::

        topics  = roll_up(subtopics, subtopic_to_topic)
        domains = roll_up(topics,    topic_to_domain)

    Two applications rather than a generic tree walker: the levels are fixed at
    three by design, and a recursive tree framework would be more machinery than
    the problem has.  Deeper hierarchies still work - chain another call.

    A key with no entry in ``parent_of`` is an error, not something to skip
    quietly, because dropping a subtopic from an aggregate silently changes the
    parent's number and nobody would see it.  Returns a dict ordered by parent
    key so that output is stable across runs.
    """
    orphans = sorted(key for key in states if key not in parent_of)
    if orphans:
        raise ValueError(f"no parent mapped for: {', '.join(orphans)}")

    grouped: dict[str, list[AbilityState]] = {}
    for key, state in states.items():
        grouped.setdefault(parent_of[key], []).append(state)

    return {parent: aggregate_ability(grouped[parent]) for parent in sorted(grouped)}


__all__ = [
    "BASE_LEARNING_RATE",
    "DEFAULT_DISCRIMINATION",
    "F_RD_CAP",
    "MAX_THETA_STEP",
    "RD_MAX",
    "RD_MIN",
    "RD_REFERENCE",
    "STAKE_BANK_ITEM",
    "STAKE_GENERATED_ITEM",
    "THETA_MAX",
    "THETA_MIN",
    "AbilityState",
    "AbilityUpdate",
    "KFactor",
    "aggregate_ability",
    "k_factor",
    "probability_correct",
    "roll_up",
    "update_ability",
    "update_uncertainty",
]
