"""Stage 2 - the weighted objective that scores the survivors.

Plan section 8.3, step 2, implemented exactly as written::

    def score_item(q, state):
        p    = sigmoid(state.theta[q.subtopic] - q.difficulty_b)
        info = p * (1 - p)                              # in [0, 0.25]

        return (
            0.40 * (info / 0.25)                        # information gain
          + 0.25 * state.jd_weight[q.topic]             # role alignment
          + 0.15 * resume_affinity(q, state.resume)     # personalisation
          + 0.15 * coverage_deficit(q.topic, state)     # quota shortfall
          - 0.10 * redundancy(q, state.asked)           # cosine to already-asked
          - 0.05 * (q.time_estimate_s / state.time_left)
        )

Why an objective at all, rather than "ask the most informative item"
--------------------------------------------------------------------

Because a pure information-maximiser is a bad interviewer.  Information is
highest where uncertainty is highest, so it would find the one subtopic it knows
least about and spend twelve questions there - ignoring the job description
entirely, never covering the blueprint, and asking two near-identical questions
back to back because both were maximally uncertain.  The candidate experiences
that as robotic and unfair, and the report it produces answers a question nobody
asked.  So measurement gets the largest single share, 0.40, and not the whole
thing.

**The six weights are a design choice, not a derived optimum.**  Nothing in this
repository has yet demonstrated that 0.40/0.25/0.15/0.15/-0.10/-0.05 beats any
other set of numbers; they encode a *priority ordering* the plan argues for, and
they are stated here so they can be argued with.  The ablation that would turn
them into an empirical claim - information-only vs. the full objective vs.
random, over simulated candidates - is plan section 8.6, and is **DEFERRED** to
Day 13.  Until it runs, no code comment, docstring or document in this project
may call these weights optimal.

Each term, and what it is for
-----------------------------

``+0.40 information``
    The primary measurement objective, normalised to ``[0, 1]`` by
    :func:`app.selection.information.normalised_information` so that it cannot
    be re-scaled by an item's discrimination.  Largest when ``b`` is near
    ``theta``.

``+0.25 JD weight``
    The candidate asked to be assessed for a specific role.  An assessment that
    drifts off the job description produces a report that is about someone else.

``+0.15 resume affinity``
    Personalisation: ask about what they say they have done.  Capped low on
    purpose - it is a product feature, not a measurement feature, and letting it
    outweigh information would mean the engine measures whatever flatters the
    resume.

``+0.15 coverage deficit``
    Pushes towards topics still behind their quota, so the blueprint is actually
    met before the budget runs out.  Together with the per-topic hard cap this
    is what stops one topic eating the interview.

``-0.10 redundancy``
    Two near-identical questions produce *correlated* evidence.  Averaging them
    looks like two independent measurements and inflates apparent confidence,
    which is worse than not asking the second one at all.

``-0.05 time cost``
    A six-minute item late in a thirty-minute session is a bad trade however
    informative it is.  Smallest weight, because the hard time filter has
    already removed everything that does not fit; this term only orders what
    remains.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Sequence
from dataclasses import dataclass

from app.selection.information import normalised_information, selection_probability
from app.selection.state import CandidateItem, ResumeProfile, SelectionState

# --------------------------------------------------------------------------
# The weights.  Plan section 8.3's table, verbatim.  Named constants rather
# than literals in the formula so that a log line, a test and a document can
# all point at the same objects - and so that changing one is a visible diff
# rather than a digit somebody edited inside an expression.
#
# DESIGN CHOICES, NOT MEASURED OPTIMA.  See the module docstring.
#
# Note that the four positive weights sum to 0.95, not 1.0 - the plan does not
# make them a probability distribution.  So a score lives in [-0.15, 0.95] and
# is a **ranking key**, never a percentage; nothing may present it as one.
# --------------------------------------------------------------------------

INFORMATION_WEIGHT = 0.40
JD_WEIGHT = 0.25
RESUME_WEIGHT = 0.15
COVERAGE_WEIGHT = 0.15
#: Subtracted, so the constant itself is positive: ``score -= 0.10 * redundancy``.
REDUNDANCY_PENALTY = 0.10
#: Subtracted, likewise.
TIME_PENALTY = 0.05


@dataclass(frozen=True, slots=True)
class ObjectiveWeights:
    """The six weights, as one value that can be varied without global state.

    **Production behaviour is unchanged**: every field defaults to the constant
    above it, :data:`DEFAULT_WEIGHTS` is that default instance, and every
    function here and in :mod:`app.selection.policy` takes ``weights`` as a
    keyword argument that defaults to it.  Calling ``score_item(item, state)``
    with no ``weights`` produces exactly the number it produced before this type
    existed, bit for bit.

    It exists so the Day 13 weight ablation (plan section 8.6) can ask "what
    happens with the coverage term switched off?" *without* mutating a module
    constant.  A simulation that monkeypatched ``COVERAGE_WEIGHT`` would leak
    across runs, be impossible to parallelise, and quietly change the behaviour
    of any other caller in the same process - which is exactly the class of bug
    an experiment must not introduce into the system it is measuring.

    **Scaling all six by the same positive factor cannot change any decision.**
    Within one selection every item is scored with the same weights, so
    multiplying them all multiplies every score, and argmax and the top-5 are
    invariant under that.  This is why an ablation needs no renormalisation:
    zeroing one weight and leaving the rest is *exactly* "remove that
    component", not "remove that component and shrink the others".
    """

    information: float = INFORMATION_WEIGHT
    jd: float = JD_WEIGHT
    resume: float = RESUME_WEIGHT
    coverage: float = COVERAGE_WEIGHT
    #: Subtracted inside :func:`score_item`, so this stays positive.
    redundancy: float = REDUNDANCY_PENALTY
    #: Subtracted likewise.
    time: float = TIME_PENALTY

    def __post_init__(self) -> None:
        for name in ("information", "jd", "resume", "coverage", "redundancy", "time"):
            value = getattr(self, name)
            if not math.isfinite(value):
                raise ValueError(f"{name} weight must be finite, got {value!r}")
            if value < 0.0:
                # A negative weight would flip a term's meaning - "prefer
                # redundant questions" - which is never what an ablation means.
                raise ValueError(f"{name} weight must not be negative, got {value!r}")

    def without(self, component: str) -> ObjectiveWeights:
        """This objective with one component switched off (its weight set to 0).

        The ablation operator.  Raises on an unknown component name rather than
        silently returning an unchanged copy, because an ablation that quietly
        ablated nothing would report the full objective's numbers under another
        label - the single most misleading thing this study could do.
        """
        if component not in COMPONENT_NAMES:
            raise ValueError(
                f"unknown component {component!r}; expected one of {', '.join(COMPONENT_NAMES)}"
            )
        return dataclasses.replace(self, **{component: 0.0})


#: The production objective.  Plan section 8.3's six numbers, unchanged.
DEFAULT_WEIGHTS = ObjectiveWeights()

#: The component names an ablation may switch off, in the order the plan lists
#: them.  Also the keys of :attr:`ScoreBreakdown.contributions`.
COMPONENT_NAMES: tuple[str, ...] = (
    "information",
    "jd",
    "resume",
    "coverage",
    "redundancy",
    "time",
)


def _clamp01(value: float) -> float:
    return 0.0 if value < 0.0 else 1.0 if value > 1.0 else value


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """Cosine of the angle between two vectors, in ``[-1, 1]``.

    Phase 2 never needed this in Python - pgvector's ``<=>`` operator does it
    inside the database - so it lands here rather than being retro-fitted into
    :mod:`app.retrieval`, which Day 12 has no business modifying.

    Both embedders in :mod:`app.retrieval.embedders` return L2-normalised
    vectors, for which this reduces to the dot product; the norms are divided
    out anyway, because a function that is only correct for normalised input is
    a trap for the next caller.  A zero vector has no direction, so its
    similarity to anything is reported as 0.0 rather than as a division by zero.

    ``math.fsum`` rather than ``sum``: 384 additions of small floats, and exact
    summation costs nothing here while making the result independent of order.
    """
    if len(left) != len(right):
        raise ValueError(f"vectors must be the same length, got {len(left)} and {len(right)}")
    dot = math.fsum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(math.fsum(a * a for a in left))
    right_norm = math.sqrt(math.fsum(b * b for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


# ---------------------------------------------------------------------------
# the three derived terms
# ---------------------------------------------------------------------------


def resume_affinity(item: CandidateItem, resume: ResumeProfile | None) -> float:
    """How close this question is to what the candidate says they have done.

    Returns a value in ``[0, 1]``, from the first of these that applies:

    1. an explicit score for the item's **subtopic** key;
    2. an explicit score for its **topic** key;
    3. the cosine similarity between the resume vector and the question vector,
       negatives clamped to 0 - "pointing the other way" and "unrelated" are the
       same statement here, and letting a negative cosine *subtract* from the
       score would turn a personalisation bonus into a penalty nobody asked for;
    4. 0.0 - no resume, or nothing in it that speaks to this question.

    Specific beats general, which is why the subtopic key is consulted first.

    **This is the minimal interface, and it is meant to be.**  Parsing a resume,
    extracting skills from it and seeding theta priors from it (plan section
    9.4) are Phase 5; :class:`app.selection.state.ResumeProfile` is the shape
    Day 12 needs from that work and nothing more.  No LLM call, no network, no
    file reading, no new service - deterministic arithmetic over data the caller
    already holds.  DEFERRED: the producer of that data.
    """
    if resume is None:
        return 0.0
    for key in (item.subtopic_key, item.topic_key):
        explicit = resume.topic_affinity.get(key)
        if explicit is not None:
            return _clamp01(explicit)
    if resume.embedding is not None and item.embedding is not None:
        return _clamp01(cosine_similarity(item.embedding, resume.embedding))
    return 0.0


def coverage_deficit(topic_key: str, state: SelectionState) -> float:
    """How far behind its quota this topic is, as a fraction in ``[0, 1]``.

    ::

        deficit = max(0, target - asked) / target

    1.0 for a topic that has had none of its required items, 0.0 once the quota
    is met, and proportionally in between - so a topic needing three more of
    four scores higher than one needing one more of four, which is the
    priority ordering the term exists to create.

    A topic with no target (not in the blueprint) and a topic whose target is
    zero both score 0.0: neither is behind on anything.  A topic that has been
    *over*-served scores 0.0 rather than a negative number, because the floor is
    what makes this term a nudge towards the underserved rather than a punishment
    of the served - and because the hard per-topic cap has already removed those
    items from the pool anyway.

    Note the division: the deficit is *relative* to the target.  A topic wanting
    1 item and having 0 is fully unserved (1.0), the same as a topic wanting 5
    and having 0 - which is correct, because both are equally far from done.

    ``coverage_targets`` is consumed, never built.  The blueprint builder that
    produces it is Day 15 and is **DEFERRED**.
    """
    target = state.coverage_targets.get(topic_key, 0)
    if target <= 0:
        return 0.0
    return state.quota_remaining(topic_key) / target


def redundancy(item: CandidateItem, asked: Sequence[CandidateItem]) -> float:
    """Semantic similarity to the most similar already-asked question, ``[0, 1]``.

    **Maximum, not mean.**  A candidate that is a near-duplicate of item 3 is
    redundant whether or not it resembles items 1, 2 and 4; averaging over ten
    asked questions would dilute that single collision into nothing, which is
    exactly the case the term exists to catch.

    Reuses Phase 2's vectors as they are stored - the same
    ``BAAI/bge-small-en-v1.5`` space that vector search runs in, written by
    ``app.retrieval.indexing``.  Nothing here embeds anything: no model is
    loaded, no network is touched, and a question with no stored vector (a bank
    ingested with ``--no-embed``) contributes no similarity rather than a
    fabricated one.

    Returns 0.0 when nothing has been asked yet, which is the first call of
    every session.  Negative cosines are clamped to 0 - two questions pointing
    in opposite directions in embedding space are not *less* than unrelated for
    this purpose.
    """
    if item.embedding is None or not asked:
        return 0.0
    similarities = [
        cosine_similarity(item.embedding, other.embedding)
        for other in asked
        if other.embedding is not None
    ]
    if not similarities:
        return 0.0
    return _clamp01(max(similarities))


def time_cost(item: CandidateItem, state: SelectionState) -> float:
    """``time_estimate_s / time_left`` - the fraction of what is left this costs.

    In ``(0, 1]`` for any item that survived the hard time filter, so late in a
    session every remaining item costs proportionally more and the term bites
    harder exactly when it should.

    With no time left the ratio is undefined; the constraint layer has already
    emptied the pool in that case, so this branch exists only to keep the
    function total, and it reports 1.0 - the value the term takes for an item
    that exactly consumes the remaining time.
    """
    if state.time_left_s <= 0.0:
        return 1.0
    return item.time_estimate_s / state.time_left_s


# ---------------------------------------------------------------------------
# the weighted sum
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ScoreBreakdown:
    """One item's score, with every term that produced it kept separately.

    Returned whole rather than as a bare float for the same reason
    :class:`app.ability.AbilityUpdate` is: plan section 8.7 requires *every*
    selection decision to be logged - pool size, the top five with their scores,
    the chosen id, theta, the information term - so that a misbehaving session
    can be replayed instead of guessed at.  A score you cannot decompose is a
    score you cannot debug.
    """

    item: CandidateItem
    #: The 2PL prediction at ``a = 1``: ``sigmoid(theta - b)``.
    p: float
    #: ``p * (1-p) / 0.25``, in ``[0, 1]``.
    information: float
    jd_weight: float
    resume_affinity: float
    coverage_deficit: float
    redundancy: float
    time_cost: float
    total: float
    #: The weights this score was computed with.  Carried so that a breakdown is
    #: self-describing: an ablation run and a production run produce the same
    #: shape, and only this field says which objective produced the number.
    weights: ObjectiveWeights = DEFAULT_WEIGHTS

    @property
    def item_id(self) -> str:
        return self.item.id

    @property
    def contributions(self) -> dict[str, float]:
        """Each term's *weighted* contribution.  Sums to ``total``.

        This is the shape a log line wants: not "information was 0.98" but
        "information contributed +0.392 of the 0.71 total", which is the number
        that actually explains why this item won.
        """
        return {
            "information": self.weights.information * self.information,
            "jd": self.weights.jd * self.jd_weight,
            "resume": self.weights.resume * self.resume_affinity,
            "coverage": self.weights.coverage * self.coverage_deficit,
            "redundancy": -self.weights.redundancy * self.redundancy,
            "time": -self.weights.time * self.time_cost,
        }


def score_item(
    item: CandidateItem,
    state: SelectionState,
    *,
    weights: ObjectiveWeights = DEFAULT_WEIGHTS,
) -> ScoreBreakdown:
    """Score one *eligible* item against the current state.

    Assumes stage 1 has run.  It will happily score an ineligible item - it is
    pure arithmetic and has no opinions - which is why
    :func:`app.selection.policy.choose_next` re-applies the hard filter rather
    than trusting its caller: a scoring function that silently returned
    ``-inf`` for a repeat would be a soft constraint wearing a hard constraint's
    clothes.

    ``p`` uses the **subtopic** theta, per the plan's
    ``state.theta[q.subtopic]``, and the ``a = 1`` sigmoid, per its
    ``sigmoid(state.theta - q.difficulty_b)`` - the item's own discrimination is
    not part of the selection score.  See
    :mod:`app.selection.information` for why the two information quantities are
    kept apart.

    ``weights`` defaults to the production objective and exists for the Day 13
    ablation; see :class:`ObjectiveWeights`.  The six *components* are always
    computed, whatever the weights are, so a breakdown from an ablated run still
    reports what the switched-off term would have said.
    """
    p = selection_probability(state.theta_for(item.subtopic_key), item.difficulty_b)
    information = normalised_information(p)
    jd = state.jd_weight(item.topic_key)
    resume = resume_affinity(item, state.resume)
    coverage = coverage_deficit(item.topic_key, state)
    overlap = redundancy(item, state.asked)
    cost = time_cost(item, state)

    total = (
        weights.information * information
        + weights.jd * jd
        + weights.resume * resume
        + weights.coverage * coverage
        - weights.redundancy * overlap
        - weights.time * cost
    )
    return ScoreBreakdown(
        item=item,
        p=p,
        information=information,
        jd_weight=jd,
        resume_affinity=resume,
        coverage_deficit=coverage,
        redundancy=overlap,
        time_cost=cost,
        total=total,
        weights=weights,
    )


def score_items(
    items: Sequence[CandidateItem],
    state: SelectionState,
    *,
    weights: ObjectiveWeights = DEFAULT_WEIGHTS,
) -> list[ScoreBreakdown]:
    """Score every item, in the order given.  Ranking is stage 3's job."""
    return [score_item(item, state, weights=weights) for item in items]


__all__ = [
    "COMPONENT_NAMES",
    "COVERAGE_WEIGHT",
    "DEFAULT_WEIGHTS",
    "INFORMATION_WEIGHT",
    "JD_WEIGHT",
    "REDUNDANCY_PENALTY",
    "RESUME_WEIGHT",
    "TIME_PENALTY",
    "ObjectiveWeights",
    "ScoreBreakdown",
    "cosine_similarity",
    "coverage_deficit",
    "redundancy",
    "resume_affinity",
    "score_item",
    "score_items",
    "time_cost",
]
