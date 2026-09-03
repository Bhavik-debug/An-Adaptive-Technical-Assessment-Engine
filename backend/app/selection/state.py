"""What the selection policy needs to know, as one immutable value.

Plan section 8.3's ``score_item(q, state)`` reads five things off ``state``:
``state.theta[q.subtopic]``, ``state.jd_weight[q.topic]``, ``state.resume``,
``state.asked`` and ``state.time_left``.  :class:`SelectionState` is exactly
those five, plus the quota targets the coverage term and the topic-cap
constraint both need.

**It stores nothing it can derive.**  How many items a topic has already had is
not a field - it is counted from ``asked``.  That is the same rule
:mod:`app.ability` follows for topic-level theta, and for the same reason: two
sources of truth drift, and reconciling them is a bug generator.

**It owns no I/O.**  Building one of these from the database - reading
``skill_states``, folding the event log, loading the blueprint from
``interview_sessions.blueprint`` - belongs to whatever owns the session, which
does not exist yet.  Until it does, a state is constructed in a test or by a
caller, and the whole selection layer stays exercisable in milliseconds.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field

from app.ability import RD_MAX, AbilityState

#: What we assume about a subtopic nobody has been asked about yet: dead centre
#: on the scale, and maximally unsure.  ``RD_MAX`` rather than something smaller
#: because claiming any precision about an unmeasured skill is a fabricated
#: number, and because a large RD is what makes the first questions on a fresh
#: subtopic move theta quickly (plan section 5.9's ``f_rd``).
#:
#: Resume-seeded priors (plan section 9.4) would replace this per subtopic.
#: DEFERRED - they need the resume subsystem, which is Phase 5.
PRIOR_ABILITY = AbilityState(theta=0.0, rd=RD_MAX)


@dataclass(frozen=True, slots=True)
class CandidateItem:
    """One question, reduced to what selection actually reads.

    A deliberately narrower view than :class:`app.models.question.Question` and
    than :class:`app.retrieval.search.QuestionRef`: selection needs the IRT
    parameters, the taxonomy keys, the time estimate and the vector, and nothing
    else.  Carrying the prose, the reference answer and the concept checklist
    through the scoring loop would be several kilobytes per item to compute a
    float with.

    The same type is used for *asked* items, because an asked question is a
    question - the redundancy term compares candidates against exactly this.

    ``embedding``
        The stored pgvector row, L2-normalised by both embedders in
        :mod:`app.retrieval.embedders`, which is what lets
        :func:`app.selection.objective.cosine_similarity` be a plain dot
        product.  ``None`` for a bank ingested with ``--no-embed``; the
        redundancy term treats a missing vector as "no evidence of similarity"
        rather than inventing one.
    """

    id: str
    topic_key: str
    subtopic_key: str
    difficulty_b: float
    time_estimate_s: int
    discrimination_a: float = 1.0
    embedding: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.difficulty_b):
            raise ValueError(f"difficulty_b must be finite, got {self.difficulty_b!r}")
        if self.time_estimate_s < 0:
            raise ValueError(f"time_estimate_s must not be negative, got {self.time_estimate_s!r}")
        if not math.isfinite(self.discrimination_a) or self.discrimination_a <= 0.0:
            raise ValueError(
                f"discrimination_a must be finite and positive, got {self.discrimination_a!r}"
            )


@dataclass(frozen=True, slots=True)
class ResumeProfile:
    """The minimal, deterministic stand-in for a parsed resume.

    **This is an interface, not a resume-processing system**, and it is scoped
    that way on purpose.  Resume ingestion, section parsing, skill extraction
    and the resume-seeded theta priors of plan section 9.4 are Phase 5;
    implementing any of them here would be building a future day under another
    name.  What Day 12 needs is one number per question in ``[0, 1]``, and
    :func:`app.selection.objective.resume_affinity` derives it from whichever of
    these two fields is populated:

    ``topic_affinity``
        An explicit score per taxonomy key - subtopic keys win over topic keys,
        because the more specific statement about a candidate is the more
        informative one.  This is the field a Phase 5 resume parser would fill.

    ``embedding``
        A vector for the resume text, in the same space as
        ``CandidateItem.embedding``.  Affinity is then the cosine similarity to
        the question's vector.  It reuses Phase 2's embedder rather than adding
        a second notion of "similar" to the codebase; nothing here calls a model
        or a network - the caller passes a vector that already exists.

    An empty profile (both fields unset) scores every question 0.0, which is the
    right reading of "we know nothing about this candidate's background": no
    question gets a personalisation bonus, and the other five terms decide.
    """

    topic_affinity: Mapping[str, float] = field(default_factory=dict)
    embedding: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        for key, value in self.topic_affinity.items():
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"topic_affinity[{key!r}] must be in [0, 1], got {value!r}")


@dataclass(frozen=True, slots=True)
class SelectionState:
    """Everything the policy reads: ability, blueprint, resume, history, clock.

    ``ability``
        Day 11's state, keyed by **subtopic** - the level it is canonically
        stored at (plan section 9.1).  A subtopic absent from the mapping falls
        back to ``prior``, so a fresh session with no measurements is a valid
        state rather than a special case.

    ``coverage_targets``
        ``topic_key -> how many items this topic must get``.  Consumed, never
        built: the blueprint builder that produces it is plan section 3, Day 15,
        and is **DEFERRED**.  Day 12's contract with it is this mapping and
        nothing more.  A topic missing from it is not required by the blueprint,
        so it has no quota left and no coverage deficit - it is simply not part
        of this interview.

    ``jd_weights``
        ``topic_key -> role alignment in [0, 1]``, straight out of the job
        description.  Also consumed rather than built (Phase 5).  A missing
        topic scores 0.0: not "unknown", but "the JD did not ask for it".

    ``asked``
        Every item already used in this session, in order.  Three of the six
        objective terms and two of the four hard constraints read it, which is
        why it is the item objects and not just their ids - the redundancy term
        needs their vectors.

    ``time_left_s``
        Seconds remaining in the interview.  Zero is legal and means no item can
        fit; negative is a caller bug and is rejected.
    """

    ability: Mapping[str, AbilityState]
    coverage_targets: Mapping[str, int]
    jd_weights: Mapping[str, float]
    time_left_s: float
    asked: tuple[CandidateItem, ...] = ()
    resume: ResumeProfile | None = None
    prior: AbilityState = PRIOR_ABILITY

    def __post_init__(self) -> None:
        if not math.isfinite(self.time_left_s):
            raise ValueError(f"time_left_s must be finite, got {self.time_left_s!r}")
        if self.time_left_s < 0.0:
            raise ValueError(f"time_left_s must not be negative, got {self.time_left_s!r}")
        for topic, target in self.coverage_targets.items():
            if target < 0:
                raise ValueError(
                    f"coverage_targets[{topic!r}] must not be negative, got {target!r}"
                )
        for topic, weight in self.jd_weights.items():
            if not math.isfinite(weight) or not 0.0 <= weight <= 1.0:
                raise ValueError(f"jd_weights[{topic!r}] must be in [0, 1], got {weight!r}")
        seen: set[str] = set()
        for item in self.asked:
            if item.id in seen:
                raise ValueError(f"asked contains {item.id!r} twice")
            seen.add(item.id)

    # -- ability -----------------------------------------------------------

    def ability_for(self, subtopic_key: str) -> AbilityState:
        """The Day 11 state for a subtopic, or the cold-start prior."""
        return self.ability.get(subtopic_key, self.prior)

    def theta_for(self, subtopic_key: str) -> float:
        return self.ability_for(subtopic_key).theta

    def rd_for(self, subtopic_key: str) -> float:
        return self.ability_for(subtopic_key).rd

    # -- history -----------------------------------------------------------

    @property
    def asked_ids(self) -> frozenset[str]:
        """Ids already used.  Derived, never stored - see the module docstring."""
        return frozenset(item.id for item in self.asked)

    def asked_count(self, topic_key: str) -> int:
        """How many items this topic has already had."""
        return sum(1 for item in self.asked if item.topic_key == topic_key)

    # -- blueprint ---------------------------------------------------------

    def quota_remaining(self, topic_key: str) -> int:
        """``target - asked``, floored at 0.  Zero for a topic with no target."""
        target = self.coverage_targets.get(topic_key, 0)
        return max(0, target - self.asked_count(topic_key))

    @property
    def topics_with_quota_left(self) -> tuple[str, ...]:
        """The ``topic_key IN (...)`` list of the plan's hard filter, sorted.

        Sorted so that the SQL bound parameters - and therefore the query plan
        cache key and any log line - are stable across runs.  Empty when the
        blueprint is empty or fully served, in which case no item is eligible;
        that is the pool-exhaustion path of plan section 8.7, not an error.
        """
        return tuple(
            sorted(topic for topic in self.coverage_targets if self.quota_remaining(topic) > 0)
        )

    def jd_weight(self, topic_key: str) -> float:
        return self.jd_weights.get(topic_key, 0.0)


__all__ = [
    "PRIOR_ABILITY",
    "CandidateItem",
    "ResumeProfile",
    "SelectionState",
]
