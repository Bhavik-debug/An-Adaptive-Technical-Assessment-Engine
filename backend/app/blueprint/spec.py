"""What a caller asks for: role, level, duration, and the topics that matter.

Plan section 3, Day 15: *"role + level + duration + JD weights -> topic quotas,
item budget, time budget"*.  This module is the left-hand side of that arrow.

The split between :class:`AssessmentSpec` and
:class:`~app.blueprint.blueprint.Blueprint` is the point of the whole package.
A spec is a **request** and may be written by hand, by an API handler, or (in
Phase 5) by a job-description parser; it says what the assessment is *for*.  A
blueprint is a **contract** and says what the engine must actually do.  Deriving
one from the other is :func:`~app.blueprint.builder.build_blueprint`, and it is
the only place the two meet.

Keeping them apart matters for one concrete reason: a spec can be under-specified
("45 minutes, standard depth") and a blueprint cannot.  If they were one type,
every consumer would have to know which fields had been filled in yet.

The three calibration tables
----------------------------

Everything derived from ``level`` and ``mode`` lives here as data, so that
"why 8 items?" has an answer with a number in it rather than a shrug.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

# ---------------------------------------------------------------------------
# errors
# ---------------------------------------------------------------------------


class BlueprintError(ValueError):
    """A blueprint or a spec is invalid.

    ``ValueError`` rather than a bespoke hierarchy, matching
    :class:`app.bank.taxonomy.TaxonomyError`: these are all "you passed
    nonsense", and an API layer maps them to 422 by type either way.

    Every message names the offending field **and** the value, because the
    caller of a builder is usually a machine and the human reading the log is
    not the one who wrote the call.
    """


# ---------------------------------------------------------------------------
# levels
# ---------------------------------------------------------------------------

#: Seniority, and the ability the assessment is aimed at.
#:
#: The number is a **theta on Day 11's [-3, +3] scale**, and it is used for
#: exactly one thing: the cold-start prior (:meth:`Blueprint.prior_ability`).
#: A senior assessment that starts every subtopic at theta = 0 spends its first
#: questions on items well below the candidate, and the difficulty window
#: ``|b - theta| <= 1.5`` is centred on that estimate.
#:
#: **``mid`` is exactly 0.0, so a mid-level blueprint produces Day 12's
#: unmodified ``PRIOR_ABILITY``.**  That is deliberate: the default changes
#: nothing, and choosing another level is an explicit, visible decision by the
#: caller rather than a silent retune of the selector.
#:
#: The values are half a difficulty band apart, which is the smallest step that
#: moves the window by a meaningful amount without asserting more precision
#: about "senior" than anyone can defend.  **They have not been validated in
#: simulation** - see ``docs/blueprint.md``, "What is deferred".
LEVEL_TARGET_THETA: Final[Mapping[str, float]] = {
    "junior": -0.5,
    "mid": 0.0,
    "senior": 0.5,
    "staff": 1.0,
}

LEVELS: Final[tuple[str, ...]] = tuple(LEVEL_TARGET_THETA)

DEFAULT_LEVEL: Final[str] = "mid"


# ---------------------------------------------------------------------------
# modes
# ---------------------------------------------------------------------------

#: How long one question is budgeted to take, per assessment mode, in seconds.
#:
#: **Calibrated against the committed bank, not guessed.**  Measured over all 60
#: items in ``data/question-bank/``, ``time_estimate_s`` has quartiles
#: p25 = 292.5, p50 = 330, p75 = 390 (mean 324).  Each mode takes one of those
#: quartiles, rounded to the 30-second granularity every authored estimate
#: already uses:
#:
#: ===========  =====  ==========================================
#: mode         s/item  what it means
#: ===========  =====  ==========================================
#: screening      300  the bank's quicker quartile - breadth first
#: standard       330  the bank's median item
#: deep_dive      390  the bank's slower quartile - fewer, harder
#: ===========  =====  ==========================================
#:
#: A 45-minute standard assessment is therefore **8 items**, not twenty.  That
#: number is smaller than Day 13's synthetic budget by design: the simulation
#: ran on a synthetic bank whose items average 120 s, and this one is derived
#: from the questions that actually exist.  See ``docs/blueprint.md``, "The
#: pacing finding".
MODE_SECONDS_PER_ITEM: Final[Mapping[str, int]] = {
    "screening": 300,
    "standard": 330,
    "deep_dive": 390,
}

MODES: Final[tuple[str, ...]] = tuple(MODE_SECONDS_PER_ITEM)

DEFAULT_MODE: Final[str] = "standard"

#: An assessment shorter than this cannot cover anything and is almost certainly
#: a units mistake (seconds passed where minutes were meant).
MIN_DURATION_MINUTES: Final[int] = 5

#: Six hours.  Not a product decision - a guard against a caller passing a
#: duration in seconds, which would otherwise silently produce a 3600-item
#: budget and an allocation loop to match.
MAX_DURATION_MINUTES: Final[int] = 360

#: No assessment is meaningful at fewer than this many questions: the stopping
#: rule needs three consecutive small updates before it can fire, and one
#: question per required topic is the floor for saying anything about coverage.
MIN_ITEM_BUDGET: Final[int] = 3

#: A ceiling, for the same reason ``MAX_DURATION_MINUTES`` exists.
MAX_ITEM_BUDGET: Final[int] = 200


# ---------------------------------------------------------------------------
# the request
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TopicRequest:
    """One topic the assessment must cover, and how much it matters.

    ``jd_weight``
        Role alignment in ``[0, 1]``, straight out of the job description.  It
        does two jobs and they are worth separating: it is the proportion used
        to divide the item budget into quotas, **and** it is passed through
        unchanged to Day 12's ``SelectionState.jd_weights``, where it is one of
        the six scored terms.  A weight of 0 on a required topic is legal and
        means "cover it, but the JD does not ask for depth"; such a topic gets
        its ``min_items`` and nothing more.

    ``subtopic_keys``
        The sub-areas that count towards this topic.  Day 11 stores theta per
        **subtopic** and Day 12 quotas it per **topic**, so something has to
        hold the mapping between them - that is
        :attr:`~app.blueprint.blueprint.Blueprint.parent_of`, which
        ``should_stop`` and ``roll_up`` both require.  Listing them here rather
        than reading the whole taxonomy is what lets a blueprint say "databases,
        but only indexing and transactions".

    ``required``
        A required topic must be covered before the assessment can claim to have
        measured what it set out to; it appears in ``required_topics``, so
        Day 12's precision stopping rule will not fire until it has been
        measured.  An optional topic still gets a quota and can still be asked -
        it simply cannot block the interview from ending.

    ``min_items`` / ``max_items``
        Floor and cap on the quota.  The floor is how "minimum coverage" is
        expressed; the cap is plan section 8.7's guard against topic starvation,
        where one topic eats a budget it was only meant to sample.
    """

    topic_key: str
    jd_weight: float = 1.0
    subtopic_keys: tuple[str, ...] = ()
    required: bool = True
    min_items: int = 0
    max_items: int | None = None

    def __post_init__(self) -> None:
        if not self.topic_key or not self.topic_key.strip():
            raise BlueprintError("topic_key must not be blank")
        if self.topic_key != self.topic_key.strip():
            raise BlueprintError(f"topic_key {self.topic_key!r} has leading or trailing whitespace")
        if not math.isfinite(self.jd_weight) or not 0.0 <= self.jd_weight <= 1.0:
            raise BlueprintError(
                f"jd_weight for {self.topic_key!r} must be a finite value in [0, 1], "
                f"got {self.jd_weight!r}"
            )
        if self.min_items < 0:
            raise BlueprintError(
                f"min_items for {self.topic_key!r} must not be negative, got {self.min_items!r}"
            )
        if self.max_items is not None:
            if self.max_items <= 0:
                raise BlueprintError(
                    f"max_items for {self.topic_key!r} must be positive when set, "
                    f"got {self.max_items!r} - use required=False to exclude a topic "
                    "rather than capping it at zero"
                )
            if self.max_items < self.min_items:
                raise BlueprintError(
                    f"max_items ({self.max_items}) is below min_items ({self.min_items}) "
                    f"for {self.topic_key!r}"
                )
        seen: set[str] = set()
        for subtopic in self.subtopic_keys:
            if not subtopic or subtopic != subtopic.strip():
                raise BlueprintError(
                    f"subtopic key {subtopic!r} under {self.topic_key!r} is blank or padded"
                )
            if subtopic in seen:
                raise BlueprintError(
                    f"subtopic {subtopic!r} is listed twice under topic {self.topic_key!r}"
                )
            seen.add(subtopic)
        if self.topic_key in seen:
            raise BlueprintError(
                f"topic {self.topic_key!r} lists itself as one of its own subtopics - "
                "taxonomy keys are unique across levels"
            )


@dataclass(frozen=True, slots=True)
class AssessmentSpec:
    """The request: what assessment is wanted, before anything is derived.

    ``item_budget`` is the one field that may be left unset, and that is the
    normal case: the plan derives it from duration and mode.  Setting it
    explicitly is supported because a caller sometimes has a fixed question
    count and wants the time budget to follow - and because Day 13's simulation
    has a 20-item budget stated by the plan rather than by a clock.  When it is
    set, the builder checks the clock can actually fit it rather than trusting
    the caller.
    """

    role: str
    topics: tuple[TopicRequest, ...]
    level: str = DEFAULT_LEVEL
    mode: str = DEFAULT_MODE
    duration_minutes: int = 45
    #: ``None`` means "derive it from ``duration_minutes`` and ``mode``".
    item_budget: int | None = None
    #: Notes for whoever reads the stored blueprint later.  Never read by code.
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.role or not self.role.strip():
            raise BlueprintError("role must not be blank")
        if self.level not in LEVEL_TARGET_THETA:
            raise BlueprintError(
                f"unknown level {self.level!r}; expected one of {', '.join(LEVELS)}"
            )
        if self.mode not in MODE_SECONDS_PER_ITEM:
            raise BlueprintError(f"unknown mode {self.mode!r}; expected one of {', '.join(MODES)}")
        if not isinstance(self.duration_minutes, int) or isinstance(self.duration_minutes, bool):
            raise BlueprintError(
                f"duration_minutes must be an int, got {type(self.duration_minutes).__name__}"
            )
        if not MIN_DURATION_MINUTES <= self.duration_minutes <= MAX_DURATION_MINUTES:
            raise BlueprintError(
                f"duration_minutes must be between {MIN_DURATION_MINUTES} and "
                f"{MAX_DURATION_MINUTES}, got {self.duration_minutes!r} - note this "
                "field is minutes, not seconds"
            )
        if not self.topics:
            raise BlueprintError(
                "an assessment must require at least one topic; an empty blueprint "
                "would make every question ineligible and end the session at turn zero"
            )
        if self.item_budget is not None and self.item_budget < MIN_ITEM_BUDGET:
            raise BlueprintError(
                f"item_budget must be at least {MIN_ITEM_BUDGET} when set, "
                f"got {self.item_budget!r}"
            )
        if self.item_budget is not None and self.item_budget > MAX_ITEM_BUDGET:
            raise BlueprintError(
                f"item_budget must not exceed {MAX_ITEM_BUDGET}, got {self.item_budget!r}"
            )

        seen_topics: set[str] = set()
        seen_subtopics: dict[str, str] = {}
        for request in self.topics:
            if request.topic_key in seen_topics:
                raise BlueprintError(
                    f"topic {request.topic_key!r} is requested twice; merge the two "
                    "entries rather than relying on which one wins"
                )
            seen_topics.add(request.topic_key)
            for subtopic in request.subtopic_keys:
                owner = seen_subtopics.get(subtopic)
                if owner is not None:
                    raise BlueprintError(
                        f"subtopic {subtopic!r} appears under both {owner!r} and "
                        f"{request.topic_key!r}; a subtopic has exactly one parent"
                    )
                seen_subtopics[subtopic] = request.topic_key
        overlap = seen_topics & set(seen_subtopics)
        if overlap:
            raise BlueprintError(
                f"key(s) {', '.join(sorted(overlap))} are used as both a topic and a "
                "subtopic; taxonomy keys are unique across levels"
            )
        if not any(request.required for request in self.topics):
            raise BlueprintError(
                "at least one topic must be required; with none, Day 12's precision "
                "stopping rule can never be satisfied and the assessment has no goal"
            )

    # -- derived -----------------------------------------------------------

    @property
    def time_budget_s(self) -> float:
        return float(self.duration_minutes * 60)

    @property
    def seconds_per_item(self) -> int:
        return MODE_SECONDS_PER_ITEM[self.mode]

    @property
    def target_theta(self) -> float:
        return LEVEL_TARGET_THETA[self.level]

    @property
    def topic_keys(self) -> tuple[str, ...]:
        return tuple(request.topic_key for request in self.topics)

    def request_for(self, topic_key: str) -> TopicRequest:
        for request in self.topics:
            if request.topic_key == topic_key:
                return request
        raise KeyError(f"no topic request for {topic_key!r}")


__all__ = [
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
    "BlueprintError",
    "TopicRequest",
]
