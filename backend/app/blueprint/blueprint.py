"""The contract between "what assessment do we need" and "which question next".

::

    AssessmentSpec        role, level, duration, JD weights      "what is wanted"
            |
    build_blueprint
            |
            v
    Blueprint             quotas, budgets, required topics       "what must happen"
            |
            +--> selection_state(...)     -> app.selection.SelectionState   (Day 12)
            +--> stopping_arguments(...)  -> app.selection.should_stop(**)  (Day 12)
            +--> to_dict()                -> interview_sessions.blueprint   (JSONB)

**A blueprint states requirements; it never selects a question.**  That boundary
is not a style preference - it is the reason the package is small.  Everything
about *which* item to ask is Day 12's, and this type's whole job is to hand
Day 12 the arguments it already declared and has been receiving from hand-built
test fixtures since it was written.

Why no changes to Day 12
------------------------

``SelectionState`` already takes ``coverage_targets``, ``jd_weights``,
``time_left_s`` and ``prior``; ``should_stop`` already takes ``required_topics``,
``parent_of``, ``item_budget`` and ``time_budget_s``.  Its docstring says of the
first two: *"Consumed, never built: the blueprint builder that produces it is
Day 15, and is DEFERRED."*  Day 15 is the producer that was deferred, so the
integration is a constructor call and not a refactor.  ``app.selection`` does
not import this package, and the dependency arrow points one way only.

Why it serialises
-----------------

``InterviewSession.blueprint`` is a JSONB column, described in
``app/models/interview.py`` as *"computed once at session creation by the
blueprint builder"*.  A session that is resumed, replayed from its event log, or
re-graded months later must be judged against the blueprint it was actually run
under - not against whatever the calibration tables say today.  So
:meth:`Blueprint.to_dict` emits JSON primitives only, :meth:`Blueprint.from_dict`
reconstructs and **re-validates**, and ``version`` records which shape was
written.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

from app.ability import RD_MAX, AbilityState
from app.blueprint.spec import (
    LEVEL_TARGET_THETA,
    MAX_ITEM_BUDGET,
    MIN_ITEM_BUDGET,
    MODE_SECONDS_PER_ITEM,
    BlueprintError,
)
from app.selection.constraints import DIFFICULTY_WINDOW
from app.selection.state import CandidateItem, ResumeProfile, SelectionState
from app.selection.stopping import RD_PRECISION_TARGET

#: Bumped when the serialised shape changes incompatibly.  A stored blueprint
#: from an older version is a migration problem to solve deliberately, not a
#: ``KeyError`` three months into a support ticket.
BLUEPRINT_VERSION: Final[int] = 1


@dataclass(frozen=True, slots=True)
class TopicPlan:
    """One topic's requirements, after the budget has been divided.

    The resolved counterpart of :class:`~app.blueprint.spec.TopicRequest`:
    everything a request could leave to the builder is now a number.
    """

    topic_key: str
    quota: int
    jd_weight: float
    required: bool
    subtopic_keys: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.topic_key.strip():
            raise BlueprintError("topic_key must not be blank")
        if self.quota < 0:
            raise BlueprintError(
                f"quota for {self.topic_key!r} must not be negative, got {self.quota!r}"
            )
        if not math.isfinite(self.jd_weight) or not 0.0 <= self.jd_weight <= 1.0:
            raise BlueprintError(
                f"jd_weight for {self.topic_key!r} must be in [0, 1], got {self.jd_weight!r}"
            )
        if self.required and self.quota == 0:
            raise BlueprintError(
                f"topic {self.topic_key!r} is required but was allocated no items; "
                "either raise the item budget, lower another topic's minimum, or "
                "mark this topic optional"
            )
        if len(set(self.subtopic_keys)) != len(self.subtopic_keys):
            raise BlueprintError(f"topic {self.topic_key!r} lists a subtopic twice")


@dataclass(frozen=True, slots=True)
class Blueprint:
    """A validated, executable assessment contract.

    Immutable, because it is the record of what a session agreed to do.  A
    blueprint that could be edited mid-session would make "did this interview
    meet its requirements?" unanswerable.

    ``target_theta``
        The ability the assessment is aimed at, from the candidate's level.  Used
        for the cold-start prior and nothing else - see :meth:`prior_ability`.

    ``difficulty_window``
        Day 12's ``|b - theta| <= window`` hard filter, carried here so the
        argument ``choose_next`` already accepts has exactly one source.  It
        defaults to Day 12's own constant and **no level or mode changes it**,
        because no evidence supports a different value; the field exists so a
        future calibration has somewhere to land, not because it varies today.

    ``rd_target``
        The precision every required topic must reach before the assessment can
        stop early.  Defaults to Day 12's 0.40.  Day 13 measured that no session
        reaches it within 20 items and Day 12's threshold was deliberately not
        moved; this field carries it rather than redefining it.
    """

    role: str
    level: str
    mode: str
    item_budget: int
    time_budget_s: float
    topics: tuple[TopicPlan, ...]
    target_theta: float = 0.0
    difficulty_window: float = DIFFICULTY_WINDOW
    rd_target: float = RD_PRECISION_TARGET
    notes: str = ""
    version: int = BLUEPRINT_VERSION

    # -- validation --------------------------------------------------------

    def __post_init__(self) -> None:
        if not self.role.strip():
            raise BlueprintError("role must not be blank")
        if self.level not in LEVEL_TARGET_THETA:
            raise BlueprintError(f"unknown level {self.level!r}")
        if self.mode not in MODE_SECONDS_PER_ITEM:
            raise BlueprintError(f"unknown mode {self.mode!r}")
        if not self.topics:
            raise BlueprintError("a blueprint must plan at least one topic")

        if not MIN_ITEM_BUDGET <= self.item_budget <= MAX_ITEM_BUDGET:
            raise BlueprintError(
                f"item_budget must be between {MIN_ITEM_BUDGET} and {MAX_ITEM_BUDGET}, "
                f"got {self.item_budget!r}"
            )
        if not math.isfinite(self.time_budget_s) or self.time_budget_s <= 0.0:
            raise BlueprintError(
                f"time_budget_s must be finite and positive, got {self.time_budget_s!r}"
            )
        if not math.isfinite(self.target_theta):
            raise BlueprintError(f"target_theta must be finite, got {self.target_theta!r}")
        if not math.isfinite(self.difficulty_window) or self.difficulty_window <= 0.0:
            raise BlueprintError(
                f"difficulty_window must be finite and positive, got {self.difficulty_window!r}"
            )
        if not math.isfinite(self.rd_target) or self.rd_target <= 0.0:
            raise BlueprintError(f"rd_target must be finite and positive, got {self.rd_target!r}")
        if self.version != BLUEPRINT_VERSION:
            raise BlueprintError(
                f"blueprint version {self.version!r} is not the version this build "
                f"understands ({BLUEPRINT_VERSION})"
            )

        seen_topics: set[str] = set()
        seen_subtopics: dict[str, str] = {}
        for plan in self.topics:
            if plan.topic_key in seen_topics:
                raise BlueprintError(f"topic {plan.topic_key!r} is planned twice")
            seen_topics.add(plan.topic_key)
            for subtopic in plan.subtopic_keys:
                owner = seen_subtopics.get(subtopic)
                if owner is not None:
                    raise BlueprintError(
                        f"subtopic {subtopic!r} appears under both {owner!r} and "
                        f"{plan.topic_key!r}"
                    )
                seen_subtopics[subtopic] = plan.topic_key
        overlap = seen_topics & set(seen_subtopics)
        if overlap:
            raise BlueprintError(
                f"key(s) {', '.join(sorted(overlap))} are used as both a topic and a subtopic"
            )

        if not any(plan.required for plan in self.topics):
            raise BlueprintError(
                "at least one topic must be required; with none, the precision "
                "stopping rule can never fire"
            )

        quota_total = sum(plan.quota for plan in self.topics)
        if quota_total > self.item_budget:
            raise BlueprintError(
                f"the topic quotas total {quota_total} items but the budget is "
                f"{self.item_budget}; a quota the session can never reach is a "
                "coverage requirement that silently fails"
            )
        if quota_total == 0:
            raise BlueprintError("every topic was allocated zero items")

        # The clock has to fit the questions.  `seconds_per_item` is the mode's
        # own pacing figure, so this is the blueprint checking its own arithmetic
        # rather than importing a second opinion about how long a question takes.
        needed = self.item_budget * MODE_SECONDS_PER_ITEM[self.mode]
        if needed > self.time_budget_s:
            raise BlueprintError(
                f"{self.item_budget} items at {MODE_SECONDS_PER_ITEM[self.mode]} s each "
                f"needs {needed} s but the time budget is {self.time_budget_s:.0f} s; "
                "the session would run out of clock before it ran out of questions"
            )

    # -- what Day 12 reads -------------------------------------------------

    @property
    def coverage_targets(self) -> dict[str, int]:
        """``topic_key -> quota``.  ``SelectionState.coverage_targets``."""
        return {plan.topic_key: plan.quota for plan in self.topics}

    @property
    def jd_weights(self) -> dict[str, float]:
        """``topic_key -> role alignment``.  ``SelectionState.jd_weights``."""
        return {plan.topic_key: plan.jd_weight for plan in self.topics}

    @property
    def required_topics(self) -> tuple[str, ...]:
        """The topics the precision stopping rule must be satisfied about.

        Sorted, so a log line and a stored blueprint agree across runs.
        """
        return tuple(sorted(plan.topic_key for plan in self.topics if plan.required))

    @property
    def topic_keys(self) -> tuple[str, ...]:
        return tuple(plan.topic_key for plan in self.topics)

    @property
    def subtopic_keys(self) -> tuple[str, ...]:
        return tuple(sub for plan in self.topics for sub in plan.subtopic_keys)

    @property
    def parent_of(self) -> dict[str, str]:
        """``subtopic_key -> topic_key`` - what ``roll_up`` and ``should_stop`` need.

        Day 11 stores theta per subtopic and Day 12 quotas per topic, so this
        mapping is the join between them.  It is derived from the plans rather
        than stored, for the reason ``SelectionState`` gives for deriving
        ``asked_ids``: two sources of truth drift.

        A blueprint whose topics list no subtopics returns an empty mapping,
        which is legal - ``roll_up`` then has nothing to aggregate and the
        precision rule cannot fire, so such a session ends on a budget.  The
        builder warns about this by refusing it only when a taxonomy was
        supplied and could have filled the subtopics in.
        """
        return {sub: plan.topic_key for plan in self.topics for sub in plan.subtopic_keys}

    def plan_for(self, topic_key: str) -> TopicPlan:
        for plan in self.topics:
            if plan.topic_key == topic_key:
                return plan
        raise KeyError(f"blueprint has no plan for topic {topic_key!r}")

    def prior_ability(self) -> AbilityState:
        """The cold start every subtopic begins from.

        ``theta = target_theta`` (from the candidate's level), ``rd = RD_MAX``.
        The uncertainty is *not* reduced by knowing the level: a level is a claim
        about where to start looking, not evidence about this person, and
        claiming precision we do not have is exactly what ``RD_MAX`` exists to
        prevent.

        At ``level = "mid"`` this returns a value equal to Day 12's
        ``PRIOR_ABILITY``, so the default blueprint changes nothing about the
        selector's behaviour.  A test pins that.
        """
        return AbilityState(theta=self.target_theta, rd=RD_MAX)

    # -- the two integration points ---------------------------------------

    def selection_state(
        self,
        *,
        ability: Mapping[str, AbilityState] | None = None,
        asked: Sequence[CandidateItem] = (),
        time_left_s: float | None = None,
        resume: ResumeProfile | None = None,
    ) -> SelectionState:
        """Build the Day 12 state this blueprint describes.

        ``time_left_s`` defaults to the whole time budget, which is the correct
        value at turn zero and wrong at every later turn - so a caller mid-session
        passes the real remaining clock.  It is a default rather than a required
        argument because "start a session from this blueprint" is the call that
        happens most and should be the shortest.

        Nothing here is a new abstraction: every argument is a field
        ``SelectionState`` has had since Day 12.
        """
        return SelectionState(
            ability=dict(ability or {}),
            coverage_targets=self.coverage_targets,
            jd_weights=self.jd_weights,
            time_left_s=self.time_budget_s if time_left_s is None else time_left_s,
            asked=tuple(asked),
            resume=resume,
            prior=self.prior_ability(),
        )

    def stopping_arguments(
        self,
        *,
        ability: Mapping[str, AbilityState],
        items_asked: int,
        time_elapsed_s: float,
        recent_theta_deltas: Sequence[float] = (),
    ) -> dict[str, Any]:
        """The keyword arguments ``app.selection.should_stop`` already takes.

        Returned as a mapping to be splatted rather than calling ``should_stop``
        here, so that this package never becomes a second place where the
        stopping *decision* lives.  The blueprint supplies the requirements; Day
        12 decides.
        """
        return {
            "ability": dict(ability),
            "parent_of": self.parent_of,
            "required_topics": self.required_topics,
            "items_asked": items_asked,
            "item_budget": self.item_budget,
            "time_elapsed_s": time_elapsed_s,
            "time_budget_s": self.time_budget_s,
            "recent_theta_deltas": tuple(recent_theta_deltas),
            "rd_target": self.rd_target,
        }

    # -- persistence -------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """JSON primitives only - the shape stored in ``interview_sessions.blueprint``.

        Lists rather than tuples, floats rather than ``Decimal``, and no nested
        dataclasses: whatever comes back out of JSONB has to reconstruct without
        a custom decoder.
        """
        return {
            "version": self.version,
            "role": self.role,
            "level": self.level,
            "mode": self.mode,
            "item_budget": self.item_budget,
            "time_budget_s": self.time_budget_s,
            "target_theta": self.target_theta,
            "difficulty_window": self.difficulty_window,
            "rd_target": self.rd_target,
            "notes": self.notes,
            "topics": [
                {
                    "topic_key": plan.topic_key,
                    "quota": plan.quota,
                    "jd_weight": plan.jd_weight,
                    "required": plan.required,
                    "subtopic_keys": list(plan.subtopic_keys),
                }
                for plan in self.topics
            ],
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> Blueprint:
        """Reconstruct a stored blueprint, **re-validating** on the way in.

        Re-validation is the point.  A row in a database is not a value this
        process produced - it may have been written by an older build, edited by
        a migration, or corrupted - and the moment to find out is at load, with a
        message naming the field, not three turns into an interview.
        """
        if not isinstance(raw, Mapping):
            raise BlueprintError(
                f"a blueprint document must be a mapping, got {type(raw).__name__}"
            )
        missing = [
            key
            for key in ("role", "level", "mode", "item_budget", "time_budget_s", "topics")
            if key not in raw
        ]
        if missing:
            raise BlueprintError(f"blueprint document is missing: {', '.join(missing)}")

        topics_raw = raw["topics"]
        if not isinstance(topics_raw, Sequence) or isinstance(topics_raw, str | bytes):
            raise BlueprintError("blueprint 'topics' must be a list")

        plans: list[TopicPlan] = []
        for entry in topics_raw:
            if not isinstance(entry, Mapping):
                raise BlueprintError("each entry of blueprint 'topics' must be a mapping")
            for key in ("topic_key", "quota", "jd_weight"):
                if key not in entry:
                    raise BlueprintError(f"a topic entry is missing {key!r}")
            plans.append(
                TopicPlan(
                    topic_key=str(entry["topic_key"]),
                    quota=int(entry["quota"]),
                    jd_weight=float(entry["jd_weight"]),
                    required=bool(entry.get("required", True)),
                    subtopic_keys=tuple(str(s) for s in entry.get("subtopic_keys", ())),
                )
            )

        return cls(
            role=str(raw["role"]),
            level=str(raw["level"]),
            mode=str(raw["mode"]),
            item_budget=int(raw["item_budget"]),
            time_budget_s=float(raw["time_budget_s"]),
            topics=tuple(plans),
            target_theta=float(raw.get("target_theta", 0.0)),
            difficulty_window=float(raw.get("difficulty_window", DIFFICULTY_WINDOW)),
            rd_target=float(raw.get("rd_target", RD_PRECISION_TARGET)),
            notes=str(raw.get("notes", "")),
            version=int(raw.get("version", BLUEPRINT_VERSION)),
        )

    # -- reporting ---------------------------------------------------------

    def describe(self) -> tuple[tuple[str, str], ...]:
        """Label/value pairs, for a log line or a report header.

        Same shape as ``ExperimentConfig.describe`` in Day 13, so the two print
        through the same table formatter.
        """
        quotas = ", ".join(f"{plan.topic_key}={plan.quota}" for plan in self.topics)
        return (
            ("role", self.role),
            ("level", self.level),
            ("mode", self.mode),
            ("item budget", str(self.item_budget)),
            ("time budget", f"{self.time_budget_s:.0f} s"),
            ("topic quotas", quotas),
            ("required topics", ", ".join(self.required_topics)),
            ("subtopics", str(len(self.subtopic_keys))),
            ("target theta / RD", f"{self.target_theta} / {RD_MAX}"),
            ("difficulty window", str(self.difficulty_window)),
            ("rd target", str(self.rd_target)),
        )


__all__ = [
    "BLUEPRINT_VERSION",
    "Blueprint",
    "TopicPlan",
]
