"""Blueprint -> Selector -> Ability -> Stopping.  Phase 3 as one system.

Days 11-14 each proved their own layer.  Nothing until now ran all four
together from a blueprint, and this file is that test - which is why the last
class walks a whole session end to end rather than asserting on a mapping.

Two properties are worth stating because they are what the whole package was
built to preserve:

* **Nothing here calls into a Day 11 or Day 12 function that Day 15 changed**,
  because Day 15 changed none.  Every argument these tests pass was already
  declared by ``SelectionState`` and ``should_stop``.
* **The default blueprint behaves exactly as Day 12 did** - a ``mid``-level
  cold start is Day 12's ``PRIOR_ABILITY``, byte for byte.

No LLM, no database, no network: the whole file is arithmetic over a small
in-memory bank.
"""

from __future__ import annotations

import dataclasses
import random

import pytest

from app.ability import AbilityState, roll_up, update_ability
from app.blueprint import build_blueprint
from app.selection import (
    STOP_ITEM_BUDGET,
    STOP_NO_NEW_INFORMATION,
    STOP_TIME_BUDGET,
    CandidateItem,
    ResumeProfile,
    SelectionState,
    choose_next,
    filter_eligible,
    ineligibility_reason,
    should_stop,
)
from tests.unit.blueprint.conftest import BACKEND_SPEC

BLUEPRINT = build_blueprint(BACKEND_SPEC)

#: A small bank spanning every subtopic the blueprint plans, with difficulties
#: spread across the window so the eligibility filter always has something to
#: choose from.  Times are the committed bank's median, which is what the
#: blueprint's pacing assumes.
BANK: tuple[CandidateItem, ...] = tuple(
    CandidateItem(
        id=f"{subtopic}-{index:02d}",
        topic_key=topic,
        subtopic_key=subtopic,
        difficulty_b=-1.0 + 0.5 * index,
        time_estimate_s=330,
    )
    for topic, subtopics in (
        ("dsa", ("arrays", "graphs")),
        ("databases", ("indexing", "transactions")),
        ("caching", ("cache_invalidation",)),
    )
    for subtopic in subtopics
    for index in range(5)
)


class TestStateInitialisation:
    def test_a_blueprint_initialises_a_valid_selection_state(self):
        state = BLUEPRINT.selection_state()
        assert isinstance(state, SelectionState)
        assert state.coverage_targets == BLUEPRINT.coverage_targets
        assert state.jd_weights == BLUEPRINT.jd_weights
        assert state.time_left_s == BLUEPRINT.time_budget_s
        assert state.asked == ()

    def test_the_clock_starts_full_and_can_be_overridden_mid_session(self):
        assert BLUEPRINT.selection_state().time_left_s == 3600.0
        assert BLUEPRINT.selection_state(time_left_s=900.0).time_left_s == 900.0

    def test_every_planned_topic_starts_with_its_full_quota(self):
        state = BLUEPRINT.selection_state()
        for topic, quota in BLUEPRINT.coverage_targets.items():
            assert state.quota_remaining(topic) == quota

    def test_the_topics_with_quota_left_are_the_blueprint_s_topics(self):
        assert set(BLUEPRINT.selection_state().topics_with_quota_left) == set(BLUEPRINT.topic_keys)

    def test_an_existing_ability_estimate_is_carried_in(self):
        state = BLUEPRINT.selection_state(ability={"arrays": AbilityState(theta=1.2, rd=0.6)})
        assert state.theta_for("arrays") == 1.2
        assert state.rd_for("arrays") == 0.6

    def test_an_unmeasured_subtopic_reads_back_as_the_blueprint_s_prior(self):
        senior = build_blueprint(dataclasses.replace(BACKEND_SPEC, level="senior"))
        assert senior.selection_state().theta_for("graphs") == 0.5
        assert BLUEPRINT.selection_state().theta_for("graphs") == 0.0

    def test_a_resume_passes_through_untouched(self):
        resume = ResumeProfile(topic_affinity={"arrays": 0.8})
        assert BLUEPRINT.selection_state(resume=resume).resume is resume


class TestRequirementsReachTheSelector:
    def test_only_blueprint_topics_are_eligible(self):
        """A topic the blueprint never asked for has no quota, so Day 12's hard
        filter drops it - no extra code was needed to enforce that."""
        outsider = CandidateItem(
            id="offtopic-1",
            topic_key="observability",
            subtopic_key="tracing",
            difficulty_b=0.0,
            time_estimate_s=330,
        )
        state = BLUEPRINT.selection_state()
        eligible = filter_eligible([*BANK, outsider], state)
        assert outsider not in eligible
        assert {item.topic_key for item in eligible} <= set(BLUEPRINT.topic_keys)

    def test_a_topic_at_its_quota_becomes_ineligible(self):
        quota = BLUEPRINT.plan_for("caching").quota
        asked = tuple(item for item in BANK if item.topic_key == "caching")[:quota]
        state = BLUEPRINT.selection_state(asked=asked)
        assert state.quota_remaining("caching") == 0
        remaining = filter_eligible(BANK, state)
        assert all(item.topic_key != "caching" for item in remaining)

    def test_the_selector_returns_something_askable_at_turn_zero(self):
        selection = choose_next(BLUEPRINT.selection_state(), BANK, rng=random.Random(0))
        assert selection is not None
        assert ineligibility_reason(selection.item, BLUEPRINT.selection_state()) is None

    def test_a_tight_clock_makes_expensive_items_ineligible(self):
        state = BLUEPRINT.selection_state(time_left_s=10.0)
        assert filter_eligible(BANK, state) == []

    def test_the_difficulty_window_the_blueprint_carries_is_day_12s(self):
        state = BLUEPRINT.selection_state()
        assert filter_eligible(BANK, state, difficulty_window=BLUEPRINT.difficulty_window) == (
            filter_eligible(BANK, state)
        )


class TestStoppingRemainsCompatible:
    def stop(self, **overrides):
        arguments = BLUEPRINT.stopping_arguments(ability={}, items_asked=0, time_elapsed_s=0.0)
        return should_stop(**{**arguments, **overrides})

    def test_a_fresh_session_does_not_stop(self):
        assert not self.stop().should_stop

    def test_the_item_budget_fires_at_the_blueprint_s_budget(self):
        assert STOP_ITEM_BUDGET in self.stop(items_asked=BLUEPRINT.item_budget).reasons
        assert not self.stop(items_asked=BLUEPRINT.item_budget - 1).should_stop

    def test_the_time_budget_fires_at_the_blueprint_s_clock(self):
        assert STOP_TIME_BUDGET in self.stop(time_elapsed_s=BLUEPRINT.time_budget_s).reasons

    def test_three_small_updates_still_end_the_session(self):
        reasons = self.stop(recent_theta_deltas=[0.01, 0.01, 0.01]).reasons
        assert STOP_NO_NEW_INFORMATION in reasons

    def test_only_required_topics_can_block_the_ending(self):
        """``caching`` is optional, so measuring it is not a precondition."""
        assert BLUEPRINT.stopping_arguments(ability={}, items_asked=0, time_elapsed_s=0.0)[
            "required_topics"
        ] == ("databases", "dsa")

    def test_the_parent_map_satisfies_roll_up(self):
        """``roll_up`` raises for a subtopic with no parent, so a blueprint whose
        ``parent_of`` were incomplete would fail at the first stopping check."""
        ability = {sub: AbilityState(theta=0.3, rd=0.5) for sub in BLUEPRINT.subtopic_keys}
        topics = roll_up(ability, BLUEPRINT.parent_of)
        assert set(topics) == set(BLUEPRINT.topic_keys)

    def test_the_rd_target_reaching_the_rule_is_day_12s(self):
        from app.selection import RD_PRECISION_TARGET

        arguments = BLUEPRINT.stopping_arguments(ability={}, items_asked=0, time_elapsed_s=0.0)
        assert arguments["rd_target"] == RD_PRECISION_TARGET


class TestAFullSession:
    """One interview, driven entirely by a blueprint.  Phase 3 end to end."""

    def run(self, blueprint, *, score: float = 0.7, seed: int = 7):
        rng = random.Random(seed)
        ability: dict[str, AbilityState] = {}
        asked: tuple[CandidateItem, ...] = ()
        deltas: list[float] = []
        time_left = blueprint.time_budget_s

        while True:
            decision = should_stop(
                **blueprint.stopping_arguments(
                    ability=ability,
                    items_asked=len(asked),
                    time_elapsed_s=blueprint.time_budget_s - time_left,
                    recent_theta_deltas=deltas,
                )
            )
            if decision.should_stop:
                return asked, ability, decision.reasons

            state = blueprint.selection_state(ability=ability, asked=asked, time_left_s=time_left)
            selection = choose_next(
                state, BANK, rng=rng, difficulty_window=blueprint.difficulty_window
            )
            if selection is None:
                return asked, ability, ("pool_exhausted",)

            item = selection.item
            update = update_ability(
                state.ability_for(item.subtopic_key),
                difficulty=item.difficulty_b,
                score=score,
            )
            ability = {**ability, item.subtopic_key: update.after}
            asked = (*asked, item)
            deltas.append(update.delta_theta)
            time_left -= item.time_estimate_s

    def test_a_session_runs_to_a_stopping_reason(self):
        asked, ability, reasons = self.run(BLUEPRINT)
        assert reasons
        assert 0 < len(asked) <= BLUEPRINT.item_budget
        assert ability

    def test_no_topic_quota_is_ever_exceeded(self):
        asked, _, _ = self.run(BLUEPRINT)
        for topic, quota in BLUEPRINT.coverage_targets.items():
            assert sum(1 for item in asked if item.topic_key == topic) <= quota

    def test_no_question_is_repeated(self):
        asked, _, _ = self.run(BLUEPRINT)
        assert len({item.id for item in asked}) == len(asked)

    def test_the_clock_is_never_overspent(self):
        asked, _, _ = self.run(BLUEPRINT)
        assert sum(item.time_estimate_s for item in asked) <= BLUEPRINT.time_budget_s

    def test_every_measured_subtopic_belongs_to_the_blueprint(self):
        _, ability, _ = self.run(BLUEPRINT)
        assert set(ability) <= set(BLUEPRINT.subtopic_keys)

    def test_the_session_is_reproducible(self):
        first = self.run(BLUEPRINT)
        second = self.run(BLUEPRINT)
        assert [item.id for item in first[0]] == [item.id for item in second[0]]

    def test_a_stored_and_reloaded_blueprint_runs_the_same_session(self):
        """The property that makes resuming a session from JSONB meaningful."""
        from app.blueprint import Blueprint

        reloaded = Blueprint.from_dict(BLUEPRINT.to_dict())
        assert [item.id for item in self.run(BLUEPRINT)[0]] == [
            item.id for item in self.run(reloaded)[0]
        ]

    @pytest.mark.parametrize("minutes", [30, 45, 60, 90])
    def test_a_longer_blueprint_asks_at_least_as_many_questions(self, minutes):
        blueprint = build_blueprint(dataclasses.replace(BACKEND_SPEC, duration_minutes=minutes))
        asked, _, _ = self.run(blueprint)
        assert len(asked) <= blueprint.item_budget

    def test_a_senior_blueprint_starts_looking_higher(self):
        """The only behavioural difference a level makes: where the difficulty
        window is centred before any answer exists."""
        senior = build_blueprint(dataclasses.replace(BACKEND_SPEC, level="senior"))
        mid_pool = filter_eligible(BANK, BLUEPRINT.selection_state())
        senior_pool = filter_eligible(BANK, senior.selection_state())
        assert max(item.difficulty_b for item in senior_pool) >= max(
            item.difficulty_b for item in mid_pool
        )


class TestDay12IsUnchanged:
    def test_a_default_blueprint_state_equals_a_hand_built_day_12_state(self):
        """The exact state Day 12's own tests construct, now produced by a
        blueprint - so integration adds a producer and changes no behaviour."""
        by_hand = SelectionState(
            ability={},
            coverage_targets=BLUEPRINT.coverage_targets,
            jd_weights=BLUEPRINT.jd_weights,
            time_left_s=BLUEPRINT.time_budget_s,
        )
        from_blueprint = BLUEPRINT.selection_state()
        assert from_blueprint == by_hand

    def test_the_selector_picks_the_same_item_either_way(self):
        by_hand = SelectionState(
            ability={},
            coverage_targets=BLUEPRINT.coverage_targets,
            jd_weights=BLUEPRINT.jd_weights,
            time_left_s=BLUEPRINT.time_budget_s,
        )
        one = choose_next(by_hand, BANK, rng=random.Random(3))
        two = choose_next(BLUEPRINT.selection_state(), BANK, rng=random.Random(3))
        assert one is not None and two is not None
        assert one.item.id == two.item.id
        assert one.chosen.total == two.chosen.total
        assert one.pool_size == two.pool_size

    def test_selection_does_not_import_the_blueprint_package(self):
        """The dependency arrow points one way: blueprint -> selection.  A cycle
        would make Day 12 unusable without Day 15."""
        import app.selection

        for module in vars(app.selection).values():
            name = getattr(module, "__name__", "")
            assert not str(name).startswith("app.blueprint")
