"""Day 11 and Day 12 together, over a whole simulated interview.

    ability state (Day 11)
            |
    candidate questions
            |
    hard filtering  ->  scoring  ->  epsilon-greedy choice
            |
      the next question
            |
    graded answer -> update_ability -> back to the top

No database, no model, no clock: the loop below is the adaptive engine running
end to end in microseconds, which is the property that makes it testable at all.

**This is not the Day 13 simulation.** There is no synthetic candidate
population, no Beta-distributed response model, no policy comparison and no
convergence measurement - those are plan section 8.6 and are deferred. What
this file asserts is that the pieces compose and that the invariants hold turn
after turn.
"""

from __future__ import annotations

import random

from app.ability import AbilityState, update_ability
from app.selection import (
    CandidateItem,
    SelectionState,
    choose_next,
    ineligibility_reason,
    should_stop,
)
from tests.unit.selection.conftest import unit_vector

PARENT_OF = {
    "caching": "systems",
    "sharding": "systems",
    "sorting": "algorithms",
    "hashing": "algorithms",
}

SUBTOPIC_TOPIC = [
    ("systems", "caching"),
    ("systems", "sharding"),
    ("algorithms", "sorting"),
    ("algorithms", "hashing"),
]

#: A bank spanning both topics and the whole difficulty range, with vectors that
#: make items in the same subtopic look similar to each other - which is what
#: gives the redundancy term something real to do.
BANK = [
    CandidateItem(
        id=f"{subtopic}-{index}",
        topic_key=topic,
        subtopic_key=subtopic,
        difficulty_b=round(-2.0 + 0.5 * index, 2),
        time_estimate_s=90 + 30 * (index % 3),
        embedding=unit_vector(
            1.0 if subtopic == "caching" else 0.0,
            1.0 if subtopic == "sharding" else 0.0,
            1.0 if subtopic == "sorting" else 0.0,
            1.0 if subtopic == "hashing" else 0.0,
            0.05 * index,
        ),
    )
    for topic, subtopic in SUBTOPIC_TOPIC
    for index in range(9)
]


def fresh_state() -> SelectionState:
    return SelectionState(
        ability={key: AbilityState(theta=0.0, rd=1.2) for key in PARENT_OF},
        coverage_targets={"systems": 3, "algorithms": 3},
        jd_weights={"systems": 0.9, "algorithms": 0.6},
        time_left_s=1800.0,
    )


def run_interview(*, seed: int, ability_of_candidate: float, turns: int = 6):
    """Ask ``turns`` questions, grading each against a fixed true ability.

    The grade is deterministic - 1.0 when the item is at or below the
    candidate's true ability, 0.0 above it - which is a crude response model on
    purpose: this file is testing the plumbing, not measuring convergence.
    """
    rng = random.Random(seed)
    state = fresh_state()
    chosen: list[CandidateItem] = []
    deltas: list[float] = []

    for _ in range(turns):
        selection = choose_next(state, BANK, rng=rng)
        if selection is None:
            break
        picked = selection.item
        chosen.append(picked)

        score = 1.0 if picked.difficulty_b <= ability_of_candidate else 0.0
        update = update_ability(
            state.ability_for(picked.subtopic_key),
            difficulty=picked.difficulty_b,
            score=score,
        )
        deltas.append(update.delta_theta)
        state = SelectionState(
            ability={**state.ability, picked.subtopic_key: update.after},
            coverage_targets=state.coverage_targets,
            jd_weights=state.jd_weights,
            asked=(*state.asked, picked),
            time_left_s=state.time_left_s - picked.time_estimate_s,
        )

    return state, chosen, deltas


class TestTheLoopHolds:
    def test_it_asks_the_questions_it_was_asked_for(self):
        _, chosen, _ = run_interview(seed=11, ability_of_candidate=0.5)
        assert len(chosen) == 6

    def test_no_question_is_ever_repeated(self):
        _, chosen, _ = run_interview(seed=11, ability_of_candidate=0.5)
        assert len({i.id for i in chosen}) == len(chosen)

    def test_no_topic_exceeds_its_quota(self):
        state, chosen, _ = run_interview(seed=11, ability_of_candidate=0.5)
        for topic, target in state.coverage_targets.items():
            assert sum(1 for i in chosen if i.topic_key == topic) <= target

    def test_every_chosen_item_was_eligible_when_it_was_chosen(self):
        """Replays the constraint check against the state as it was each turn."""
        state = fresh_state()
        rng = random.Random(7)
        for _ in range(6):
            selection = choose_next(state, BANK, rng=rng)
            if selection is None:
                break
            picked = selection.item
            assert ineligibility_reason(picked, state) is None
            update = update_ability(
                state.ability_for(picked.subtopic_key),
                difficulty=picked.difficulty_b,
                score=1.0,
            )
            state = SelectionState(
                ability={**state.ability, picked.subtopic_key: update.after},
                coverage_targets=state.coverage_targets,
                jd_weights=state.jd_weights,
                asked=(*state.asked, picked),
                time_left_s=state.time_left_s - picked.time_estimate_s,
            )

    def test_every_chosen_item_sits_inside_the_difficulty_window(self):
        state = fresh_state()
        rng = random.Random(3)
        for _ in range(6):
            selection = choose_next(state, BANK, rng=rng)
            assert selection is not None
            picked = selection.item
            assert abs(picked.difficulty_b - state.theta_for(picked.subtopic_key)) <= 1.5
            update = update_ability(
                state.ability_for(picked.subtopic_key),
                difficulty=picked.difficulty_b,
                score=0.0,
            )
            state = SelectionState(
                ability={**state.ability, picked.subtopic_key: update.after},
                coverage_targets=state.coverage_targets,
                jd_weights=state.jd_weights,
                asked=(*state.asked, picked),
                time_left_s=state.time_left_s - picked.time_estimate_s,
            )

    def test_the_whole_run_is_reproducible_from_a_seed(self):
        first = run_interview(seed=99, ability_of_candidate=1.0)[1]
        second = run_interview(seed=99, ability_of_candidate=1.0)[1]
        assert [i.id for i in first] == [i.id for i in second]

    def test_uncertainty_falls_as_the_interview_proceeds(self):
        """Day 11's RD update, driven by Day 12's choices."""
        state, chosen, _ = run_interview(seed=5, ability_of_candidate=0.5)
        touched = {i.subtopic_key for i in chosen}
        assert touched
        for subtopic in touched:
            assert state.rd_for(subtopic) < 1.2


class TestCoverageActuallySpreads:
    def test_selection_does_not_park_on_one_topic(self):
        """The coverage-deficit term and the per-topic cap, working together:
        a six-item interview with quotas of 3 and 3 must touch both topics."""
        _, chosen, _ = run_interview(seed=11, ability_of_candidate=0.5)
        assert {i.topic_key for i in chosen} == {"systems", "algorithms"}

    def test_the_pool_empties_once_every_quota_is_served(self):
        state, _, _ = run_interview(seed=11, ability_of_candidate=0.5)
        assert state.topics_with_quota_left == ()
        assert choose_next(state, BANK, rng=random.Random(0)) is None


class TestStoppingSeesTheSameState:
    def test_the_item_budget_ends_the_run_the_selection_layer_would_continue(self):
        state, chosen, deltas = run_interview(seed=11, ability_of_candidate=0.5)
        decision = should_stop(
            ability=state.ability,
            parent_of=PARENT_OF,
            required_topics=list(state.coverage_targets),
            items_asked=len(chosen),
            item_budget=6,
            time_elapsed_s=1800.0 - state.time_left_s,
            time_budget_s=1800.0,
            recent_theta_deltas=deltas,
        )
        assert decision.should_stop is True
        assert "item_budget_reached" in decision.reasons

    def test_a_fresh_interview_does_not_stop_before_it_starts(self):
        state = fresh_state()
        decision = should_stop(
            ability=state.ability,
            parent_of=PARENT_OF,
            required_topics=list(state.coverage_targets),
            items_asked=0,
            item_budget=12,
            time_elapsed_s=0.0,
            time_budget_s=1800.0,
            recent_theta_deltas=[],
        )
        assert decision.should_stop is False
