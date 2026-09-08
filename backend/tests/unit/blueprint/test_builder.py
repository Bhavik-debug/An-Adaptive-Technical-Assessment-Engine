"""Turning a request into a contract: derivation, and everything it refuses.

The builder's job is small - four divisions - so most of this file is the
refusals.  That ratio is deliberate: a blueprint is written to a database once
and every later question is selected against it, so an invalid one that gets
through is not a bad interview, it is a bad interview nobody can explain
afterwards.

Every rejection test names the *reason* in its method name, so a failure reads
as "we stopped rejecting duplicate topics" rather than "test_invalid_7 failed".
"""

from __future__ import annotations

import dataclasses

import pytest

from app.bank.taxonomy import load_taxonomy
from app.blueprint import (
    LEVEL_TARGET_THETA,
    MODE_SECONDS_PER_ITEM,
    AssessmentSpec,
    Blueprint,
    BlueprintError,
    TopicRequest,
    build_blueprint,
    derive_item_budget,
)
from tests.unit.blueprint.conftest import BACKEND_SPEC

TAXONOMY = load_taxonomy()


def spec(**overrides) -> AssessmentSpec:
    return dataclasses.replace(BACKEND_SPEC, **overrides)


# ---------------------------------------------------------------------------
# valid blueprints
# ---------------------------------------------------------------------------


class TestNormalBlueprints:
    def test_a_multi_topic_blueprint_builds(self):
        blueprint = build_blueprint(BACKEND_SPEC)
        assert blueprint.role == "Backend Engineer"
        assert blueprint.topic_keys == ("dsa", "databases", "caching")
        assert blueprint.item_budget == 10  # 60 min / 330 s, floored
        assert blueprint.time_budget_s == 3600.0

    def test_the_quotas_spend_the_whole_item_budget(self):
        blueprint = build_blueprint(BACKEND_SPEC)
        assert sum(blueprint.coverage_targets.values()) == blueprint.item_budget

    def test_a_heavier_jd_weight_buys_at_least_as_many_items(self):
        quotas = build_blueprint(BACKEND_SPEC).coverage_targets
        assert quotas["dsa"] >= quotas["databases"] >= quotas["caching"]

    def test_the_jd_weights_reach_the_selector_unchanged(self):
        """The weight divides the budget *and* is a scored term in Day 12; it
        must not be normalised on the way through."""
        blueprint = build_blueprint(BACKEND_SPEC)
        assert blueprint.jd_weights == {"dsa": 0.9, "databases": 0.7, "caching": 0.4}

    def test_only_required_topics_are_required(self):
        blueprint = build_blueprint(BACKEND_SPEC)
        assert blueprint.required_topics == ("databases", "dsa")
        assert "caching" in blueprint.topic_keys

    def test_an_optional_topic_still_gets_a_quota(self):
        """Optional means "cannot block the ending", not "never asked"."""
        assert build_blueprint(BACKEND_SPEC).plan_for("caching").quota >= 1

    def test_subtopics_become_the_parent_map_roll_up_needs(self):
        blueprint = build_blueprint(BACKEND_SPEC)
        assert blueprint.parent_of == {
            "arrays": "dsa",
            "graphs": "dsa",
            "indexing": "databases",
            "transactions": "databases",
            "cache_invalidation": "caching",
        }

    def test_a_minimum_is_honoured_even_against_a_low_weight(self):
        blueprint = build_blueprint(
            spec(
                topics=(
                    TopicRequest("dsa", jd_weight=0.9, subtopic_keys=("arrays",)),
                    TopicRequest(
                        "caching", jd_weight=0.1, subtopic_keys=("cache_eviction",), min_items=4
                    ),
                )
            )
        )
        assert blueprint.plan_for("caching").quota >= 4

    def test_a_maximum_caps_a_topic_and_the_rest_absorb_it(self):
        blueprint = build_blueprint(
            spec(
                topics=(
                    TopicRequest("dsa", jd_weight=0.9, subtopic_keys=("arrays",), max_items=2),
                    TopicRequest("databases", jd_weight=0.1, subtopic_keys=("indexing",)),
                )
            )
        )
        assert blueprint.plan_for("dsa").quota == 2
        assert sum(blueprint.coverage_targets.values()) == blueprint.item_budget

    def test_a_single_topic_blueprint_is_valid(self):
        blueprint = build_blueprint(
            spec(topics=(TopicRequest("dsa", jd_weight=1.0, subtopic_keys=("arrays", "graphs")),))
        )
        assert blueprint.coverage_targets == {"dsa": blueprint.item_budget}


class TestBudgetDerivation:
    @pytest.mark.parametrize(
        ("mode", "minutes", "expected"),
        [
            ("screening", 30, 6),
            ("screening", 45, 9),
            ("standard", 30, 5),
            ("standard", 45, 8),
            ("standard", 60, 10),
            ("deep_dive", 45, 6),
            ("deep_dive", 90, 13),
        ],
    )
    def test_the_item_budget_is_the_clock_divided_by_the_mode_pace(self, mode, minutes, expected):
        assert derive_item_budget(spec(mode=mode, duration_minutes=minutes)) == expected

    def test_it_floors_rather_than_rounds(self):
        """27 minutes buys 4.909 standard questions.  Rounding would promise a
        fifth that the clock cannot finish and Day 12's time constraint would
        drop - after it had already been counted in a quota."""
        assert derive_item_budget(spec(mode="standard", duration_minutes=27)) == 4

    def test_a_stated_item_budget_wins_over_the_derived_one(self):
        blueprint = build_blueprint(spec(item_budget=20, duration_minutes=180))
        assert blueprint.item_budget == 20

    def test_the_time_budget_is_the_duration_in_seconds(self):
        assert build_blueprint(spec(duration_minutes=45)).time_budget_s == 2700.0

    def test_every_mode_and_level_produces_a_valid_blueprint(self):
        for mode in MODE_SECONDS_PER_ITEM:
            for level in LEVEL_TARGET_THETA:
                blueprint = build_blueprint(spec(mode=mode, level=level, duration_minutes=90))
                assert blueprint.mode == mode
                assert blueprint.level == level
                assert blueprint.target_theta == LEVEL_TARGET_THETA[level]


# ---------------------------------------------------------------------------
# invalid specs
# ---------------------------------------------------------------------------


class TestRejectedSpecs:
    def test_an_empty_assessment_is_rejected(self):
        with pytest.raises(BlueprintError, match="at least one topic"):
            AssessmentSpec(role="Backend Engineer", topics=())

    def test_a_blank_role_is_rejected(self):
        with pytest.raises(BlueprintError, match="role must not be blank"):
            AssessmentSpec(role="   ", topics=(TopicRequest("dsa"),))

    def test_a_duplicate_topic_is_rejected(self):
        with pytest.raises(BlueprintError, match="requested twice"):
            AssessmentSpec(role="r", topics=(TopicRequest("dsa", 0.9), TopicRequest("dsa", 0.5)))

    def test_a_subtopic_listed_twice_under_one_topic_is_rejected(self):
        with pytest.raises(BlueprintError, match="listed twice under topic"):
            TopicRequest("dsa", subtopic_keys=("arrays", "arrays"))

    def test_a_subtopic_shared_by_two_topics_is_rejected(self):
        """A subtopic has exactly one parent; two would make ``parent_of``
        ambiguous and ``roll_up`` would silently pick one."""
        with pytest.raises(BlueprintError, match="appears under both"):
            AssessmentSpec(
                role="r",
                topics=(
                    TopicRequest("dsa", subtopic_keys=("arrays",)),
                    TopicRequest("databases", subtopic_keys=("arrays",)),
                ),
            )

    def test_a_key_used_as_both_a_topic_and_a_subtopic_is_rejected(self):
        with pytest.raises(BlueprintError, match="both a topic and a subtopic"):
            AssessmentSpec(
                role="r",
                topics=(
                    TopicRequest("dsa", subtopic_keys=("arrays",)),
                    TopicRequest("databases", subtopic_keys=("dsa",)),
                ),
            )

    def test_a_topic_that_is_its_own_subtopic_is_rejected(self):
        with pytest.raises(BlueprintError, match="itself as one of its own subtopics"):
            TopicRequest("dsa", subtopic_keys=("dsa",))

    def test_a_negative_minimum_is_rejected(self):
        with pytest.raises(BlueprintError, match="min_items .* must not be negative"):
            TopicRequest("dsa", min_items=-1)

    def test_a_zero_maximum_is_rejected(self):
        """Capping a topic at zero is a confusing way to say "leave it out"."""
        with pytest.raises(BlueprintError, match="max_items .* must be positive"):
            TopicRequest("dsa", max_items=0)

    def test_a_maximum_below_its_minimum_is_rejected(self):
        with pytest.raises(BlueprintError, match="below min_items"):
            TopicRequest("dsa", min_items=5, max_items=2)

    @pytest.mark.parametrize("weight", [-0.1, 1.1, float("nan"), float("inf")])
    def test_an_out_of_range_jd_weight_is_rejected(self, weight):
        with pytest.raises(BlueprintError, match="jd_weight .* in \\[0, 1\\]"):
            TopicRequest("dsa", jd_weight=weight)

    def test_an_unknown_level_is_rejected(self):
        with pytest.raises(BlueprintError, match="unknown level"):
            AssessmentSpec(role="r", topics=(TopicRequest("dsa"),), level="principal")

    def test_an_unknown_mode_is_rejected(self):
        with pytest.raises(BlueprintError, match="unknown mode"):
            AssessmentSpec(role="r", topics=(TopicRequest("dsa"),), mode="lightning")

    @pytest.mark.parametrize("minutes", [0, -30, 4, 361, 3600])
    def test_an_impossible_duration_is_rejected(self, minutes):
        with pytest.raises(BlueprintError, match="duration_minutes"):
            AssessmentSpec(role="r", topics=(TopicRequest("dsa"),), duration_minutes=minutes)

    def test_a_duration_in_seconds_is_caught_by_the_ceiling(self):
        """2700 "minutes" is the classic units mistake; the guard names it."""
        with pytest.raises(BlueprintError, match="minutes, not seconds"):
            AssessmentSpec(role="r", topics=(TopicRequest("dsa"),), duration_minutes=2700)

    def test_a_non_integer_duration_is_rejected(self):
        with pytest.raises(BlueprintError, match="must be an int"):
            AssessmentSpec(role="r", topics=(TopicRequest("dsa"),), duration_minutes=45.5)

    def test_an_all_optional_assessment_is_rejected(self):
        """With nothing required, the precision stopping rule can never fire."""
        with pytest.raises(BlueprintError, match="at least one topic must be required"):
            AssessmentSpec(
                role="r",
                topics=(
                    TopicRequest("dsa", required=False),
                    TopicRequest("databases", required=False),
                ),
            )

    @pytest.mark.parametrize("budget", [0, -5, 2, 201])
    def test_an_out_of_range_stated_item_budget_is_rejected(self, budget):
        with pytest.raises(BlueprintError, match="item_budget"):
            AssessmentSpec(role="r", topics=(TopicRequest("dsa"),), item_budget=budget)


class TestRejectedDerivations:
    def test_a_duration_too_short_for_the_mode_is_rejected(self):
        with pytest.raises(BlueprintError, match="below the minimum"):
            build_blueprint(spec(mode="deep_dive", duration_minutes=15))

    def test_a_stated_budget_the_clock_cannot_hold_is_rejected(self):
        """The caller's claim is checked rather than quietly overruled."""
        with pytest.raises(BlueprintError, match="run out of clock"):
            build_blueprint(spec(item_budget=30, duration_minutes=45))

    def test_minimums_that_exceed_the_budget_are_rejected(self):
        with pytest.raises(BlueprintError, match="minimums total"):
            build_blueprint(
                spec(
                    duration_minutes=30,  # 5 items
                    topics=(
                        TopicRequest("dsa", 0.9, ("arrays",), min_items=4),
                        TopicRequest("databases", 0.7, ("indexing",), min_items=4),
                    ),
                )
            )

    def test_maximums_that_cannot_absorb_the_budget_are_rejected(self):
        with pytest.raises(BlueprintError, match="maximums total"):
            build_blueprint(
                spec(
                    duration_minutes=90,  # 16 items
                    topics=(
                        TopicRequest("dsa", 0.9, ("arrays",), max_items=2),
                        TopicRequest("databases", 0.7, ("indexing",), max_items=2),
                    ),
                )
            )

    def test_more_required_topics_than_items_is_rejected(self):
        """Every required topic needs at least one question, so a 5-item budget
        cannot carry six required topics."""
        with pytest.raises(BlueprintError, match="minimums total"):
            build_blueprint(
                spec(
                    duration_minutes=30,  # 5 items
                    topics=tuple(
                        TopicRequest(f"t{index}", 0.5, (f"s{index}",)) for index in range(6)
                    ),
                )
            )


# ---------------------------------------------------------------------------
# the taxonomy check
# ---------------------------------------------------------------------------


class TestAgainstTheRealTaxonomy:
    def test_a_blueprint_over_committed_keys_builds(self):
        blueprint = build_blueprint(
            AssessmentSpec(
                role="Backend Engineer",
                duration_minutes=60,
                topics=(TopicRequest("databases", 0.8), TopicRequest("caching", 0.5)),
            ),
            taxonomy=TAXONOMY,
        )
        assert blueprint.topic_keys == ("databases", "caching")

    def test_an_unlisted_topic_is_filled_in_from_the_taxonomy(self):
        """ "Cover databases" means all of databases; a topic with no subtopics
        would contribute nothing to ``roll_up`` and the precision rule."""
        blueprint = build_blueprint(
            AssessmentSpec(role="r", topics=(TopicRequest("caching", 1.0),)),
            taxonomy=TAXONOMY,
        )
        assert set(blueprint.parent_of) == {
            "cache_invalidation",
            "cache_eviction",
            "cache_stampede",
        }

    def test_an_explicit_subset_of_subtopics_is_kept(self):
        blueprint = build_blueprint(
            AssessmentSpec(
                role="r",
                topics=(TopicRequest("databases", 1.0, subtopic_keys=("indexing",)),),
            ),
            taxonomy=TAXONOMY,
        )
        assert tuple(blueprint.parent_of) == ("indexing",)

    def test_a_topic_typo_is_rejected(self):
        with pytest.raises(BlueprintError, match="not in the taxonomy"):
            build_blueprint(
                AssessmentSpec(role="r", topics=(TopicRequest("databses", 1.0),)),
                taxonomy=TAXONOMY,
            )

    def test_a_subtopic_used_as_a_topic_is_rejected(self):
        with pytest.raises(BlueprintError, match="is a subtopic in the taxonomy, not a topic"):
            build_blueprint(
                AssessmentSpec(role="r", topics=(TopicRequest("indexing", 1.0),)),
                taxonomy=TAXONOMY,
            )

    def test_a_subtopic_under_the_wrong_parent_is_rejected(self):
        with pytest.raises(BlueprintError, match="is not a subtopic of"):
            build_blueprint(
                AssessmentSpec(
                    role="r",
                    topics=(TopicRequest("databases", 1.0, subtopic_keys=("arrays",)),),
                ),
                taxonomy=TAXONOMY,
            )

    def test_an_unknown_subtopic_is_rejected(self):
        with pytest.raises(BlueprintError, match="subtopic 'nope' is not in the taxonomy"):
            build_blueprint(
                AssessmentSpec(
                    role="r",
                    topics=(TopicRequest("databases", 1.0, subtopic_keys=("nope",)),),
                ),
                taxonomy=TAXONOMY,
            )

    def test_the_check_is_optional(self):
        """The simulation builds blueprints over a synthetic taxonomy that is
        not in the committed file; requiring it would couple an experiment to a
        dataset it does not use."""
        assert build_blueprint(BACKEND_SPEC).topic_keys == ("dsa", "databases", "caching")


# ---------------------------------------------------------------------------
# properties
# ---------------------------------------------------------------------------


class TestProperties:
    def test_the_same_spec_always_produces_the_same_blueprint(self):
        assert build_blueprint(BACKEND_SPEC) == build_blueprint(BACKEND_SPEC)

    def test_building_does_not_mutate_the_spec(self):
        before = dataclasses.astuple(BACKEND_SPEC)
        build_blueprint(BACKEND_SPEC)
        assert dataclasses.astuple(BACKEND_SPEC) == before

    def test_topic_order_follows_the_spec_rather_than_the_allocation(self):
        """A report lists topics in the order the JD did; the quota division
        sorts internally and must not leak that ordering out."""
        reordered = spec(topics=tuple(reversed(BACKEND_SPEC.topics)))
        assert build_blueprint(reordered).topic_keys == ("caching", "databases", "dsa")

    def test_no_quota_is_ever_negative(self):
        for minutes in range(20, 181, 10):
            for mode in MODE_SECONDS_PER_ITEM:
                blueprint = build_blueprint(spec(mode=mode, duration_minutes=minutes))
                assert all(quota >= 0 for quota in blueprint.coverage_targets.values())

    def test_a_longer_assessment_never_asks_fewer_questions(self):
        budgets = [
            build_blueprint(spec(duration_minutes=minutes)).item_budget
            for minutes in range(30, 181, 5)
        ]
        assert budgets == sorted(budgets)

    def test_the_quotas_always_spend_the_whole_budget(self):
        for minutes in range(20, 181, 10):
            blueprint = build_blueprint(spec(duration_minutes=minutes))
            assert sum(blueprint.coverage_targets.values()) == blueprint.item_budget

    def test_an_invalid_blueprint_cannot_be_constructed_at_all(self):
        """Validation lives in ``__post_init__``, so there is no window in which
        a half-built blueprint exists and could be stored."""
        with pytest.raises(BlueprintError):
            Blueprint(
                role="r",
                level="mid",
                mode="standard",
                item_budget=10,
                time_budget_s=3600.0,
                topics=(),
            )
