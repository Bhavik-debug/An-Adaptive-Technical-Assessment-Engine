"""The contract value: its accessors, its serialisation, and its own validation.

Two things get tested hard here.

**Serialisation**, because ``interview_sessions.blueprint`` is a JSONB column and
a session resumed next week must be judged against the blueprint it actually ran
under.  A round trip that quietly loses ``required`` would turn a required topic
optional between two halves of one interview.

**Construction-time validation**, because a ``Blueprint`` is frozen and validated
in ``__post_init__``, so there is no window in which a half-valid one exists.
A caller cannot build an invalid blueprint and fix it later - which is the
property that lets every consumer skip re-checking.
"""

from __future__ import annotations

import dataclasses
import json

import pytest

from app.ability import RD_MAX
from app.blueprint import (
    BLUEPRINT_VERSION,
    Blueprint,
    BlueprintError,
    TopicPlan,
    build_blueprint,
)
from app.selection import PRIOR_ABILITY, RD_PRECISION_TARGET
from app.selection.constraints import DIFFICULTY_WINDOW
from tests.unit.blueprint.conftest import BACKEND_SPEC

BLUEPRINT = build_blueprint(BACKEND_SPEC)


def plan(**overrides) -> TopicPlan:
    base = {"topic_key": "dsa", "quota": 4, "jd_weight": 0.9, "required": True}
    return TopicPlan(**{**base, **overrides})


def blueprint(**overrides) -> Blueprint:
    base = {
        "role": "Backend Engineer",
        "level": "mid",
        "mode": "standard",
        "item_budget": 8,
        "time_budget_s": 2700.0,
        "topics": (plan(quota=8),),
    }
    return Blueprint(**{**base, **overrides})


class TestAccessors:
    def test_it_hands_the_selector_exactly_the_mappings_it_declares(self):
        """10 items; the two required topics take a floor of 1 each, and the
        remaining 8 divide 0.9 / 0.7 / 0.4 into 3, 2, 1 with two spares going to
        the largest fractions (databases 0.8, then caching 0.6 ahead of dsa 0.6
        on the name tie-break)."""
        assert BLUEPRINT.coverage_targets == {"dsa": 4, "databases": 4, "caching": 2}
        assert BLUEPRINT.jd_weights == {"dsa": 0.9, "databases": 0.7, "caching": 0.4}

    def test_required_topics_is_sorted_for_a_stable_log_line(self):
        assert BLUEPRINT.required_topics == ("databases", "dsa")

    def test_parent_of_is_derived_rather_than_stored(self):
        """Two sources of truth drift; ``SelectionState`` derives ``asked_ids``
        for the same reason."""
        assert BLUEPRINT.parent_of["arrays"] == "dsa"
        assert set(BLUEPRINT.parent_of) == set(BLUEPRINT.subtopic_keys)

    def test_plan_for_names_the_topic_it_could_not_find(self):
        with pytest.raises(KeyError, match="no plan for topic 'nope'"):
            BLUEPRINT.plan_for("nope")

    def test_describe_lists_what_the_session_agreed_to(self):
        labels = dict(BLUEPRINT.describe())
        assert labels["role"] == "Backend Engineer"
        assert labels["item budget"] == "10"
        assert labels["topic quotas"] == "dsa=4, databases=4, caching=2"


class TestPrior:
    def test_a_mid_level_blueprint_reproduces_day_12s_prior_exactly(self):
        """The default changes nothing about the selector.  Choosing another
        level is an explicit decision by the caller, not a silent retune."""
        assert BLUEPRINT.prior_ability() == PRIOR_ABILITY

    @pytest.mark.parametrize(
        ("level", "theta"),
        [("junior", -0.5), ("mid", 0.0), ("senior", 0.5), ("staff", 1.0)],
    )
    def test_the_level_moves_theta_and_never_the_uncertainty(self, level, theta):
        """A level says where to start looking; it is not evidence about this
        person, so RD stays at its maximum."""
        state = build_blueprint(dataclasses.replace(BACKEND_SPEC, level=level)).prior_ability()
        assert state.theta == theta
        assert state.rd == RD_MAX

    def test_the_default_carries_day_12s_own_constants(self):
        assert BLUEPRINT.difficulty_window == DIFFICULTY_WINDOW
        assert BLUEPRINT.rd_target == RD_PRECISION_TARGET


class TestSerialisation:
    def test_a_round_trip_preserves_the_whole_blueprint(self):
        assert Blueprint.from_dict(BLUEPRINT.to_dict()) == BLUEPRINT

    def test_it_survives_real_json(self):
        """JSONB goes through an encoder; tuples come back as lists."""
        restored = Blueprint.from_dict(json.loads(json.dumps(BLUEPRINT.to_dict())))
        assert restored == BLUEPRINT

    def test_the_document_holds_json_primitives_only(self):
        def check(value: object) -> None:
            if isinstance(value, dict):
                for key, inner in value.items():
                    assert isinstance(key, str)
                    check(inner)
            elif isinstance(value, list):
                for inner in value:
                    check(inner)
            else:
                assert isinstance(value, str | int | float | bool) or value is None

        check(BLUEPRINT.to_dict())

    def test_the_version_is_recorded(self):
        assert BLUEPRINT.to_dict()["version"] == BLUEPRINT_VERSION

    def test_an_unknown_version_is_refused_rather_than_guessed(self):
        document = BLUEPRINT.to_dict()
        document["version"] = BLUEPRINT_VERSION + 1
        with pytest.raises(BlueprintError, match="is not the version this build understands"):
            Blueprint.from_dict(document)

    def test_a_stored_blueprint_is_revalidated_on_the_way_in(self):
        """A row is not a value this process produced.  It may have been written
        by an older build or edited by a migration."""
        document = BLUEPRINT.to_dict()
        document["item_budget"] = 0
        with pytest.raises(BlueprintError, match="item_budget must be between"):
            Blueprint.from_dict(document)

    def test_a_corrupted_quota_is_caught(self):
        document = BLUEPRINT.to_dict()
        document["topics"][0]["quota"] = -3
        with pytest.raises(BlueprintError, match="must not be negative"):
            Blueprint.from_dict(document)

    def test_missing_fields_are_named(self):
        document = BLUEPRINT.to_dict()
        del document["role"]
        del document["item_budget"]
        with pytest.raises(BlueprintError, match="missing: role, item_budget"):
            Blueprint.from_dict(document)

    def test_a_missing_topic_field_is_named(self):
        document = BLUEPRINT.to_dict()
        del document["topics"][0]["quota"]
        with pytest.raises(BlueprintError, match="missing 'quota'"):
            Blueprint.from_dict(document)

    def test_a_non_mapping_document_is_rejected(self):
        with pytest.raises(BlueprintError, match="must be a mapping"):
            Blueprint.from_dict(["not", "a", "blueprint"])  # type: ignore[arg-type]

    def test_topics_must_be_a_list(self):
        document = BLUEPRINT.to_dict()
        document["topics"] = "dsa"
        with pytest.raises(BlueprintError, match="'topics' must be a list"):
            Blueprint.from_dict(document)

    def test_serialisation_does_not_mutate_the_blueprint(self):
        before = dataclasses.astuple(BLUEPRINT)
        document = BLUEPRINT.to_dict()
        document["topics"].clear()
        assert dataclasses.astuple(BLUEPRINT) == before


class TestValidation:
    def test_an_empty_blueprint_is_rejected(self):
        with pytest.raises(BlueprintError, match="at least one topic"):
            blueprint(topics=())

    def test_a_required_topic_with_no_quota_is_rejected(self):
        """A required topic that can never be asked is a coverage requirement
        that silently fails."""
        with pytest.raises(BlueprintError, match="required but was allocated no items"):
            plan(quota=0, required=True)

    def test_an_optional_topic_may_have_no_quota(self):
        assert plan(quota=0, required=False).quota == 0

    def test_a_negative_quota_is_rejected(self):
        with pytest.raises(BlueprintError, match="must not be negative"):
            plan(quota=-1)

    def test_quotas_that_exceed_the_item_budget_are_rejected(self):
        with pytest.raises(BlueprintError, match="quotas total 12 items but the budget is 8"):
            blueprint(topics=(plan(quota=12),))

    def test_a_budget_the_clock_cannot_hold_is_rejected(self):
        with pytest.raises(BlueprintError, match="run out of clock"):
            blueprint(item_budget=20, time_budget_s=600.0, topics=(plan(quota=20),))

    def test_an_all_optional_blueprint_is_rejected(self):
        with pytest.raises(BlueprintError, match="at least one topic must be required"):
            blueprint(topics=(plan(quota=8, required=False),))

    def test_a_duplicate_topic_plan_is_rejected(self):
        with pytest.raises(BlueprintError, match="planned twice"):
            blueprint(topics=(plan(quota=4), plan(quota=4)))

    def test_a_shared_subtopic_is_rejected(self):
        with pytest.raises(BlueprintError, match="appears under both"):
            blueprint(
                item_budget=8,
                topics=(
                    plan(topic_key="dsa", quota=4, subtopic_keys=("arrays",)),
                    plan(topic_key="databases", quota=4, subtopic_keys=("arrays",)),
                ),
            )

    def test_a_key_used_at_two_levels_is_rejected(self):
        with pytest.raises(BlueprintError, match="both a topic and a subtopic"):
            blueprint(
                item_budget=8,
                topics=(
                    plan(topic_key="dsa", quota=4, subtopic_keys=("arrays",)),
                    plan(topic_key="databases", quota=4, subtopic_keys=("dsa",)),
                ),
            )

    def test_a_subtopic_listed_twice_is_rejected(self):
        with pytest.raises(BlueprintError, match="lists a subtopic twice"):
            plan(subtopic_keys=("arrays", "arrays"))

    @pytest.mark.parametrize("weight", [-0.1, 1.5, float("nan")])
    def test_an_out_of_range_jd_weight_is_rejected(self, weight):
        with pytest.raises(BlueprintError, match="jd_weight"):
            plan(jd_weight=weight)

    @pytest.mark.parametrize(
        ("field", "value", "message"),
        [
            ("role", "  ", "role must not be blank"),
            ("level", "principal", "unknown level"),
            ("mode", "lightning", "unknown mode"),
            ("item_budget", 0, "item_budget must be between"),
            ("item_budget", 999, "item_budget must be between"),
            ("time_budget_s", 0.0, "time_budget_s must be finite and positive"),
            ("time_budget_s", float("inf"), "time_budget_s must be finite and positive"),
            ("target_theta", float("nan"), "target_theta must be finite"),
            ("difficulty_window", 0.0, "difficulty_window must be finite and positive"),
            ("rd_target", -1.0, "rd_target must be finite and positive"),
        ],
    )
    def test_invalid_fields_are_rejected_by_name(self, field, value, message):
        with pytest.raises(BlueprintError, match=message):
            blueprint(**{field: value})

    def test_it_is_immutable(self):
        with pytest.raises(dataclasses.FrozenInstanceError):
            BLUEPRINT.role = "someone else"  # type: ignore[misc]
