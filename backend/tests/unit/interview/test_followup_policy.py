"""Plan section 8.5's follow-up policy, as a pure function - no LLM anywhere."""

from __future__ import annotations

import json
import math

import pytest

from app.ability import RD_MAX, AbilityState, update_ability
from app.interview.followup import (
    CAP_ITEM,
    CAP_SESSION,
    RULE_AMBIGUOUS_BAND,
    RULE_CONCEPT_ABSENT,
    RULE_FACTUAL_ERROR,
    RULE_HIGH_UNCERTAINTY,
    RuleStatus,
    decide,
    policy_input,
)

#: A three-concept item in bank order: one essential concept, two lesser.
EXPECTED = [("eviction", 2), ("stampede", 3), ("invalidation", 1)]
CALM = {"score": 0.9, "rd": 0.5}  # outside the band, below the RD threshold


def judgment(labels=None, errors=()):
    labels = labels or {"eviction": "covered", "stampede": "covered", "invalidation": "covered"}
    return {
        "concepts": [
            {"key": k, "label": v, "evidence": None if v == "absent" else "quoted span"}
            for k, v in labels.items()
        ],
        "errors": [{"claim": "some claim", "severity": s} for s in errors],
    }


def run(j="default", *, score=0.9, rd=0.5, item=0, session=0, expected=EXPECTED):
    if j == "default":
        j = judgment()
    return decide(
        policy_input(
            judgment=j,
            expected=expected,
            score=score,
            rd=rd,
            item_follow_ups=item,
            session_follow_ups=session,
        )
    )


def statuses(decision):
    return {o.rule: o.status for o in decision.outcomes}


# ---------------------------------------------------------------------------
# Each trigger on its own
# ---------------------------------------------------------------------------


def test_no_trigger_no_follow_up():
    d = run()
    assert (d.follow_up, d.reason, d.concept_key, d.capped_by) == (False, None, None, None)
    assert set(statuses(d).values()) == {RuleStatus.NOT_FIRED}


def test_high_weight_concept_absent():
    d = run(judgment({"eviction": "covered", "stampede": "absent", "invalidation": "covered"}))
    assert (d.follow_up, d.reason, d.concept_key) == (True, RULE_CONCEPT_ABSENT, "stampede")


def test_a_low_weight_absent_concept_is_not_enough():
    d = run(judgment({"eviction": "absent", "stampede": "covered", "invalidation": "absent"}))
    assert not d.follow_up
    assert statuses(d)[RULE_CONCEPT_ABSENT] is RuleStatus.NOT_FIRED


def test_partial_is_not_absent():
    d = run(judgment({"eviction": "covered", "stampede": "partial", "invalidation": "covered"}))
    assert not d.follow_up


def test_major_factual_error():
    d = run(judgment(errors=("major",)))
    assert (d.follow_up, d.reason) == (True, RULE_FACTUAL_ERROR)


def test_minor_errors_do_not_trigger():
    d = run(judgment(errors=("minor", "minor")))
    assert not d.follow_up


@pytest.mark.parametrize("score", [0.45, 0.5, 0.6, 0.65])
def test_ambiguous_band_is_inclusive(score):
    d = run(score=score)
    assert (d.follow_up, d.reason) == (True, RULE_AMBIGUOUS_BAND)


@pytest.mark.parametrize("score", [0.0, 0.4499, 0.6501, 1.0])
def test_outside_the_band(score):
    assert not run(score=score).follow_up


def test_high_uncertainty_is_strict():
    assert run(rd=0.7001).reason == RULE_HIGH_UNCERTAINTY
    assert not run(rd=0.7).follow_up


def test_rules_fire_in_the_plans_order():
    everything = judgment(
        {"eviction": "covered", "stampede": "absent", "invalidation": "absent"}, ("major",)
    )
    assert run(everything, score=0.5, rd=1.0).reason == RULE_CONCEPT_ABSENT
    assert run(judgment(errors=("major",)), score=0.5, rd=1.0).reason == RULE_FACTUAL_ERROR
    assert run(score=0.5, rd=1.0).reason == RULE_AMBIGUOUS_BAND


def test_tie_between_essential_concepts_breaks_on_bank_order():
    expected = [("alpha", 3), ("beta", 3), ("gamma", 1)]
    j = judgment({"alpha": "absent", "beta": "absent", "gamma": "covered"})
    assert run(j, expected=expected).concept_key == "alpha"
    assert run(j, expected=list(reversed(expected))).concept_key == "beta"


def test_non_concept_reasons_still_target_the_weakest_concept():
    j = judgment({"eviction": "absent", "stampede": "partial", "invalidation": "covered"})
    d = run(j, rd=1.0)  # stampede is partial, so concept_absent does not fire
    assert d.reason == RULE_HIGH_UNCERTAINTY and d.concept_key == "eviction"
    assert run(rd=1.0).concept_key is None  # all covered: probe for depth


# ---------------------------------------------------------------------------
# Missing, null, malformed, contradictory: insufficient evidence, never "poor"
# ---------------------------------------------------------------------------


def test_no_judgment_at_all():
    d = run(None, score=None, rd=None)
    assert not d.follow_up
    assert set(statuses(d).values()) == {RuleStatus.UNAVAILABLE}


@pytest.mark.parametrize("bad", ["not a dict", 42, [], {"concepts": "nope", "errors": 7}])
def test_malformed_judgments(bad):
    d = run(bad)
    assert not d.follow_up
    assert statuses(d)[RULE_FACTUAL_ERROR] is RuleStatus.UNAVAILABLE


def test_an_unlabelled_essential_concept_is_not_absent():
    j = {"concepts": [{"key": "eviction", "label": "covered"}], "errors": []}
    d = run(j)
    assert not d.follow_up
    assert statuses(d)[RULE_CONCEPT_ABSENT] is RuleStatus.UNAVAILABLE


def test_an_invalid_label_is_unavailable_not_absent():
    j = judgment({"eviction": "covered", "stampede": "ABSENT??", "invalidation": "covered"})
    assert not run(j).follow_up


def test_a_judgment_naming_concepts_the_item_lacks_is_contradictory():
    j = judgment({"eviction": "covered", "stampede": "absent", "invented": "absent"})
    d = run(j)
    assert statuses(d)[RULE_CONCEPT_ABSENT] is RuleStatus.UNAVAILABLE
    assert d.trace()["rules"][0]["unknown_concepts"] == ["invented"]
    assert not d.follow_up


@pytest.mark.parametrize("weight", [None, 0, 4, "3", 3.0])
def test_an_out_of_scale_weight_is_not_essential(weight):
    expected = [("eviction", 2), ("stampede", weight), ("invalidation", 1)]
    j = judgment({"eviction": "covered", "stampede": "absent", "invalidation": "covered"})
    assert not run(j, expected=expected).follow_up


@pytest.mark.parametrize("score", [None, float("nan"), float("inf"), -0.1, 1.5])
def test_missing_or_invalid_score(score):
    d = run(score=score)
    assert statuses(d)[RULE_AMBIGUOUS_BAND] is RuleStatus.UNAVAILABLE
    assert not d.follow_up


@pytest.mark.parametrize("rd", [None, float("nan"), 0.0, -1.0])
def test_missing_or_invalid_rd(rd):
    d = run(rd=rd)
    assert statuses(d)[RULE_HIGH_UNCERTAINTY] is RuleStatus.UNAVAILABLE
    assert not d.follow_up


# ---------------------------------------------------------------------------
# Caps
# ---------------------------------------------------------------------------


def test_two_per_item():
    trigger = {"score": 0.5}
    assert run(item=1, **trigger).follow_up
    d = run(item=2, **trigger)
    assert (d.follow_up, d.capped_by) == (False, CAP_ITEM)


def test_six_per_session():
    assert run(score=0.5, session=5).follow_up
    d = run(score=0.5, session=6)
    assert (d.follow_up, d.capped_by) == (False, CAP_SESSION)


def test_a_capped_decision_still_records_what_fired():
    d = run(score=0.5, item=2)
    assert statuses(d)[RULE_AMBIGUOUS_BAND] is RuleStatus.FIRED


# ---------------------------------------------------------------------------
# Determinism and the trace
# ---------------------------------------------------------------------------


def test_identical_inputs_identical_decisions():
    j = judgment(
        {"eviction": "partial", "stampede": "absent", "invalidation": "covered"}, ("minor",)
    )
    decisions = [run(j, score=0.55, rd=0.9) for _ in range(20)]
    assert len({json.dumps(d.trace(), sort_keys=True) for d in decisions}) == 1


def test_the_trace_is_keys_and_numbers():
    d = run(judgment(errors=("major",)), score=0.61234567, rd=0.98765)
    trace = d.trace()
    assert trace["reason"] == RULE_FACTUAL_ERROR
    assert [r["rule"] for r in trace["rules"]] == [
        RULE_CONCEPT_ABSENT,
        RULE_FACTUAL_ERROR,
        RULE_AMBIGUOUS_BAND,
        RULE_HIGH_UNCERTAINTY,
    ]
    assert trace["rules"][1]["major_errors"] == 1
    assert trace["rules"][2]["score"] == 0.6123
    blob = json.dumps(trace)
    assert "some claim" not in blob and "quoted span" not in blob  # no candidate text


# ---------------------------------------------------------------------------
# The measured consequence of the plan's RD threshold
# ---------------------------------------------------------------------------


def test_rd_stays_above_the_threshold_for_five_observations():
    """With RD_MAX = 1.3 and well-targeted items (p = 0.5), Day 11's update
    leaves RD above 0.7 after observations 1-5 (1.09, 0.957, 0.863, 0.793,
    0.737) and below it from the 6th (0.691). Follow-up answers are
    observations too, so they lower RD as well.

    So high_uncertainty fires on essentially every early item and the caps
    (2/item, 6/session) are what bound early follow-ups. Pinned so that a change
    to either constant is a visible decision, not a silent behaviour change.
    """
    state = AbilityState(theta=0.0, rd=RD_MAX)
    rds = []
    for _ in range(6):
        state = update_ability(state, difficulty=0.0, score=0.5).after
        rds.append(state.rd)
    assert all(rd > 0.7 for rd in rds[:5]) and rds[5] < 0.7
    assert math.isclose(rds[0], 1.09, abs_tol=1e-3)
    assert math.isclose(rds[4], 0.7369, abs_tol=1e-3)
