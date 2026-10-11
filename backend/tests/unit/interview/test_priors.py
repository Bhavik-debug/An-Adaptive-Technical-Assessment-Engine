"""Resume-seeded priors (plan section 9.4): exact table values, hard caps, and
what the evidence does to them afterwards."""

from __future__ import annotations

import math

import pytest

from app.ability import RD_MAX, AbilityState, probability_correct, update_ability
from app.blueprint import build_blueprint
from app.ingestion.profile import CandidateProfile, parse_profile
from app.interview import EventLogCorrupted, fold
from app.interview.priors import (
    PRIOR_OFFSET_MAX,
    PRIOR_RD_MIN,
    compute_priors,
    priors_payload,
)
from tests.unit.blueprint.conftest import BACKEND_SPEC
from tests.unit.interview.scenario import HAPPY_PATH, SESSION_ID, as_stored

BP = build_blueprint(BACKEND_SPEC)  # subtopics: arrays, graphs, indexing, transactions, ...
T0 = BP.target_theta


def profile(*skills, **over):
    return CandidateProfile.model_validate(
        {
            "skills": [{"name": n, "evidence": e, "subtopic_keys": list(k)} for n, e, k in skills],
            **over,
        }
    )


def test_no_profile_or_no_skills_means_default_priors():
    assert compute_priors(None, BP) == {}
    assert compute_priors(profile(), BP) == {}


def test_the_plans_table_exactly():
    p = compute_priors(
        profile(
            ("Arrays", "listed", ["arrays"]),
            ("Graphs", "project", ["graphs"]),
            ("Postgres indexes", "professional", ["indexing"]),
        ),
        BP,
    )
    assert (p["arrays"].theta - T0, p["arrays"].rd) == pytest.approx((0.0, 1.2))
    assert (p["graphs"].theta - T0, p["graphs"].rd) == pytest.approx((0.25, 1.1))
    assert (p["indexing"].theta - T0, p["indexing"].rd) == pytest.approx((0.40, 1.0))
    # Mentioned nothing about transactions: "not mentioned at all".
    assert (p["transactions"].theta - T0, p["transactions"].rd) == pytest.approx((-0.10, 1.3))
    assert all(a.n_observations == 0 for a in p.values())
    assert set(p) == set(BP.subtopic_keys)


def test_conflicting_claims_take_the_strongest_evidence_still_capped():
    p = compute_priors(
        profile(("Indexes", "listed", ["indexing"]), ("B-trees", "professional", ["indexing"])),
        BP,
    )
    assert p["indexing"].theta - T0 == pytest.approx(0.40)


def test_adversarial_profile_cannot_exceed_the_caps():
    claims = [(f"Skill {i}", "professional", list(BP.subtopic_keys)[:5]) for i in range(60)]
    p = compute_priors(profile(*claims, years_experience=60, headline="World's best"), BP)
    for state in p.values():
        assert state.theta - T0 <= PRIOR_OFFSET_MAX + 1e-12
        assert PRIOR_RD_MIN <= state.rd <= RD_MAX


def test_years_and_headline_are_not_inputs():
    a = compute_priors(profile(("Arrays", "project", ["arrays"])), BP)
    b = compute_priors(
        profile(("Arrays", "project", ["arrays"]), years_experience=40, headline="CTO"), BP
    )
    assert a == b


def test_subtopics_outside_the_blueprint_or_unknown_are_ignored():
    p = compute_priors(profile(("Kafka", "professional", ["kafka_streams", "nonsense"])), BP)
    assert all(math.isclose(a.theta - T0, -0.10) for a in p.values())


def test_deterministic():
    prof = profile(("Arrays", "project", ["arrays"]), ("Graphs", "listed", ["graphs"]))
    assert priors_payload(compute_priors(prof, BP)) == priors_payload(compute_priors(prof, BP))


@pytest.mark.parametrize(
    "raw",
    [
        {"skills": "not a list"},
        {"skills": [{"name": "", "evidence": "listed"}]},
        {"skills": [{"name": "Go", "evidence": "god-tier"}]},
        {"skills": [], "years_experience": -3},
        {"skills": [], "extra": "field"},
        "a string",
        42,
    ],
)
def test_malformed_stored_profiles_degrade_to_no_profile(raw):
    assert parse_profile(raw) is None
    assert compute_priors(parse_profile(raw), BP) == {}


# ---------------------------------------------------------------------------
# The log refuses a prior more confident than the table allows
# ---------------------------------------------------------------------------


def _log(priors):
    steps = list(HAPPY_PATH[:2])
    steps[1] = (steps[1][0], {"priors": priors, "prior_source": "resume_profile"})
    return as_stored(steps)


def test_recorded_priors_fold_into_state():
    state = fold(SESSION_ID, _log({"arrays": {"theta": 0.4, "rd": 1.0}}))
    assert state.priors == {"arrays": AbilityState(theta=0.4, rd=1.0)}
    assert state.ability == {}  # a prior is not an observation


@pytest.mark.parametrize(
    "bad", [{"theta": 0.4, "rd": 0.5}, {"theta": 9.0, "rd": 1.0}, {"theta": 0.0, "rd": 2.0}]
)
def test_the_log_refuses_out_of_range_priors(bad):
    with pytest.raises(EventLogCorrupted):
        fold(SESSION_ID, _log({"arrays": bad}))


# ---------------------------------------------------------------------------
# Evidence after the prior - measured, not assumed
# ---------------------------------------------------------------------------


def _trajectory(start: AbilityState, scorer, n=6):
    s, out = start, []
    for _ in range(n):
        b = s.theta
        s = update_ability(s, difficulty=b, score=scorer(b)).after
        out.append(s.theta)
    return out


def test_observed_answers_still_drive_the_estimate():
    """A professional claim (+0.40) followed by failing answers: theta falls
    below the level target within two items and keeps falling. The claim sets
    the start; it cannot hold the estimate up against evidence."""
    claimed = AbilityState(theta=T0 + 0.40, rd=1.0)
    path = _trajectory(claimed, lambda b: 0.0)
    assert path[1] < T0 and all(b < a for a, b in zip(path, path[1:], strict=False))


def test_measured_washout_of_a_false_claim():
    """MEASURED, and weaker than plan section 9.4 hopes ("washed out within 2-3
    answers ... converges to the truth within 4 items").

    A weak candidate (true theta = -1) answering at their expected score, items
    targeted at the current estimate. Gap between the claimed (+0.40, RD 1.0)
    and the unclaimed (0.00, RD 1.3) trajectories after items 1..6. The gap
    shrinks, but is still ~0.22 after four items. Pinned so a change to the
    table or the update is a visible decision - see docs/ingestion.md §5.
    """
    claimed = _trajectory(AbilityState(theta=0.40, rd=1.0), lambda b: probability_correct(-1.0, b))
    neutral = _trajectory(AbilityState(theta=0.0, rd=1.3), lambda b: probability_correct(-1.0, b))
    gaps = [round(a - b, 3) for a, b in zip(claimed, neutral, strict=True)]
    assert gaps == [0.332, 0.284, 0.254, 0.222, 0.193, 0.167]
