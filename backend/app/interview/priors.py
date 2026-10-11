"""Resume-seeded, capped ability priors - plan section 9.4.

    | Evidence in the resume                          | Prior θ (offset) | Prior RD |
    |-------------------------------------------------|-----------------:|---------:|
    | Skill merely listed                             |            0.00  |    1.2   |
    | Skill used in a described project               |           +0.25  |    1.1   |
    | Skill in professional experience with specifics |           +0.40  |    1.0   |
    | Not mentioned at all                            |           −0.10  |    1.3   |

    "Cap the resume prior at +0.4 ... A resume is a claim, not evidence."

**Every number above is the plan's; none is new.**  How they are applied:

* **Relative to the level target.**  The plan's θ scale puts 0 at "the target
  level for the role", and the blueprint already starts every subtopic at
  ``target_theta`` (``Blueprint.prior_ability``).  So a prior is
  ``target_theta + offset`` - a resume can move the starting point at most 0.4
  above where a candidate *without* a resume starts, and at most 0.10 below it.
* **Caps, enforced in code whatever the input says:** offset clamped to
  ``[-0.10, +0.40]``; RD clamped to ``[1.0, RD_MAX]`` (a claim may never make
  the engine *more* certain than RD 1.0, the table's tightest value); θ clamped
  to the model's range.  A later bug or a hand-edited row cannot exceed them,
  and the event log refuses a recorded prior with RD < 1.0 (``state.py``).
* **Only a candidate-confirmed profile counts.**  A parser suggestion seeds
  nothing (``resumes.profile`` vs ``resumes.extracted``).
* **Strongest stated evidence per subtopic.**  If two skills map to one
  subtopic ("Redis" listed, "caching" used professionally) the subtopic gets
  the stronger row - still capped at +0.40.
* **"Not mentioned" (−0.10) needs a resume that mentions something.**  A
  confirmed profile with no skills is treated as no resume (default priors):
  the −0.10 row encodes "this resume talks about other things", which an empty
  profile does not say.
* ``years_experience`` and ``headline`` are not inputs.  Years of experience
  are not competence, and nothing in the plan maps them to θ.

**Why this cannot override evidence.**  A prior is only where the Day-11
update *starts*.  Its RD is 1.0-1.3 - the uncertainty of one or zero
observations - so the K-factor's ``f_rd`` term stays large and observed answers
move θ quickly.  ``test_priors.py`` plays a candidate who claims professional
expertise but answers like a weak one and shows the claimed and unclaimed
trajectories converge within four items, which is the plan's own acceptance
check ("confirm θ converges to the truth within 4 items").

Pure: same profile + same blueprint -> the same priors, always.
"""

from __future__ import annotations

from collections.abc import Mapping

from app.ability import RD_MAX, THETA_MAX, THETA_MIN, AbilityState
from app.blueprint import Blueprint
from app.ingestion.profile import CandidateProfile

#: plan section 9.4, evidence -> (theta offset, RD)
EVIDENCE_PRIORS: Mapping[str, tuple[float, float]] = {
    "professional": (0.40, 1.0),
    "project": (0.25, 1.1),
    "listed": (0.00, 1.2),
}
NOT_MENTIONED = (-0.10, 1.3)
_RANK = {"listed": 0, "project": 1, "professional": 2}

#: The caps.  Upper offset is the plan's "+0.4"; the lower offset and the RD
#: floor are the table's own extremes, so no input can leave the table's range.
PRIOR_OFFSET_MAX = 0.40
PRIOR_OFFSET_MIN = -0.10
PRIOR_RD_MIN = 1.0


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def compute_priors(
    profile: CandidateProfile | None, blueprint: Blueprint
) -> dict[str, AbilityState]:
    """Per-subtopic priors for the blueprint's subtopics, or ``{}`` (defaults)."""
    if profile is None or not profile.skills:
        return {}
    wanted = set(blueprint.subtopic_keys)
    best: dict[str, str] = {}
    for skill in profile.skills:
        for key in skill.subtopic_keys:
            if key in wanted and _RANK[skill.evidence] >= _RANK.get(best.get(key, ""), -1):
                best[key] = skill.evidence
    priors: dict[str, AbilityState] = {}
    for key in sorted(wanted):
        offset, rd = EVIDENCE_PRIORS[best[key]] if key in best else NOT_MENTIONED
        offset = _clamp(offset, PRIOR_OFFSET_MIN, PRIOR_OFFSET_MAX)
        rd = _clamp(rd, PRIOR_RD_MIN, RD_MAX)
        theta = _clamp(blueprint.target_theta + offset, THETA_MIN, THETA_MAX)
        priors[key] = AbilityState(theta=theta, rd=rd, n_observations=0)
    return priors


def priors_payload(priors: Mapping[str, AbilityState]) -> dict[str, dict[str, float]]:
    """The form recorded in ``PREPARATION_STARTED.priors``."""
    return {
        k: {"theta": round(a.theta, 6), "rd": round(a.rd, 6)} for k, a in sorted(priors.items())
    }


__all__ = [
    "EVIDENCE_PRIORS",
    "NOT_MENTIONED",
    "PRIOR_OFFSET_MAX",
    "PRIOR_OFFSET_MIN",
    "PRIOR_RD_MIN",
    "compute_priors",
    "priors_payload",
]
