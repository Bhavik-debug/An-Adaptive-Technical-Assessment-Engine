"""The candidate profile: what the candidate confirmed, as a validated contract.

The parser's output (``resumes.extracted``) is a *suggestion*.  This model is
what the candidate submits with ``PATCH /api/resumes/{id}/profile`` - typically
the suggestion, reviewed and corrected - and it is the only thing that can seed
priors (``app.interview.priors``).

Even confirmed, every field is a **claim**: the candidate saying it does not
make it true, so its only effect is a capped prior that the first few answers
outweigh (plan section 9.4).  ``years_experience`` and ``headline`` are kept for
display and the report; neither feeds any number.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

Evidence = Literal["listed", "project", "professional"]

MAX_SKILLS = 60


class ProfileSkill(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: Annotated[str, Field(min_length=1, max_length=80)]
    evidence: Evidence = "listed"
    subtopic_keys: list[Annotated[str, Field(min_length=1, max_length=64)]] = Field(
        default_factory=list, max_length=5
    )

    @field_validator("name")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("must not be blank")
        return v.strip()

    @field_validator("subtopic_keys")
    @classmethod
    def _unique(cls, v: list[str]) -> list[str]:
        return list(dict.fromkeys(v))


class CandidateProfile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    headline: Annotated[str, Field(max_length=120)] | None = None
    skills: list[ProfileSkill] = Field(default_factory=list, max_length=MAX_SKILLS)
    years_experience: float | None = Field(default=None, ge=0, le=60)

    def unknown_subtopics(self, known: set[str]) -> list[str]:
        return sorted({k for s in self.skills for k in s.subtopic_keys if k not in known})


def parse_profile(raw: Any) -> CandidateProfile | None:
    """A stored profile, validated - or ``None`` if absent or malformed.

    Malformed stored data (it should not exist: the API validates on write)
    degrades to "no profile", i.e. the default priors - never to an exception
    and never to a partially trusted profile.
    """
    if raw is None:
        return None
    try:
        return CandidateProfile.model_validate(raw)
    except ValidationError:
        return None


__all__ = ["MAX_SKILLS", "CandidateProfile", "Evidence", "ProfileSkill", "parse_profile"]
