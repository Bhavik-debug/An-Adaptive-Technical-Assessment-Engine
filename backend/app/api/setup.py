"""What the interview setup wizard may offer (Day 26).

``GET /api/setup-options`` returns the committed taxonomy's topics and
subtopics, and the levels, modes and duration bounds the blueprint builder
accepts.  The frontend renders exactly these, so it can never offer a choice
the API would reject with a 422, and the vocabulary has one source of truth
(``data/question-bank/taxonomy.json`` and ``app.blueprint.spec``) instead of a
copy in TypeScript.

Read-only, and it exposes nothing about selection: no per-level target
ability, no seconds-per-item, no difficulty window.
"""

from __future__ import annotations

import dataclasses
from functools import lru_cache

from fastapi import APIRouter
from pydantic import BaseModel

from app.bank.taxonomy import Taxonomy, load_taxonomy
from app.blueprint.spec import (
    DEFAULT_LEVEL,
    DEFAULT_MODE,
    LEVELS,
    MAX_DURATION_MINUTES,
    MIN_DURATION_MINUTES,
    MODES,
    AssessmentSpec,
)
from app.deps import CurrentUser

router = APIRouter(prefix="/api", tags=["setup"])

#: ``AssessmentSpec.duration_minutes``' own default, read rather than copied.
DEFAULT_DURATION_MINUTES: int = next(
    f.default for f in dataclasses.fields(AssessmentSpec) if f.name == "duration_minutes"
)  # type: ignore[assignment]


class SubtopicOption(BaseModel):
    key: str
    display_name: str


class TopicOption(BaseModel):
    key: str
    display_name: str
    domain: str
    subtopics: list[SubtopicOption]


class SetupOptions(BaseModel):
    topics: list[TopicOption]
    levels: list[str]
    modes: list[str]
    default_level: str
    default_mode: str
    default_duration_minutes: int
    min_duration_minutes: int
    max_duration_minutes: int


@lru_cache(maxsize=1)
def _taxonomy() -> Taxonomy:
    return load_taxonomy()


@router.get("/setup-options", summary="Topics, levels and modes the setup wizard may offer")
async def setup_options(user: CurrentUser) -> SetupOptions:
    del user  # authenticated like every other /api route; the answer is not per-user
    tax = _taxonomy()
    return SetupOptions(
        topics=[
            TopicOption(
                key=topic.key,
                display_name=topic.display_name,
                domain=topic.domain,
                subtopics=[
                    SubtopicOption(key=s.key, display_name=s.display_name)
                    for s in tax.children_of(topic.key)
                ],
            )
            for topic in tax.topics()
        ],
        levels=list(LEVELS),
        modes=list(MODES),
        default_level=DEFAULT_LEVEL,
        default_mode=DEFAULT_MODE,
        default_duration_minutes=DEFAULT_DURATION_MINUTES,
        min_duration_minutes=MIN_DURATION_MINUTES,
        max_duration_minutes=MAX_DURATION_MINUTES,
    )


__all__ = ["router"]
