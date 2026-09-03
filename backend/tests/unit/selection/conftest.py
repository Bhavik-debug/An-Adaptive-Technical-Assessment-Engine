"""Builders shared by the Day 12 selection tests.

Every test in this package is offline and pure - no database, no model, no
clock - so the fixtures here are plain constructors rather than pytest
fixtures, which keeps each test's inputs visible at its own call site.
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence

from app.ability import AbilityState
from app.selection import CandidateItem, ResumeProfile, SelectionState


def item(
    item_id: str = "q1",
    *,
    topic: str = "systems",
    subtopic: str = "caching",
    b: float = 0.0,
    seconds: int = 120,
    a: float = 1.0,
    embedding: Sequence[float] | None = None,
) -> CandidateItem:
    return CandidateItem(
        id=item_id,
        topic_key=topic,
        subtopic_key=subtopic,
        difficulty_b=b,
        time_estimate_s=seconds,
        discrimination_a=a,
        embedding=None if embedding is None else tuple(embedding),
    )


def state(
    *,
    ability: Mapping[str, AbilityState] | None = None,
    theta: float | None = None,
    targets: Mapping[str, int] | None = None,
    jd: Mapping[str, float] | None = None,
    asked: Sequence[CandidateItem] = (),
    time_left: float = 600.0,
    resume: ResumeProfile | None = None,
) -> SelectionState:
    """A state with sane defaults; pass ``theta`` for the single-subtopic case."""
    if ability is None:
        ability = {"caching": AbilityState(theta=0.0 if theta is None else theta, rd=0.9)}
    return SelectionState(
        ability=ability,
        coverage_targets={"systems": 3} if targets is None else targets,
        jd_weights={"systems": 1.0} if jd is None else jd,
        asked=tuple(asked),
        time_left_s=time_left,
        resume=resume,
    )


def unit_vector(*components: float) -> tuple[float, ...]:
    """L2-normalise, so the test vectors live where the real embeddings do."""
    norm = math.sqrt(sum(c * c for c in components))
    return tuple(c / norm for c in components)


class ScriptedRandom(random.Random):
    """A ``random.Random`` whose ``random()`` returns a scripted sequence.

    Subclassing the real class rather than passing a duck type keeps the
    production signature honest (``rng: random.Random``) while letting a test
    say exactly which branch it wants: ``ScriptedRandom([0.99])`` never
    explores, ``ScriptedRandom([0.0])`` always does.

    Once the script runs out it falls through to the seeded generator, which
    matters because ``Random.choice`` on a subclass reaches ``random()`` too -
    so the exploration draw stays a real uniform draw over the top five, and
    only the explore/exploit branch is dictated.
    """

    def __init__(self, draws: Sequence[float], *, seed: int = 0) -> None:
        super().__init__(seed)
        self._draws = list(draws)
        self.calls = 0

    def random(self) -> float:
        self.calls += 1
        if self._draws:
            return self._draws.pop(0)
        return super().random()
