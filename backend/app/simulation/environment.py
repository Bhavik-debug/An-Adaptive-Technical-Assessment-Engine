"""The synthetic world: a question bank, and 200 candidates with hidden ability.

Plan section 8.6, step 1.  Everything here is generated from one integer seed,
so an entire experiment is reproducible from ``ExperimentConfig.seed`` alone.

The seeding discipline
----------------------

Python's built-in ``hash()`` is salted per process, so a seed derived from it
would change between runs of the same command.  :func:`derive_seed` uses BLAKE2b
over the string form of its parts instead, which is stable across processes,
machines and Python versions.

Every stochastic component draws from its **own** stream, named by what it is
for::

    derive_seed(seed, "bank")                     the question bank
    derive_seed(seed, "population")               ground-truth abilities
    derive_seed(seed, "resume", candidate_id)     the synthetic resume
    derive_seed(seed, "response", cand_id, q_id)  one graded answer
    derive_seed(seed, "policy", strategy, cand)   a policy's own randomness

Separate streams are not tidiness.  If the response noise and the policy's
epsilon-greedy draw shared a generator, then asking a *different* question would
shift every later response - and the three strategies would be answering
different exams.  Because the response stream is keyed by *(candidate, item)*
and nothing else, candidate 7 answering item ``systems-caching-19`` produces the
same score whoever asked it, whenever they asked it, under whatever policy.
That is the technique known as **common random numbers**, and it is the single
strongest fairness control in this experiment.

What is hidden from whom
------------------------

::

    SyntheticCandidate.true_theta          the ground truth
            |
            | read ONLY by app.simulation.response
            v
    a graded score in [0, 1]
            |
            | the only channel out of the simulator
            v
    app.ability.update_ability  ->  SelectionState  ->  the policy

A strategy is handed a ``SelectionState`` and the bank.  Neither carries the
candidate, so there is no reference through which ground truth could be read
even by accident - a structural guarantee rather than a promise, and one the
tests check by construction.
"""

from __future__ import annotations

import hashlib
import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from app.selection import CandidateItem, ResumeProfile
from app.simulation.config import (
    BANK_DIFFICULTY_JITTER,
    BANK_DIFFICULTY_MAX,
    BANK_DIFFICULTY_MIN,
    BANK_DISCRIMINATION_MAX,
    BANK_DISCRIMINATION_MIN,
    BANK_TIME_ESTIMATES_S,
    CANDIDATE_OVERALL_SD,
    CANDIDATE_SUBTOPIC_SD,
    CANDIDATE_THETA_MAX,
    CANDIDATE_THETA_MIN,
    EMBEDDING_DIM,
    EMBEDDING_SPREAD,
    RESUME_AFFINITY_MAX,
    RESUME_AFFINITY_MIN,
    RESUME_SUBTOPICS,
    ExperimentConfig,
)


def derive_seed(*parts: object) -> int:
    """A stable 63-bit seed from any parts, via BLAKE2b.

    Not ``hash()``: that is salted per process, so the "same" experiment would
    differ between two runs of the same script and the reproducibility claim
    would be false in exactly the way nobody checks.
    """
    digest = hashlib.blake2b("|".join(str(p) for p in parts).encode("utf-8"), digest_size=8)
    return int.from_bytes(digest.digest(), "big") >> 1


def _clamp(value: float, low: float, high: float) -> float:
    return low if value < low else high if value > high else value


# ---------------------------------------------------------------------------
# the bank
# ---------------------------------------------------------------------------


def _subtopic_axis(subtopics: Sequence[str], subtopic: str, dim: int) -> list[float]:
    """A unit vector for a subtopic: one axis of the embedding space.

    Deterministic from the subtopic's position, and orthogonal to every other
    subtopic's axis, so "same subtopic" and "close in vector space" mean the
    same thing before any noise is added.  Requires ``dim >= len(subtopics)``.
    """
    axis = [0.0] * dim
    axis[subtopics.index(subtopic) % dim] = 1.0
    return axis


def _item_embedding(axis: Sequence[float], rng: random.Random, spread: float) -> tuple[float, ...]:
    """The subtopic's axis, nudged, then L2-normalised.

    Normalised because both real embedders return unit vectors and
    ``cosine_similarity`` is then the dot product - the simulation should not
    hand the redundancy term a geometry the production one never sees.
    """
    noisy = [value + rng.gauss(0.0, spread) for value in axis]
    norm = math.sqrt(sum(v * v for v in noisy))
    if norm == 0.0:  # pragma: no cover - probability zero, but a zero vector has no direction
        return tuple(axis)
    return tuple(v / norm for v in noisy)


def build_bank(config: ExperimentConfig) -> list[CandidateItem]:
    """The synthetic question bank, as production ``CandidateItem`` values.

    ``CandidateItem`` rather than a simulation-local type on purpose: the
    adaptive strategy is the *real* Day 12 selector, and it must be fed exactly
    what a real session would feed it.  A parallel item type would be the first
    step towards a parallel selector.

    Within each subtopic, difficulty is spread evenly across
    ``[-2.5, +2.5]`` and then jittered, so every subtopic has items inside the
    ``|b - theta| <= 1.5`` window for any candidate the population can produce.
    Discrimination, time estimate and the embedding wobble are drawn from the
    bank's own stream.  Ids are stable and readable: ``arrays-07``.
    """
    rng = random.Random(derive_seed(config.seed, "bank"))
    subtopics = config.subtopics
    if len(subtopics) > EMBEDDING_DIM:
        raise ValueError(
            f"embedding space has {EMBEDDING_DIM} dimensions but the taxonomy has "
            f"{len(subtopics)} subtopics; each subtopic needs its own axis"
        )

    items: list[CandidateItem] = []
    per = config.items_per_subtopic
    span = BANK_DIFFICULTY_MAX - BANK_DIFFICULTY_MIN
    for subtopic in subtopics:
        topic = config.parent_of[subtopic]
        axis = _subtopic_axis(subtopics, subtopic, EMBEDDING_DIM)
        for index in range(per):
            even = BANK_DIFFICULTY_MIN + span * (index / (per - 1) if per > 1 else 0.5)
            difficulty = _clamp(
                even + rng.uniform(-BANK_DIFFICULTY_JITTER, BANK_DIFFICULTY_JITTER),
                BANK_DIFFICULTY_MIN,
                BANK_DIFFICULTY_MAX,
            )
            items.append(
                CandidateItem(
                    id=f"{subtopic}-{index:02d}",
                    topic_key=topic,
                    subtopic_key=subtopic,
                    difficulty_b=difficulty,
                    time_estimate_s=rng.choice(BANK_TIME_ESTIMATES_S),
                    discrimination_a=rng.uniform(BANK_DISCRIMINATION_MIN, BANK_DISCRIMINATION_MAX),
                    embedding=_item_embedding(axis, rng, EMBEDDING_SPREAD),
                )
            )
    return items


# ---------------------------------------------------------------------------
# the population
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SyntheticCandidate:
    """One simulated person: a hidden ability vector and a resume.

    ``true_theta`` is the **ground truth**.  It is read by exactly one function
    in this package, :func:`app.simulation.response.graded_score`, and never
    reaches a strategy: a strategy is passed a ``SelectionState``, and a
    ``SelectionState`` holds no reference to a candidate.

    ``resume`` is deliberately drawn from a stream that never sees
    ``true_theta``.  A resume that correlated with true ability would give the
    adaptive policy a signal the baselines do not use, and "CAT wins" would stop
    being a statement about selection.  See ``config.RESUME_SUBTOPICS``.
    """

    id: str
    true_theta: Mapping[str, float]
    resume: ResumeProfile

    def theta(self, subtopic_key: str) -> float:
        try:
            return self.true_theta[subtopic_key]
        except KeyError as exc:  # pragma: no cover - a taxonomy/candidate mismatch
            raise KeyError(f"no ground truth for subtopic {subtopic_key!r}") from exc


def build_candidate(config: ExperimentConfig, index: int) -> SyntheticCandidate:
    """Candidate ``index``, generated independently of every other candidate.

    Independence matters for reproducibility: candidate 7 is the same person
    whether the run asked for 10 candidates or 200, because its stream is keyed
    by its own id rather than by its position in a shared sequence.

    Ability is drawn hierarchically - one overall level per candidate, then a
    per-subtopic deviation - so that subtopics correlate the way a real person's
    skills do without collapsing into a single number.  Values are clamped into
    the range the bank can actually probe.
    """
    candidate_id = f"c{index:03d}"
    rng = random.Random(derive_seed(config.seed, "population", candidate_id))
    overall = rng.gauss(0.0, CANDIDATE_OVERALL_SD)
    true_theta = {
        subtopic: _clamp(
            overall + rng.gauss(0.0, CANDIDATE_SUBTOPIC_SD),
            CANDIDATE_THETA_MIN,
            CANDIDATE_THETA_MAX,
        )
        for subtopic in config.subtopics
    }

    # A separate stream, so the resume cannot encode the abilities above even
    # through the shared position of a shared generator.
    resume_rng = random.Random(derive_seed(config.seed, "resume", candidate_id))
    mentioned = resume_rng.sample(
        list(config.subtopics), k=min(RESUME_SUBTOPICS, len(config.subtopics))
    )
    resume = ResumeProfile(
        topic_affinity={
            subtopic: resume_rng.uniform(RESUME_AFFINITY_MIN, RESUME_AFFINITY_MAX)
            for subtopic in sorted(mentioned)
        }
    )
    return SyntheticCandidate(id=candidate_id, true_theta=true_theta, resume=resume)


def build_population(config: ExperimentConfig) -> list[SyntheticCandidate]:
    """``config.candidate_count`` candidates, in id order."""
    return [build_candidate(config, index) for index in range(config.candidate_count)]


@dataclass(frozen=True, slots=True)
class SyntheticEnvironment:
    """The bank and the population together - what every strategy shares.

    Built once per experiment and passed to all three policies, which is the
    mechanical form of "same bank, same candidates, same difficulties, same
    discriminations, same topic distribution".
    """

    config: ExperimentConfig
    bank: tuple[CandidateItem, ...]
    candidates: tuple[SyntheticCandidate, ...]

    @property
    def items_by_id(self) -> dict[str, CandidateItem]:
        return {item.id: item for item in self.bank}


def build_environment(config: ExperimentConfig) -> SyntheticEnvironment:
    """Everything the experiment needs, from one seed."""
    return SyntheticEnvironment(
        config=config,
        bank=tuple(build_bank(config)),
        candidates=tuple(build_population(config)),
    )


__all__ = [
    "SyntheticCandidate",
    "SyntheticEnvironment",
    "build_bank",
    "build_candidate",
    "build_environment",
    "build_population",
    "derive_seed",
]
