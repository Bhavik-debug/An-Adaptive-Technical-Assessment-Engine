"""Shared material for the grading tests.

Everything here is synthetic and written for the purpose. No real candidate
text, no real resume, no real answer - the same rule that governs
``fixtures/llm/`` governs anything that could end up quoted in a failure
message or copied into a recording.
"""

from __future__ import annotations

from typing import Any

import pytest

from app.bank.schema import ExpectedConcept

#: One question's checklist, deliberately NOT in sorted key order: several tests
#: are about the fact that the pipeline canonicalises the order, and a fixture
#: that was already sorted would make them pass for the wrong reason.
CONCEPTS: tuple[ExpectedConcept, ...] = (
    ExpectedConcept(
        key="stampede",
        weight=2,
        hint="names the thundering-herd problem on a cold cache and a mitigation",
    ),
    ExpectedConcept(
        key="cache_invalidation",
        weight=3,
        hint="says how a stale entry is removed or refreshed, not merely that it can be stale",
    ),
    ExpectedConcept(
        key="eviction",
        weight=1,
        hint="names an eviction policy and the bound it operates against",
    ),
)

MISCONCEPTIONS: tuple[str, ...] = (
    "Believes Redis offers ACID transactions the way a relational database does",
    "Believes a TTL alone guarantees correctness after a write",
)

QUESTION = (
    "A read-heavy catalogue service is hitting its database hard. Walk me through how "
    "you would put a cache in front of it: what you would cache, how entries stop "
    "being stale, and what happens when the cache is cold."
)

ANSWER = (
    "I would go write-through so the cache never serves a stale row, keep LRU with a "
    "4 GB memory cap, and put a per-key lock in front of a miss so a cold start does "
    "not send every request to the database at once."
)


def judgment_payload(**overrides: Any) -> dict[str, Any]:
    """A valid ``GradeJudgment`` as JSON, with per-test overrides.

    A dict rather than a built model, because most of these tests are about
    what ``model_validate`` accepts and rejects - which is exactly the step a
    pre-built model would have skipped.
    """
    payload: dict[str, Any] = {
        "concepts": [
            {
                "key": "cache_invalidation",
                "label": "covered",
                "evidence": "I would go write-through so the cache never serves a stale row",
            },
            {
                "key": "eviction",
                "label": "covered",
                "evidence": "keep LRU with a 4 GB memory cap",
            },
            {
                "key": "stampede",
                "label": "partial",
                "evidence": "put a per-key lock in front of a miss",
            },
        ],
        "errors": [],
        "misconceptions_hit": [],
        "rubric": {
            "structure": 3,
            "tradeoff_awareness": 2,
            "specificity": 3,
            "communication": 3,
        },
        "grader_confidence": 0.82,
    }
    payload.update(overrides)
    return payload


@pytest.fixture
def concepts() -> tuple[ExpectedConcept, ...]:
    return CONCEPTS


@pytest.fixture
def misconceptions() -> tuple[str, ...]:
    return MISCONCEPTIONS
