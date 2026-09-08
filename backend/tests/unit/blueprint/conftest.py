"""A realistic spec for the Day 15 tests, and the pieces to vary it.

One shared fixture rather than a builder in every file: a blueprint has enough
fields that constructing one inline three times would bury the thing each test
is actually about.
"""

from __future__ import annotations

from app.blueprint import AssessmentSpec, TopicRequest

#: The spec most tests start from: a real role, three topics with unequal JD
#: weights, one optional, and subtopics stated explicitly so the tests do not
#: need the committed taxonomy file.
BACKEND_SPEC = AssessmentSpec(
    role="Backend Engineer",
    level="mid",
    mode="standard",
    duration_minutes=60,
    topics=(
        TopicRequest(
            topic_key="dsa",
            jd_weight=0.9,
            subtopic_keys=("arrays", "graphs"),
        ),
        TopicRequest(
            topic_key="databases",
            jd_weight=0.7,
            subtopic_keys=("indexing", "transactions"),
        ),
        TopicRequest(
            topic_key="caching",
            jd_weight=0.4,
            subtopic_keys=("cache_invalidation",),
            required=False,
        ),
    ),
)
