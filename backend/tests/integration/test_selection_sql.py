"""The Day 12 hard constraints, executed by a real PostgreSQL over the real bank.

The unit tests prove the *rules* are right and that the statement compiles with
every value bound. Only this can prove the two expressions of those rules agree:
that ``ABS(difficulty_b - CASE subtopic_key ... END) <= 1.5`` in SQL selects
exactly the rows ``ineligibility_reason`` would keep in Python, over real
``REAL`` columns with real rounding.

That agreement is the point. ``constraints.py`` deliberately carries the rule
twice - once as the specification and once as the fast path - and two
implementations of one rule drift unless something checks them against each
other. This is that something.

Skips when the compose stack is down (see ``tests/integration/conftest.py``);
``REQUIRE_INTEGRATION=1`` in CI turns that skip into a failure.
"""

from __future__ import annotations

import random

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from app.ability import AbilityState
from app.bank.ingest import ingest_bank
from app.bank.loader import validate_bank
from app.bank.paths import BANK_DIR
from app.bank.taxonomy import load_taxonomy
from app.models.question import Question
from app.retrieval.embedders import HashingEmbedder
from app.selection import (
    CandidateItem,
    SelectionState,
    eligible_items,
    filter_eligible,
    ineligibility_reason,
    select_next_item,
)


@pytest.fixture(scope="module")
def taxonomy():
    return load_taxonomy()


@pytest.fixture(scope="module")
def report(taxonomy):
    report = validate_bank(BANK_DIR, taxonomy)
    assert report.ok, "\n".join(report.errors[:10])
    return report


@pytest_asyncio.fixture
async def bank(db_engine: AsyncEngine, report, taxonomy):
    factory = async_sessionmaker(db_engine, expire_on_commit=False)
    async with factory() as session:
        result = await ingest_bank(session, report, taxonomy, embedder=HashingEmbedder())
        await session.commit()
    return result


@pytest_asyncio.fixture
async def session(db_engine: AsyncEngine, bank):
    factory = async_sessionmaker(db_engine, expire_on_commit=False)
    async with factory() as opened:
        yield opened


async def all_items(session: AsyncSession) -> list[CandidateItem]:
    """Every question in the bank, as the selection layer sees them."""
    rows = (
        await session.execute(
            select(
                Question.id,
                Question.topic_key,
                Question.subtopic_key,
                Question.difficulty_b,
                Question.time_estimate_s,
                Question.discrimination_a,
                Question.embedding,
            ).order_by(Question.id)
        )
    ).all()
    return [
        CandidateItem(
            id=row.id,
            topic_key=row.topic_key,
            subtopic_key=row.subtopic_key,
            difficulty_b=float(row.difficulty_b),
            time_estimate_s=int(row.time_estimate_s),
            discrimination_a=float(row.discrimination_a),
            embedding=None if row.embedding is None else tuple(float(v) for v in row.embedding),
        )
        for row in rows
    ]


def state_over(items: list[CandidateItem], **overrides) -> SelectionState:
    """A state whose blueprint covers every topic the bank actually has."""
    topics = sorted({i.topic_key for i in items})
    subtopics = sorted({i.subtopic_key for i in items})
    kwargs = {
        "ability": {key: AbilityState(theta=0.0, rd=1.0) for key in subtopics},
        "coverage_targets": dict.fromkeys(topics, 3),
        "jd_weights": dict.fromkeys(topics, 0.7),
        "time_left_s": 1800.0,
    }
    kwargs.update(overrides)
    return SelectionState(**kwargs)


class TestSqlAgreesWithPython:
    async def test_the_two_filters_select_the_same_rows(self, session):
        items = await all_items(session)
        state = state_over(items)
        from_sql = await eligible_items(session, state)
        from_python = filter_eligible(items, state)
        assert [i.id for i in from_sql] == [i.id for i in from_python]

    async def test_they_still_agree_mid_interview(self, session):
        """Asked ids, a partly-served quota, a moved theta and a shrunken clock,
        all at once - the state a real session is actually in."""
        items = await all_items(session)
        asked = tuple(items[:4])
        topics = sorted({i.topic_key for i in items})
        subtopics = sorted({i.subtopic_key for i in items})
        state = state_over(
            items,
            ability={
                key: AbilityState(theta=-1.0 + 0.4 * index, rd=0.7)
                for index, key in enumerate(subtopics)
            },
            coverage_targets={topic: 2 for topic in topics},
            asked=asked,
            time_left_s=240.0,
        )
        from_sql = await eligible_items(session, state)
        from_python = filter_eligible(items, state)
        assert [i.id for i in from_sql] == [i.id for i in from_python]
        assert from_sql  # the fixture must not be trivially empty

    async def test_the_rows_come_back_with_their_stored_vectors(self, session):
        items = await all_items(session)
        from_sql = await eligible_items(session, state_over(items))
        assert from_sql
        assert all(i.embedding is not None and len(i.embedding) == 384 for i in from_sql)


class TestEachConstraintOverRealRows:
    async def test_an_asked_question_never_comes_back(self, session):
        items = await all_items(session)
        state = state_over(items)
        first = (await eligible_items(session, state))[0]

        after = state_over(items, asked=(first,))
        returned = await eligible_items(session, after)
        assert first.id not in {i.id for i in returned}

    async def test_only_topics_with_quota_left_come_back(self, session):
        items = await all_items(session)
        topics = sorted({i.topic_key for i in items})
        state = state_over(items, coverage_targets={topics[0]: 2})
        returned = await eligible_items(session, state)
        assert returned
        assert {i.topic_key for i in returned} == {topics[0]}

    async def test_a_topic_at_its_quota_disappears(self, session):
        items = await all_items(session)
        topics = sorted({i.topic_key for i in items})
        in_topic = [i for i in items if i.topic_key == topics[0]][:2]
        state = state_over(items, coverage_targets={topics[0]: 2}, asked=tuple(in_topic))
        assert await eligible_items(session, state) == []

    async def test_every_returned_row_is_inside_the_difficulty_window(self, session):
        items = await all_items(session)
        state = state_over(
            items,
            ability={
                key: AbilityState(theta=1.0, rd=0.6) for key in {i.subtopic_key for i in items}
            },
        )
        for row in await eligible_items(session, state):
            assert abs(row.difficulty_b - 1.0) <= 1.5

    async def test_nothing_longer_than_the_remaining_time_comes_back(self, session):
        items = await all_items(session)
        shortest = min(i.time_estimate_s for i in items)
        state = state_over(items, time_left_s=float(shortest))
        returned = await eligible_items(session, state)
        assert returned
        assert all(i.time_estimate_s <= shortest for i in returned)

    async def test_no_time_left_returns_nothing(self, session):
        items = await all_items(session)
        assert await eligible_items(session, state_over(items, time_left_s=0.0)) == []

    async def test_the_limit_is_respected(self, session):
        items = await all_items(session)
        assert len(await eligible_items(session, state_over(items), limit=5)) == 5


class TestSelectNextItemEndToEnd:
    async def test_it_returns_an_eligible_item(self, session):
        items = await all_items(session)
        state = state_over(items)
        selection = await select_next_item(session, state, rng=random.Random(4))
        assert selection is not None
        assert ineligibility_reason(selection.item, state) is None

    async def test_it_is_reproducible_under_a_seed(self, session):
        items = await all_items(session)
        state = state_over(items)
        first = await select_next_item(session, state, rng=random.Random(4))
        second = await select_next_item(session, state, rng=random.Random(4))
        assert first is not None and second is not None
        assert first.item_id == second.item_id

    async def test_it_returns_none_when_the_blueprint_is_served(self, session):
        items = await all_items(session)
        state = state_over(items, coverage_targets={})
        assert await select_next_item(session, state, rng=random.Random(4)) is None
