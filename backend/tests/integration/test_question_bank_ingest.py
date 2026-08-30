"""The real dataset, loaded into a real Postgres by the real migration.

The unit tests prove the files are internally consistent.  Only this can prove
they are *ingestible*: that every taxonomy key satisfies the foreign keys, that
``expected_concepts`` survives the round trip through JSONB, and that the
generated ``tsv`` column is populated - none of which a validator running on
JSON can tell you.

Skips when the compose stack is down (see ``tests/integration/conftest.py``);
``REQUIRE_INTEGRATION=1`` in CI turns that skip into a failure.
"""

from __future__ import annotations

import pytest
import pytest_asyncio
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker

from app.bank.ingest import ingest_bank
from app.bank.loader import validate_bank
from app.bank.paths import BANK_DIR
from app.bank.taxonomy import load_taxonomy
from app.models.question import Question
from app.models.taxonomy import Topic


@pytest.fixture(scope="module")
def taxonomy():
    return load_taxonomy()


@pytest.fixture(scope="module")
def report(taxonomy):
    report = validate_bank(BANK_DIR, taxonomy)
    assert report.ok, "\n".join(report.errors[:10])
    return report


@pytest_asyncio.fixture
async def ingested(db_engine: AsyncEngine, report, taxonomy):
    factory = async_sessionmaker(db_engine, expire_on_commit=False)
    async with factory() as session:
        result = await ingest_bank(session, report, taxonomy)
        await session.commit()
    return result


async def test_the_whole_bank_ingests(ingested, report, taxonomy, db_engine: AsyncEngine):
    assert ingested.questions_written == report.count
    assert ingested.topics_written == len(taxonomy.rows())
    async with db_engine.connect() as conn:
        assert await conn.scalar(select(func.count()).select_from(Question)) == report.count
        assert await conn.scalar(select(func.count()).select_from(Topic)) == len(taxonomy.rows())


async def test_ingesting_twice_changes_nothing(ingested, report, taxonomy, db_engine: AsyncEngine):
    """Idempotence: a dataset that can only be loaded into an empty database drifts."""
    factory = async_sessionmaker(db_engine, expire_on_commit=False)
    async with factory() as session:
        await ingest_bank(session, report, taxonomy)
        await session.commit()
    async with db_engine.connect() as conn:
        assert await conn.scalar(select(func.count()).select_from(Question)) == report.count


async def test_expected_concepts_survive_the_round_trip(ingested, report, db_engine: AsyncEngine):
    """JSONB is where a silently-dropped field would show up."""
    expected = {loaded.item.id: loaded.item for loaded in report.items}
    sample_id = sorted(expected)[0]
    async with db_engine.connect() as conn:
        row = (
            await conn.execute(
                select(Question.expected_concepts, Question.reference_answer).where(
                    Question.id == sample_id
                )
            )
        ).one()
    concepts, reference = row
    source = expected[sample_id]
    assert [c["key"] for c in concepts] == source.concept_keys
    assert all(1 <= c["weight"] <= 3 and c["hint"] for c in concepts)
    assert reference == source.reference_answer


async def test_the_generated_search_vector_is_populated(ingested, db_engine: AsyncEngine):
    """`tsv` is a GENERATED column: Postgres fills it, and ingest must not try to."""
    async with db_engine.connect() as conn:
        empty = await conn.scalar(
            text("SELECT count(*) FROM questions WHERE tsv IS NULL OR tsv = ''::tsvector")
        )
    assert empty == 0


async def test_only_reviewed_skips_the_drafts(report, taxonomy, db_engine: AsyncEngine):
    """The production posture: unreviewed items never reach a database serving candidates."""
    reviewed = sum(1 for loaded in report.items if loaded.item.review_status == "reviewed")
    factory = async_sessionmaker(db_engine, expire_on_commit=False)
    async with factory() as session:
        result = await ingest_bank(session, report, taxonomy, only_reviewed=True)
        await session.commit()
    assert result.questions_written == reviewed
    async with db_engine.connect() as conn:
        assert await conn.scalar(select(func.count()).select_from(Question)) == reviewed
