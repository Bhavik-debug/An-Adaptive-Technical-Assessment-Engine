"""Retrieval against a real Postgres, over the real 60-question bank.

The unit tests prove fusion is correct arithmetic and the recipe is
deterministic.  Only this can prove the parts fit: that ``vector(384)`` accepts
what the embedder produces, that ``<=>`` orders the way the code assumes, that
the generated ``tsvector`` covers the concepts, and that re-ingesting does not
re-embed.

**Which embedder these use, and why.**  ``HashingEmbedder`` - deterministic,
no download, no network.  It matches words rather than meaning, which is enough
to test every *mechanism* here: ranking, SQL, fusion, idempotency.  It is not
enough to test that a paraphrase is found, and no test in this file claims
otherwise; that claim lives in ``tests/unit/retrieval/test_real_model.py``,
against the real model, behind ``-m embeddings``.

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
from app.retrieval.embedders import HashingEmbedder
from app.retrieval.indexing import document_text_for_item, embed_questions
from app.retrieval.search import hybrid_search, lexical_search, vector_search


@pytest.fixture(scope="module")
def taxonomy():
    return load_taxonomy()


@pytest.fixture(scope="module")
def report(taxonomy):
    report = validate_bank(BANK_DIR, taxonomy)
    assert report.ok, "\n".join(report.errors[:10])
    return report


@pytest.fixture
def embedder():
    return HashingEmbedder()


@pytest_asyncio.fixture
async def bank(db_engine: AsyncEngine, report, taxonomy, embedder):
    """The whole bank, ingested and embedded, in one transaction."""
    factory = async_sessionmaker(db_engine, expire_on_commit=False)
    async with factory() as session:
        result = await ingest_bank(session, report, taxonomy, embedder=embedder)
        await session.commit()
    return result


@pytest_asyncio.fixture
async def session(db_engine: AsyncEngine):
    factory = async_sessionmaker(db_engine, expire_on_commit=False)
    async with factory() as opened:
        yield opened


class TestIngestWritesVectors:
    async def test_every_question_gets_an_embedding(self, bank, report, session):
        assert bank.embeddings is not None
        assert bank.embeddings.embedded == report.count
        assert bank.embeddings.missing_rows == []
        stored = await session.scalar(
            select(func.count()).select_from(Question).where(Question.embedding.isnot(None))
        )
        assert stored == report.count

    async def test_the_model_and_fingerprint_are_recorded(self, bank, session, embedder):
        rows = (
            await session.execute(
                select(Question.embedding_model, Question.embedding_text_sha256).limit(5)
            )
        ).all()
        for model_id, fingerprint in rows:
            assert model_id == embedder.model_id
            assert fingerprint and len(fingerprint) == 64

    async def test_the_stored_vector_has_the_right_width(self, bank, session, embedder):
        vector = await session.scalar(select(Question.embedding).limit(1))
        assert len(vector) == embedder.dimension

    async def test_the_search_document_is_written_and_contains_the_concepts(
        self, bank, report, session
    ):
        item = next(loaded.item for loaded in report.items if loaded.item.id == "sys-cache-003")
        stored = await session.scalar(
            select(Question.search_document).where(Question.id == item.id)
        )
        assert stored == document_text_for_item(item)
        assert "cache stampede" in stored, "humanised concept keys are searchable"

    async def test_ingesting_without_an_embedder_still_writes_the_search_document(
        self, db_engine, report, taxonomy, session
    ):
        """Lexical search must not depend on whether embeddings were generated."""
        factory = async_sessionmaker(db_engine, expire_on_commit=False)
        async with factory() as writing:
            result = await ingest_bank(writing, report, taxonomy)
            await writing.commit()
        assert result.embeddings is None
        missing = await session.scalar(
            select(func.count()).select_from(Question).where(Question.search_document.is_(None))
        )
        assert missing == 0


class TestIdempotency:
    async def test_re_ingesting_does_not_re_embed(self, bank, report, taxonomy, db_engine):
        """The expensive half must be skipped when nothing changed."""
        factory = async_sessionmaker(db_engine, expire_on_commit=False)
        async with factory() as session:
            again = await ingest_bank(session, report, taxonomy, embedder=HashingEmbedder())
            await session.commit()
        assert again.embeddings is not None
        assert again.embeddings.embedded == 0
        assert again.embeddings.reused == report.count

    async def test_re_ingesting_does_not_duplicate_rows(self, bank, report, taxonomy, db_engine):
        factory = async_sessionmaker(db_engine, expire_on_commit=False)
        async with factory() as session:
            await ingest_bank(session, report, taxonomy, embedder=HashingEmbedder())
            await session.commit()
            total = await session.scalar(select(func.count()).select_from(Question))
        assert total == report.count

    async def test_force_re_embeds_everything(self, bank, report, taxonomy, db_engine):
        factory = async_sessionmaker(db_engine, expire_on_commit=False)
        async with factory() as session:
            forced = await ingest_bank(
                session, report, taxonomy, embedder=HashingEmbedder(), reembed=True
            )
            await session.commit()
        assert forced.embeddings is not None
        assert forced.embeddings.embedded == report.count
        assert forced.embeddings.reused == 0

    async def test_a_changed_model_invalidates_the_stored_vectors(
        self, bank, report, taxonomy, db_engine
    ):
        """Two models produce two incomparable spaces; mixing them is silently wrong."""
        other = HashingEmbedder(model_id="hashing-v2")
        factory = async_sessionmaker(db_engine, expire_on_commit=False)
        async with factory() as session:
            switched = await embed_questions(
                session, other, [loaded.item for loaded in report.items]
            )
            await session.commit()
        assert switched.embedded == report.count
        assert switched.reused == 0

    async def test_an_edited_question_is_re_embedded_and_the_others_are_not(
        self, bank, report, session, embedder
    ):
        """The fingerprint's whole purpose: find the one row that changed."""
        edited = report.items[0].item.model_copy(
            update={"text": "A completely rewritten question."}
        )
        others = [loaded.item for loaded in report.items[1:]]
        await session.execute(
            text("UPDATE questions SET text = :t, search_document = :d WHERE id = :i"),
            {
                "t": edited.text,
                "d": document_text_for_item(edited),
                "i": edited.id,
            },
        )
        result = await embed_questions(session, embedder, [edited, *others])
        assert result.embedded == 1
        assert result.reused == len(others)


class TestVectorSearch:
    async def test_it_returns_ranked_results(self, bank, session, embedder):
        hits = await vector_search(session, embedder, "database index performance", limit=5)
        assert len(hits) == 5
        assert [hit.rank for hit in hits] == [1, 2, 3, 4, 5]

    async def test_results_are_ordered_by_descending_similarity(self, bank, session, embedder):
        hits = await vector_search(session, embedder, "cache invalidation", limit=10)
        scores = [hit.score for hit in hits]
        assert scores == sorted(scores, reverse=True)

    async def test_similarity_is_reported_not_distance(self, bank, session, embedder):
        """`<=>` gives distance; every score in this module means "higher is better"."""
        hits = await vector_search(session, embedder, "transactions", limit=3)
        assert all(-1.0 <= hit.score <= 1.0 for hit in hits)

    async def test_a_question_finds_itself_first(self, bank, report, session, embedder):
        """The strongest available signal that the column, operator and recipe agree."""
        item = next(loaded.item for loaded in report.items if loaded.item.id == "db-index-002")
        hits = await vector_search(session, embedder, document_text_for_item(item), limit=1)
        assert hits[0].question.id == item.id
        assert hits[0].score == pytest.approx(1.0, abs=1e-4)

    async def test_it_is_deterministic(self, bank, session, embedder):
        first = await vector_search(session, embedder, "replication lag", limit=10)
        second = await vector_search(session, embedder, "replication lag", limit=10)
        assert [h.question.id for h in first] == [h.question.id for h in second]

    async def test_the_limit_is_respected(self, bank, session, embedder):
        assert len(await vector_search(session, embedder, "anything", limit=3)) == 3

    async def test_unembedded_rows_are_excluded(self, bank, session, embedder):
        """An unembedded question is unknown, not distant - returning it would hide a bug."""
        await session.execute(text("UPDATE questions SET embedding = NULL WHERE id = 'db-tx-001'"))
        hits = await vector_search(session, embedder, "ACID atomicity durability", limit=60)
        assert "db-tx-001" not in {hit.question.id for hit in hits}
        assert len(hits) == 59

    async def test_it_carries_the_metadata_needed_to_judge_a_result(self, bank, session, embedder):
        hit = (await vector_search(session, embedder, "sharding", limit=1))[0]
        assert hit.question.topic_key and hit.question.subtopic_key
        assert hit.question.text
        assert -3.0 <= hit.question.difficulty_b <= 3.0


class TestLexicalSearch:
    async def test_an_exact_phrase_finds_its_question(self, bank, session):
        hits = await lexical_search(session, "circuit breaker", limit=5)
        assert hits, "a phrase taken from a question must match it"
        assert "sys-ft-002" in {hit.question.id for hit in hits}

    async def test_results_are_ordered_by_descending_rank(self, bank, session):
        hits = await lexical_search(session, "index", limit=20)
        scores = [hit.score for hit in hits]
        assert scores == sorted(scores, reverse=True)

    async def test_stemming_means_a_singular_query_matches_a_plural(self, bank, session):
        singular = {hit.question.id for hit in await lexical_search(session, "index", limit=30)}
        plural = {hit.question.id for hit in await lexical_search(session, "indexes", limit=30)}
        assert singular & plural

    async def test_it_searches_concepts_not_only_the_question_prose(self, bank, session):
        """'thundering herd' appears only in sys-cache-003's expected_concepts."""
        hits = await lexical_search(session, "thundering herd", limit=5)
        assert "sys-cache-003" in {hit.question.id for hit in hits}

    async def test_a_query_matching_nothing_returns_nothing(self, bank, session):
        assert await lexical_search(session, "xylophone marsupial", limit=10) == []

    async def test_a_query_of_only_stop_words_returns_nothing(self, bank, session):
        assert await lexical_search(session, "the and of", limit=10) == []

    async def test_punctuation_and_quotes_do_not_raise(self, bank, session):
        """`websearch_to_tsquery` accepts anything a person types; `to_tsquery` would error."""
        for query in ['"database index"', "index OR cache", "a & b | c", "!!!", "index -cache"]:
            await lexical_search(session, query, limit=5)

    async def test_it_is_deterministic(self, bank, session):
        first = await lexical_search(session, "index", limit=20)
        second = await lexical_search(session, "index", limit=20)
        assert [h.question.id for h in first] == [h.question.id for h in second]


class TestHybridSearch:
    async def test_it_fuses_both_sources(self, bank, session, embedder):
        outcome = await hybrid_search(session, embedder, "database indexes", final_k=10)
        assert outcome.vector_candidates > 0
        assert outcome.lexical_candidates > 0
        assert any(hit.sources == ("vector", "lexical") for hit in outcome.results)

    async def test_results_are_ordered_by_descending_rrf_score(self, bank, session, embedder):
        outcome = await hybrid_search(session, embedder, "cache invalidation", final_k=10)
        scores = [hit.rrf_score for hit in outcome.results]
        assert scores == sorted(scores, reverse=True)

    async def test_a_result_found_by_both_outranks_one_found_by_a_single_source(
        self, bank, session, embedder
    ):
        """The property that justifies fusing at all."""
        outcome = await hybrid_search(session, embedder, "deadlock lock ordering", final_k=10)
        both = [h for h in outcome.results if len(h.sources) == 2]
        one = [h for h in outcome.results if len(h.sources) == 1]
        if both and one:
            assert min(h.rrf_score for h in both) > max(h.rrf_score for h in one)

    async def test_every_result_carries_the_evidence_for_its_position(
        self, bank, session, embedder
    ):
        outcome = await hybrid_search(session, embedder, "index", final_k=10)
        for hit in outcome.results:
            assert hit.sources, "a result came from somewhere"
            if "vector" in hit.sources:
                assert hit.vector_rank is not None and hit.vector_similarity is not None
            else:
                assert hit.vector_rank is None and hit.vector_similarity is None
            if "lexical" in hit.sources:
                assert hit.lexical_rank is not None and hit.lexical_score is not None
            else:
                assert hit.lexical_rank is None and hit.lexical_score is None

    async def test_candidate_k_is_larger_than_final_k(self, bank, session, embedder):
        """A document ranked 25th by one source and 2nd by the other must reach fusion."""
        outcome = await hybrid_search(
            session, embedder, "index", vector_k=30, lexical_k=30, final_k=5
        )
        assert len(outcome.results) == 5
        assert outcome.fused_candidates > 5

    async def test_truncated_candidates_are_counted_not_hidden(self, bank, session, embedder):
        outcome = await hybrid_search(session, embedder, "index", final_k=5)
        assert outcome.truncated == outcome.fused_candidates - len(outcome.results)
        assert outcome.truncated > 0

    async def test_it_is_deterministic(self, bank, session, embedder):
        first = await hybrid_search(session, embedder, "replication", final_k=10)
        second = await hybrid_search(session, embedder, "replication", final_k=10)
        assert [h.id for h in first.results] == [h.id for h in second.results]
        assert [h.rrf_score for h in first.results] == [h.rrf_score for h in second.results]

    async def test_it_reports_timings_for_each_stage(self, bank, session, embedder):
        outcome = await hybrid_search(session, embedder, "index", final_k=5)
        assert set(outcome.timings_ms) == {
            "embed",
            "vector_sql",
            "lexical_sql",
            "fusion",
            "total",
        }
        assert all(value >= 0 for value in outcome.timings_ms.values())

    async def test_no_duplicate_questions_in_the_final_list(self, bank, session, embedder):
        outcome = await hybrid_search(session, embedder, "database index cache", final_k=10)
        ids = [hit.id for hit in outcome.results]
        assert len(ids) == len(set(ids))


class TestWeakAndEmptyQueries:
    async def test_a_nonsense_query_still_returns_vector_results(self, bank, session, embedder):
        """Vector search always returns its nearest K; "nearest" is not "relevant"."""
        outcome = await hybrid_search(session, embedder, "xylophone marsupial", final_k=5)
        assert outcome.lexical_candidates == 0
        assert len(outcome.results) == 5
        assert all(hit.sources == ("vector",) for hit in outcome.results)

    async def test_an_empty_query_does_not_raise(self, bank, session, embedder):
        outcome = await hybrid_search(session, embedder, "", final_k=5)
        assert outcome.lexical_candidates == 0
        assert isinstance(outcome.results, list)

    async def test_a_whitespace_query_does_not_raise(self, bank, session, embedder):
        outcome = await hybrid_search(session, embedder, "   \n  ", final_k=5)
        assert isinstance(outcome.results, list)

    async def test_an_empty_bank_returns_nothing_rather_than_failing(
        self, db_engine, session, embedder
    ):
        await session.execute(text("DELETE FROM questions"))
        outcome = await hybrid_search(session, embedder, "database index", final_k=5)
        assert outcome.results == []
        assert outcome.fused_candidates == 0


class TestTheDatabaseObjects:
    async def test_the_hnsw_index_exists_with_the_cosine_operator_class(self, session):
        definition = await session.scalar(
            text(
                "SELECT indexdef FROM pg_indexes " "WHERE indexname = 'ix_questions_embedding_hnsw'"
            )
        )
        assert definition is not None
        assert "hnsw" in definition.lower()
        assert "vector_cosine_ops" in definition

    async def test_the_gin_index_on_the_tsvector_exists(self, session):
        definition = await session.scalar(
            text("SELECT indexdef FROM pg_indexes WHERE indexname = 'ix_questions_tsv'")
        )
        assert definition is not None
        assert "gin" in definition.lower()

    async def test_the_tsvector_is_generated_and_covers_the_search_document(self, bank, session):
        """A generated column cannot drift from what it indexes."""
        row = await session.scalar(
            text("SELECT tsv::text FROM questions WHERE id = 'sys-cache-003'")
        )
        assert "stamped" in row, "the concept key reached the index, stemmed"

    async def test_the_hnsw_index_is_usable_by_the_planner(self, bank, session, embedder):
        """At 60 rows a sequential scan is genuinely cheaper, so the planner picks it.

        That is correct, and it means "the index is never used" is not evidence
        that it works. Forcing the choice proves the index, the operator class
        and the query are compatible - which is the part that could be wrong.
        """
        vector = str(embedder.embed_query("database index"))
        await session.execute(text("SET LOCAL enable_seqscan = off"))
        plan = "\n".join(
            row[0]
            for row in (
                await session.execute(
                    text(
                        "EXPLAIN SELECT id FROM questions WHERE embedding IS NOT NULL "
                        "ORDER BY embedding <=> CAST(:v AS vector) LIMIT 10"
                    ),
                    {"v": vector},
                )
            ).all()
        )
        assert "ix_questions_embedding_hnsw" in plan
