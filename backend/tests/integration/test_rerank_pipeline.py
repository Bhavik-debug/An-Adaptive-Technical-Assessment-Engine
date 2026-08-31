"""The full two-stage pipeline against a real Postgres and the real 60 questions.

    query -> hybrid retrieval (vector + lexical + RRF) -> candidates
          -> reranker -> reordered top K

The unit tests prove the ordering logic in isolation. Only this can prove the
stages actually compose: that stage 1's ``RetrievedQuestion`` is what stage 2
consumes, that ``search_document`` survives the SQL into the reranker's input,
and that a failing reranker still returns a usable ranking.

**Which reranker these use.** ``LexicalOverlapReranker`` - deterministic, no
download, no network. It scores word overlap, which is enough to test every
mechanism here. It cannot judge relevance, and no test in this file claims it
can; that claim lives in ``tests/unit/retrieval/test_real_reranker.py`` behind
``-m embeddings``.

Skips when the compose stack is down (see ``tests/integration/conftest.py``);
``REQUIRE_INTEGRATION=1`` in CI turns that skip into a failure.
"""

from __future__ import annotations

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker

from app.bank.ingest import ingest_bank
from app.bank.loader import validate_bank
from app.bank.paths import BANK_DIR
from app.bank.taxonomy import load_taxonomy
from app.retrieval.embedders import HashingEmbedder
from app.retrieval.pipeline import search_and_rerank
from app.retrieval.rerank import rerank_document
from app.retrieval.rerankers import LexicalOverlapReranker


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


@pytest.fixture
def reranker():
    return LexicalOverlapReranker()


@pytest_asyncio.fixture
async def bank(db_engine: AsyncEngine, report, taxonomy, embedder):
    factory = async_sessionmaker(db_engine, expire_on_commit=False)
    async with factory() as session:
        await ingest_bank(session, report, taxonomy, embedder=embedder)
        await session.commit()


@pytest_asyncio.fixture
async def session(db_engine: AsyncEngine):
    factory = async_sessionmaker(db_engine, expire_on_commit=False)
    async with factory() as opened:
        yield opened


class ExplodingReranker:
    model_id = "exploding"

    def score_pairs(self, query, documents):
        raise RuntimeError("onnxruntime session failed to initialise")


class ReversingReranker:
    """Scores in reverse candidate order, so any effect is unmistakable."""

    model_id = "reversing"

    def score_pairs(self, query, documents):
        return [float(i) for i in range(len(documents))]


class TestTheStagesCompose:
    async def test_it_returns_reranked_results(self, bank, session, embedder, reranker):
        result = await search_and_rerank(
            session, embedder, "database index performance", reranker=reranker, final_k=5
        )
        assert result.reranked is True
        assert result.fallback_reason is None
        assert len(result.results) == 5
        assert result.reranker_model_id == "lexical-overlap-v1"

    async def test_stage_one_generates_more_candidates_than_are_returned(
        self, bank, session, embedder, reranker
    ):
        """The reranker can only promote what stage 1 handed it."""
        result = await search_and_rerank(
            session, embedder, "index", reranker=reranker, candidate_k=40, final_k=5
        )
        assert result.candidates_generated > 5
        assert result.candidates_scored == result.candidates_generated

    async def test_the_hybrid_result_is_kept_for_comparison(
        self, bank, session, embedder, reranker
    ):
        """Day 10's ablation needs both orders from one run."""
        result = await search_and_rerank(
            session, embedder, "cache invalidation", reranker=reranker, final_k=5
        )
        assert result.hybrid.results
        assert result.hybrid.vector_candidates > 0

    async def test_the_reranker_actually_changes_the_order(self, bank, session, embedder):
        result = await search_and_rerank(
            session, embedder, "index", reranker=ReversingReranker(), candidate_k=10, final_k=10
        )
        hybrid_order = [c.id for c in result.hybrid.results[:10]]
        assert [r.id for r in result.results] == list(reversed(hybrid_order))

    async def test_ranks_before_and_after_are_both_reported(self, bank, session, embedder):
        result = await search_and_rerank(
            session, embedder, "index", reranker=ReversingReranker(), candidate_k=5, final_k=5
        )
        top = result.results[0]
        assert top.rerank_rank == 1
        assert top.retrieval_rank == 5
        assert top.rank_delta == 4
        assert result.promoted

    async def test_the_model_sees_the_same_document_stage_one_indexed(
        self, bank, session, embedder
    ):
        """Otherwise stage 2 could demote what stage 2 cannot see stage 1 found."""
        seen: list[str] = []

        class Capturing:
            model_id = "capturing"

            def score_pairs(self, query, documents):
                seen.extend(documents)
                return [0.0] * len(documents)

        result = await search_and_rerank(
            session, embedder, "thundering herd", reranker=Capturing(), candidate_k=40, final_k=5
        )
        assert seen, "the reranker was called"
        assert any("Concepts:" in document for document in seen)
        for candidate, document in zip(result.hybrid.results, seen, strict=True):
            assert document == rerank_document(candidate.question)


class TestTopKAndDeterminism:
    async def test_final_k_is_respected(self, bank, session, embedder, reranker):
        result = await search_and_rerank(
            session, embedder, "index", reranker=reranker, candidate_k=20, final_k=3
        )
        assert len(result.results) == 3

    async def test_candidate_k_caps_what_is_scored(self, bank, session, embedder, reranker):
        result = await search_and_rerank(
            session, embedder, "index", reranker=reranker, candidate_k=7, final_k=3
        )
        assert result.candidates_scored == 7

    async def test_the_pipeline_is_deterministic(self, bank, session, embedder, reranker):
        first = await search_and_rerank(session, embedder, "replication lag", reranker=reranker)
        second = await search_and_rerank(session, embedder, "replication lag", reranker=reranker)
        assert [r.id for r in first.results] == [r.id for r in second.results]
        assert [r.rerank_score for r in first.results] == [r.rerank_score for r in second.results]

    async def test_results_are_ordered_by_descending_score(self, bank, session, embedder, reranker):
        result = await search_and_rerank(
            session, embedder, "database transactions", reranker=reranker
        )
        scores = [r.rerank_score for r in result.results]
        assert scores == sorted(scores, reverse=True)

    async def test_no_duplicate_questions_survive_the_pipeline(
        self, bank, session, embedder, reranker
    ):
        result = await search_and_rerank(
            session, embedder, "database index cache", reranker=reranker
        )
        ids = [r.id for r in result.results]
        assert len(ids) == len(set(ids))


class TestFallback:
    async def test_a_failing_reranker_still_returns_the_hybrid_order(self, bank, session, embedder):
        result = await search_and_rerank(
            session, embedder, "database index", reranker=ExplodingReranker(), final_k=5
        )
        assert result.reranked is False
        assert [r.id for r in result.results] == [c.id for c in result.hybrid.results[:5]]

    async def test_the_failure_is_visible_in_the_result(self, bank, session, embedder):
        """Never silently pretend reranking happened."""
        result = await search_and_rerank(
            session, embedder, "database index", reranker=ExplodingReranker()
        )
        assert result.fallback_reason is not None
        assert "RuntimeError" in result.fallback_reason
        assert all(r.rerank_score is None for r in result.results)

    async def test_no_reranker_runs_stage_one_only(self, bank, session, embedder):
        result = await search_and_rerank(session, embedder, "database index", final_k=5)
        assert result.reranked is False
        assert result.fallback_reason == "reranking disabled"
        assert len(result.results) == 5
        assert result.candidates_scored == 0

    async def test_disabled_and_broken_are_distinguishable(self, bank, session, embedder):
        off = await search_and_rerank(session, embedder, "index")
        broken = await search_and_rerank(session, embedder, "index", reranker=ExplodingReranker())
        assert off.fallback_reason != broken.fallback_reason


class TestWeakAndEmptyQueries:
    async def test_a_query_matching_nothing_lexically_still_reranks(
        self, bank, session, embedder, reranker
    ):
        result = await search_and_rerank(
            session, embedder, "xylophone marsupial", reranker=reranker, final_k=5
        )
        assert result.hybrid.lexical_candidates == 0
        assert result.reranked is True
        assert len(result.results) == 5

    async def test_an_empty_bank_returns_nothing_without_failing(self, session, embedder, reranker):
        from sqlalchemy import text

        await session.execute(text("DELETE FROM questions"))
        result = await search_and_rerank(session, embedder, "database", reranker=reranker)
        assert result.results == []
        assert result.candidates_scored == 0
        assert result.reranked is True, "nothing failed; there was nothing to order"

    async def test_an_empty_query_does_not_raise(self, bank, session, embedder, reranker):
        result = await search_and_rerank(session, embedder, "", reranker=reranker, final_k=5)
        assert isinstance(result.results, list)


class TestTimings:
    async def test_each_stage_is_timed_separately(self, bank, session, embedder, reranker):
        """The two-stage cost argument only means something if both are measured."""
        result = await search_and_rerank(session, embedder, "index", reranker=reranker)
        assert set(result.timings_ms) == {"hybrid", "rerank", "total"}
        assert all(v >= 0 for v in result.timings_ms.values())
