"""The evaluation runner, end to end against a real Postgres and the real bank.

The unit tests prove the metrics are arithmetically right. Only this can prove
the *runner* is right: that each mode calls the retrieval function it claims to,
that every mode is scored over the same depth, and that a summary really is the
mean of its per-query outcomes.

**Which models these use.** The deterministic stand-ins - ``HashingEmbedder``
and ``LexicalOverlapReranker`` - so the default suite needs no download. That
means **nothing here asserts a quality number**. A test claiming "hybrid beats
vector" against a hashing embedder would be asserting something about the
stand-in, not about retrieval. The real quality measurement is the committed
ablation report, produced by ``scripts/run_retrieval_eval.py`` with real models.

Skips when the compose stack is down; ``REQUIRE_INTEGRATION=1`` in CI turns that
skip into a failure.
"""

from __future__ import annotations

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker

from app.bank.ingest import ingest_bank
from app.bank.loader import validate_bank
from app.bank.paths import BANK_DIR
from app.bank.taxonomy import load_taxonomy
from app.evaluation.dataset import load_eval_dataset
from app.evaluation.report import ablation_table, by_kind_table, markdown_report, per_query_table
from app.evaluation.runner import ALL_MODES, run_evaluation
from app.retrieval.embedders import HashingEmbedder
from app.retrieval.rerankers import LexicalOverlapReranker


@pytest.fixture(scope="module")
def taxonomy():
    return load_taxonomy()


@pytest.fixture(scope="module")
def bank_report(taxonomy):
    report = validate_bank(BANK_DIR, taxonomy)
    assert report.ok, "\n".join(report.errors[:10])
    return report


@pytest.fixture(scope="module")
def dataset(bank_report):
    return load_eval_dataset(known_question_ids={x.item.id for x in bank_report.items})


@pytest.fixture
def embedder():
    return HashingEmbedder()


@pytest.fixture
def reranker():
    return LexicalOverlapReranker()


@pytest_asyncio.fixture
async def bank(db_engine: AsyncEngine, bank_report, taxonomy, embedder):
    factory = async_sessionmaker(db_engine, expire_on_commit=False)
    async with factory() as session:
        await ingest_bank(session, bank_report, taxonomy, embedder=embedder)
        await session.commit()


@pytest_asyncio.fixture
async def session(db_engine: AsyncEngine):
    factory = async_sessionmaker(db_engine, expire_on_commit=False)
    async with factory() as opened:
        yield opened


@pytest_asyncio.fixture
async def report(bank, session, dataset, embedder, reranker):
    return await run_evaluation(
        session,
        dataset,
        embedder=embedder,
        reranker=reranker,
        depth=10,
        real_models=False,
        question_count=60,
    )


class TestTheRunnerCoversEverything:
    async def test_all_four_modes_are_evaluated(self, report):
        assert [s.mode for s in report.summaries] == list(ALL_MODES)

    async def test_every_query_is_run_through_every_mode(self, report, dataset):
        for summary in report.summaries:
            assert summary.queries == len(dataset)
        assert len(report.outcomes) == len(dataset) * len(ALL_MODES)

    async def test_the_reranker_mode_is_dropped_when_no_reranker_is_given(
        self, bank, session, dataset, embedder
    ):
        outcome = await run_evaluation(session, dataset, embedder=embedder, depth=10)
        assert [s.mode for s in outcome.summaries] == ["vector", "lexical", "hybrid"]

    async def test_only_the_requested_modes_run(self, bank, session, dataset, embedder):
        outcome = await run_evaluation(session, dataset, embedder=embedder, modes=["lexical"])
        assert [s.mode for s in outcome.summaries] == ["lexical"]


class TestFairness:
    async def test_no_mode_returns_more_than_the_scored_depth(self, report):
        """A mode allowed a longer list would win on recall for that reason alone."""
        for outcome in report.outcomes:
            assert len(outcome.retrieved) <= report.config.depth

    async def test_recall_at_5_is_a_prefix_of_the_same_list(self, report):
        """Not a second retrieval run, which could reorder between measurements."""
        for outcome in report.outcomes:
            assert outcome.recall_at_5 <= outcome.recall_at_10 + 1e-9

    async def test_no_mode_returns_duplicate_ids(self, report):
        for outcome in report.outcomes:
            assert len(outcome.retrieved) == len(set(outcome.retrieved))


class TestScoresAreConsistent:
    async def test_every_metric_is_in_range(self, report):
        for outcome in report.outcomes:
            assert 0.0 <= outcome.recall_at_5 <= 1.0
            assert 0.0 <= outcome.recall_at_10 <= 1.0
            assert 0.0 <= outcome.reciprocal_rank <= 1.0
            assert 0.0 <= outcome.ndcg <= 1.0

    async def test_a_summary_is_the_mean_of_its_outcomes(self, report):
        for summary in report.summaries:
            rows = [o for o in report.outcomes if o.mode == summary.mode]
            assert summary.mrr == pytest.approx(sum(o.reciprocal_rank for o in rows) / len(rows))
            assert summary.ndcg == pytest.approx(sum(o.ndcg for o in rows) / len(rows))

    async def test_the_reciprocal_rank_matches_the_reported_first_rank(self, report):
        for outcome in report.outcomes:
            if outcome.first_rank is None:
                assert outcome.reciprocal_rank == 0.0
            else:
                assert outcome.reciprocal_rank == pytest.approx(1 / outcome.first_rank)

    async def test_a_miss_is_recorded_when_nothing_relevant_was_found(self, report):
        for summary in report.summaries:
            rows = [o for o in report.outcomes if o.mode == summary.mode]
            expected = sorted(o.query_id for o in rows if o.first_rank is None)
            assert summary.misses == expected


class TestDeterminism:
    async def test_two_runs_produce_identical_rankings(
        self, bank, session, dataset, embedder, reranker
    ):
        """The whole evaluation is reproducible, or its numbers mean nothing."""
        first = await run_evaluation(
            session, dataset, embedder=embedder, reranker=reranker, depth=10
        )
        second = await run_evaluation(
            session, dataset, embedder=embedder, reranker=reranker, depth=10
        )
        assert [o.retrieved for o in first.outcomes] == [o.retrieved for o in second.outcomes]
        assert [s.mrr for s in first.summaries] == [s.mrr for s in second.summaries]


class TestTheConfigIsRecorded:
    async def test_it_records_the_models_and_k_values(self, report):
        config = report.config
        assert config.embedder_model == "hashing-v1"
        assert config.reranker_model == "lexical-overlap-v1"
        assert config.depth == 10
        assert config.question_count == 60

    async def test_stand_in_runs_are_flagged_as_not_a_quality_measurement(self, report):
        """Confusing a plumbing check for a quality number is the error to prevent."""
        assert report.config.real_models is False


class TestTheReportRenders:
    async def test_the_ablation_table_has_a_row_per_mode(self, report):
        table = ablation_table(report)
        for summary in report.summaries:
            assert summary.label in table

    async def test_the_markdown_report_names_the_stand_ins(self, report, dataset):
        text = markdown_report(report, dataset)
        assert "DETERMINISTIC" in text
        assert "hashing-v1" in text

    async def test_the_per_query_table_lists_every_query(self, report, dataset):
        table = per_query_table(report, dataset)
        for item in dataset:
            assert item.id in table

    async def test_the_by_kind_table_lists_every_kind(self, report, dataset):
        table = by_kind_table(report, dataset)
        for kind in dataset.by_kind():
            assert kind in table


class TestFailureIsLoud:
    async def test_a_broken_reranker_fails_the_evaluation_rather_than_scoring_a_fallback(
        self, bank, session, dataset, embedder
    ):
        """Scoring a fallback would credit the cross-encoder with the hybrid's ranking."""

        class Exploding:
            model_id = "exploding"

            def score_pairs(self, query, documents):
                raise RuntimeError("model file is corrupt")

        with pytest.raises(RuntimeError, match="reranking did not run"):
            await run_evaluation(
                session,
                dataset,
                embedder=embedder,
                reranker=Exploding(),
                modes=["hybrid_rerank"],
            )
