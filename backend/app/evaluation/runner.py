"""Running the evaluation set through each retrieval mode, and scoring it.

    evaluation queries
            |
            +----> vector only ---------+
            +----> lexical only --------+
            +----> hybrid (RRF) --------+---> Recall@K / MRR / nDCG@K
            +----> hybrid + reranker ---+              |
                                                       v
                                              four-row ablation table

**An ablation study** is this: take the finished system, remove or replace one
component at a time, and measure what each removal costs. It is the only way to
answer "is the reranker earning its 100 ms?" - as opposed to assuming it must,
because it is the more sophisticated model.

**Every number here comes from a real retrieval run.** Nothing is simulated. The
modes call exactly the Day 8 and Day 9 functions the application calls; this
module adds no retrieval logic of its own, and could not, because it holds no
SQL and no model.

**Fairness rules**, which matter more than they look:

* Every mode is asked for the same ``depth`` results, so Recall@10 compares
  like with like. A mode allowed to return 20 while another returns 8 would
  win on recall for that reason alone.
* Recall@5 is computed from the *first five of the same list*, not from a
  second retrieval run with ``k=5``. Re-running would let a mode reorder its
  own results between the two measurements.
* The reranker's candidate pool is separate from ``depth``: it scores
  ``candidate_k`` candidates and returns the best ``depth``.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from sqlalchemy.ext.asyncio import AsyncSession

from app.bank.paths import REPO_ROOT
from app.evaluation.dataset import EvalDataset, EvalQuery
from app.evaluation.metrics import (
    first_relevant_rank,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)
from app.retrieval.embedding import Embedder
from app.retrieval.pipeline import search_and_rerank
from app.retrieval.rerank import DEFAULT_CANDIDATE_K, Reranker
from app.retrieval.rrf import DEFAULT_RRF_K
from app.retrieval.search import (
    DEFAULT_LEXICAL_K,
    DEFAULT_VECTOR_K,
    hybrid_search,
    lexical_search,
    vector_search,
)

ModeName = Literal["vector", "lexical", "hybrid", "hybrid_rerank"]

#: How deep every mode's result list is cut before scoring. 10 because the plan
#: reports Recall@10 / nDCG@10, and because Recall@5 is then a prefix of the
#: same list rather than a second run.
DEFAULT_DEPTH = 10

MODE_LABELS: dict[ModeName, str] = {
    "vector": "Vector only",
    "lexical": "Lexical only",
    "hybrid": "Hybrid RRF",
    "hybrid_rerank": "Hybrid + reranker",
}

ALL_MODES: tuple[ModeName, ...] = ("vector", "lexical", "hybrid", "hybrid_rerank")


@dataclass(frozen=True, slots=True)
class QueryOutcome:
    """One query, through one mode, with its scores and the evidence behind them."""

    query_id: str
    mode: ModeName
    retrieved: list[str]
    #: 1-based position of the first relevant question, or None if not found in
    #: the retrieved depth. The number a human reads when a query looks wrong.
    first_rank: int | None
    recall_at_5: float
    recall_at_10: float
    reciprocal_rank: float
    ndcg: float
    elapsed_ms: float


@dataclass(frozen=True, slots=True)
class ModeSummary:
    """One row of the ablation table."""

    mode: ModeName
    label: str
    queries: int
    recall_at_5: float
    recall_at_10: float
    mrr: float
    ndcg: float
    mean_ms: float
    total_ms: float
    #: Queries where nothing relevant appeared in the retrieved depth.
    misses: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class EvalConfig:
    """Everything needed to reproduce a run. Written into the report verbatim."""

    embedder_model: str
    reranker_model: str | None
    depth: int
    vector_k: int
    lexical_k: int
    candidate_k: int
    rrf_k: float
    #: False when a deterministic stand-in was used instead of a real model -
    #: the difference between "measured the system" and "measured the plumbing".
    real_models: bool
    dataset_path: str
    question_count: int


@dataclass(frozen=True, slots=True)
class EvalReport:
    """The whole run: per-query outcomes, per-mode summaries, and the config."""

    config: EvalConfig
    summaries: list[ModeSummary]
    outcomes: list[QueryOutcome]

    def outcomes_for(self, query_id: str) -> dict[ModeName, QueryOutcome]:
        return {o.mode: o for o in self.outcomes if o.query_id == query_id}

    def summary_for(self, mode: ModeName) -> ModeSummary:
        return next(s for s in self.summaries if s.mode == mode)


async def _retrieve(
    mode: ModeName,
    session: AsyncSession,
    embedder: Embedder,
    reranker: Reranker | None,
    query: str,
    *,
    depth: int,
    vector_k: int,
    lexical_k: int,
    candidate_k: int,
    rrf_k: float,
) -> list[str]:
    """Ranked question ids from one mode. The only place a mode is defined."""
    if mode == "vector":
        hits = await vector_search(session, embedder, query, limit=depth)
        return [hit.question.id for hit in hits]
    if mode == "lexical":
        hits = await lexical_search(session, query, limit=depth)
        return [hit.question.id for hit in hits]
    if mode == "hybrid":
        outcome = await hybrid_search(
            session,
            embedder,
            query,
            vector_k=vector_k,
            lexical_k=lexical_k,
            final_k=depth,
            rrf_k=rrf_k,
        )
        return [item.id for item in outcome.results]
    if reranker is None:
        raise ValueError("mode 'hybrid_rerank' needs a reranker")
    result = await search_and_rerank(
        session,
        embedder,
        query,
        reranker=reranker,
        vector_k=vector_k,
        lexical_k=lexical_k,
        candidate_k=candidate_k,
        final_k=depth,
        rrf_k=rrf_k,
    )
    if not result.reranked:
        # Scoring a fallback as if it were a rerank would credit the
        # cross-encoder with the hybrid's ranking. Loud, not silent.
        raise RuntimeError(f"reranking did not run: {result.fallback_reason}")
    return [item.id for item in result.results]


def score(query: EvalQuery, retrieved: Sequence[str], *, depth: int) -> dict[str, float]:
    """Every metric for one query's ranked list. Pure - no I/O, no model."""
    relevant = query.relevant_ids
    return {
        "recall_at_5": recall_at_k(retrieved, relevant, 5),
        "recall_at_10": recall_at_k(retrieved, relevant, 10),
        "reciprocal_rank": reciprocal_rank(retrieved, relevant, k=depth),
        "ndcg": ndcg_at_k(retrieved, query.relevant, depth),
    }


async def run_evaluation(
    session: AsyncSession,
    dataset: EvalDataset,
    *,
    embedder: Embedder,
    reranker: Reranker | None = None,
    modes: Sequence[ModeName] = ALL_MODES,
    depth: int = DEFAULT_DEPTH,
    vector_k: int = DEFAULT_VECTOR_K,
    lexical_k: int = DEFAULT_LEXICAL_K,
    candidate_k: int = DEFAULT_CANDIDATE_K,
    rrf_k: float = DEFAULT_RRF_K,
    real_models: bool = True,
    question_count: int = 0,
) -> EvalReport:
    """Run every query through every mode and score the results.

    Modes run sequentially and per query, so a slow mode cannot distort another
    mode's timing. ``elapsed_ms`` is wall-clock for that one retrieval, warm -
    model loading happens before the first query is scored.
    """
    selected = [m for m in modes if m != "hybrid_rerank" or reranker is not None]
    outcomes: list[QueryOutcome] = []

    for mode in selected:
        for query in dataset:
            started = time.perf_counter()
            retrieved = await _retrieve(
                mode,
                session,
                embedder,
                reranker,
                query.query,
                depth=depth,
                vector_k=vector_k,
                lexical_k=lexical_k,
                candidate_k=candidate_k,
                rrf_k=rrf_k,
            )
            elapsed = (time.perf_counter() - started) * 1000
            scores = score(query, retrieved, depth=depth)
            outcomes.append(
                QueryOutcome(
                    query_id=query.id,
                    mode=mode,
                    retrieved=list(retrieved),
                    first_rank=first_relevant_rank(retrieved[:depth], query.relevant_ids),
                    elapsed_ms=elapsed,
                    recall_at_5=scores["recall_at_5"],
                    recall_at_10=scores["recall_at_10"],
                    reciprocal_rank=scores["reciprocal_rank"],
                    ndcg=scores["ndcg"],
                )
            )

    summaries = [_summarise(mode, outcomes) for mode in selected]
    config = EvalConfig(
        embedder_model=embedder.model_id,
        reranker_model=reranker.model_id if reranker is not None else None,
        depth=depth,
        vector_k=vector_k,
        lexical_k=lexical_k,
        candidate_k=candidate_k,
        rrf_k=rrf_k,
        real_models=real_models,
        # Repo-relative: the report is a committed artefact, and an absolute
        # path bakes one machine's home directory into a file everyone reads.
        dataset_path=_relative(dataset.path),
        question_count=question_count,
    )
    return EvalReport(config=config, summaries=summaries, outcomes=outcomes)


def _relative(path: Path) -> str:
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:  # a dataset outside the repo, e.g. a tmp_path in tests
        return path.name


def _summarise(mode: ModeName, outcomes: Sequence[QueryOutcome]) -> ModeSummary:
    rows = [o for o in outcomes if o.mode == mode]
    n = len(rows)
    if n == 0:  # pragma: no cover - a mode with no queries cannot be selected
        return ModeSummary(mode, MODE_LABELS[mode], 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    total_ms = sum(o.elapsed_ms for o in rows)
    return ModeSummary(
        mode=mode,
        label=MODE_LABELS[mode],
        queries=n,
        recall_at_5=sum(o.recall_at_5 for o in rows) / n,
        recall_at_10=sum(o.recall_at_10 for o in rows) / n,
        mrr=sum(o.reciprocal_rank for o in rows) / n,
        ndcg=sum(o.ndcg for o in rows) / n,
        mean_ms=total_ms / n,
        total_ms=total_ms,
        misses=sorted(o.query_id for o in rows if o.first_rank is None),
    )


__all__ = [
    "ALL_MODES",
    "DEFAULT_DEPTH",
    "MODE_LABELS",
    "EvalConfig",
    "EvalReport",
    "ModeName",
    "ModeSummary",
    "QueryOutcome",
    "run_evaluation",
    "score",
]
