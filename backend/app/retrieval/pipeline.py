"""The two-stage retrieval pipeline: hybrid candidates, then cross-encoder.

                              QUERY
                                |
                 +--------------+--------------+
                 |     STAGE 1  (Day 8)        |
                 |  vector + lexical + RRF     |   indexed, milliseconds
                 +--------------+--------------+
                                |
                     ~40 candidate questions
                                |
                 +--------------+--------------+
                 |     STAGE 2  (Day 9)        |
                 |  cross-encoder relevance    |   one model pass per candidate
                 +--------------+--------------+
                                |
                        top 8, reordered

**Why this file exists rather than a `rerank=True` flag on `hybrid_search`.**
The two stages must stay separable. Later work changes how candidates are chosen
- Phase 3 adds difficulty and coverage constraints to selection - and none of
that should require touching the reranker, or vice versa. ``search.py`` does not
import ``rerank``; ``rerank`` does not run queries; this module is the only place
that knows both exist.

The composition is deliberately thin. If it starts growing policy, that policy
belongs in whichever layer it is really about.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from sqlalchemy.ext.asyncio import AsyncSession

from app.retrieval.embedding import Embedder
from app.retrieval.rerank import (
    DEFAULT_CANDIDATE_K,
    DEFAULT_RERANK_FINAL_K,
    RerankedQuestion,
    Reranker,
    rerank_candidates,
)
from app.retrieval.rrf import DEFAULT_RRF_K
from app.retrieval.search import (
    DEFAULT_LEXICAL_K,
    DEFAULT_VECTOR_K,
    HybridSearchResult,
    hybrid_search,
)

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class PipelineResult:
    """The final ranking, plus everything needed to explain how it was reached."""

    query: str
    results: list[RerankedQuestion]
    #: Stage 1's own result, unmodified. Keeping it is what makes "the reranker
    #: moved this from 7th to 1st" answerable, and it is what Day 10's ablation
    #: will compare against.
    hybrid: HybridSearchResult
    reranked: bool
    reranker_model_id: str | None = None
    #: Set if and only if `reranked` is False.
    fallback_reason: str | None = None
    candidates_generated: int = 0
    candidates_scored: int = 0
    timings_ms: dict[str, float] = field(default_factory=dict)

    @property
    def promoted(self) -> list[RerankedQuestion]:
        """Results the reranker moved up. Empty when it did not run."""
        return [r for r in self.results if r.rank_delta > 0]


async def search_and_rerank(
    session: AsyncSession,
    embedder: Embedder,
    query: str,
    *,
    reranker: Reranker | None = None,
    vector_k: int = DEFAULT_VECTOR_K,
    lexical_k: int = DEFAULT_LEXICAL_K,
    candidate_k: int = DEFAULT_CANDIDATE_K,
    final_k: int = DEFAULT_RERANK_FINAL_K,
    rrf_k: float = DEFAULT_RRF_K,
) -> PipelineResult:
    """Run stage 1, then stage 2, and return the reordered top ``final_k``.

    **The three K values, which are three different things.**

    * ``vector_k`` / ``lexical_k`` - how many each *retriever* proposes before
      fusion (Day 8's candidate K).
    * ``candidate_k`` - how many fused candidates the cross-encoder scores.
      This is what ``hybrid_search`` is asked for, so it replaces Day 8's
      ``final_k`` when reranking is on.
    * ``final_k`` - how many are returned.

    ``candidate_k`` must be comfortably larger than ``final_k``: the reranker can
    only reorder what stage 1 gave it, so a relevant question stage 1 ranked 30th
    is unreachable if only 10 candidates were generated. That is the failure mode
    the two-stage design exists to avoid, and it is why the plan says 40 -> 8.

    ``reranker=None`` runs stage 1 only and returns the RRF order, with
    ``reranked=False`` and a reason. That is the same path taken when the model
    cannot load, so "reranking is off" and "reranking broke" are represented
    identically to a caller and distinguishably in the reason string.
    """
    started = time.perf_counter()
    hybrid = await hybrid_search(
        session,
        embedder,
        query,
        vector_k=vector_k,
        lexical_k=lexical_k,
        final_k=candidate_k,
        rrf_k=rrf_k,
    )
    after_hybrid = time.perf_counter()

    if reranker is None:
        outcome_results = [
            RerankedQuestion(
                candidate=candidate,
                rerank_score=None,
                rerank_rank=position,
                retrieval_rank=position,
            )
            for position, candidate in enumerate(hybrid.results[:final_k], start=1)
        ]
        return PipelineResult(
            query=query,
            results=outcome_results,
            hybrid=hybrid,
            reranked=False,
            fallback_reason="reranking disabled",
            candidates_generated=len(hybrid.results),
            timings_ms={
                "hybrid": (after_hybrid - started) * 1000,
                "rerank": 0.0,
                "total": (time.perf_counter() - started) * 1000,
            },
        )

    outcome = rerank_candidates(reranker, query, hybrid.results, final_k=final_k)
    finished = time.perf_counter()

    if not outcome.reranked:
        # Loud in the logs, honest in the result. A reranker that silently stops
        # working looks exactly like one that is working, which is the worst
        # possible failure for a component whose whole job is ordering quality.
        log.warning(
            "reranking unavailable, serving hybrid order",
            extra={"reason": outcome.fallback_reason, "query_length": len(query)},
        )

    return PipelineResult(
        query=query,
        results=outcome.results,
        hybrid=hybrid,
        reranked=outcome.reranked,
        reranker_model_id=outcome.model_id,
        fallback_reason=outcome.fallback_reason,
        candidates_generated=len(hybrid.results),
        candidates_scored=outcome.candidates_scored,
        timings_ms={
            "hybrid": (after_hybrid - started) * 1000,
            "rerank": (finished - after_hybrid) * 1000,
            "total": (finished - started) * 1000,
        },
    )


def format_pipeline_results(result: PipelineResult, *, width: int = 62) -> str:
    """A readable table of a pipeline run, for scripts and manual verification."""
    header = (
        f"{'#':>2}  {'id':<20} {'score':>9} {'was':>4} {'move':>5}  "
        f"{'rrf':>7} {'vec':>4} {'lex':>4}  question"
    )
    lines = [header, "-" * (len(header) + width - 8)]
    for hit in result.results:
        candidate = hit.candidate
        move = f"{hit.rank_delta:+d}" if hit.rank_delta else "-"
        score = "-" if hit.rerank_score is None else f"{hit.rerank_score:.4f}"
        lines.append(
            f"{hit.rerank_rank:>2}  {hit.id:<20} {score:>9} {hit.retrieval_rank:>4} {move:>5}  "
            f"{candidate.rrf_score:>7.5f} "
            f"{_dash(candidate.vector_rank):>4} {_dash(candidate.lexical_rank):>4}  "
            f"{candidate.question.text[:width]}"
        )
    return "\n".join(lines)


def _dash(value: int | None) -> str:
    return "-" if value is None else str(value)


__all__ = ["PipelineResult", "format_pipeline_results", "search_and_rerank"]
