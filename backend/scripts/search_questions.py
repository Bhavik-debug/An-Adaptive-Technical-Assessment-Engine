"""Search the question bank from the command line (Days 8-9).

    cd backend
    python scripts/search_questions.py "how do database indexes improve performance"
    python scripts/search_questions.py "MVCC" --mode lexical
    python scripts/search_questions.py "speed up slow lookups" --mode vector
    python scripts/search_questions.py "how do I stop stale reads" --no-rerank
    python scripts/search_questions.py "cache invalidation" --explain

A **developer tool**, not the product.  Plan section 3 puts
``GET /questions/search`` in the Phase 2 exit gate, and it is neither Day 8's nor
Day 9's - those two build the retrieval layer itself.  This script exists so the
layer can be driven and seen by hand before anything is exposed over HTTP, and it
calls exactly the same functions the endpoint will.

``--mode`` runs one retriever alone, which is the fastest way to see *why*
hybrid is worth having: try a query with the same words as a question
(``lexical`` wins) and then one that means the same thing in different words
(``vector`` wins).

In ``hybrid`` mode the Day 9 cross-encoder reranks the candidates by default.
``--no-rerank`` shows stage 1 alone, and the ``was`` / ``move`` columns show what
stage 2 changed - the clearest way to see whether reranking earns its ~10 ms per
candidate on this bank.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:  # so `python scripts/...` works without install
    sys.path.insert(0, str(BACKEND))

from sqlalchemy import func, select  # noqa: E402
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine  # noqa: E402

from app.config import Settings  # noqa: E402
from app.models.question import Question  # noqa: E402
from app.retrieval.embedders import EmbeddingBackendUnavailable, build_embedder  # noqa: E402
from app.retrieval.pipeline import format_pipeline_results, search_and_rerank  # noqa: E402
from app.retrieval.rerankers import RerankerUnavailable, build_reranker  # noqa: E402
from app.retrieval.search import (  # noqa: E402
    RetrievedQuestion,
    ScoredHit,
    format_results,
    hybrid_search,
    lexical_search,
    vector_search,
)

DOTENV = BACKEND.parent / ".env"


def _as_results(hits: list[ScoredHit]) -> list[RetrievedQuestion]:
    """Show a single-retriever run in the same table as a hybrid one."""
    return [
        RetrievedQuestion(
            question=hit.question,
            rrf_score=0.0,
            vector_rank=hit.rank,
            vector_similarity=hit.score,
        )
        for hit in hits
    ]


def _explain(results: list[RetrievedQuestion]) -> None:
    print("")
    print("what each result was matched on:")
    for hit in results:
        print("")
        print(f"--- {hit.id}  sources={','.join(hit.sources) or 'none'}")
        print(f"    rrf={hit.rrf_score:.6f}")
        print(f"    {hit.question.topic_key} / {hit.question.subtopic_key}")


async def _run(settings: Settings, database_url: str, args: argparse.Namespace) -> int:
    engine = create_async_engine(database_url)
    try:
        factory = async_sessionmaker(engine, expire_on_commit=False)
        async with factory() as session:
            total = await session.scalar(select(func.count()).select_from(Question))
            embedded = await session.scalar(
                select(func.count()).select_from(Question).where(Question.embedding.isnot(None))
            )
            print(f"bank: {total} questions, {embedded} with an embedding")
            if args.mode in {"hybrid", "vector"} and not embedded:
                print(
                    "no embeddings stored. Run: " "python scripts/ingest_question_bank.py",
                    file=sys.stderr,
                )
                return 1

            if args.mode == "lexical":
                hits = await lexical_search(session, args.query, limit=args.final_k)
                results = [
                    RetrievedQuestion(
                        question=hit.question,
                        rrf_score=0.0,
                        lexical_rank=hit.rank,
                        lexical_score=hit.score,
                    )
                    for hit in hits
                ]
                print(f"lexical only - {len(results)} result(s)")
                print(format_results(results))
                return 0

            embedder = build_embedder(settings)
            print(f"embedder: {embedder.model_id} ({embedder.dimension} dimensions)")

            if args.mode == "vector":
                hits = await vector_search(session, embedder, args.query, limit=args.final_k)
                print(f"vector only - {len(hits)} result(s)")
                print(format_results(_as_results(hits)))
                return 0

            # --- stage 1 only ------------------------------------------------
            if not args.rerank:
                outcome = await hybrid_search(
                    session,
                    embedder,
                    args.query,
                    vector_k=args.vector_k,
                    lexical_k=args.lexical_k,
                    final_k=args.final_k,
                    rrf_k=settings.retrieval_rrf_k,
                )
                print(
                    f"candidates: {outcome.vector_candidates} vector, "
                    f"{outcome.lexical_candidates} lexical, "
                    f"{outcome.fused_candidates} distinct after fusion, "
                    f"{outcome.truncated} beyond the final {args.final_k}"
                )
                timings = "  ".join(f"{k}={v:.1f}ms" for k, v in outcome.timings_ms.items())
                print(f"timings: {timings}")
                print("")
                print(format_results(outcome.results))
                if args.explain:
                    _explain(outcome.results[: args.explain_n])
                return 0

            # --- both stages -------------------------------------------------
            reranker = build_reranker(settings)
            print(f"reranker: {reranker.model_id}")
            result = await search_and_rerank(
                session,
                embedder,
                args.query,
                reranker=reranker,
                vector_k=args.vector_k,
                lexical_k=args.lexical_k,
                candidate_k=args.candidate_k,
                final_k=args.final_k,
                rrf_k=settings.retrieval_rrf_k,
            )
            print(
                f"stage 1: {result.hybrid.vector_candidates} vector, "
                f"{result.hybrid.lexical_candidates} lexical, "
                f"{result.candidates_generated} candidates"
            )
            if result.reranked:
                print(f"stage 2: {result.candidates_scored} pairs scored by the cross-encoder")
            else:
                print(f"stage 2: NOT RUN - {result.fallback_reason}")
                print("         the ranking below is stage 1's, unchanged")
            timings = "  ".join(f"{k}={v:.1f}ms" for k, v in result.timings_ms.items())
            print(f"timings: {timings}")
            print("")
            print(format_pipeline_results(result))
            moved = result.promoted
            if moved:
                print("")
                print(
                    "promoted by reranking: "
                    + ", ".join(f"{m.id} ({m.retrieval_rank}->{m.rerank_rank})" for m in moved)
                )
            if args.explain:
                _explain([m.candidate for m in result.results[: args.explain_n]])
            return 0
    finally:
        await engine.dispose()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("query", help="a natural-language search query")
    parser.add_argument(
        "--mode",
        choices=("hybrid", "vector", "lexical"),
        default="hybrid",
        help="run one retriever alone instead of fusing both (default: hybrid)",
    )
    parser.add_argument(
        "--no-rerank",
        dest="rerank",
        action="store_false",
        help="stage 1 only: show the RRF order without the cross-encoder",
    )
    parser.add_argument("--vector-k", type=int, default=None, help="vector candidates")
    parser.add_argument("--lexical-k", type=int, default=None, help="lexical candidates")
    parser.add_argument(
        "--candidate-k", type=int, default=None, help="candidates handed to the reranker"
    )
    parser.add_argument("--final-k", type=int, default=None, help="results to return")
    parser.add_argument("--explain", action="store_true", help="show per-result detail")
    parser.add_argument("--explain-n", type=int, default=3)
    parser.add_argument("--database-url", default=None)
    args = parser.parse_args()

    settings = Settings(_env_file=DOTENV if DOTENV.exists() else None)  # type: ignore[call-arg]
    args.vector_k = args.vector_k or settings.retrieval_vector_k
    args.lexical_k = args.lexical_k or settings.retrieval_lexical_k
    args.candidate_k = args.candidate_k or settings.rerank_candidate_k
    # Reranking only applies to the fused list; a single-retriever run is a
    # diagnostic of that retriever, and reranking it would obscure what it did.
    args.rerank = args.rerank and args.mode == "hybrid" and settings.rerank_enabled
    if args.rerank:
        args.final_k = args.final_k or settings.rerank_final_k
    else:
        args.final_k = args.final_k or settings.retrieval_final_k
    database_url = args.database_url or settings.database_url

    try:
        return asyncio.run(_run(settings, database_url, args))
    except EmbeddingBackendUnavailable as exc:
        print(f"embedding backend unavailable: {exc}", file=sys.stderr)
        return 1
    except RerankerUnavailable as exc:
        # Reachable only because this script builds the reranker eagerly to print
        # its name; inside the pipeline a load failure degrades to stage 1's order.
        print(f"reranker unavailable: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
