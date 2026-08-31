"""Run the retrieval evaluation and print the ablation table (Day 10).

    cd backend
    python scripts/run_retrieval_eval.py                    # all four modes, real models
    python scripts/run_retrieval_eval.py --per-query        # plus the per-query rank table
    python scripts/run_retrieval_eval.py --write-report     # also write evals/reports/
    python scripts/run_retrieval_eval.py --no-rerank        # skip the slow stage
    python scripts/run_retrieval_eval.py --stand-ins        # deterministic, no model download

**This measures; it never tunes.** No number produced here is fed back into the
retrieval system. An evaluation set the system has been fitted to has stopped
being a measurement of anything.

**Reproducibility.** Every input is fixed: the committed dataset, the committed
question bank, and two models pinned by name. Both models are deterministic on
CPU, and at 60 rows PostgreSQL chooses an exact sequential scan over the
approximate HNSW index, so re-running produces identical rankings. Only the
timings vary. The configuration block printed above the table records everything
needed to reproduce a run.

Requires the bank ingested and embedded (`python scripts/ingest_question_bank.py`)
and, for the default real-model run, the optional extra:
``pip install -e "./backend[embeddings]"``.
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

from app.bank.loader import validate_bank  # noqa: E402
from app.bank.taxonomy import load_taxonomy  # noqa: E402
from app.config import Settings  # noqa: E402
from app.evaluation.dataset import REPORTS_DIR, EvalQueryError, load_eval_dataset  # noqa: E402
from app.evaluation.report import (  # noqa: E402
    ablation_table,
    by_kind_table,
    config_block,
    markdown_report,
    per_query_table,
)
from app.evaluation.runner import ALL_MODES, DEFAULT_DEPTH, run_evaluation  # noqa: E402
from app.models.question import Question  # noqa: E402
from app.retrieval.embedders import (  # noqa: E402
    EmbeddingBackendUnavailable,
    HashingEmbedder,
    build_embedder,
)
from app.retrieval.embedding import Embedder  # noqa: E402
from app.retrieval.rerank import Reranker  # noqa: E402
from app.retrieval.rerankers import (  # noqa: E402
    LexicalOverlapReranker,
    RerankerUnavailable,
    build_reranker,
)

DOTENV = BACKEND.parent / ".env"


async def _run(settings: Settings, database_url: str, args: argparse.Namespace) -> int:
    bank = validate_bank(taxonomy=load_taxonomy())
    if not bank.ok:
        print(f"the question bank does not validate ({len(bank.errors)} errors)", file=sys.stderr)
        return 1
    question_ids = {loaded.item.id for loaded in bank.items}

    try:
        dataset = load_eval_dataset(known_question_ids=question_ids)
    except EvalQueryError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    embedder: Embedder
    reranker: Reranker | None
    if args.stand_ins:
        embedder, reranker = HashingEmbedder(), LexicalOverlapReranker()
    else:
        embedder = build_embedder(settings)
        reranker = None if args.no_rerank else build_reranker(settings)

    engine = create_async_engine(database_url)
    try:
        factory = async_sessionmaker(engine, expire_on_commit=False)
        async with factory() as session:
            embedded = await session.scalar(
                select(func.count()).select_from(Question).where(Question.embedding.isnot(None))
            )
            if not embedded:
                print(
                    "no embeddings stored. Run: python scripts/ingest_question_bank.py",
                    file=sys.stderr,
                )
                return 1
            if args.stand_ins and embedded:
                # The stored vectors came from bge-small; a hashing query vector
                # is not comparable with them, so vector-mode numbers would be
                # noise dressed up as a measurement.
                print(
                    "warning: --stand-ins uses a hashing embedder against vectors that were\n"
                    "         written by a different model. Use this to check the runner works,\n"
                    "         NOT to judge retrieval quality.\n",
                    file=sys.stderr,
                )

            # Load the models before the first timed query, so `ms/query` measures
            # retrieval rather than a one-off model load.
            embedder.embed_query("warm up")
            if reranker is not None:
                reranker.score_pairs("warm up", ["a document"])

            modes = [m for m in ALL_MODES if not (m == "hybrid_rerank" and reranker is None)]
            report = await run_evaluation(
                session,
                dataset,
                embedder=embedder,
                reranker=reranker,
                modes=modes,
                depth=args.depth,
                vector_k=settings.retrieval_vector_k,
                lexical_k=settings.retrieval_lexical_k,
                candidate_k=settings.rerank_candidate_k,
                rrf_k=settings.retrieval_rrf_k,
                real_models=not args.stand_ins,
                question_count=len(question_ids),
            )
    finally:
        await engine.dispose()

    print(config_block(report))
    print(f"queries      : {len(dataset)}")
    print()
    print(ablation_table(report))
    print()
    print("MRR by query kind")
    print(by_kind_table(report, dataset))

    if args.per_query:
        print()
        print("rank of the first relevant question (- = not in the top " f"{args.depth})")
        print(per_query_table(report, dataset))

    misses = {s.label: s.misses for s in report.summaries if s.misses}
    if misses:
        print()
        print("misses (nothing relevant retrieved):")
        for label, ids in misses.items():
            print(f"  {label:<20} {', '.join(ids)}")

    if args.write_report:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        target = REPORTS_DIR / (
            "retrieval_ablation.md" if not args.stand_ins else "retrieval_ablation_standins.md"
        )
        target.write_text(markdown_report(report, dataset), encoding="utf-8")
        print()
        print(f"wrote {target}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--depth", type=int, default=DEFAULT_DEPTH, help="results scored per mode")
    parser.add_argument("--per-query", action="store_true", help="print the per-query rank table")
    parser.add_argument("--write-report", action="store_true", help="write evals/reports/")
    parser.add_argument("--no-rerank", action="store_true", help="skip the cross-encoder mode")
    parser.add_argument(
        "--stand-ins",
        action="store_true",
        help="deterministic in-repo models; checks the runner, not retrieval quality",
    )
    parser.add_argument("--database-url", default=None)
    args = parser.parse_args()

    settings = Settings(_env_file=DOTENV if DOTENV.exists() else None)  # type: ignore[call-arg]
    database_url = args.database_url or settings.database_url

    try:
        return asyncio.run(_run(settings, database_url, args))
    except EmbeddingBackendUnavailable as exc:
        print(f"embedding backend unavailable: {exc}", file=sys.stderr)
        return 1
    except RerankerUnavailable as exc:
        print(f"reranker unavailable: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
