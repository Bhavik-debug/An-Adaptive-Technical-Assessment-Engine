"""Load ``data/question-bank/`` into Postgres.

    cd backend
    python scripts/ingest_question_bank.py --dry-run
    python scripts/ingest_question_bank.py
    python scripts/ingest_question_bank.py --only-reviewed   # the production posture

Idempotent: an upsert keyed on the readable question id, so running it twice is
running it once.  It refuses to write anything if the dataset does not validate,
because a half-ingested bank is worse than no bank - the retrieval layer cannot
tell the difference.

Deletions are never automatic.  An item removed from the files is *reported*;
``turns.question_id`` references ``questions.id``, and a silent cascade would
erase interview history to tidy up a dataset edit.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:  # so `python scripts/...` works without install
    sys.path.insert(0, str(BACKEND))

from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine  # noqa: E402

from app.bank.ingest import ingest_bank  # noqa: E402
from app.bank.loader import validate_bank  # noqa: E402
from app.bank.taxonomy import TaxonomyError, load_taxonomy  # noqa: E402
from app.config import Settings  # noqa: E402

DOTENV = BACKEND.parent / ".env"


async def _run(database_url: str, only_reviewed: bool) -> int:
    taxonomy = load_taxonomy()
    report = validate_bank(taxonomy=taxonomy)
    if not report.ok:
        print(f"{len(report.errors)} validation error(s); nothing was written", file=sys.stderr)
        for error in report.errors[:20]:
            print(f"  x {error}", file=sys.stderr)
        return 1

    engine = create_async_engine(database_url)
    try:
        factory = async_sessionmaker(engine, expire_on_commit=False)
        async with factory() as session:
            result = await ingest_bank(session, report, taxonomy, only_reviewed=only_reviewed)
            await session.commit()
    finally:
        await engine.dispose()

    print(f"ingested: {result.summary}")
    if result.unreviewed_question_ids:
        print(
            f"\nwarning: {len(result.unreviewed_question_ids)} ingested item(s) are drafts "
            "awaiting human review. Re-run with --only-reviewed to load reviewed items only."
        )
    if result.orphaned_question_ids:
        print(
            "\nnote: these question ids exist in the database but not in the files, "
            "and were left alone:\n  " + ", ".join(result.orphaned_question_ids)
        )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--database-url",
        default=None,
        help="defaults to DATABASE_URL from the environment / .env",
    )
    parser.add_argument(
        "--only-reviewed",
        action="store_true",
        help="skip items that are not marked reviewed",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="validate and report; touch no database"
    )
    args = parser.parse_args()

    try:
        taxonomy = load_taxonomy()
    except TaxonomyError as exc:
        print(f"taxonomy: {exc}", file=sys.stderr)
        return 1

    if args.dry_run:
        report = validate_bank(taxonomy=taxonomy)
        status = "valid" if report.ok else f"{len(report.errors)} error(s)"
        pending = len(report.unreviewed())
        print(
            f"dry run: {report.count} items across {len(report.files)} files - {status}; "
            f"{len(taxonomy.rows())} taxonomy rows; {pending} awaiting review"
        )
        for error in report.errors[:20]:
            print(f"  x {error}", file=sys.stderr)
        return 0 if report.ok else 1

    database_url = args.database_url
    if database_url is None:
        settings = Settings(_env_file=DOTENV if DOTENV.exists() else None)  # type: ignore[call-arg]
        database_url = settings.database_url

    return asyncio.run(_run(database_url, args.only_reviewed))


if __name__ == "__main__":
    raise SystemExit(main())
