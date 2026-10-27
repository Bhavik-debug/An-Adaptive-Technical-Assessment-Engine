"""Operator tool: list and re-queue failed grading turns and failed reports.

A turn whose grading failed (``GRADING_FAILED``) leaves its session in
``GRADING``, and a report whose statistics could not be built leaves its
session in ``REPORTING``. Neither is retried on its own, by design: the cause
(an outage, a bad key) has to be fixed first. This is the documented operator
action (docs/turn-api.md, docs/reporting.md) as a command, using the same
tested functions as the test suite:
``app.interview.grading.requeue_failed_turn`` and
``app.reporting.jobs.requeue_failed_report``. The running API's background
worker then picks the job up.

Prints ids, error codes and timestamps only - never answer text.

    cd backend
    # From the host, point at the compose Postgres (the .env URL names the
    # compose service, which only resolves inside Docker):
    URL=postgresql+asyncpg://USER:PASS@localhost:5433/DB
    python scripts/requeue_failed.py --database-url $URL --list
    python scripts/requeue_failed.py --database-url $URL turn <session_id> <turn_id>
    python scripts/requeue_failed.py --database-url $URL report <session_id>

On Windows PowerShell, set the variable with ``$URL = "..."`` and run the
venv's ``..\\.venv\\Scripts\\python`` instead (docs/local-development.md).
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import uuid
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:  # so `python scripts/...` works without install
    sys.path.insert(0, str(BACKEND))

import sqlalchemy as sa  # noqa: E402
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine  # noqa: E402

from app.config import DatabaseSettings  # noqa: E402
from app.interview.grading import requeue_failed_turn  # noqa: E402
from app.models import Turn  # noqa: E402
from app.models.interview import GRADING_FAILED  # noqa: E402
from app.models.reporting import REPORT_FAILED, Report  # noqa: E402
from app.reporting.jobs import requeue_failed_report  # noqa: E402

DOTENV = BACKEND.parent / ".env"


async def _run(database_url: str, args: argparse.Namespace) -> int:
    engine = create_async_engine(database_url, hide_parameters=True)
    maker = async_sessionmaker(engine, expire_on_commit=False)
    try:
        if args.command == "list":
            async with maker() as db:
                turns = (
                    await db.execute(
                        sa.select(
                            Turn.session_id, Turn.turn_id, Turn.grading_error, Turn.created_at
                        ).where(Turn.grading_status == GRADING_FAILED)
                    )
                ).all()
                reports = (
                    await db.execute(
                        sa.select(Report.session_id, Report.error).where(
                            Report.status == REPORT_FAILED
                        )
                    )
                ).all()
            print(f"failed turns: {len(turns)}")
            for t in turns:
                print(f"  turn  session={t.session_id} turn={t.turn_id} error={t.grading_error}")
            print(f"failed reports: {len(reports)}")
            for r in reports:
                print(f"  report session={r.session_id} error={r.error}")
            return 0
        if args.command == "turn":
            ok = await requeue_failed_turn(
                maker, session_id=uuid.UUID(args.session_id), turn_id=uuid.UUID(args.turn_id)
            )
        else:
            ok = await requeue_failed_report(maker, session_id=uuid.UUID(args.session_id))
        print("re-queued" if ok else "nothing to re-queue (not found, or not in a failed state)")
        return 0 if ok else 1
    finally:
        await engine.dispose()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--database-url", help="default: DATABASE_URL from the root .env")
    parser.add_argument("--list", action="store_true", help="list failed turns and reports")
    sub = parser.add_subparsers(dest="command")
    turn = sub.add_parser("turn", help="re-queue one failed turn")
    turn.add_argument("session_id")
    turn.add_argument("turn_id")
    report = sub.add_parser("report", help="re-queue one failed report")
    report.add_argument("session_id")
    args = parser.parse_args()
    if args.list:
        args.command = "list"
    if not args.command:
        parser.print_help()
        return 2
    database_url = (
        args.database_url
        or DatabaseSettings(
            _env_file=DOTENV if DOTENV.exists() else None  # type: ignore[call-arg]
        ).database_url
    )
    return asyncio.run(_run(database_url, args))


if __name__ == "__main__":
    sys.exit(main())
