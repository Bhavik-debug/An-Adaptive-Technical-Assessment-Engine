"""Phase 5 gate 3: "Killing the API mid-session and restarting rebuilds session
state from the event log."

A real OS-level kill: the API runs as its own process (``api_process.py``) and
is killed with ``Popen.kill()`` - TerminateProcess / SIGKILL, no shutdown hooks
- **while a grade is in flight** (claimed, lease held, model call under way;
the scripted model is slowed to 3 s so the kill lands inside it).  A new
process is then started on the same Postgres and Redis.

What must hold afterwards:

* the accepted turn is not lost: the new process re-claims it after the dead
  claim's lease expires and grades it - **once** (one ANSWER_GRADED, attempt 2);
* the log replays to a valid state that equals the service's view;
* the session is not stuck: the candidate answers the next question and it is
  graded too.

Leases are shortened for the test (timeout 10 s + margin 1 s) so the drill takes
seconds, not minutes; the recovery mechanism is unchanged.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Iterator

import httpx
import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker

from app.bank.ingest import ingest_bank
from app.bank.loader import validate_bank
from app.bank.paths import BANK_DIR
from app.bank.taxonomy import load_taxonomy
from app.interview import InterviewState, fold, load_state
from app.interview.store import read_events
from app.retrieval.embedders import HashingEmbedder
from tests.integration.api_process import ApiProcess, start_api

DRILL_ENV = {
    "LLM_STUB_SCRIPTED_DELAY_S": "3",
    "GRADING_TIMEOUT_S": "10",
    "GRADING_LEASE_MARGIN_S": "1",
}
STRONG = (
    "I would start from the access pattern and the data volume, pick the structure that "
    "makes the hot query cheap, and measure it in production before and after."
)


@pytest_asyncio.fixture
async def bank(db_engine: AsyncEngine):
    taxonomy = load_taxonomy()
    report = validate_bank(BANK_DIR, taxonomy)
    async with async_sessionmaker(db_engine, expire_on_commit=False)() as s:
        await ingest_bank(s, report, taxonomy, embedder=HashingEmbedder())
        await s.commit()


@pytest.fixture
def processes(tmp_path) -> Iterator[list[ApiProcess]]:
    started: list[ApiProcess] = []
    yield started
    for p in started:
        p.stop()


async def _turn(engine, turn_id):
    async with engine.connect() as conn:
        return (
            await conn.execute(
                text("SELECT grading_status, grading_attempts FROM turns WHERE turn_id = :t"),
                {"t": uuid.UUID(turn_id)},
            )
        ).one_or_none()


async def _wait(predicate, within: float, step: float = 0.1):
    async with asyncio.timeout(within):
        while not await predicate():  # noqa: ASYNC110 - polling a database
            await asyncio.sleep(step)


async def test_killing_the_api_mid_grade_loses_and_duplicates_nothing(
    migrated_database_url, redis_url, tmp_path, bank, db_engine, processes
):
    maker = async_sessionmaker(db_engine, expire_on_commit=False)
    first = start_api(migrated_database_url, redis_url, tmp_path, **DRILL_ENV)
    processes.append(first)
    async with httpx.AsyncClient(base_url=first.base_url, timeout=30) as http:
        reg = await http.post(
            "/api/auth/register", json={"email": "kill@example.com", "password": "correct horse 9"}
        )
        headers = {"Authorization": f"Bearer {reg.json()['access_token']}"}
        spec = {"role": "Backend", "topics": [{"topic_key": "dsa"}, {"topic_key": "databases"}]}
        sid = (await http.post("/api/sessions", json=spec, headers=headers)).json()["session_id"]
        await http.post(f"/api/sessions/{sid}/start", headers=headers)
        turn_id = str(uuid.uuid4())
        accepted = await http.post(
            f"/api/sessions/{sid}/turns",
            json={"turn_id": turn_id, "answer": STRONG},
            headers=headers,
        )
    assert accepted.status_code == 202

    # Wait until the grade is in flight, then kill the process.
    async def processing():
        row = await _turn(db_engine, turn_id)
        return row is not None and row.grading_status == "processing"

    await _wait(processing, within=15)
    exit_code = first.kill()
    assert exit_code != 0  # killed, not exited cleanly
    row = await _turn(db_engine, turn_id)
    assert (row.grading_status, row.grading_attempts) == ("processing", 1)
    async with maker() as db:
        events = await read_events(db, uuid.UUID(sid))
    assert "ANSWER_GRADED" not in [e.type for e in events]  # nothing half-committed

    # A brand-new process on the same database and Redis.
    second = start_api(migrated_database_url, redis_url, tmp_path, **DRILL_ENV)
    processes.append(second)

    async def graded():
        r = await _turn(db_engine, turn_id)
        return r.grading_status == "graded"

    await _wait(graded, within=45)
    row = await _turn(db_engine, turn_id)
    assert row.grading_attempts == 2  # the dead claim, then the recovery

    session_id = uuid.UUID(sid)
    async with maker() as db:
        events = await read_events(db, session_id)
        uid = await db.scalar(
            text("SELECT user_id FROM interview_sessions WHERE id = :s"), {"s": session_id}
        )
        rebuilt = fold(session_id, events)
        assert await load_state(db, None, session_id=session_id, user_id=uid) == rebuilt
    types = [e.type for e in events]
    assert types.count("ANSWER_SUBMITTED") == 1 and types.count("ANSWER_GRADED") == 1

    # Not stuck: wait for the next question (follow-up or next item) and answer it.
    async def asking():
        async with maker() as db:
            s = fold(session_id, await read_events(db, session_id))
        return s.state is InterviewState.ASKING

    await _wait(asking, within=30)
    next_turn = str(uuid.uuid4())
    async with httpx.AsyncClient(base_url=second.base_url, timeout=30) as http:
        r = await http.post(
            f"/api/sessions/{sid}/turns",
            json={"turn_id": next_turn, "answer": STRONG},
            headers=headers,
        )
    assert r.status_code == 202

    async def next_graded():
        row = await _turn(db_engine, next_turn)
        return row is not None and row.grading_status == "graded"

    await _wait(next_graded, within=30)
    async with maker() as db:
        final = await read_events(db, session_id)
    assert [e.type for e in final].count("ANSWER_GRADED") == 2
    fold(session_id, final)  # still a valid log
