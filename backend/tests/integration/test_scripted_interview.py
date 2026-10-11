"""Phase 5 gates 1 and 2, end to end: a scripted candidate drives a complete
interview through the public API of a real API *process*, and the database is
then audited against what the client saw.

Model: the scripted stub (``app/simulation/scripted_llm.py`` - read its rule).
Everything else is production code, in the deployment shape.
"""

from __future__ import annotations

import importlib.util
import uuid
from collections.abc import Iterator
from decimal import Decimal

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
from tests.integration.api_process import BACKEND_DIR, ApiProcess, start_api

_spec = importlib.util.spec_from_file_location(
    "scripted_interview", BACKEND_DIR / "scripts" / "scripted_interview.py"
)
assert _spec and _spec.loader
driver = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(driver)


@pytest_asyncio.fixture
async def bank(db_engine: AsyncEngine):
    taxonomy = load_taxonomy()
    report = validate_bank(BANK_DIR, taxonomy)
    async with async_sessionmaker(db_engine, expire_on_commit=False)() as s:
        await ingest_bank(s, report, taxonomy, embedder=HashingEmbedder())
        await s.commit()


@pytest.fixture
def api(migrated_database_url, redis_url, tmp_path, bank) -> Iterator[ApiProcess]:
    process = start_api(migrated_database_url, redis_url, tmp_path)
    try:
        yield process
    finally:
        process.stop()


async def test_a_complete_ten_question_interview_through_the_public_api(api, db_engine):
    evidence = await driver.run_interview(api.base_url)

    # --- what the client saw -------------------------------------------------
    assert evidence["primary_questions"] == 10, api.log_tail()
    assert 0 <= evidence["follow_ups"] <= 6
    report = evidence["report"]
    assert report["status"] == "ready"
    assert report["narrative_source"] in {"model", "template"}
    assert evidence["answer_after_completion"] == 409  # COMPLETED accepts nothing
    assert evidence["gate2"] == {
        "turn_id": evidence["turns"][0]["turn_id"],
        "sequential_retry": 200,
        "concurrent_retries": [200, 200, 200, 200],
        "different_answer_same_key": 409,
    }

    # --- the database, audited -----------------------------------------------
    sid = uuid.UUID(evidence["session_id"])
    maker = async_sessionmaker(db_engine, expire_on_commit=False)
    async with maker() as db:
        events = await read_events(db, sid)
    types = [e.type for e in events]
    turns = [t["turn_id"] for t in evidence["turns"]]

    # One submission event and one grade per logical turn - retries added none.
    submitted = [e.payload["turn_id"] for e in events if e.type == "ANSWER_SUBMITTED"]
    graded = [e.payload["turn_id"] for e in events if e.type == "ANSWER_GRADED"]
    assert submitted == turns and graded == turns
    assert types.count("GRADING_FAILED") == 0
    primaries = [
        e for e in events if e.type == "QUESTION_ASKED" and not e.payload.get("is_follow_up")
    ]
    assert len(primaries) == 10
    assert len({e.payload["question_id"] for e in primaries}) == 10  # no repeats
    assert types.count("FOLLOW_UP_STARTED") == evidence["follow_ups"] + types.count(
        "FOLLOW_UP_ABANDONED"
    )
    assert types[-2:] == ["REPORTING_STARTED", "SESSION_COMPLETED"]
    assert events[-2].payload["reasons"] == ["item_budget_reached"]

    # Replay from session creation reaches COMPLETED, and equals the service view.
    state = fold(sid, events)
    assert state.state is InterviewState.COMPLETED
    async with maker() as db:
        uid = await db.scalar(
            text("SELECT user_id FROM interview_sessions WHERE id=:s"), {"s": sid}
        )
        assert await load_state(db, None, session_id=sid, user_id=uid) == state

    # Exactly one report, linked from the terminal event, built from the log.
    async with db_engine.connect() as conn:
        rows = (
            await conn.execute(
                text("SELECT id, stats, total_tokens FROM reports WHERE session_id=:s"), {"s": sid}
            )
        ).all()
        session_row = (
            await conn.execute(
                text("SELECT state, total_tokens, cost_usd FROM interview_sessions WHERE id=:s"),
                {"s": sid},
            )
        ).one()
        grade_tokens = await conn.scalar(
            text(
                "SELECT coalesce(sum((c->>'input_tokens')::int + (c->>'output_tokens')::int), 0) "
                "FROM turns t, jsonb_array_elements(t.grade->'calls') c WHERE t.session_id=:s"
            ),
            {"s": sid},
        )
        attempts = (
            (
                await conn.execute(
                    text("SELECT grading_attempts FROM turns WHERE session_id=:s"), {"s": sid}
                )
            )
            .scalars()
            .all()
        )
    assert len(rows) == 1
    assert str(rows[0].id) == events[-1].payload["report_id"] == report["report_id"]
    assert session_row.state == "COMPLETED"
    assert len(attempts) == len(turns) and set(attempts) == {1}

    stats = rows[0].stats
    assert stats["counts"]["primary_questions"] == 10
    assert stats["counts"]["answers_graded"] == len(turns)
    assert stats["log_seq"] == events[-2].seq
    scores = {e.payload["turn_id"]: e.payload["score"] for e in events if e.type == "ANSWER_GRADED"}
    primary_turns = [t["turn_id"] for t in evidence["turns"] if not t["follow_up"]]
    assert [q["score"] for q in stats["questions"]] == [round(scores[t], 4) for t in primary_turns]

    # Accounting: each committed grade once, plus the report call once.
    assert session_row.total_tokens == grade_tokens + rows[0].total_tokens
    assert session_row.cost_usd == Decimal(session_row.cost_usd)  # NUMERIC, not float

    # The report reaches the client exactly as stored.
    assert report["stats"] == stats


async def test_the_report_is_deterministic_from_the_log(api, db_engine):
    """Rebuilding the statistics from the committed log gives the stored stats."""
    from app.blueprint import Blueprint
    from app.reporting.stats import QuestionInfo, aggregate

    evidence = await driver.run_interview(api.base_url)
    sid = uuid.UUID(evidence["session_id"])
    maker = async_sessionmaker(db_engine, expire_on_commit=False)
    async with maker() as db:
        events = await read_events(db, sid)
        bp = await db.scalar(
            text("SELECT blueprint FROM interview_sessions WHERE id=:s"), {"s": sid}
        )
        stored = await db.scalar(text("SELECT stats FROM reports WHERE session_id=:s"), {"s": sid})
        rows = (await db.execute(text("SELECT id, topic_key, subtopic_key FROM questions"))).all()
    questions = {r.id: QuestionInfo(r.topic_key, r.subtopic_key) for r in rows}
    upto = [e for e in events if e.seq <= stored["log_seq"]]
    rebuilt = [
        aggregate(
            events=upto, session_id=sid, blueprint=Blueprint.from_dict(bp), questions=questions
        )
        for _ in range(3)
    ]
    assert all(r == stored for r in rebuilt)
