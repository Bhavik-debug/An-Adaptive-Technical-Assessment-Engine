"""The report job and route against real Postgres: once, bounded, truthful."""

from __future__ import annotations

import asyncio
import uuid
from decimal import Decimal

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from app.interview import EventType, InterviewState, append_event, fold, load_state
from app.interview.grading import RetryPolicy
from app.interview.store import read_events
from app.interview.turns import submit_turn
from app.llm.errors import ProviderAuthError, ProviderTimeoutError
from app.reporting.jobs import (
    ReportOutcome,
    claim_next_report,
    ensure_report_job,
    process_report,
    requeue_failed_report,
    run_once_report,
)
from app.reporting.narrative import NarrativeRejected, ReportNarrative
from tests.integration.interview_support import (
    insert_question,
    session_in_asking,
    user,
)

FAST = RetryPolicy(max_attempts=3, base_delay_s=0.01, max_delay_s=0.02, timeout_s=5.0)
GOOD = ReportNarrative(summary="The assessment covered the planned topics; see the figures below.")


@pytest_asyncio.fixture
async def maker(db_engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(db_engine, expire_on_commit=False)


async def reporting_session(maker, db_engine, email="r@example.com"):
    """A session driven to REPORTING through the service layer, with its job."""
    await insert_question(db_engine, f"q-{email}")
    uid = await user(db_engine, email)
    sid = await session_in_asking(maker, uid, f"q-{email}")
    turn = uuid.uuid4()
    await submit_turn(
        maker, None, session_id=sid, user_id=uid, turn_id=turn, answer_text="An answer."
    )
    for event_type, payload in (
        (EventType.ANSWER_GRADED, {"turn_id": str(turn), "score": 0.5}),
        (EventType.REPORTING_STARTED, {"reasons": ["item_budget_reached"]}),
    ):
        async with maker() as db:
            await append_event(
                db, None, session_id=sid, user_id=uid, event_type=event_type, payload=payload
            )
    async with maker() as db:
        await ensure_report_job(db, sid)
        await db.commit()
    return uid, sid


class Narrator:
    def __init__(self, *script):
        self.script, self.calls = list(script), 0

    async def __call__(self, stats, job):
        self.calls += 1
        step = self.script.pop(0) if len(self.script) > 1 else self.script[0]
        if isinstance(step, BaseException):
            raise step
        from app.llm.client import CallMeta
        from app.llm.tasks import TaskName
        from app.llm.types import ModelTier

        meta = CallMeta(
            task=TaskName.FINAL_REPORT,
            prompt_version="v1",
            prompt_fingerprint="f",
            schema_fingerprint="s",
            provider="fake",
            model="fake",
            tier=ModelTier.MID,
            temperature=0.3,
            input_tokens=100,
            output_tokens=40,
            reasoning_tokens=0,
            cost_usd=Decimal("0.000010"),
            price_known=True,
            cache_hit=False,
            schema_retry_count=0,
            failover_count=0,
            structured_mode="fake",
            reasoning_enabled=False,
            latency_ms=1,
        )
        return step, meta


async def _report(db_engine, sid):
    async with db_engine.connect() as conn:
        return (
            await conn.execute(
                text(
                    "SELECT id, status, attempts, narrative_source, narrative_error, error, "
                    "stats IS NOT NULL AS has_stats, total_tokens FROM reports WHERE session_id=:s"
                ),
                {"s": sid},
            )
        ).all()


async def _events(maker, sid):
    async with maker() as db:
        return await read_events(db, sid)


# ---------------------------------------------------------------------------
# The job
# ---------------------------------------------------------------------------


async def test_completion_is_one_transaction(maker, db_engine):
    uid, sid = await reporting_session(maker, db_engine)
    assert await run_once_report(maker, None, narrative_fn=Narrator(GOOD), policy=FAST) == (
        ReportOutcome.COMPLETED
    )
    [row] = await _report(db_engine, sid)
    assert (row.status, row.narrative_source, row.has_stats, row.total_tokens) == (
        "ready",
        "model",
        True,
        140,
    )
    events = await _events(maker, sid)
    assert events[-1].type == "SESSION_COMPLETED"
    assert events[-1].payload == {"report_id": str(row.id)}
    async with maker() as db:
        state = await load_state(db, None, session_id=sid, user_id=uid)
    assert state.state is InterviewState.COMPLETED == fold(sid, events).state
    async with db_engine.connect() as conn:
        tokens = await conn.scalar(
            text("SELECT total_tokens FROM interview_sessions WHERE id=:s"), {"s": sid}
        )
    assert tokens == 140  # the report call, once (the grade here was hand-written)


async def test_concurrent_workers_complete_once(maker, db_engine):
    _, sid = await reporting_session(maker, db_engine)
    narrator = Narrator(GOOD)
    outcomes = await asyncio.gather(
        *(run_once_report(maker, None, narrative_fn=narrator, policy=FAST) for _ in range(5))
    )
    assert sorted(o or "" for o in outcomes) == ["", "", "", "", ReportOutcome.COMPLETED]
    assert narrator.calls == 1
    assert len(await _report(db_engine, sid)) == 1
    assert [e.type for e in await _events(maker, sid)].count("SESSION_COMPLETED") == 1


async def test_duplicate_delivery_is_fenced(maker, db_engine):
    _, sid = await reporting_session(maker, db_engine)
    job = await claim_next_report(maker, FAST)
    first = await process_report(maker, None, job, narrative_fn=Narrator(GOOD), policy=FAST)
    second = await process_report(maker, None, job, narrative_fn=Narrator(GOOD), policy=FAST)
    assert (first, second) == (ReportOutcome.COMPLETED, ReportOutcome.STALE)
    async with db_engine.connect() as conn:
        tokens = await conn.scalar(
            text("SELECT total_tokens FROM interview_sessions WHERE id=:s"), {"s": sid}
        )
    assert tokens == 140  # charged once


async def test_transient_then_success(maker, db_engine):
    _, sid = await reporting_session(maker, db_engine)
    narrator = Narrator(ProviderTimeoutError("slow", provider="x"), GOOD)
    assert await run_once_report(maker, None, narrative_fn=narrator, policy=FAST) == (
        ReportOutcome.RETRY_SCHEDULED
    )
    await asyncio.sleep(0.05)
    assert await run_once_report(maker, None, narrative_fn=narrator, policy=FAST) == (
        ReportOutcome.COMPLETED
    )
    [row] = await _report(db_engine, sid)
    assert (row.narrative_source, row.attempts) == ("model", 2)


@pytest.mark.parametrize(
    ("failure", "code"),
    [
        (ProviderAuthError("401", provider="x"), "provider_auth"),
        (NarrativeRejected("invented_number"), "narrative_invented_number"),
        (ValueError("boom"), "internal_error"),
    ],
)
async def test_permanent_failures_complete_with_the_labelled_template(
    maker, db_engine, failure, code
):
    _, sid = await reporting_session(maker, db_engine)
    narrator = Narrator(failure)
    assert await run_once_report(maker, None, narrative_fn=narrator, policy=FAST) == (
        ReportOutcome.COMPLETED
    )
    [row] = await _report(db_engine, sid)
    assert (row.status, row.narrative_source, row.narrative_error) == ("ready", "template", code)
    assert row.total_tokens == 0  # no model narrative was kept, none is charged
    assert narrator.calls == 1


async def test_transient_failures_are_bounded_then_template(maker, db_engine):
    _, sid = await reporting_session(maker, db_engine)
    narrator = Narrator(ProviderTimeoutError("slow", provider="x"))
    outcomes = []
    for _ in range(10):
        o = await run_once_report(maker, None, narrative_fn=narrator, policy=FAST)
        if o is None:
            await asyncio.sleep(0.03)
            continue
        outcomes.append(o)
        if o == ReportOutcome.COMPLETED:
            break
    assert outcomes == [ReportOutcome.RETRY_SCHEDULED] * 2 + [ReportOutcome.COMPLETED]
    assert narrator.calls == FAST.max_attempts
    [row] = await _report(db_engine, sid)
    assert (row.narrative_source, row.narrative_error) == ("template", "retry_exhausted")


async def test_a_crash_looping_report_does_not_call_the_model_again(maker, db_engine):
    _, sid = await reporting_session(maker, db_engine)
    async with db_engine.begin() as conn:
        await conn.execute(
            text(
                "UPDATE reports SET status='processing', attempts=3, "
                "lease_expires_at = now() - interval '1 second' WHERE session_id=:s"
            ),
            {"s": sid},
        )
    narrator = Narrator(GOOD)
    assert await run_once_report(maker, None, narrative_fn=narrator, policy=FAST) == (
        ReportOutcome.COMPLETED
    )
    assert narrator.calls == 0
    [row] = await _report(db_engine, sid)
    assert row.narrative_source == "template"


async def test_statistics_unavailable_means_failed_not_completed(maker, db_engine):
    """A report job for a session not in REPORTING: failed with a code; the
    session is NOT completed; an operator can requeue."""
    await insert_question(db_engine, "q-x")
    uid = await user(db_engine, "x@example.com")
    sid = await session_in_asking(maker, uid, "q-x")
    async with maker() as db:
        await ensure_report_job(db, sid)
        await db.commit()
    assert await run_once_report(maker, None, narrative_fn=Narrator(GOOD), policy=FAST) == (
        ReportOutcome.FAILED
    )
    [row] = await _report(db_engine, sid)
    assert (row.status, row.error) == ("failed", "not_reporting")
    assert (await _events(maker, sid))[-1].type == "QUESTION_ASKED"
    assert await requeue_failed_report(maker, session_id=sid) is True


async def test_an_unexpected_stats_error_fails_the_report_instead_of_looping(
    maker, db_engine, monkeypatch
):
    """Not ReportNotReady/EventLogCorrupted: previously the exception escaped to
    the worker loop, the row stayed `processing`, and every lease re-claimed it,
    forever. Now it is a failed report an operator can re-queue."""
    import app.reporting.jobs as jobs

    _, sid = await reporting_session(maker, db_engine)

    async def boom(*_args, **_kwargs):
        raise RuntimeError("unexpected")

    monkeypatch.setattr(jobs, "build_stats", boom)
    assert await run_once_report(maker, None, narrative_fn=Narrator(GOOD), policy=FAST) == (
        ReportOutcome.FAILED
    )
    [row] = await _report(db_engine, sid)
    assert (row.status, row.error) == ("failed", "internal_error")
    # The session is not falsely completed, and the report can be retried.
    assert (await _events(maker, sid))[-1].type != "SESSION_COMPLETED"
    assert await requeue_failed_report(maker, session_id=sid) is True


async def test_the_requeue_script_lists_and_requeues_a_failed_report(
    maker, db_engine, migrated_database_url, monkeypatch
):
    """scripts/requeue_failed.py, run as an operator would: lists the failed
    report by id and code only, re-queues it once, and refuses a second time."""
    import subprocess
    import sys
    from pathlib import Path

    import app.reporting.jobs as jobs

    _, sid = await reporting_session(maker, db_engine)

    async def boom(*_args, **_kwargs):
        raise RuntimeError("unexpected")

    monkeypatch.setattr(jobs, "build_stats", boom)
    await run_once_report(maker, None, narrative_fn=Narrator(GOOD), policy=FAST)

    backend = Path(__file__).resolve().parents[2]

    def script(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(  # noqa: S603 - fixed argv, no shell
            [sys.executable, "scripts/requeue_failed.py", "--database-url", migrated_database_url]
            + list(args),
            cwd=backend,
            capture_output=True,
            text=True,
            timeout=60,
        )

    listed = await asyncio.to_thread(script, "--list")
    assert listed.returncode == 0, listed.stderr
    assert "failed reports: 1" in listed.stdout
    assert f"report session={sid} error=internal_error" in listed.stdout
    assert "An answer." not in listed.stdout  # ids and codes only, never answer text

    requeued = await asyncio.to_thread(script, "report", str(sid))
    assert (requeued.returncode, requeued.stdout.strip()) == (0, "re-queued")
    [row] = await _report(db_engine, sid)
    assert row.status == "pending"

    again = await asyncio.to_thread(script, "report", str(sid))
    assert again.returncode == 1
    assert "nothing to re-queue" in again.stdout


async def test_the_ready_constraint_refuses_a_half_written_report(db_engine, maker):
    _, sid = await reporting_session(maker, db_engine)
    from sqlalchemy.exc import DBAPIError

    with pytest.raises(DBAPIError, match="ck_reports_ready_has_content"):
        async with db_engine.begin() as conn:
            await conn.execute(
                text("UPDATE reports SET status='ready' WHERE session_id=:s"), {"s": sid}
            )


async def test_one_report_per_session(db_engine, maker):
    _, sid = await reporting_session(maker, db_engine)
    for _ in range(3):
        async with maker() as db:
            await ensure_report_job(db, sid)
            await db.commit()
    assert len(await _report(db_engine, sid)) == 1


# ---------------------------------------------------------------------------
# The route
# ---------------------------------------------------------------------------


async def _register(client, engine, email):
    r = await client.post(
        "/api/auth/register", json={"email": email, "password": "correct horse 9"}
    )
    async with engine.connect() as conn:
        uid = await conn.scalar(text("SELECT id FROM users WHERE email=:e"), {"e": email})
    return uid, {"Authorization": f"Bearer {r.json()['access_token']}"}


async def test_report_route_lifecycle_and_ownership(client, db_engine, maker):
    uid, headers = await _register(client, db_engine, "owner@example.com")
    await insert_question(db_engine, "q-route")
    sid = await session_in_asking(maker, uid, "q-route")
    url = f"/api/sessions/{sid}/report"
    assert (await client.get(url, headers=headers)).status_code == 409  # not finished
    turn = uuid.uuid4()
    await submit_turn(maker, None, session_id=sid, user_id=uid, turn_id=turn, answer_text="x")
    for et, p in (
        (EventType.ANSWER_GRADED, {"turn_id": str(turn), "score": 0.5}),
        (EventType.REPORTING_STARTED, {"reasons": ["item_budget_reached"]}),
    ):
        async with maker() as db:
            await append_event(db, None, session_id=sid, user_id=uid, event_type=et, payload=p)

    # Concurrent GETs while REPORTING: 202 each, and exactly one job row.
    responses = await asyncio.gather(*(client.get(url, headers=headers) for _ in range(5)))
    assert {r.status_code for r in responses} == {202}
    assert len(await _report(db_engine, sid)) == 1

    _, intruder = await _register(client, db_engine, "intruder@example.com")
    assert (await client.get(url, headers=intruder)).status_code == 404
    assert (await client.get(url)).status_code == 401

    await run_once_report(maker, None, narrative_fn=Narrator(GOOD), policy=FAST)
    ready = await client.get(url, headers=headers)
    assert ready.status_code == 200
    body = ready.json()
    assert body["status"] == "ready" and body["narrative_source"] == "model"
    assert body["stats"]["counts"]["answers_graded"] == 1
    assert "An answer" not in str(body)  # answer text is never in a report
    again = await client.get(url, headers=headers)
    assert again.json() == body  # idempotent read

    # COMPLETED accepts no new answer.
    late = await client.post(
        f"/api/sessions/{sid}/turns",
        json={"turn_id": str(uuid.uuid4()), "answer": "late"},
        headers=headers,
    )
    assert late.status_code == 409


async def test_report_route_failed_is_500_with_the_typed_body(client, db_engine, maker):
    """The report screen (Day 28) tells "failed" from any other server error by
    the body: a failed report is a 500 whose body is the report, status
    "failed", with no stats or narrative to show. Ownership still applies."""
    uid, headers = await _register(client, db_engine, "failed-owner@example.com")
    await insert_question(db_engine, "q-failed")
    sid = await session_in_asking(maker, uid, "q-failed")
    turn = uuid.uuid4()
    await submit_turn(maker, None, session_id=sid, user_id=uid, turn_id=turn, answer_text="x")
    for et, p in (
        (EventType.ANSWER_GRADED, {"turn_id": str(turn), "score": 0.5}),
        (EventType.REPORTING_STARTED, {"reasons": ["item_budget_reached"]}),
    ):
        async with maker() as db:
            await append_event(db, None, session_id=sid, user_id=uid, event_type=et, payload=p)
    url = f"/api/sessions/{sid}/report"
    assert (await client.get(url, headers=headers)).status_code == 202  # job created
    async with db_engine.begin() as conn:
        await conn.execute(
            text(
                "UPDATE reports SET status='failed', error='stats_unavailable' "
                "WHERE session_id=:s"
            ),
            {"s": sid},
        )

    failed = await client.get(url, headers=headers)
    assert failed.status_code == 500
    body = failed.json()
    assert body["status"] == "failed" and body["error"] == "stats_unavailable"
    assert body["stats"] is None and body["narrative"] is None

    _, intruder = await _register(client, db_engine, "failed-intruder@example.com")
    assert (await client.get(url, headers=intruder)).status_code == 404


async def test_sse_announces_completion_with_the_report_id(maker, db_engine):
    from app.interview.stream import event_stream

    uid, sid = await reporting_session(maker, db_engine)
    await run_once_report(maker, None, narrative_fn=Narrator(GOOD), policy=FAST)
    chunks = [
        c
        async for c in event_stream(
            maker,
            session_id=sid,
            user_id=uid,
            last_event_id=-1,
            poll_interval_s=0.05,
            keepalive_s=60,
            is_disconnected=lambda: asyncio.sleep(0, False),
        )
    ]  # the stream ends by itself after session_complete
    last = chunks[-1]
    assert "event: session_complete" in last
    [row] = await _report(db_engine, sid)
    assert str(row.id) in last and "COMPLETED" in last


async def test_a_stale_report_worker_cannot_complete_over_the_new_claim(maker, db_engine):
    """A's lease expires and B re-claims; A finishes first. The FSM alone would
    still allow exactly one completion - but it must be the *current* owner's.
    (Without the attempt fence, A's stale attempt would complete the report.)"""
    _, sid = await reporting_session(maker, db_engine)
    job_a = await claim_next_report(maker, FAST)
    async with db_engine.begin() as conn:
        await conn.execute(
            text("UPDATE reports SET lease_expires_at = now() - interval '1 second'")
        )
    job_b = await claim_next_report(maker, FAST)
    assert (job_a.attempt, job_b.attempt) == (1, 2)
    assert await process_report(maker, None, job_a, narrative_fn=Narrator(GOOD), policy=FAST) == (
        ReportOutcome.STALE
    )
    assert [e.type for e in await _events(maker, sid)][-1] == "REPORTING_STARTED"
    assert await process_report(maker, None, job_b, narrative_fn=Narrator(GOOD), policy=FAST) == (
        ReportOutcome.COMPLETED
    )
    [row] = await _report(db_engine, sid)
    assert (row.status, row.attempts) == ("ready", 2)
