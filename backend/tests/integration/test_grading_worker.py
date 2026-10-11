"""The grading worker against real Postgres: once, bounded, fenced, recoverable.

The "grader" in most tests is scripted - it returns a real ``GradeResult`` (built
once by the real pipeline replaying the committed Day-18 recording) or raises a
real exception from the LLM error taxonomy.  The last section runs the real
``grade_answer`` inside the real app's worker, end to end, offline.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator
from decimal import Decimal

import pytest
import pytest_asyncio
from asgi_lifespan import LifespanManager
from httpx import ASGITransport, AsyncClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from app.bank.schema import ExpectedConcept
from app.config import get_settings
from app.grading.grader import GradingContractError, GradingInputError
from app.grading.pipeline import GradeResult, grade_answer
from app.interview import EventType, InterviewState, fold, load_state
from app.interview.grading import (
    GradingJob,
    Outcome,
    RetryPolicy,
    charge_for,
    claim_next,
    classify_error,
    process_job,
    requeue_failed_turn,
    run_once,
)
from app.interview.store import read_events
from app.interview.turns import submit_turn
from app.llm.errors import (
    AllProvidersFailedError,
    ProviderAuthError,
    ProviderRateLimitedError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    SchemaValidationFailedError,
)
from tests.integration.interview_support import (
    RECORDED,
    STRONG_ANSWER,
    SUBTOPIC,
    event_types,
    insert_question,
    session_cost,
    session_in_asking,
    turn_rows,
    user,
)
from tests.unit.grading.test_recorded_grading import replay_router
from tests.unit.llm.conftest import MemoryCache

FAST = RetryPolicy(max_attempts=3, base_delay_s=0.01, max_delay_s=0.05, timeout_s=5.0)


@pytest_asyncio.fixture
async def maker(db_engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(db_engine, expire_on_commit=False)


@pytest_asyncio.fixture
async def real_result(env) -> GradeResult:
    """One real grade: the real pipeline over the committed recording."""
    env()
    inputs = RECORDED["grade_answer_strong"]
    concepts = [
        ExpectedConcept(key=line[2:].split(": ", 1)[0], weight=1, hint=line[2:].split(": ", 1)[1])
        for line in inputs["expected_concepts"].splitlines()
    ]
    return await grade_answer(
        question_text=inputs["question"],
        expected_concepts=concepts,
        answer_text=inputs["answer"],
        common_misconceptions=[m[2:] for m in inputs["misconceptions"].splitlines()],
        router=replay_router(),
        cache=MemoryCache(),
        settings=get_settings(),
    )


@pytest_asyncio.fixture
async def submitted(maker, db_engine):
    """A session in GRADING with one pending turn."""
    await insert_question(db_engine)
    uid = await user(db_engine)
    sid = await session_in_asking(maker, uid)
    turn_id = uuid.uuid4()
    await submit_turn(
        maker, None, session_id=sid, user_id=uid, turn_id=turn_id, answer_text=STRONG_ANSWER
    )
    return uid, sid, turn_id


class Scripted:
    """A grader that plays a script: each entry is a result or an exception."""

    def __init__(self, *script):
        self.script = list(script)
        self.calls: list[GradingJob] = []

    async def __call__(self, job: GradingJob) -> GradeResult:
        self.calls.append(job)
        step = self.script.pop(0) if len(self.script) > 1 else self.script[0]
        if isinstance(step, BaseException):
            raise step
        return step


async def _state(maker, uid, sid):
    async with maker() as db:
        return await load_state(db, None, session_id=sid, user_id=uid)


async def _make_claimable(engine: AsyncEngine, sid) -> None:
    async with engine.begin() as conn:
        await conn.execute(
            text("UPDATE turns SET next_attempt_at = now() WHERE session_id = :s"), {"s": sid}
        )


# ---------------------------------------------------------------------------
# Success
# ---------------------------------------------------------------------------


async def test_a_grade_is_committed_once_with_its_event_ability_and_cost(
    maker, db_engine, submitted, real_result
):
    uid, sid, turn_id = submitted
    grader = Scripted(real_result)
    assert await run_once(maker, None, grade_fn=grader, policy=FAST) == Outcome.GRADED

    rows = await turn_rows(db_engine, sid)
    assert rows[0]["grading_status"] == "graded"
    assert rows[0]["score"] == pytest.approx(real_result.score)
    assert rows[0]["grading_attempts"] == 1 and rows[0]["graded_at"] is not None
    assert (await event_types(db_engine, sid)).count("ANSWER_GRADED") == 1

    state = await _state(maker, uid, sid)
    assert state.ability[SUBTOPIC].n_observations == 1
    # Day 23: the decision is committed with the grade. After a subtopic's first
    # observation RD is still > 0.7, so plan 8.5's high_uncertainty rule fires
    # and the session moves to PROBING with a pending wording job.
    assert state.state is InterviewState.PROBING
    assert state.ability[SUBTOPIC].rd > 0.7
    async with db_engine.connect() as conn:
        decision = await conn.scalar(
            text("SELECT policy_decision FROM turns WHERE session_id = :s"), {"s": sid}
        )
        jobs = (
            await conn.execute(
                text("SELECT status, ordinal FROM follow_ups WHERE session_id = :s"), {"s": sid}
            )
        ).all()
    assert decision["follow_up"] is True and decision["reason"] == "high_uncertainty"
    assert [tuple(j) for j in jobs] == [("pending", 1)]

    cost, tokens = charge_for(real_result)
    assert tokens > 0
    assert await session_cost(db_engine, sid) == (cost.quantize(Decimal("0.000001")), tokens)

    # The grader received the DB question, misconceptions included (migration 0004).
    job = grader.calls[0]
    assert job.common_misconceptions and len(job.expected_concepts) == 3
    assert job.answer_text == STRONG_ANSWER


async def test_the_stored_grade_has_evidence_but_no_prompt(
    maker, db_engine, submitted, real_result
):
    _, sid, _ = submitted
    await run_once(maker, None, grade_fn=Scripted(real_result), policy=FAST)
    async with db_engine.connect() as conn:
        grade = await conn.scalar(text("SELECT grade FROM turns WHERE session_id=:s"), {"s": sid})
    assert set(grade) == {"score", "judgment", "breakdown", "gate", "calls"}
    assert "question" not in str(grade["calls"]) and "prompt_version" in grade["calls"][0]


async def test_an_empty_queue_claims_nothing(maker, db_engine):
    assert await claim_next(maker, FAST) is None


# ---------------------------------------------------------------------------
# Duplicate delivery and concurrency
# ---------------------------------------------------------------------------


async def test_duplicate_delivery_commits_one_grade(maker, db_engine, submitted, real_result):
    uid, sid, _ = submitted
    job = await claim_next(maker, FAST)
    grader = Scripted(real_result)
    first = await process_job(maker, None, job, grade_fn=grader, policy=FAST)
    second = await process_job(maker, None, job, grade_fn=grader, policy=FAST)
    assert (first, second) == (Outcome.GRADED, Outcome.STALE)
    # The provider was asked twice - that is the honest cost of at-least-once
    # delivery - but only one grade and one charge were committed.
    assert len(grader.calls) == 2
    assert (await event_types(db_engine, sid)).count("ANSWER_GRADED") == 1
    cost, tokens = charge_for(real_result)
    assert (await session_cost(db_engine, sid))[1] == tokens
    assert (await _state(maker, uid, sid)).ability[SUBTOPIC].n_observations == 1


async def test_competing_workers_claim_a_turn_once(maker, db_engine, submitted, real_result):
    _, sid, _ = submitted
    grader = Scripted(real_result)
    outcomes = await asyncio.gather(
        *(run_once(maker, None, grade_fn=grader, policy=FAST) for _ in range(6))
    )
    assert sorted(o or "" for o in outcomes) == ["", "", "", "", "", Outcome.GRADED]
    assert len(grader.calls) == 1
    assert (await event_types(db_engine, sid)).count("ANSWER_GRADED") == 1


async def test_accounting_is_applied_at_most_once(
    maker, db_engine, submitted, real_result, monkeypatch
):
    """TEST INSTRUMENTATION, not a real charge: counts how often the success
    path *applies* accounting, across a retry, a duplicate delivery and a
    competing worker. This codebase's only accounting is the session's
    cost_usd/total_tokens (see docs/turn-api.md §6)."""
    from app.interview import grading

    applied = {"n": 0}
    real_charge = grading.charge_for

    def counting(result):
        applied["n"] += 1
        return real_charge(result)

    monkeypatch.setattr(grading, "charge_for", counting)
    _, sid, _ = submitted
    grader = Scripted(ProviderTimeoutError("slow", provider="stub"), real_result)
    assert await run_once(maker, None, grade_fn=grader, policy=FAST) == Outcome.RETRY_SCHEDULED
    await _make_claimable(db_engine, sid)
    job = await claim_next(maker, FAST)
    await asyncio.gather(
        process_job(maker, None, job, grade_fn=grader, policy=FAST),
        process_job(maker, None, job, grade_fn=grader, policy=FAST),
    )
    assert applied["n"] == 1
    assert (await session_cost(db_engine, sid))[1] == charge_for(real_result)[1]


# ---------------------------------------------------------------------------
# Retries and failures
# ---------------------------------------------------------------------------


async def test_transient_then_success(maker, db_engine, submitted, real_result):
    uid, sid, _ = submitted
    policy = RetryPolicy(max_attempts=3, base_delay_s=0.3, max_delay_s=1.0, timeout_s=5.0)
    grader = Scripted(ProviderRateLimitedError("429", provider="stub"), real_result)

    assert await run_once(maker, None, grade_fn=grader, policy=policy) == Outcome.RETRY_SCHEDULED
    rows = await turn_rows(db_engine, sid)
    assert rows[0]["grading_status"] == "pending"
    assert rows[0]["grading_error"] == "provider_ratelimited"
    # Backoff is real: the turn is not claimable until next_attempt_at.
    assert await run_once(maker, None, grade_fn=grader, policy=policy) is None
    await asyncio.sleep(0.4)
    assert await run_once(maker, None, grade_fn=grader, policy=policy) == Outcome.GRADED
    rows = await turn_rows(db_engine, sid)
    assert rows[0]["grading_attempts"] == 2 and rows[0]["grading_error"] is None
    assert "GRADING_FAILED" not in await event_types(db_engine, sid)


@pytest.mark.parametrize(
    ("error", "code"),
    [
        (GradingContractError("wrong keys"), "grader_contract_violation"),
        (GradingInputError("empty"), "ungradeable_input"),
        (ProviderAuthError("401", provider="stub"), "provider_auth"),
        (
            SchemaValidationFailedError(attempts=3, last_error="x", raw_output="y"),
            "schema_validation_failed",
        ),
        (KeyError("surprise"), "internal_error"),
    ],
)
async def test_permanent_errors_fail_after_one_attempt(maker, db_engine, submitted, error, code):
    uid, sid, turn_id = submitted
    grader = Scripted(error)
    assert await run_once(maker, None, grade_fn=grader, policy=FAST) == Outcome.FAILED
    assert len(grader.calls) == 1
    rows = await turn_rows(db_engine, sid)
    assert (rows[0]["grading_status"], rows[0]["grading_error"]) == ("failed", code)
    assert rows[0]["score"] is None
    assert await run_once(maker, None, grade_fn=grader, policy=FAST) is None  # not retried

    async with maker() as db:
        events = await read_events(db, sid)
    failed = [e for e in events if e.type == "GRADING_FAILED"]
    assert len(failed) == 1 and "ANSWER_GRADED" not in [e.type for e in events]
    assert failed[0].payload == {
        "turn_id": str(turn_id),
        "error_code": code,
        "attempts": 1,
        "retries_exhausted": False,
    }
    state = await _state(maker, uid, sid)
    assert state.state is InterviewState.GRADING and not state.current_graded
    assert state.ability == {}
    assert await session_cost(db_engine, sid) == (Decimal("0.000000"), 0)


async def test_retries_are_bounded(maker, db_engine, submitted):
    uid, sid, _ = submitted
    grader = Scripted(ProviderUnavailableError("503", provider="stub"))
    outcomes = []
    for _ in range(10):
        await _make_claimable(db_engine, sid)
        outcome = await run_once(maker, None, grade_fn=grader, policy=FAST)
        if outcome is None:
            break
        outcomes.append(outcome)
    assert outcomes == [Outcome.RETRY_SCHEDULED, Outcome.RETRY_SCHEDULED, Outcome.FAILED]
    assert len(grader.calls) == FAST.max_attempts == 3
    rows = await turn_rows(db_engine, sid)
    assert (rows[0]["grading_status"], rows[0]["grading_attempts"]) == ("failed", 3)
    async with maker() as db:
        failed = [e for e in await read_events(db, sid) if e.type == "GRADING_FAILED"]
    assert failed[0].payload["retries_exhausted"] is True
    assert not (await _state(maker, uid, sid)).current_graded


async def test_a_grader_that_hangs_times_out_as_transient(maker, db_engine, submitted):
    _, sid, _ = submitted

    async def hang(job):
        await asyncio.sleep(10)

    policy = RetryPolicy(max_attempts=1, base_delay_s=0.01, max_delay_s=0.01, timeout_s=0.2)
    assert await run_once(maker, None, grade_fn=hang, policy=policy) == Outcome.FAILED
    rows = await turn_rows(db_engine, sid)
    assert rows[0]["grading_error"] == "timeout"


def test_backoff_is_bounded_and_monotonic():
    policy = RetryPolicy(max_attempts=10, base_delay_s=2.0, max_delay_s=30.0)
    delays = [policy.delay_after(a) for a in range(1, 11)]
    assert delays[:4] == [2.0, 4.0, 8.0, 16.0]
    assert max(delays) == 30.0 and delays == sorted(delays)


@pytest.mark.parametrize(
    ("error", "transient"),
    [
        (ProviderTimeoutError("t", provider="p"), True),
        (ProviderRateLimitedError("r", provider="p"), True),
        (ProviderUnavailableError("u", provider="p"), True),
        (AllProvidersFailedError({"p": "down"}), True),
        (TimeoutError(), True),
        (ConnectionResetError(), True),
        (ProviderAuthError("a", provider="p"), False),
        (GradingContractError("c"), False),
        (ValueError("v"), False),
    ],
)
def test_classification(error, transient):
    code, is_transient = classify_error(error)
    assert is_transient is transient
    assert code and " " not in code and "p]" not in code  # a key, not a message


# ---------------------------------------------------------------------------
# Crash recovery and fencing
# ---------------------------------------------------------------------------


async def test_a_crashed_claim_is_recovered_after_its_lease(
    maker, db_engine, submitted, real_result
):
    """A worker claims and dies. Nothing happens until the lease expires; then
    another worker re-claims, and the dead one's late result is fenced out."""
    uid, sid, _ = submitted
    dead_job = await claim_next(maker, FAST)  # ...and the process dies here
    assert await claim_next(maker, FAST) is None  # lease still valid
    async with db_engine.begin() as conn:  # time passes
        await conn.execute(text("UPDATE turns SET lease_expires_at = now() - interval '1 second'"))
    assert (
        await run_once(maker, None, grade_fn=Scripted(real_result), policy=FAST) == Outcome.GRADED
    )
    rows = await turn_rows(db_engine, sid)
    assert rows[0]["grading_attempts"] == 2

    # The "dead" worker was only slow; its result must not land.
    late = await process_job(maker, None, dead_job, grade_fn=Scripted(real_result), policy=FAST)
    assert late == Outcome.STALE
    assert (await event_types(db_engine, sid)).count("ANSWER_GRADED") == 1


async def test_a_stale_worker_cannot_finish_while_the_new_claim_is_in_flight(
    maker, db_engine, submitted, real_result
):
    """The interleaving the attempt fence exists for: A's lease expires, B
    re-claims (still processing), and A finishes *first*. A must be refused
    even though the turn is still 'processing' - only B's attempt may land."""
    uid, sid, _ = submitted
    job_a = await claim_next(maker, FAST)
    async with db_engine.begin() as conn:
        await conn.execute(text("UPDATE turns SET lease_expires_at = now() - interval '1 second'"))
    job_b = await claim_next(maker, FAST)
    assert (job_a.attempt, job_b.attempt) == (1, 2)

    assert await process_job(maker, None, job_a, grade_fn=Scripted(real_result), policy=FAST) == (
        Outcome.STALE
    )
    assert "ANSWER_GRADED" not in await event_types(db_engine, sid)
    assert await process_job(maker, None, job_b, grade_fn=Scripted(real_result), policy=FAST) == (
        Outcome.GRADED
    )
    rows = await turn_rows(db_engine, sid)
    assert (rows[0]["grading_status"], rows[0]["grading_attempts"]) == ("graded", 2)
    assert (await event_types(db_engine, sid)).count("ANSWER_GRADED") == 1


async def test_crash_looping_turns_stop_at_the_bound(maker, db_engine, submitted):
    """Every claim crashed: attempts are used up without a successful call, and
    the next claim fails the turn instead of calling the model again."""
    _, sid, _ = submitted
    async with db_engine.begin() as conn:
        await conn.execute(
            text(
                "UPDATE turns SET grading_status='processing', grading_attempts=3, "
                "lease_expires_at = now() - interval '1 second'"
            )
        )
    grader = Scripted(ProviderTimeoutError("t", provider="stub"))
    assert await run_once(maker, None, grade_fn=grader, policy=FAST) == Outcome.FAILED
    assert grader.calls == []
    assert (await turn_rows(db_engine, sid))[0]["grading_error"] == "retry_exhausted"


async def test_a_failed_turn_can_be_requeued_and_graded(maker, db_engine, submitted, real_result):
    uid, sid, turn_id = submitted
    await run_once(maker, None, grade_fn=Scripted(GradingInputError("x")), policy=FAST)
    assert await requeue_failed_turn(maker, session_id=sid, turn_id=turn_id) is True
    assert await requeue_failed_turn(maker, session_id=sid, turn_id=turn_id) is False
    assert (
        await run_once(maker, None, grade_fn=Scripted(real_result), policy=FAST) == Outcome.GRADED
    )
    assert (await event_types(db_engine, sid))[-3:] == [
        "GRADING_FAILED",
        "ANSWER_GRADED",
        "FOLLOW_UP_STARTED",  # Day 23: the decision after the (late) grade
    ]
    async with maker() as db:
        assert fold(sid, await read_events(db, sid)).current_graded


async def test_a_redis_outage_does_not_block_the_grade(maker, db_engine, submitted, real_result):
    from redis.asyncio import Redis

    from app.interview import SessionStateCache

    _, sid, _ = submitted
    dead = Redis.from_url("redis://127.0.0.1:1/0", socket_connect_timeout=0.2)
    try:
        cache = SessionStateCache(dead, ttl_s=60, timeout_s=0.3)
        outcome = await run_once(maker, cache, grade_fn=Scripted(real_result), policy=FAST)
    finally:
        await dead.aclose()
    assert outcome == Outcome.GRADED


async def test_a_grade_the_log_refuses_becomes_a_failure_not_a_force(
    maker, db_engine, submitted, real_result
):
    """If the log already moved past this turn (here: graded by someone else),
    the worker records a failure instead of forcing a second grade in."""
    uid, sid, turn_id = submitted
    job = await claim_next(maker, FAST)
    from app.interview import append_event

    async with maker() as db:
        await append_event(
            db,
            None,
            session_id=sid,
            user_id=uid,
            event_type=EventType.ANSWER_GRADED,
            payload={"turn_id": str(turn_id), "score": 0.1},
        )
    outcome = await process_job(maker, None, job, grade_fn=Scripted(real_result), policy=FAST)
    assert outcome == Outcome.FAILED
    assert (await turn_rows(db_engine, sid))[0]["grading_error"] == "state_conflict"
    assert (await event_types(db_engine, sid)).count("ANSWER_GRADED") == 1


# ---------------------------------------------------------------------------
# End to end: the real app, the real worker, the real grader, offline
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def app_factory(env, migrated_database_url, redis_url) -> AsyncIterator:
    from app.cache import dispose_redis
    from app.db import dispose_engine
    from app.llm import dispose_llm
    from app.main import create_app

    async def start(worker: bool):
        await dispose_llm()
        await dispose_engine()
        await dispose_redis()
        env(
            {
                "DATABASE_URL": migrated_database_url,
                "REDIS_URL": redis_url,
                "GRADING_WORKER_ENABLED": "true" if worker else "false",
                "GRADING_POLL_INTERVAL_S": "0.1",
                # The offline stub, STRICT: a request with no recording fails
                # loudly instead of being synthesised.
                "LLM_PROVIDER_ORDER": "stub",
                "LLM_CACHE_ENABLED": "false",
            }
        )
        from app.config import get_settings as _gs

        _gs.cache_clear()
        return create_app()

    yield start
    await dispose_llm()
    await dispose_engine()
    await dispose_redis()


async def _register(http, engine, email="e2e@example.com"):
    r = await http.post(
        "/api/auth/register", json={"email": email, "password": "correct horse battery"}
    )
    async with engine.connect() as conn:
        uid = await conn.scalar(text("SELECT id FROM users WHERE email=:e"), {"e": email})
    return uid, {"Authorization": f"Bearer {r.json()['access_token']}"}


async def _wait_for_status(http, url, headers, want, within=20.0):
    deadline = asyncio.get_running_loop().time() + within
    while True:
        body = (await http.get(url, headers=headers)).json()
        if body["status"] in want or asyncio.get_running_loop().time() > deadline:
            return body
        await asyncio.sleep(0.1)


async def test_end_to_end_submit_then_the_real_grader_grades(
    app_factory, db_engine, maker, real_result
):
    await insert_question(db_engine)
    app = await app_factory(worker=True)
    async with LifespanManager(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as http:
            uid, headers = await _register(http, db_engine)
            sid = await session_in_asking(maker, uid)
            body = {"turn_id": str(uuid.uuid4()), "answer": STRONG_ANSWER}
            accepted = await http.post(f"/api/sessions/{sid}/turns", json=body, headers=headers)
            assert accepted.status_code == 202
            final = await _wait_for_status(
                http, accepted.json()["status_url"], headers, {"graded", "failed"}
            )
            retry = await http.post(f"/api/sessions/{sid}/turns", json=body, headers=headers)
    assert final["status"] == "graded", final
    # The same number the real pipeline produced from the same recording.
    assert final["score"] == pytest.approx(real_result.score)
    assert retry.status_code == 200 and retry.json()["score"] == final["score"]
    assert (await event_types(db_engine, sid)).count("ANSWER_GRADED") == 1


async def test_accepted_work_survives_an_api_restart(app_factory, db_engine, maker, real_result):
    """Submitted while no worker ran; the app stops; a fresh app (with its
    worker) starts and grades it. Nothing about the turn lived in memory."""
    await insert_question(db_engine)
    app = await app_factory(worker=False)
    async with LifespanManager(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as http:
            uid, headers = await _register(http, db_engine)
            sid = await session_in_asking(maker, uid)
            body = {"turn_id": str(uuid.uuid4()), "answer": STRONG_ANSWER}
            accepted = await http.post(f"/api/sessions/{sid}/turns", json=body, headers=headers)
    assert accepted.status_code == 202
    assert (await turn_rows(db_engine, sid))[0]["grading_status"] == "pending"

    app2 = await app_factory(worker=True)
    async with LifespanManager(app2):
        async with AsyncClient(transport=ASGITransport(app=app2), base_url="http://t") as http:
            final = await _wait_for_status(
                http, accepted.json()["status_url"], headers, {"graded", "failed"}
            )
    assert final["status"] == "graded", final
    assert final["score"] == pytest.approx(real_result.score)


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------


async def _until_graded(engine, sid, within=5.0):
    async with asyncio.timeout(within):
        while (await turn_rows(engine, sid))[0]["grading_status"] != "graded":  # noqa: ASYNC110
            await asyncio.sleep(0.05)


async def test_the_worker_loop_drains_and_stops(maker, db_engine, submitted, real_result):
    from app.interview.grading import run_worker

    _, sid, _ = submitted
    stop = asyncio.Event()
    task = asyncio.create_task(
        run_worker(
            maker,
            None,
            grade_fn=Scripted(real_result),
            policy=FAST,
            poll_interval_s=0.05,
            stop=stop,
        )
    )
    await _until_graded(db_engine, sid)
    stop.set()
    await asyncio.wait_for(task, timeout=2)


async def test_the_worker_loop_survives_a_failing_iteration(
    maker, db_engine, submitted, real_result, monkeypatch, caplog
):
    """A database blip in one iteration is logged; the worker keeps going."""
    from app.interview import grading

    _, sid, _ = submitted
    real_run_once = grading.run_once
    calls = {"n": 0}

    async def flaky(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ConnectionRefusedError("database restarting")
        return await real_run_once(*args, **kwargs)

    monkeypatch.setattr(grading, "run_once", flaky)
    stop = asyncio.Event()
    task = asyncio.create_task(
        grading.run_worker(
            maker,
            None,
            grade_fn=Scripted(real_result),
            policy=FAST,
            poll_interval_s=0.05,
            stop=stop,
        )
    )
    await _until_graded(db_engine, sid)
    stop.set()
    await asyncio.wait_for(task, timeout=2)
    assert "grading worker iteration failed" in caplog.text
