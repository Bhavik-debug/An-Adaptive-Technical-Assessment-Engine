"""POST /api/sessions/{id}/turns: one logical turn per turn_id, atomically.

Real Postgres, real Redis, the real app.  The grading worker is OFF here
(``GRADING_WORKER_ENABLED=false`` in the test environment), so these tests see
exactly what submission writes and nothing the worker would add later.
"""

from __future__ import annotations

import asyncio
import uuid

import pytest
import pytest_asyncio
from httpx import AsyncClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from app.interview import EventType, IllegalTransition, SessionNotFound, append_event
from app.interview.turns import TurnIdConflict, submit_turn
from tests.integration.interview_support import (
    QUESTION_ID,
    STRONG_ANSWER,
    event_types,
    insert_question,
    session_in_asking,
    turn_rows,
    user,
)

PASSWORD = "correct horse battery staple"


@pytest_asyncio.fixture
async def maker(db_engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(db_engine, expire_on_commit=False)


@pytest_asyncio.fixture
async def question(db_engine) -> str:
    return await insert_question(db_engine)


async def _register(client: AsyncClient, engine: AsyncEngine, email: str):
    response = await client.post("/api/auth/register", json={"email": email, "password": PASSWORD})
    assert response.status_code == 201, response.text
    async with engine.connect() as conn:
        uid = await conn.scalar(text("SELECT id FROM users WHERE email = :e"), {"e": email})
    return uid, {"Authorization": f"Bearer {response.json()['access_token']}"}


@pytest_asyncio.fixture
async def candidate(client, db_engine):
    return await _register(client, db_engine, "candidate@example.com")


@pytest_asyncio.fixture
async def asking(maker, candidate, question) -> uuid.UUID:
    return await session_in_asking(maker, candidate[0])


def _url(sid: uuid.UUID) -> str:
    return f"/api/sessions/{sid}/turns"


def _body(turn_id: uuid.UUID | None = None, answer: str = STRONG_ANSWER) -> dict[str, str]:
    return {"turn_id": str(turn_id or uuid.uuid4()), "answer": answer}


# ---------------------------------------------------------------------------
# The happy path and the 202 contract
# ---------------------------------------------------------------------------


async def test_a_new_submission_is_accepted_with_202(client, db_engine, candidate, asking):
    body = _body()
    response = await client.post(_url(asking), json=body, headers=candidate[1])
    assert response.status_code == 202, response.text
    data = response.json()
    assert data["turn_id"] == body["turn_id"]
    assert data["status"] == "pending"
    assert data["score"] is None and data["attempts"] == 0
    assert data["status_url"] == f"/api/sessions/{asking}/turns/{body['turn_id']}"
    assert data["stream_url"] == f"/api/sessions/{asking}/stream"
    assert response.headers["Location"] == data["status_url"]
    assert "Idempotent-Replayed" not in response.headers

    # Durable: exactly one turn and exactly one ANSWER_SUBMITTED, committed.
    rows = await turn_rows(db_engine, asking)
    assert len(rows) == 1 and str(rows[0]["turn_id"]) == body["turn_id"]
    assert rows[0]["grading_status"] == "pending"
    assert (await event_types(db_engine, asking)).count("ANSWER_SUBMITTED") == 1
    async with db_engine.connect() as conn:
        state = await conn.scalar(
            text("SELECT state FROM interview_sessions WHERE id = :s"), {"s": asking}
        )
        question_id = await conn.scalar(
            text("SELECT question_id FROM turns WHERE session_id = :s"), {"s": asking}
        )
    assert state == "GRADING"
    assert question_id == QUESTION_ID


async def test_the_status_endpoint_reports_the_same_turn(client, candidate, asking):
    body = _body()
    await client.post(_url(asking), json=body, headers=candidate[1])
    response = await client.get(f"{_url(asking)}/{body['turn_id']}", headers=candidate[1])
    assert response.status_code == 200
    assert response.json()["status"] == "pending"


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------


async def test_a_retry_returns_the_original_turn_and_writes_nothing(
    client, db_engine, candidate, asking
):
    body = _body()
    first = await client.post(_url(asking), json=body, headers=candidate[1])
    events_before = await event_types(db_engine, asking)
    second = await client.post(_url(asking), json=body, headers=candidate[1])

    assert second.status_code == 200
    assert second.headers["Idempotent-Replayed"] == "true"
    assert second.json() == first.json()
    assert len(await turn_rows(db_engine, asking)) == 1
    assert await event_types(db_engine, asking) == events_before


async def test_a_retry_after_grading_returns_the_graded_result(
    client, db_engine, candidate, asking
):
    body = _body()
    await client.post(_url(asking), json=body, headers=candidate[1])
    async with db_engine.begin() as conn:  # as the worker would have left it
        await conn.execute(
            text(
                "UPDATE turns SET grading_status = 'graded', score = 0.75, graded_at = now(), "
                "grading_attempts = 1 WHERE session_id = :s"
            ),
            {"s": asking},
        )
    again = await client.post(_url(asking), json=body, headers=candidate[1])
    assert again.status_code == 200
    assert again.json()["status"] == "graded"
    assert again.json()["score"] == 0.75
    assert len(await turn_rows(db_engine, asking)) == 1


async def test_same_turn_id_with_a_different_answer_is_a_conflict(
    client, db_engine, candidate, asking
):
    turn_id = uuid.uuid4()
    await client.post(_url(asking), json=_body(turn_id), headers=candidate[1])
    events_before = await event_types(db_engine, asking)
    response = await client.post(
        _url(asking), json=_body(turn_id, "A completely different answer."), headers=candidate[1]
    )
    assert response.status_code == 409
    assert "different answer" in response.json()["detail"]
    # The original is untouched and nothing new was written.
    rows = await turn_rows(db_engine, asking)
    assert len(rows) == 1 and rows[0]["answer_text"] == STRONG_ANSWER
    assert await event_types(db_engine, asking) == events_before


async def test_whitespace_differences_are_different_submissions(client, candidate, asking):
    turn_id = uuid.uuid4()
    await client.post(_url(asking), json=_body(turn_id), headers=candidate[1])
    response = await client.post(
        _url(asking), json=_body(turn_id, STRONG_ANSWER + " "), headers=candidate[1]
    )
    assert response.status_code == 409


async def test_simultaneous_duplicates_make_one_turn_and_one_event(
    client, db_engine, candidate, asking
):
    body = _body()
    responses = await asyncio.gather(
        *(client.post(_url(asking), json=body, headers=candidate[1]) for _ in range(8))
    )
    codes = sorted(r.status_code for r in responses)
    assert codes == [200] * 7 + [202], codes
    assert len({r.json()["turn_id"] for r in responses}) == 1
    assert len(await turn_rows(db_engine, asking)) == 1
    assert (await event_types(db_engine, asking)).count("ANSWER_SUBMITTED") == 1


async def test_simultaneous_different_turn_ids_admit_exactly_one(
    client, db_engine, candidate, asking
):
    """Two different answers racing for one question: the FSM lets one in."""
    responses = await asyncio.gather(
        *(client.post(_url(asking), json=_body(), headers=candidate[1]) for _ in range(6))
    )
    codes = sorted(r.status_code for r in responses)
    assert codes == [202] + [409] * 5, codes
    assert len(await turn_rows(db_engine, asking)) == 1


async def test_service_level_concurrent_duplicates(maker, db_engine, question):
    """Same race without HTTP in the way: the lock, not the client, serialises."""
    uid = await user(db_engine)
    sid = await session_in_asking(maker, uid)
    turn_id = uuid.uuid4()
    results = await asyncio.gather(
        *(
            submit_turn(
                maker, None, session_id=sid, user_id=uid, turn_id=turn_id, answer_text="An answer."
            )
            for _ in range(10)
        )
    )
    assert sum(r.created for r in results) == 1
    assert {(r.turn.turn_id, r.turn.created_at) for r in results} == {
        (results[0].turn.turn_id, results[0].turn.created_at)
    }


async def test_concurrent_mismatched_duplicates(maker, db_engine, question):
    uid = await user(db_engine)
    sid = await session_in_asking(maker, uid)
    turn_id = uuid.uuid4()
    results = await asyncio.gather(
        *(
            submit_turn(
                maker, None, session_id=sid, user_id=uid, turn_id=turn_id, answer_text=f"Answer {i}"
            )
            for i in range(5)
        ),
        return_exceptions=True,
    )
    created = [r for r in results if not isinstance(r, BaseException)]
    assert len(created) == 1 and created[0].created
    assert all(isinstance(r, TurnIdConflict) for r in results if isinstance(r, BaseException))
    assert len(await turn_rows(db_engine, sid)) == 1


async def test_on_conflict_do_nothing_is_the_backstop(maker, db_engine, question, monkeypatch):
    """If the lock is bypassed, the second insert hits the unique constraint,
    does nothing, rolls back its own staged event, and returns the winner."""
    import sqlalchemy as sa

    from app.interview import store, turns
    from app.models.interview import InterviewSession

    uid = await user(db_engine)
    sid = await session_in_asking(maker, uid)

    async def unlocked(db, *, session_id, user_id):
        return await db.scalar(
            sa.select(InterviewSession).where(
                InterviewSession.id == session_id, InterviewSession.user_id == user_id
            )
        )

    monkeypatch.setattr(store, "lock_session", unlocked)
    monkeypatch.setattr(turns, "lock_session", unlocked)
    real_find = turns._find
    calls = {"n": 0}

    async def blind_first_find(db, session_id, turn_id):
        calls["n"] += 1
        if calls["n"] == 1:  # pretend the row is not there yet
            return None
        return await real_find(db, session_id, turn_id)

    turn_id = uuid.uuid4()
    first = await submit_turn(
        maker, None, session_id=sid, user_id=uid, turn_id=turn_id, answer_text="Same."
    )
    # Second submission: _find is blind on its first call, so it stages an event
    # and attempts the insert; ON CONFLICT DO NOTHING returns no row.
    monkeypatch.setattr(turns, "_find", blind_first_find)
    # Put the log back in ASKING-equivalent shape is impossible (append-only), so
    # stage_append itself would refuse from GRADING. Patch it to a no-op stager.
    from app.interview.state import SessionState

    async def stage_noop(db, **kw):
        return SessionState(
            session_id=sid,
            seq=99,
            current_question_id=QUESTION_ID,
            current_item_id=QUESTION_ID,
            state=None,
        )

    monkeypatch.setattr(turns, "stage_append", stage_noop)
    second = await submit_turn(
        maker, None, session_id=sid, user_id=uid, turn_id=turn_id, answer_text="Same."
    )
    assert first.created and not second.created
    assert second.turn.created_at == first.turn.created_at
    assert len(await turn_rows(db_engine, sid)) == 1


# ---------------------------------------------------------------------------
# Rejections write nothing
# ---------------------------------------------------------------------------


async def test_submission_outside_asking_is_409(client, db_engine, maker, candidate, question):
    async with maker() as db:
        from app.interview import create_session
        from tests.integration.interview_support import BLUEPRINT

        sid = (await create_session(db, None, user_id=candidate[0], blueprint=BLUEPRINT)).session_id
    response = await client.post(_url(sid), json=_body(), headers=candidate[1])
    assert response.status_code == 409
    assert "not allowed from CREATED" in response.json()["detail"]
    assert await turn_rows(db_engine, sid) == []
    assert await event_types(db_engine, sid) == ["SESSION_CREATED"]


async def test_second_answer_while_grading_is_409(client, db_engine, candidate, asking):
    await client.post(_url(asking), json=_body(), headers=candidate[1])
    response = await client.post(_url(asking), json=_body(), headers=candidate[1])
    assert response.status_code == 409
    assert len(await turn_rows(db_engine, asking)) == 1


async def test_a_failure_after_staging_rolls_back_turn_and_event(
    maker, db_engine, question, monkeypatch
):
    """Turn and ANSWER_SUBMITTED commit together or not at all."""
    from app.interview import turns

    uid = await user(db_engine)
    sid = await session_in_asking(maker, uid)
    before = await event_types(db_engine, sid)

    async def boom(*args, **kwargs):
        raise RuntimeError("simulated commit failure")

    monkeypatch.setattr(turns, "commit_and_cache", boom)
    with pytest.raises(RuntimeError, match="simulated"):
        await submit_turn(
            maker, None, session_id=sid, user_id=uid, turn_id=uuid.uuid4(), answer_text="x"
        )
    assert await turn_rows(db_engine, sid) == []
    assert await event_types(db_engine, sid) == before
    async with db_engine.connect() as conn:
        state = await conn.scalar(
            text("SELECT state FROM interview_sessions WHERE id=:s"), {"s": sid}
        )
    assert state == "ASKING"


async def test_a_follow_up_without_stored_wording_is_refused(maker, db_engine, question):
    """Only a pre-Day-23 log can ask a follow-up without its text; there is
    nothing to grade the answer against, so it is refused, not graded blind."""
    uid = await user(db_engine)
    sid = await session_in_asking(maker, uid)
    first = uuid.uuid4()
    await submit_turn(maker, None, session_id=sid, user_id=uid, turn_id=first, answer_text="x")
    for event_type, payload in (
        (EventType.ANSWER_GRADED, {"turn_id": str(first), "score": 0.5}),
        (EventType.FOLLOW_UP_STARTED, {}),
        (EventType.QUESTION_ASKED, {"is_follow_up": True}),
    ):
        async with maker() as db:
            await append_event(
                db, None, session_id=sid, user_id=uid, event_type=event_type, payload=payload
            )
    from app.interview.turns import FollowUpTurnsUnsupported

    before = await event_types(db_engine, sid)
    with pytest.raises(FollowUpTurnsUnsupported):
        await submit_turn(
            maker, None, session_id=sid, user_id=uid, turn_id=uuid.uuid4(), answer_text="y"
        )
    assert await event_types(db_engine, sid) == before
    assert len(await turn_rows(db_engine, sid)) == 1


async def test_service_raises_domain_errors(maker, db_engine, question):
    uid = await user(db_engine)
    with pytest.raises(SessionNotFound):
        await submit_turn(
            maker, None, session_id=uuid.uuid4(), user_id=uid, turn_id=uuid.uuid4(), answer_text="x"
        )
    async with maker() as db:
        from app.interview import create_session
        from tests.integration.interview_support import BLUEPRINT

        sid = (await create_session(db, None, user_id=uid, blueprint=BLUEPRINT)).session_id
    with pytest.raises(IllegalTransition):
        await submit_turn(
            maker, None, session_id=sid, user_id=uid, turn_id=uuid.uuid4(), answer_text="x"
        )


# ---------------------------------------------------------------------------
# Authentication, ownership, validation
# ---------------------------------------------------------------------------


async def test_no_token_is_401(client, asking):
    response = await client.post(_url(asking), json=_body())
    assert response.status_code == 401


async def test_a_bad_token_is_401(client, asking):
    response = await client.post(
        _url(asking), json=_body(), headers={"Authorization": "Bearer not-a-jwt"}
    )
    assert response.status_code == 401


async def test_another_users_session_is_404_and_untouched(client, db_engine, asking):
    _, intruder = await _register(client, db_engine, "intruder@example.com")
    before = await event_types(db_engine, asking)
    response = await client.post(_url(asking), json=_body(), headers=intruder)
    assert response.status_code == 404
    assert response.json()["detail"] == "Session not found"
    assert await event_types(db_engine, asking) == before
    assert await turn_rows(db_engine, asking) == []


async def test_an_unknown_session_is_the_same_404(client, candidate, question):
    response = await client.post(_url(uuid.uuid4()), json=_body(), headers=candidate[1])
    assert response.status_code == 404
    assert response.json()["detail"] == "Session not found"


async def test_another_users_turn_status_is_404(client, db_engine, candidate, asking):
    body = _body()
    await client.post(_url(asking), json=body, headers=candidate[1])
    _, intruder = await _register(client, db_engine, "intruder@example.com")
    response = await client.get(f"{_url(asking)}/{body['turn_id']}", headers=intruder)
    assert response.status_code == 404


@pytest.mark.parametrize(
    "body",
    [
        {"turn_id": "not-a-uuid", "answer": "x"},
        {"turn_id": str(uuid.uuid4()), "answer": ""},
        {"turn_id": str(uuid.uuid4()), "answer": "   \n\t"},
        {"turn_id": str(uuid.uuid4()), "answer": "x" * 8001},
        {"turn_id": str(uuid.uuid4())},
        {"answer": "x"},
        {"turn_id": str(uuid.uuid4()), "answer": "x", "score": 1.0},
    ],
)
async def test_invalid_bodies_are_422_and_write_nothing(client, db_engine, candidate, asking, body):
    response = await client.post(_url(asking), json=body, headers=candidate[1])
    assert response.status_code == 422
    assert await turn_rows(db_engine, asking) == []


async def test_a_client_supplied_score_is_refused(client, candidate, asking):
    """The worker must never trust client-supplied grading - so the request
    schema does not even have a place to put one."""
    body = {**_body(), "score": 1.0}
    response = await client.post(_url(asking), json=body, headers=candidate[1])
    assert response.status_code == 422


async def test_answers_are_not_logged(client, candidate, asking, caplog):
    secret = "UNIQUE-ANSWER-MARKER-7f3a " + STRONG_ANSWER
    await client.post(_url(asking), json=_body(answer=secret), headers=candidate[1])
    await client.post(_url(asking), json=_body(answer=secret), headers=candidate[1])
    assert "UNIQUE-ANSWER-MARKER-7f3a" not in caplog.text


async def test_redis_down_does_not_block_submission(maker, db_engine, question):
    from redis.asyncio import Redis

    from app.interview import SessionStateCache

    dead = Redis.from_url("redis://127.0.0.1:1/0", socket_connect_timeout=0.2)
    try:
        cache = SessionStateCache(dead, ttl_s=60, timeout_s=0.3)
        uid = await user(db_engine)
        sid = await session_in_asking(maker, uid)
        result = await submit_turn(
            maker, cache, session_id=sid, user_id=uid, turn_id=uuid.uuid4(), answer_text="x"
        )
    finally:
        await dead.aclose()
    assert result.created
    assert (await event_types(db_engine, sid))[-1] == "ANSWER_SUBMITTED"
