"""GET /api/sessions/{id}/stream against real Postgres - and a real socket.

httpx's in-memory ``ASGITransport`` buffers a response until it ends, so it
cannot observe a *live* stream. The live tests therefore start the real app
under a real uvicorn server on a free localhost port and read it over TCP -
which is also the only honest way to test a client disconnect.
"""

from __future__ import annotations

import asyncio
import json
import socket
import uuid
from collections.abc import AsyncIterator

import httpx
import pytest_asyncio
import uvicorn
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from app.interview import EventType, append_event
from app.interview.stream import event_stream
from app.interview.turns import submit_turn
from tests.integration.interview_support import (
    STRONG_ANSWER,
    insert_question,
    session_in_asking,
    user,
)


@pytest_asyncio.fixture
async def maker(db_engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(db_engine, expire_on_commit=False)


async def _never() -> bool:
    return False


def _parse_stream(raw: str) -> list[dict]:
    out = []
    for block in raw.split("\n\n"):
        if not block.strip() or block.startswith(":"):
            continue
        fields = dict(line.split(": ", 1) for line in block.split("\n"))
        out.append(
            {
                "id": int(fields["id"]) if "id" in fields else None,
                "event": fields["event"],
                "data": json.loads(fields["data"]),
            }
        )
    return out


async def _drive_to_completion(maker, uid, sid) -> None:
    """Take a session in ASKING (seq 2) to COMPLETED (seq 6)."""
    turn = uuid.uuid4()
    await submit_turn(maker, None, session_id=sid, user_id=uid, turn_id=turn, answer_text="x")
    for event_type, payload in (
        (EventType.ANSWER_GRADED, {"turn_id": str(turn), "score": 0.5}),
        (EventType.REPORTING_STARTED, {"reasons": ["item_budget_reached"]}),
        (EventType.SESSION_COMPLETED, {}),
    ):
        async with maker() as db:
            await append_event(
                db, None, session_id=sid, user_id=uid, event_type=event_type, payload=payload
            )


async def _collect(gen, limit=50) -> list[dict]:
    raw = ""
    async for chunk in gen:
        raw += chunk
        if raw.count("\n\n") >= limit:
            break
    return _parse_stream(raw)


def _stream(maker, sid, uid, after=-1, **kw):
    return event_stream(
        maker,
        session_id=sid,
        user_id=uid,
        last_event_id=after,
        poll_interval_s=kw.get("poll", 0.05),
        keepalive_s=kw.get("keepalive", 60.0),
        is_disconnected=kw.get("is_disconnected", _never),
    )


@pytest_asyncio.fixture
async def completed(maker, db_engine):
    await insert_question(db_engine)
    uid = await user(db_engine)
    sid = await session_in_asking(maker, uid)
    await _drive_to_completion(maker, uid, sid)
    return uid, sid


# ---------------------------------------------------------------------------
# The generator against the real log
# ---------------------------------------------------------------------------


async def test_full_replay_in_order_then_the_stream_ends(maker, completed):
    uid, sid = completed
    messages = await _collect(_stream(maker, sid, uid))
    assert [m["id"] for m in messages] == list(range(7))
    assert [m["event"] for m in messages] == [
        "state_changed",
        "state_changed",
        "next_question",
        "grading_started",
        "grade_ready",
        "state_changed",
        "session_complete",
    ]
    assert messages[4]["data"]["score"] == 0.5
    assert messages[-1]["data"]["state"] == "COMPLETED"


async def test_last_event_id_resumes_exactly_after_it(maker, completed):
    uid, sid = completed
    messages = await _collect(_stream(maker, sid, uid, after=3))
    assert [m["id"] for m in messages] == [4, 5, 6]


async def test_a_last_event_id_at_the_head_replays_nothing_and_ends(maker, completed):
    uid, sid = completed
    assert await _collect(_stream(maker, sid, uid, after=6)) == []


async def test_live_events_arrive_as_they_are_committed(maker, db_engine):
    await insert_question(db_engine)
    uid = await user(db_engine)
    sid = await session_in_asking(maker, uid)
    gen = _stream(maker, sid, uid, after=2)

    async def produce():
        await asyncio.sleep(0.2)
        await _drive_to_completion(maker, uid, sid)

    producer = asyncio.create_task(produce())
    messages = await asyncio.wait_for(_collect(gen), timeout=10)
    await producer
    assert [m["event"] for m in messages] == [
        "grading_started",
        "grade_ready",
        "state_changed",
        "session_complete",
    ]


async def test_disconnect_ends_the_stream(maker, db_engine):
    await insert_question(db_engine)
    uid = await user(db_engine)
    sid = await session_in_asking(maker, uid)
    flag = {"gone": False}

    async def disconnected():
        return flag["gone"]

    gen = _stream(maker, sid, uid, is_disconnected=disconnected)
    first = await anext(gen)
    assert "state_changed" in first
    flag["gone"] = True
    remaining = [chunk async for chunk in gen]
    # At most the messages already read in the same poll; then it stops.
    assert all("event:" in c for c in remaining)


async def test_keepalive_comments_when_idle(maker, db_engine):
    await insert_question(db_engine)
    uid = await user(db_engine)
    sid = await session_in_asking(maker, uid)
    gen = _stream(maker, sid, uid, after=2, keepalive=0.1)
    chunk = await asyncio.wait_for(anext(gen), timeout=5)
    assert chunk == ": keepalive\n\n"
    await gen.aclose()


async def test_a_deleted_session_ends_the_stream(maker, db_engine):
    await insert_question(db_engine)
    uid = await user(db_engine)
    sid = await session_in_asking(maker, uid)
    gen = _stream(maker, sid, uid, after=2)

    async def delete_later():
        await asyncio.sleep(0.2)
        async with db_engine.begin() as conn:
            await conn.execute(text("DELETE FROM users WHERE id = :u"), {"u": uid})

    task = asyncio.create_task(delete_later())
    remaining = await asyncio.wait_for(_collect(gen), timeout=5)
    await task
    assert remaining == []


async def test_another_users_stream_yields_nothing(maker, completed, db_engine):
    _, sid = completed
    stranger = await user(db_engine, "stranger@example.com")
    assert await _collect(_stream(maker, sid, stranger)) == []


async def test_a_corrupted_log_sends_an_error_event_and_ends(maker, db_engine):
    await insert_question(db_engine)
    uid = await user(db_engine)
    sid = await session_in_asking(maker, uid)
    async with db_engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO interview_events (session_id, seq, type, payload) "
                "VALUES (:s, 3, 'SESSION_COMPLETED', '{}'::jsonb)"
            ),
            {"s": sid},
        )
    messages = await _collect(_stream(maker, sid, uid))
    assert messages[-1] == {
        "id": None,
        "event": "error",
        "data": {"code": "session_state_unavailable"},
    }


# ---------------------------------------------------------------------------
# HTTP: auth, ownership, headers (ASGITransport is fine for a stream that ends)
# ---------------------------------------------------------------------------


async def _register(client, engine, email):
    r = await client.post(
        "/api/auth/register", json={"email": email, "password": "correct horse battery"}
    )
    async with engine.connect() as conn:
        uid = await conn.scalar(text("SELECT id FROM users WHERE email=:e"), {"e": email})
    return uid, {"Authorization": f"Bearer {r.json()['access_token']}"}


async def test_http_stream_of_a_completed_session(client, db_engine, maker):
    await insert_question(db_engine)
    uid, headers = await _register(client, db_engine, "s@example.com")
    sid = await session_in_asking(maker, uid)
    await _drive_to_completion(maker, uid, sid)
    response = await client.get(
        f"/api/sessions/{sid}/stream", headers={**headers, "Last-Event-ID": "2"}
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["cache-control"] == "no-cache"
    messages = _parse_stream(response.text)
    assert [m["id"] for m in messages] == [3, 4, 5, 6]


async def test_http_stream_requires_auth(client, db_engine, maker, completed):
    _, sid = completed
    assert (await client.get(f"/api/sessions/{sid}/stream")).status_code == 401


async def test_http_stream_of_someone_elses_session_is_404(client, db_engine, completed):
    _, sid = completed
    _, headers = await _register(client, db_engine, "intruder@example.com")
    response = await client.get(f"/api/sessions/{sid}/stream", headers=headers)
    assert response.status_code == 404


async def test_http_bad_last_event_id_is_400(client, db_engine, maker):
    await insert_question(db_engine)
    uid, headers = await _register(client, db_engine, "b@example.com")
    sid = await session_in_asking(maker, uid)
    response = await client.get(
        f"/api/sessions/{sid}/stream", headers={**headers, "Last-Event-ID": "nope"}
    )
    assert response.status_code == 400


# ---------------------------------------------------------------------------
# Live, over a real socket, with the real worker grading in the background
# ---------------------------------------------------------------------------


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@pytest_asyncio.fixture
async def live_server(env, migrated_database_url, redis_url) -> AsyncIterator[str]:
    from app.cache import dispose_redis
    from app.config import get_settings
    from app.db import dispose_engine
    from app.llm import dispose_llm
    from app.main import create_app

    await dispose_llm()
    await dispose_engine()
    await dispose_redis()
    env(
        {
            "DATABASE_URL": migrated_database_url,
            "REDIS_URL": redis_url,
            "GRADING_WORKER_ENABLED": "true",
            "GRADING_POLL_INTERVAL_S": "0.1",
            "SSE_POLL_INTERVAL_S": "0.1",
            "LLM_PROVIDER_ORDER": "stub",
            "LLM_CACHE_ENABLED": "false",
        }
    )
    get_settings.cache_clear()
    port = _free_port()
    server = uvicorn.Server(
        uvicorn.Config(
            create_app(), host="127.0.0.1", port=port, log_level="warning", lifespan="on"
        )
    )
    task = asyncio.create_task(server.serve())
    while not server.started:  # noqa: ASYNC110 - uvicorn exposes no "started" event
        await asyncio.sleep(0.05)
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    await asyncio.wait_for(task, timeout=15)
    await dispose_llm()
    await dispose_engine()
    await dispose_redis()


async def _read_events(response, want: int, within: float = 20.0) -> list[dict]:
    raw = ""
    async with asyncio.timeout(within):
        async for chunk in response.aiter_text():
            raw += chunk
            parsed = _parse_stream(raw)
            if len(parsed) >= want:
                return parsed
    return _parse_stream(raw)


async def test_live_stream_shows_grading_started_then_grade_ready(live_server, db_engine, maker):
    await insert_question(db_engine)
    async with httpx.AsyncClient(base_url=live_server, timeout=30) as http:
        uid, headers = await _register(http, db_engine, "live@example.com")
        sid = await session_in_asking(maker, uid)
        async with http.stream(
            "GET", f"/api/sessions/{sid}/stream", headers={**headers, "Last-Event-ID": "2"}
        ) as stream:
            assert stream.status_code == 200
            turn_id = str(uuid.uuid4())
            submitted = await http.post(
                f"/api/sessions/{sid}/turns",
                json={"turn_id": turn_id, "answer": STRONG_ANSWER},
                headers=headers,
            )
            assert submitted.status_code == 202
            messages = await _read_events(stream, want=2)
    assert [m["event"] for m in messages[:2]] == ["grading_started", "grade_ready"]
    assert [m["id"] for m in messages[:2]] == [3, 4]
    assert messages[1]["data"]["turn_id"] == turn_id
    assert 0.0 <= messages[1]["data"]["score"] <= 1.0
    # Nothing but the whitelisted fields crossed the wire.
    assert set(messages[1]["data"]) == {"seq", "state", "turn_id", "score"}


async def test_live_reconnect_with_last_event_id_misses_nothing(live_server, db_engine, maker):
    await insert_question(db_engine)
    async with httpx.AsyncClient(base_url=live_server, timeout=30) as http:
        uid, headers = await _register(http, db_engine, "reconnect@example.com")
        sid = await session_in_asking(maker, uid)
        async with http.stream("GET", f"/api/sessions/{sid}/stream", headers=headers) as first:
            seen = await _read_events(first, want=3)
        last = seen[-1]["id"]  # disconnected here, after seq 2
        # While nobody is listening, the turn is submitted and graded.
        await http.post(
            f"/api/sessions/{sid}/turns",
            json={"turn_id": str(uuid.uuid4()), "answer": STRONG_ANSWER},
            headers=headers,
        )
        await asyncio.sleep(1.5)
        async with http.stream(
            "GET", f"/api/sessions/{sid}/stream", headers={**headers, "Last-Event-ID": str(last)}
        ) as second:
            resumed = await _read_events(second, want=2)
    assert last == 2
    assert [m["id"] for m in resumed[:2]] == [3, 4]
    assert [m["event"] for m in resumed[:2]] == ["grading_started", "grade_ready"]


async def test_live_client_disconnect_releases_the_server(live_server, db_engine, maker):
    """After the client goes away, the server's stream stops: shutting the
    server down afterwards completes promptly (the fixture waits on it)."""
    await insert_question(db_engine)
    async with httpx.AsyncClient(base_url=live_server, timeout=30) as http:
        uid, headers = await _register(http, db_engine, "bye@example.com")
        sid = await session_in_asking(maker, uid)
        for _ in range(3):
            async with http.stream("GET", f"/api/sessions/{sid}/stream", headers=headers) as s:
                await _read_events(s, want=1)
    async with db_engine.connect() as conn:
        # No connection is left idle-in-transaction by an abandoned stream.
        stuck = await conn.scalar(
            text(
                "SELECT count(*) FROM pg_stat_activity "
                "WHERE state = 'idle in transaction' AND datname = current_database()"
            )
        )
    assert stuck == 0
