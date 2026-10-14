"""GET /api/sessions/{id} (Day 27): the interview screen's snapshot.

Real Postgres, Redis and app. The snapshot must be the caller's own session
only, carry the question in front of the candidate, and nothing the candidate
must not see (grades, ability, selection internals).
"""

from __future__ import annotations

import json
import uuid

import pytest_asyncio
from httpx import AsyncClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from app.interview import EventType, append_event, create_session
from tests.integration.interview_support import (
    BLUEPRINT,
    QUESTION_ID,
    STRONG_ANSWER,
    insert_question,
)

PASSWORD = "correct horse battery staple"
QUESTION_TEXT = "How would you cache a slow query?\n\n    SELECT * FROM orders;"

#: Keys that would expose grading, ability or selection internals.
FORBIDDEN = (
    "blueprint",
    "ability",
    "theta",
    "score",
    "scores",
    "difficulty",
    "difficulty_window",
    "target_theta",
    "item_budget",
    "stop_reasons",
    "reference_answer",
    "expected_concepts",
    "priors",
)


@pytest_asyncio.fixture
async def maker(db_engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(db_engine, expire_on_commit=False)


async def _register(client: AsyncClient, engine: AsyncEngine, email: str):
    response = await client.post("/api/auth/register", json={"email": email, "password": PASSWORD})
    assert response.status_code == 201, response.text
    async with engine.connect() as conn:
        uid = await conn.scalar(text("SELECT id FROM users WHERE email = :e"), {"e": email})
    return uid, {"Authorization": f"Bearer {response.json()['access_token']}"}


@pytest_asyncio.fixture
async def candidate(client, db_engine):
    return await _register(client, db_engine, "candidate@example.com")


async def _created(maker, user_id) -> uuid.UUID:
    async with maker() as db:
        return (await create_session(db, None, user_id=user_id, blueprint=BLUEPRINT)).session_id


async def _asking(maker, db_engine, user_id) -> uuid.UUID:
    await insert_question(db_engine)
    sid = await _created(maker, user_id)
    for event_type, payload in (
        (EventType.PREPARATION_STARTED, None),
        (EventType.QUESTION_ASKED, {"question_id": QUESTION_ID, "text": QUESTION_TEXT}),
    ):
        async with maker() as db:
            await append_event(
                db, None, session_id=sid, user_id=user_id, event_type=event_type, payload=payload
            )
    return sid


def _leaks(body: dict) -> list[str]:
    dumped = json.dumps(body)
    return [key for key in FORBIDDEN if f'"{key}"' in dumped]


async def test_requires_auth(client):
    assert (await client.get(f"/api/sessions/{uuid.uuid4()}")).status_code == 401


async def test_unknown_and_foreign_sessions_are_the_same_404(client, db_engine, maker, candidate):
    sid = await _created(maker, candidate[0])
    _, stranger = await _register(client, db_engine, "stranger@example.com")
    for headers, target in ((stranger, sid), (candidate[1], uuid.uuid4())):
        r = await client.get(f"/api/sessions/{target}", headers=headers)
        assert r.status_code == 404
        assert r.json() == {"detail": "Session not found"}


async def test_a_malformed_id_is_422(client, candidate):
    r = await client.get("/api/sessions/not-a-uuid", headers=candidate[1])
    assert r.status_code == 422


async def test_a_created_session_has_no_question_yet(client, maker, candidate):
    sid = await _created(maker, candidate[0])
    r = await client.get(f"/api/sessions/{sid}", headers=candidate[1])
    assert r.status_code == 200
    body = r.json()
    assert body["state"] == "CREATED" and body["seq"] == 0
    assert body["question"] is None
    assert body["questions_asked"] == 0 and body["follow_ups_asked"] == 0
    assert body["current_turn_id"] is None
    assert body["stream_url"] == f"/api/sessions/{sid}/stream"
    assert _leaks(body) == []


async def test_an_asking_session_carries_the_question_verbatim(client, db_engine, maker, candidate):
    sid = await _asking(maker, db_engine, candidate[0])
    body = (await client.get(f"/api/sessions/{sid}", headers=candidate[1])).json()
    assert body["state"] == "ASKING" and body["seq"] == 2
    assert body["question"] == {
        "question_id": QUESTION_ID,
        "is_follow_up": False,
        "text": QUESTION_TEXT,  # newlines and indentation preserved
    }
    assert body["questions_asked"] == 1
    assert body["role"] == BLUEPRINT.to_dict()["role"]
    assert body["topic_keys"]
    assert _leaks(body) == []


async def test_while_grading_it_names_the_turn_and_keeps_the_question(
    client, db_engine, maker, candidate
):
    sid = await _asking(maker, db_engine, candidate[0])
    turn_id = str(uuid.uuid4())
    submitted = await client.post(
        f"/api/sessions/{sid}/turns",
        json={"turn_id": turn_id, "answer": STRONG_ANSWER},
        headers=candidate[1],
    )
    assert submitted.status_code == 202

    body = (await client.get(f"/api/sessions/{sid}", headers=candidate[1])).json()
    assert body["state"] == "GRADING" and body["seq"] == 3
    assert body["current_turn_id"] == turn_id
    assert body["question"]["text"] == QUESTION_TEXT
    # The answer text is never echoed back.
    assert STRONG_ANSWER not in json.dumps(body)
    assert _leaks(body) == []


async def test_reading_the_snapshot_writes_nothing(client, db_engine, maker, candidate):
    sid = await _asking(maker, db_engine, candidate[0])
    for _ in range(3):
        await client.get(f"/api/sessions/{sid}", headers=candidate[1])
    async with db_engine.connect() as conn:
        count = await conn.scalar(
            text("SELECT count(*) FROM interview_events WHERE session_id = :s"), {"s": sid}
        )
    assert count == 3
