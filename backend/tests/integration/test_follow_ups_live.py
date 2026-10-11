"""Day 23 end to end over a real socket: the public API, the app's own worker,
the REAL grading pipeline, and SSE.

Offline: the one question the Day-18 grading recordings were made against is
inserted under real taxonomy keys, so ``grade_answer`` replays the real model's
recorded judgment. The follow-up probe has no recording, so the stub runs in
``synthesize`` mode and produces a schema-valid probe (marked ``stub``). Every
decision in between - the policy, the caps, selection - is the production code.
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
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker

from app.bank.taxonomy import load_taxonomy
from app.models.question import Question
from tests.integration.interview_support import RECORDED, STRONG_ANSWER, _concepts

QUESTION_ID = "day23-live-cache"


async def _seed(engine: AsyncEngine) -> None:
    async with engine.begin() as conn:
        for node in load_taxonomy().rows():
            await conn.execute(
                text(
                    "INSERT INTO topics (key, parent_key, display_name, domain) "
                    "VALUES (:k, :p, :n, :d) ON CONFLICT DO NOTHING"
                ),
                {"k": node.key, "p": node.parent_key, "n": node.display_name, "d": node.domain},
            )
    inputs = RECORDED["grade_answer_strong"]
    maker = async_sessionmaker(engine, expire_on_commit=False)
    async with maker() as db:
        db.add(
            Question(
                id=QUESTION_ID,
                topic_key="caching",
                subtopic_key="cache_invalidation",
                text=inputs["question"],
                difficulty_b=0.0,
                discrimination_a=1.0,
                expected_concepts=_concepts(inputs["expected_concepts"]),
                common_misconceptions=[m[2:] for m in inputs["misconceptions"].splitlines()],
                reference_answer="A reference answer used only by tests.",
                time_estimate_s=300,
                source="test",
            )
        )
        await db.commit()


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@pytest_asyncio.fixture
async def server(env, migrated_database_url, redis_url) -> AsyncIterator[str]:
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
            "LLM_STUB_ON_MISSING": "synthesize",
            "LLM_CACHE_ENABLED": "false",
        }
    )
    get_settings.cache_clear()
    port = _free_port()
    srv = uvicorn.Server(
        uvicorn.Config(
            create_app(), host="127.0.0.1", port=port, log_level="warning", lifespan="on"
        )
    )
    task = asyncio.create_task(srv.serve())
    while not srv.started:  # noqa: ASYNC110 - uvicorn exposes no "started" event
        await asyncio.sleep(0.05)
    yield f"http://127.0.0.1:{port}"
    srv.should_exit = True
    await asyncio.wait_for(task, timeout=15)
    await dispose_llm()
    await dispose_engine()
    await dispose_redis()


def _parse(raw: str) -> list[dict]:
    out = []
    for block in raw.split("\n\n"):
        if block.startswith("id:"):
            f = dict(line.split(": ", 1) for line in block.split("\n"))
            out.append({"id": int(f["id"]), "event": f["event"], "data": json.loads(f["data"])})
    return out


async def test_public_api_follow_up_round_trip(server, db_engine):
    await _seed(db_engine)
    async with httpx.AsyncClient(base_url=server, timeout=30) as http:
        reg = await http.post(
            "/api/auth/register", json={"email": "live@example.com", "password": "correct horse 9"}
        )
        headers = {"Authorization": f"Bearer {reg.json()['access_token']}"}
        spec = {
            "role": "Backend Engineer",
            "topics": [
                {"topic_key": "caching", "jd_weight": 1.0, "subtopic_keys": ["cache_invalidation"]}
            ],
        }
        created = await http.post("/api/sessions", json=spec, headers=headers)
        assert created.status_code == 201, created.text
        sid = created.json()["session_id"]
        started = await http.post(f"/api/sessions/{sid}/start", headers=headers)
        assert started.status_code == 200
        assert started.json()["question"]["question_id"] == QUESTION_ID

        async with http.stream(
            "GET", f"/api/sessions/{sid}/stream", headers={**headers, "Last-Event-ID": "2"}
        ) as stream:
            accepted = await http.post(
                f"/api/sessions/{sid}/turns",
                json={"turn_id": str(uuid.uuid4()), "answer": STRONG_ANSWER},
                headers=headers,
            )
            assert accepted.status_code == 202
            raw = ""
            async with asyncio.timeout(30):
                async for chunk in stream.aiter_text():
                    raw += chunk
                    if '"is_follow_up":true' in raw:
                        break
        messages = _parse(raw)
        names = [m["event"] for m in messages]
        assert names[:4] == ["grading_started", "grade_ready", "follow_up", "next_question"]
        probe = messages[3]["data"]
        assert probe["is_follow_up"] is True and probe["question_id"] is None
        assert probe["text"] and len(probe["text"]) >= 10
        assert set(messages[2]["data"]) == {"seq", "state"}

        # The follow-up is answerable through the same public endpoint.
        follow_up_turn = str(uuid.uuid4())
        answered = await http.post(
            f"/api/sessions/{sid}/turns",
            json={"turn_id": follow_up_turn, "answer": "Then every request misses at once."},
            headers=headers,
        )
        assert answered.status_code == 202
        assert answered.json()["question_id"] == QUESTION_ID  # the parent item

    async with db_engine.connect() as conn:
        row = (
            await conn.execute(
                text(
                    "SELECT t.is_follow_up, t.question_text, f.status, f.reason "
                    "FROM turns t JOIN follow_ups f ON f.session_id = t.session_id "
                    "WHERE t.turn_id = :t"
                ),
                {"t": uuid.UUID(follow_up_turn)},
            )
        ).one()
    assert row.is_follow_up is True and row.question_text == probe["text"]
    assert (row.status, row.reason) == ("ready", "high_uncertainty")
