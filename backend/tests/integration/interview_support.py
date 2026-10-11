"""Shared set-up for the Day-22 turn, worker and SSE tests.

The question inserted here is the one the Day-18 grading recordings were made
against (``fixtures/recording_plans/grade_answer.json``), reconstructed field by
field - so a turn answered with a recorded answer is graded by the *real*
``grade_answer`` pipeline, replaying the real model's recorded reply offline.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from app.blueprint import build_blueprint
from app.interview import EventType, append_event, create_session
from app.models.question import Question
from tests.unit.blueprint.conftest import BACKEND_SPEC

BACKEND_DIR = Path(__file__).resolve().parents[2]
PLAN = json.loads(
    (BACKEND_DIR / "fixtures" / "recording_plans" / "grade_answer.json").read_text(encoding="utf-8")
)
RECORDED: dict[str, dict[str, str]] = {e["name"]: e["inputs"] for e in PLAN["requests"]}

QUESTION_ID = "day22-cache-001"
SUBTOPIC = "day22_caching_sub"
BLUEPRINT = build_blueprint(BACKEND_SPEC)
STRONG_ANSWER = RECORDED["grade_answer_strong"]["answer"]
PARTIAL_ANSWER = RECORDED["grade_answer_partial"]["answer"]


def _concepts(rendered: str) -> list[dict[str, Any]]:
    out = []
    for line in rendered.splitlines():
        key, hint = line[2:].split(": ", 1)
        out.append({"key": key, "weight": 1, "hint": hint})
    return out


async def insert_question(engine: AsyncEngine, question_id: str = QUESTION_ID) -> str:
    inputs = RECORDED["grade_answer_strong"]
    async with engine.begin() as conn:
        for key, parent in (
            ("day22_dom", None),
            ("day22_topic", "day22_dom"),
            (SUBTOPIC, "day22_topic"),
        ):
            await conn.execute(
                text(
                    "INSERT INTO topics (key, parent_key, display_name, domain) "
                    "VALUES (:k, :p, :k, 'day22_dom') ON CONFLICT DO NOTHING"
                ),
                {"k": key, "p": parent},
            )
    maker = async_sessionmaker(engine, expire_on_commit=False)
    async with maker() as db:
        db.add(
            Question(
                id=question_id,
                topic_key="day22_topic",
                subtopic_key=SUBTOPIC,
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
    return question_id


async def user(engine: AsyncEngine, email: str = "candidate@example.com") -> uuid.UUID:
    async with engine.begin() as conn:
        return await conn.scalar(  # type: ignore[no-any-return]
            text("INSERT INTO users (email, password_hash) VALUES (:e, 'h') RETURNING id"),
            {"e": email},
        )


async def session_in_asking(
    maker: async_sessionmaker[AsyncSession], user_id: uuid.UUID, question_id: str = QUESTION_ID
) -> uuid.UUID:
    """A session driven CREATED -> PREPARING -> ASKING through the Day-21 service
    layer.  No HTTP route does this yet (session creation/start is not Day 22)."""
    async with maker() as db:
        sid = (await create_session(db, None, user_id=user_id, blueprint=BLUEPRINT)).session_id
    async with maker() as db:
        await append_event(
            db, None, session_id=sid, user_id=user_id, event_type=EventType.PREPARATION_STARTED
        )
    async with maker() as db:
        await append_event(
            db,
            None,
            session_id=sid,
            user_id=user_id,
            event_type=EventType.QUESTION_ASKED,
            payload={"question_id": question_id},
        )
    return sid


async def event_types(engine: AsyncEngine, sid: uuid.UUID) -> list[str]:
    async with engine.connect() as conn:
        rows = await conn.execute(
            text("SELECT type FROM interview_events WHERE session_id = :s ORDER BY seq"),
            {"s": sid},
        )
        return [r[0] for r in rows]


async def turn_rows(engine: AsyncEngine, sid: uuid.UUID) -> list[dict[str, Any]]:
    async with engine.connect() as conn:
        rows = await conn.execute(
            text(
                "SELECT turn_id, answer_text, grading_status, grading_attempts, grading_error, "
                "score, graded_at FROM turns WHERE session_id = :s ORDER BY created_at"
            ),
            {"s": sid},
        )
        return [dict(r._mapping) for r in rows]


async def session_cost(engine: AsyncEngine, sid: uuid.UUID) -> tuple[Any, int]:
    async with engine.connect() as conn:
        row = (
            await conn.execute(
                text("SELECT cost_usd, total_tokens FROM interview_sessions WHERE id = :s"),
                {"s": sid},
            )
        ).one()
        return row[0], row[1]
