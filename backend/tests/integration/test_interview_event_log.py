"""The event log against a real Postgres: append, order, refuse, roll back, race.

Everything here goes through the real migration (0003 included) and the real
service functions. Redis is not involved - ``cache=None`` - so these prove the
durable half on its own; the cache has its own file.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
import pytest_asyncio
from alembic import command
from alembic.config import Config
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from app.blueprint import build_blueprint
from app.interview import (
    EventLogCorrupted,
    EventType,
    IllegalTransition,
    InterviewState,
    InvalidEventPayload,
    SeqConflict,
    SessionNotFound,
    append_event,
    create_session,
    fold,
    load_state,
    stage_append,
)
from app.interview.store import read_events
from tests.unit.blueprint.conftest import BACKEND_SPEC
from tests.unit.interview.scenario import HAPPY_PATH

E = EventType
S = InterviewState
BACKEND_DIR = Path(__file__).resolve().parents[2]
BLUEPRINT = build_blueprint(BACKEND_SPEC)


@pytest_asyncio.fixture
async def maker(db_engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(db_engine, expire_on_commit=False)


async def _user(engine: AsyncEngine, email: str = "candidate@example.com") -> uuid.UUID:
    async with engine.begin() as conn:
        return await conn.scalar(  # type: ignore[no-any-return]
            text("INSERT INTO users (email, password_hash) VALUES (:e, 'h') RETURNING id"),
            {"e": email},
        )


@pytest_asyncio.fixture
async def owner(db_engine: AsyncEngine) -> uuid.UUID:
    return await _user(db_engine)


@pytest_asyncio.fixture
async def session_id(maker, owner) -> AsyncIterator[uuid.UUID]:
    async with maker() as db:
        state = await create_session(db, None, user_id=owner, blueprint=BLUEPRINT)
    yield state.session_id


async def _append(maker, sid, owner, event_type, payload=None, **kw):
    async with maker() as db:
        return await append_event(
            db, None, session_id=sid, user_id=owner, event_type=event_type, payload=payload, **kw
        )


async def _events(engine: AsyncEngine, sid: uuid.UUID) -> list[tuple[int, str]]:
    async with engine.connect() as conn:
        rows = await conn.execute(
            text("SELECT seq, type FROM interview_events WHERE session_id = :s ORDER BY seq"),
            {"s": sid},
        )
        return [(r[0], r[1]) for r in rows]


async def _session_row(engine: AsyncEngine, sid: uuid.UUID):
    async with engine.connect() as conn:
        return (
            await conn.execute(
                text("SELECT state, ended_at, blueprint FROM interview_sessions WHERE id = :s"),
                {"s": sid},
            )
        ).one()


# ---------------------------------------------------------------------------
# Creation and the happy path
# ---------------------------------------------------------------------------


async def test_create_writes_the_row_and_seq_0(db_engine, maker, owner):
    async with maker() as db:
        state = await create_session(db, None, user_id=owner, blueprint=BLUEPRINT)
    assert (state.state, state.seq) == (S.CREATED, 0)
    assert await _events(db_engine, state.session_id) == [(0, "SESSION_CREATED")]
    row = await _session_row(db_engine, state.session_id)
    assert row.state == "CREATED"
    assert row.ended_at is None
    assert row.blueprint == BLUEPRINT.to_dict()


async def test_full_interview_is_persisted_in_order_and_replays(
    db_engine, maker, owner, session_id
):
    states = []
    for event_type, payload in HAPPY_PATH[1:]:
        states.append(await _append(maker, session_id, owner, event_type, payload))

    rows = await _events(db_engine, session_id)
    assert rows == [(i, t.value) for i, (t, _) in enumerate(HAPPY_PATH)]
    assert [s.seq for s in states] == list(range(1, len(HAPPY_PATH)))

    # The fold of what is in the table equals the state the last append returned.
    async with maker() as db:
        replayed = fold(session_id, await read_events(db, session_id))
        loaded = await load_state(db, None, session_id=session_id, user_id=owner)
    assert replayed == states[-1] == loaded
    assert replayed.state is S.COMPLETED

    # The projection tracked every step and the session is closed.
    row = await _session_row(db_engine, session_id)
    assert row.state == "COMPLETED"
    assert row.ended_at is not None


async def test_projection_follows_each_transition(db_engine, maker, owner, session_id):
    for event_type, payload in HAPPY_PATH[1:5]:
        state = await _append(maker, session_id, owner, event_type, payload)
        assert (await _session_row(db_engine, session_id)).state == state.state.value


async def test_payload_is_stored_canonically(db_engine, maker, owner, session_id):
    for event_type, payload in HAPPY_PATH[1:4]:
        await _append(maker, session_id, owner, event_type, payload)
    async with db_engine.connect() as conn:
        payload = await conn.scalar(
            text("SELECT payload FROM interview_events WHERE session_id = :s AND seq = 2"),
            {"s": session_id},
        )
    assert payload == {"question_id": "dsa-001", "is_follow_up": False}


# ---------------------------------------------------------------------------
# Rejections leave nothing behind (transaction rollback)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("event_type", "payload", "error"),
    [
        (E.ANSWER_SUBMITTED, {"turn_id": str(uuid.uuid4())}, IllegalTransition),
        (E.SESSION_COMPLETED, {}, IllegalTransition),
        (E.SESSION_CREATED, {}, IllegalTransition),
        (E.PREPARATION_STARTED, {"note": "x"}, InvalidEventPayload),
    ],
)
async def test_rejected_events_write_nothing(
    db_engine, maker, owner, session_id, event_type, payload, error
):
    with pytest.raises(error):
        await _append(maker, session_id, owner, event_type, payload)
    assert await _events(db_engine, session_id) == [(0, "SESSION_CREATED")]
    assert (await _session_row(db_engine, session_id)).state == "CREATED"


async def test_a_failure_after_staging_rolls_back_the_event_and_projection(
    db_engine, maker, owner, session_id
):
    """The event row and the projection update are one unit with the commit."""

    class Boom(Exception):
        pass

    async with maker() as db:
        with pytest.raises(Boom):
            try:
                await stage_append(
                    db,
                    session_id=session_id,
                    user_id=owner,
                    event_type=E.PREPARATION_STARTED,
                    payload={},
                )
                raise Boom
            except Boom:
                await db.rollback()
                raise
    assert await _events(db_engine, session_id) == [(0, "SESSION_CREATED")]
    assert (await _session_row(db_engine, session_id)).state == "CREATED"
    # And the session is still usable afterwards.
    assert (await _append(maker, session_id, owner, E.PREPARATION_STARTED)).seq == 1


async def test_completed_sessions_accept_nothing(db_engine, maker, owner, session_id):
    for event_type, payload in HAPPY_PATH[1:]:
        await _append(maker, session_id, owner, event_type, payload)
    for event_type in EventType:
        with pytest.raises(IllegalTransition):
            await _append(maker, session_id, owner, event_type, {})
    assert len(await _events(db_engine, session_id)) == len(HAPPY_PATH)


# ---------------------------------------------------------------------------
# Ownership
# ---------------------------------------------------------------------------


async def test_another_user_cannot_append_or_read(db_engine, maker, session_id):
    intruder = await _user(db_engine, "intruder@example.com")
    with pytest.raises(SessionNotFound):
        await _append(maker, session_id, intruder, E.PREPARATION_STARTED)
    async with maker() as db:
        with pytest.raises(SessionNotFound):
            await load_state(db, None, session_id=session_id, user_id=intruder)
    assert len(await _events(db_engine, session_id)) == 1


async def test_unknown_session_is_not_found(maker, owner):
    with pytest.raises(SessionNotFound):
        await _append(maker, uuid.uuid4(), owner, E.PREPARATION_STARTED)


# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------


async def test_expected_seq_detects_a_lost_update(maker, owner, session_id):
    state = await _append(maker, session_id, owner, E.PREPARATION_STARTED, expected_seq=0)
    with pytest.raises(SeqConflict) as exc:
        await _append(
            maker, session_id, owner, E.QUESTION_ASKED, {"question_id": "q-1"}, expected_seq=0
        )
    assert (exc.value.expected, exc.value.actual) == (0, state.seq)


async def test_concurrent_conflicting_transitions_admit_exactly_one(
    db_engine, maker, owner, session_id
):
    """Ten requests race to move CREATED -> PREPARING. The row lock serialises
    them; the first wins and every other one sees PREPARING and is refused."""
    results = await asyncio.gather(
        *(_append(maker, session_id, owner, E.PREPARATION_STARTED) for _ in range(10)),
        return_exceptions=True,
    )
    winners = [r for r in results if not isinstance(r, BaseException)]
    losers = [r for r in results if isinstance(r, BaseException)]
    assert len(winners) == 1
    assert all(isinstance(r, IllegalTransition) for r in losers), losers
    assert await _events(db_engine, session_id) == [
        (0, "SESSION_CREATED"),
        (1, "PREPARATION_STARTED"),
    ]


async def test_concurrent_legal_appends_get_distinct_gapless_seqs(db_engine, maker, owner):
    """Different questions asked concurrently from SELECTING cannot all be legal
    - but each racer that wins must get the next seq, with no gaps."""
    async with maker() as db:
        sid = (await create_session(db, None, user_id=owner, blueprint=BLUEPRINT)).session_id
    for event_type, payload in HAPPY_PATH[1:5]:
        await _append(maker, sid, owner, event_type, payload)
    await _append(maker, sid, owner, E.SELECTION_STARTED)  # seq 5, SELECTING

    results = await asyncio.gather(
        *(
            _append(maker, sid, owner, E.QUESTION_ASKED, {"question_id": f"q-{i}"})
            for i in range(8)
        ),
        return_exceptions=True,
    )
    assert sum(not isinstance(r, BaseException) for r in results) == 1
    seqs = [seq for seq, _ in await _events(db_engine, sid)]
    assert seqs == list(range(len(seqs)))


async def test_unique_seq_is_the_backstop_without_the_lock(
    db_engine, maker, owner, session_id, monkeypatch
):
    """If a future code path forgot the row lock, two writers would fold the
    same head and both write seq 1. The unique index must refuse the second,
    and the service must report that as a SeqConflict - not a 500, and not a
    forked history."""
    import sqlalchemy as sa

    from app.interview import store
    from app.models.interview import InterviewSession

    async def unlocked(db, *, session_id, user_id):
        return await db.scalar(
            sa.select(InterviewSession).where(
                InterviewSession.id == session_id, InterviewSession.user_id == user_id
            )
        )

    monkeypatch.setattr(store, "lock_session", unlocked)

    first = maker()
    await stage_append(
        first, session_id=session_id, user_id=owner, event_type=E.PREPARATION_STARTED, payload={}
    )  # seq 1 flushed, not committed

    # The second writer folds the committed log (seq 0 only), so it also picks
    # seq 1; its INSERT then waits on the first transaction's index entry.
    second = asyncio.create_task(_append(maker, session_id, owner, E.PREPARATION_STARTED))
    await asyncio.sleep(0.3)
    assert not second.done()
    await first.commit()
    await first.close()

    with pytest.raises(SeqConflict):
        await second
    assert await _events(db_engine, session_id) == [
        (0, "SESSION_CREATED"),
        (1, "PREPARATION_STARTED"),
    ]


# ---------------------------------------------------------------------------
# Append-only, enforced by the database (migration 0003)
# ---------------------------------------------------------------------------


async def test_update_is_rejected_by_the_database(db_engine, session_id):
    with pytest.raises(DBAPIError, match="append-only"):
        async with db_engine.begin() as conn:
            await conn.execute(
                text("UPDATE interview_events SET type = 'X' WHERE session_id = :s"),
                {"s": session_id},
            )
    assert await _events(db_engine, session_id) == [(0, "SESSION_CREATED")]


async def test_direct_delete_is_rejected_by_the_database(db_engine, session_id):
    with pytest.raises(DBAPIError, match="append-only"):
        async with db_engine.begin() as conn:
            await conn.execute(
                text("DELETE FROM interview_events WHERE session_id = :s"), {"s": session_id}
            )
    assert len(await _events(db_engine, session_id)) == 1


async def test_cascade_delete_from_the_session_still_works(db_engine, owner, session_id):
    """Deleting an account must still delete its history."""
    async with db_engine.begin() as conn:
        await conn.execute(text("DELETE FROM users WHERE id = :u"), {"u": owner})
    assert await _events(db_engine, session_id) == []


async def test_negative_seq_is_rejected_by_the_database(db_engine, session_id):
    with pytest.raises(DBAPIError, match="ck_interview_events_seq_non_negative"):
        async with db_engine.begin() as conn:
            await conn.execute(
                text(
                    "INSERT INTO interview_events (session_id, seq, type, payload) "
                    "VALUES (:s, -1, 'SESSION_CREATED', '{}'::jsonb)"
                ),
                {"s": session_id},
            )


# ---------------------------------------------------------------------------
# A corrupted log is refused, not extended
# ---------------------------------------------------------------------------


async def test_corrupted_log_blocks_reads_and_appends(db_engine, maker, owner, session_id):
    async with db_engine.begin() as conn:  # bypass the service, as a bad tool would
        await conn.execute(
            text(
                "INSERT INTO interview_events (session_id, seq, type, payload) "
                "VALUES (:s, 1, 'ANSWER_GRADED', '{}'::jsonb)"
            ),
            {"s": session_id},
        )
    async with maker() as db:
        with pytest.raises(EventLogCorrupted) as exc:
            await load_state(db, None, session_id=session_id, user_id=owner)
    assert exc.value.seq == 1
    with pytest.raises(EventLogCorrupted):
        await _append(maker, session_id, owner, E.PREPARATION_STARTED)
    assert len(await _events(db_engine, session_id)) == 2


# ---------------------------------------------------------------------------
# Migration 0003 rolls back and re-applies
# ---------------------------------------------------------------------------


def _alembic(url: str) -> Config:
    cfg = Config(str(BACKEND_DIR / "alembic.ini"))
    cfg.set_main_option("script_location", str(BACKEND_DIR / "migrations"))
    cfg.set_main_option("sqlalchemy.url", url)
    return cfg


async def _trigger_and_check(engine: AsyncEngine) -> tuple[int, int]:
    async with engine.connect() as conn:
        triggers = await conn.scalar(
            text(
                "SELECT count(*) FROM pg_trigger "
                "WHERE tgname = 'trg_interview_events_append_only'"
            )
        )
        checks = await conn.scalar(
            text(
                "SELECT count(*) FROM pg_constraint "
                "WHERE conname = 'ck_interview_events_seq_non_negative'"
            )
        )
    return int(triggers or 0), int(checks or 0)


async def test_migration_0003_downgrades_and_upgrades(db_engine, migrated_database_url, session_id):
    cfg = _alembic(migrated_database_url)
    assert await _trigger_and_check(db_engine) == (1, 1)
    try:
        await asyncio.to_thread(command.downgrade, cfg, "0002")
        assert await _trigger_and_check(db_engine) == (0, 0)
        # Without the trigger an UPDATE goes through - proving the trigger is
        # what blocked it, not something else.
        async with db_engine.begin() as conn:
            await conn.execute(
                text("UPDATE interview_events SET payload = '{}'::jsonb WHERE session_id = :s"),
                {"s": session_id},
            )
    finally:
        await asyncio.to_thread(command.upgrade, cfg, "head")
    assert await _trigger_and_check(db_engine) == (1, 1)
