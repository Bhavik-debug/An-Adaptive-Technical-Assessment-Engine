"""The session-state cache against a real Redis and a real Postgres.

What these establish that the unit tests cannot: the Lua compare-and-set is
really atomic and monotonic, TTLs are really set, a real server-side stall
really times out, and every way the cache can be wrong - absent, behind, ahead,
garbage, unreachable - ends with the state Postgres says, never Redis.
"""

from __future__ import annotations

import asyncio
import json
import random
import uuid
from collections.abc import AsyncIterator

import pytest
import pytest_asyncio
from redis.asyncio import Redis
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from app.blueprint import build_blueprint
from app.interview import (
    CacheStatus,
    EventType,
    InterviewState,
    SessionNotFound,
    SessionStateCache,
    append_event,
    cache_key,
    create_session,
    fold,
    load_state,
)
from tests.unit.blueprint.conftest import BACKEND_SPEC
from tests.unit.interview.scenario import HAPPY_PATH, as_stored

E = EventType
BLUEPRINT = build_blueprint(BACKEND_SPEC)


@pytest_asyncio.fixture
async def redis(redis_url: str) -> AsyncIterator[Redis]:
    client = Redis.from_url(redis_url, decode_responses=True)
    yield client
    await client.aclose()


@pytest_asyncio.fixture
async def cache(redis: Redis) -> SessionStateCache:
    return SessionStateCache(redis, ttl_s=14_400, timeout_s=0.5)


@pytest_asyncio.fixture
async def dead_cache() -> AsyncIterator[SessionStateCache]:
    """A cache whose Redis is not there - port 1 refuses connections."""
    client = Redis.from_url("redis://127.0.0.1:1/0", socket_connect_timeout=0.2)
    yield SessionStateCache(client, ttl_s=14_400, timeout_s=0.5)
    await client.aclose()


@pytest_asyncio.fixture
async def maker(db_engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(db_engine, expire_on_commit=False)


@pytest_asyncio.fixture
async def owner(db_engine: AsyncEngine) -> uuid.UUID:
    async with db_engine.begin() as conn:
        return await conn.scalar(  # type: ignore[no-any-return]
            text("INSERT INTO users (email, password_hash) VALUES ('c@x.com', 'h') RETURNING id")
        )


async def _create(maker, cache, owner) -> uuid.UUID:
    async with maker() as db:
        return (await create_session(db, cache, user_id=owner, blueprint=BLUEPRINT)).session_id


async def _append(maker, cache, sid, owner, n_from: int, n_to: int):
    state = None
    for event_type, payload in HAPPY_PATH[n_from:n_to]:
        async with maker() as db:
            state = await append_event(
                db, cache, session_id=sid, user_id=owner, event_type=event_type, payload=payload
            )
    return state


async def _load(maker, cache, sid, owner):
    async with maker() as db:
        return await load_state(db, cache, session_id=sid, user_id=owner)


async def _cached_seq(redis: Redis, sid: uuid.UUID) -> int | None:
    raw = await redis.hget(cache_key(sid), "seq")
    return int(raw) if raw is not None else None


# ---------------------------------------------------------------------------
# Hit, miss, and staying at the head
# ---------------------------------------------------------------------------


async def test_create_and_append_keep_the_cache_at_the_head(redis, cache, maker, owner):
    sid = await _create(maker, cache, owner)
    assert await _cached_seq(redis, sid) == 0
    state = await _append(maker, cache, sid, owner, 1, 6)
    assert await _cached_seq(redis, sid) == state.seq == 5
    read = await cache.get(sid)
    assert read.status is CacheStatus.HIT and read.state == state
    ttl = await redis.ttl(cache_key(sid))
    assert 14_000 < ttl <= 14_400


async def test_a_hit_is_served_from_redis(redis, cache, maker, owner):
    """Proves the hit path really skips the fold: a cached state at the head is
    returned as-is (here, marked so it is recognisable)."""
    sid = await _create(maker, cache, owner)
    await _append(maker, cache, sid, owner, 1, 3)
    data = json.loads(await redis.hget(cache_key(sid), "state"))
    data["stop_reasons"] = ["served_from_cache"]
    await redis.hset(cache_key(sid), "state", json.dumps(data))
    assert (await _load(maker, cache, sid, owner)).stop_reasons == ("served_from_cache",)


async def test_a_miss_rebuilds_from_postgres_and_repopulates(redis, cache, maker, owner):
    sid = await _create(maker, cache, owner)
    expected = await _append(maker, cache, sid, owner, 1, 8)
    await redis.delete(cache_key(sid))
    assert (await cache.get(sid)).status is CacheStatus.MISS
    assert await _load(maker, cache, sid, owner) == expected
    assert await _cached_seq(redis, sid) == expected.seq


async def test_recovery_after_a_full_cache_flush(redis, cache, maker, owner):
    """The 'Redis restarted empty' case: every session rebuilds, none is lost."""
    sids = [await _create(maker, cache, owner) for _ in range(3)]
    expected = [await _append(maker, cache, sid, owner, 1, 4 + i) for i, sid in enumerate(sids)]
    await redis.flushdb()
    for sid, state in zip(sids, expected, strict=True):
        assert await _load(maker, cache, sid, owner) == state
        assert await _cached_seq(redis, sid) == state.seq


# ---------------------------------------------------------------------------
# Stale, ahead, garbage
# ---------------------------------------------------------------------------


async def test_a_stale_cache_is_never_returned(redis, cache, maker, owner):
    """Appends made without the cache (e.g. Redis was briefly down) leave it
    behind; the reader sees cached seq < head and rebuilds."""
    sid = await _create(maker, cache, owner)
    await _append(maker, cache, sid, owner, 1, 3)  # cache at seq 2
    newest = await _append(maker, None, sid, owner, 3, 6)  # log at seq 5
    assert await _cached_seq(redis, sid) == 2
    loaded = await _load(maker, cache, sid, owner)
    assert loaded == newest and loaded.seq == 5
    assert await _cached_seq(redis, sid) == 5


async def test_a_cache_ahead_of_the_log_is_replaced(redis, cache, maker, owner):
    """Only a database restore produces this. Postgres wins."""
    sid = await _create(maker, cache, owner)
    truth = await _append(maker, cache, sid, owner, 1, 3)
    phantom = fold(sid, as_stored(list(HAPPY_PATH[:9])))  # seq 8 - never happened
    await redis.delete(cache_key(sid))
    assert await cache.store(phantom)
    assert await _load(maker, cache, sid, owner) == truth
    assert await _cached_seq(redis, sid) == truth.seq


@pytest.mark.parametrize(
    "corrupt",
    [
        lambda r, k: r.hset(k, mapping={"seq": "2", "state": "{garbage"}),
        lambda r, k: r.hset(k, mapping={"seq": "nope", "state": "{}"}),
        lambda r, k: r.set(k, "a plain string, not a hash"),
        lambda r, k: r.hset(k, mapping={"seq": "2", "state": json.dumps({"v": 999})}),
    ],
)
async def test_garbage_entries_are_replaced_with_the_truth(redis, cache, maker, owner, corrupt):
    sid = await _create(maker, cache, owner)
    truth = await _append(maker, cache, sid, owner, 1, 3)
    await redis.delete(cache_key(sid))
    await corrupt(redis, cache_key(sid))
    assert await _load(maker, cache, sid, owner) == truth
    read = await cache.get(sid)
    assert read.status is CacheStatus.HIT and read.state == truth


# ---------------------------------------------------------------------------
# Out-of-order writes (the Lua compare-and-set)
# ---------------------------------------------------------------------------


async def test_out_of_order_writes_keep_the_newest(redis, cache):
    sid = uuid.uuid4()
    states = [fold(sid, as_stored(list(HAPPY_PATH[:n]))) for n in range(1, len(HAPPY_PATH) + 1)]
    shuffled = states[:]
    random.Random(21).shuffle(shuffled)
    results = await asyncio.gather(*(cache.store(s) for s in shuffled))
    assert any(results)
    read = await cache.get(sid)
    assert read.state == states[-1]


async def test_an_old_writer_cannot_overwrite_a_new_one(redis, cache):
    sid = uuid.uuid4()
    new = fold(sid, as_stored(list(HAPPY_PATH[:7])))
    old = fold(sid, as_stored(list(HAPPY_PATH[:4])))
    assert await cache.store(new) is True
    assert await cache.store(old) is False
    assert await cache.store(new) is False  # equal seq: nothing to do
    assert (await cache.get(sid)).state == new


async def test_concurrent_appends_leave_the_cache_at_the_head(redis, cache, maker, owner):
    """Racing appends + racing cache writes: whatever interleaving happens, the
    cache ends at the durable head and equals the fold of the log."""
    sid = await _create(maker, cache, owner)
    await _append(maker, cache, sid, owner, 1, 5)  # seq 4: GRADING, graded

    async def try_append(event_type):
        async with maker() as db:
            try:
                return await append_event(
                    db, cache, session_id=sid, user_id=owner, event_type=event_type, payload={}
                )
            except Exception:  # noqa: BLE001 - losers of the race are expected
                return None

    await asyncio.gather(*(try_append(E.SELECTION_STARTED) for _ in range(6)))
    head = await _load(maker, None, sid, owner)
    assert head.state is InterviewState.SELECTING
    assert await _cached_seq(redis, sid) == head.seq
    assert (await cache.get(sid)).state == head


# ---------------------------------------------------------------------------
# Redis unavailable or slow: durable writes are unaffected
# ---------------------------------------------------------------------------


async def test_redis_down_does_not_block_or_lose_appends(dead_cache, maker, owner, caplog):
    sid = await _create(maker, dead_cache, owner)
    state = await _append(maker, dead_cache, sid, owner, 1, 6)
    assert state.seq == 5
    assert await _load(maker, dead_cache, sid, owner) == state
    assert "session cache" in caplog.text


async def test_recovery_when_redis_comes_back(redis, cache, dead_cache, maker, owner):
    sid = await _create(maker, dead_cache, owner)
    state = await _append(maker, dead_cache, sid, owner, 1, 6)
    assert await _cached_seq(redis, sid) is None  # nothing reached Redis
    assert await _load(maker, cache, sid, owner) == state  # Redis "back"
    assert await _cached_seq(redis, sid) == state.seq


async def test_a_stalled_redis_times_out_into_a_rebuild(redis, maker, owner):
    """CLIENT PAUSE makes the real server stop answering for 400 ms."""
    fast = SessionStateCache(redis, ttl_s=14_400, timeout_s=0.05)
    sid = await _create(maker, fast, owner)
    state = await _append(maker, fast, sid, owner, 1, 4)

    side = Redis.from_url(_url(redis))
    try:
        await side.execute_command("CLIENT", "PAUSE", "400", "ALL")
        read = await fast.get(sid)
        assert read.status is CacheStatus.UNAVAILABLE
        # A load during the stall still answers, from Postgres.
        assert await _load(maker, fast, sid, owner) == state
    finally:
        await side.execute_command("CLIENT", "UNPAUSE")
        await side.aclose()
    await asyncio.sleep(0.05)
    # The stalled request may have left a zombie reply on that connection; a
    # fresh pool proves the cache is usable again once Redis answers.
    client = Redis.from_url(_url(redis), decode_responses=True)
    try:
        fresh = SessionStateCache(client, ttl_s=14_400, timeout_s=0.5)
        assert (await fresh.get(sid)).state == state
    finally:
        await client.aclose()


def _url(redis: Redis) -> str:
    kw = redis.connection_pool.connection_kwargs
    return f"redis://{kw['host']}:{kw['port']}/{kw.get('db', 0)}"


# ---------------------------------------------------------------------------
# The cache never bypasses ownership
# ---------------------------------------------------------------------------


async def test_a_cached_session_is_still_owner_only(db_engine, redis, cache, maker, owner):
    sid = await _create(maker, cache, owner)
    await _append(maker, cache, sid, owner, 1, 3)
    assert (await cache.get(sid)).status is CacheStatus.HIT
    async with db_engine.begin() as conn:
        intruder = await conn.scalar(
            text("INSERT INTO users (email, password_hash) VALUES ('i@x.com','h') RETURNING id")
        )
    with pytest.raises(SessionNotFound):
        await _load(maker, cache, sid, intruder)
