"""The session-state cache's failure semantics, against a scripted fake Redis.

These prove the *code paths*: what each kind of failure turns into. They do not
prove anything about Redis itself - the Lua compare-and-set, real TTLs and a
real restart are exercised against a real server in
``tests/integration/test_session_cache.py``.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import ResponseError

from app.interview import CacheStatus, SessionStateCache, cache_key, fold
from tests.unit.interview.scenario import SESSION_ID, prefix


class FakeRedis:
    """Just the three commands the cache uses, with injectable failures."""

    def __init__(self) -> None:
        self.data: dict[str, dict[str, str]] = {}
        self.ttl: dict[str, int] = {}
        self.fail_with: BaseException | None = None
        self.delay_s = 0.0
        self.calls: list[str] = []

    async def _maybe_fail(self, name: str) -> None:
        self.calls.append(name)
        if self.delay_s:
            await asyncio.sleep(self.delay_s)
        if self.fail_with is not None:
            raise self.fail_with

    async def hgetall(self, key: str) -> dict[str, str]:
        await self._maybe_fail("hgetall")
        return dict(self.data.get(key, {}))

    async def delete(self, key: str) -> int:
        await self._maybe_fail("delete")
        return 1 if self.data.pop(key, None) is not None else 0

    async def eval(self, script: str, numkeys: int, key: str, seq: str, blob: str, ttl: str) -> int:
        # Mirrors the Lua script's semantics in Python.
        await self._maybe_fail("eval")
        current = self.data.get(key, {}).get("seq")
        try:
            cur = int(current) if current is not None else None
        except ValueError:
            cur = None
        if cur is not None and cur >= int(seq):
            return 0
        self.data[key] = {"seq": seq, "state": blob}
        self.ttl[key] = int(ttl)
        return 1


def _cache(fake: FakeRedis, timeout_s: float = 0.5) -> SessionStateCache:
    return SessionStateCache(fake, ttl_s=14_400, timeout_s=timeout_s)  # type: ignore[arg-type]


STATE_3 = fold(SESSION_ID, prefix(3))
STATE_5 = fold(SESSION_ID, prefix(5))


async def test_miss_on_an_empty_cache():
    read = await _cache(FakeRedis()).get(SESSION_ID)
    assert read.status is CacheStatus.MISS
    assert read.state is None


async def test_store_then_hit_round_trips_the_state():
    fake = FakeRedis()
    cache = _cache(fake)
    assert await cache.store(STATE_5) is True
    read = await cache.get(SESSION_ID)
    assert read.status is CacheStatus.HIT
    assert read.state == STATE_5
    assert fake.ttl[cache_key(SESSION_ID)] == 14_400
    assert cache_key(SESSION_ID) == f"sess:{SESSION_ID}"


async def test_an_older_state_never_overwrites_a_newer_one():
    cache = _cache(FakeRedis())
    assert await cache.store(STATE_5)
    assert await cache.store(STATE_3) is False
    assert (await cache.get(SESSION_ID)).state == STATE_5


async def test_replace_overrides_monotonicity():
    cache = _cache(FakeRedis())
    await cache.store(STATE_5)
    assert await cache.replace(STATE_3) is True
    assert (await cache.get(SESSION_ID)).state == STATE_3


async def test_refuses_to_cache_the_pre_creation_state():
    from app.interview import SessionState

    with pytest.raises(ValueError):
        await _cache(FakeRedis()).store(SessionState.initial(SESSION_ID))


@pytest.mark.parametrize(
    "entry",
    [
        {"seq": "5"},  # no state field
        {"seq": "5", "state": "{not json"},
        {"seq": "five", "state": "{}"},
        {"seq": "5", "state": json.dumps({"v": 999})},
    ],
)
async def test_unreadable_entries_are_invalid(entry):
    fake = FakeRedis()
    fake.data[cache_key(SESSION_ID)] = entry
    assert (await _cache(fake).get(SESSION_ID)).status is CacheStatus.INVALID


async def test_seq_field_disagreeing_with_the_state_is_invalid():
    fake = FakeRedis()
    fake.data[cache_key(SESSION_ID)] = {"seq": "9", "state": json.dumps(STATE_5.to_dict())}
    assert (await _cache(fake).get(SESSION_ID)).status is CacheStatus.INVALID


async def test_a_state_for_another_session_is_invalid():
    other = fold(__import__("uuid").uuid4(), prefix(5))
    fake = FakeRedis()
    fake.data[cache_key(SESSION_ID)] = {"seq": "4", "state": json.dumps(other.to_dict())}
    assert (await _cache(fake).get(SESSION_ID)).status is CacheStatus.INVALID


async def test_wrongtype_is_invalid_not_unavailable():
    fake = FakeRedis()
    fake.fail_with = ResponseError("WRONGTYPE Operation against a key holding the wrong kind")
    assert (await _cache(fake).get(SESSION_ID)).status is CacheStatus.INVALID


@pytest.mark.parametrize(
    "error", [RedisConnectionError("refused"), OSError("network down"), TimeoutError()]
)
async def test_connection_failures_degrade_to_unavailable(error, caplog):
    fake = FakeRedis()
    fake.fail_with = error
    cache = _cache(fake)
    assert (await cache.get(SESSION_ID)).status is CacheStatus.UNAVAILABLE
    assert await cache.store(STATE_5) is False
    assert await cache.invalidate(SESSION_ID) is False
    assert await cache.replace(STATE_5) is False
    assert "session cache" in caplog.text


async def test_a_slow_redis_times_out_as_unavailable():
    fake = FakeRedis()
    fake.delay_s = 0.2
    cache = _cache(fake, timeout_s=0.05)
    assert (await cache.get(SESSION_ID)).status is CacheStatus.UNAVAILABLE
    assert await cache.store(STATE_5) is False


@pytest.mark.parametrize(("ttl", "timeout"), [(0, 0.5), (-1, 0.5), (60, 0.0)])
def test_unbounded_or_zero_limits_are_rejected(ttl, timeout):
    with pytest.raises(ValueError):
        SessionStateCache(FakeRedis(), ttl_s=ttl, timeout_s=timeout)  # type: ignore[arg-type]


def test_from_settings_uses_the_configured_policy(env):
    from app.config import get_settings

    env({"SESSION_CACHE_TTL_S": "600", "SESSION_CACHE_TIMEOUT_S": "0.25"})
    cache = SessionStateCache.from_settings(FakeRedis(), get_settings())  # type: ignore[arg-type]
    assert (cache._ttl_s, cache._timeout_s) == (600, 0.25)


def test_default_policy_is_the_plans_four_hours(env):
    from app.config import get_settings

    env()
    settings = get_settings()
    assert settings.session_cache_ttl_s == 4 * 3600
    assert settings.session_cache_timeout_s == 0.5


async def test_payload_values_are_plain_json(monkeypatch: Any):
    """The stored blob is canonical JSON - sorted keys, no NaN."""
    fake = FakeRedis()
    await _cache(fake).store(STATE_5)
    blob = fake.data[cache_key(SESSION_ID)]["state"]
    assert json.loads(blob) == STATE_5.to_dict()
    assert blob == json.dumps(json.loads(blob), separators=(",", ":"), sort_keys=True)
