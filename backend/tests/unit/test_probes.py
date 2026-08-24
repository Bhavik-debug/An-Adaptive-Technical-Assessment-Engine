"""Dependency probes fail loudly rather than silently reporting healthy."""

from __future__ import annotations

import pytest

import app.cache as cache
import app.db as db
from app.config import get_settings


async def test_database_probe_without_initialisation_raises(env):
    env()
    await db.dispose_engine()
    with pytest.raises(RuntimeError, match="not initialised"):
        await db.check_database(timeout_s=0.1)


async def test_redis_probe_without_initialisation_raises(env):
    env()
    await cache.dispose_redis()
    with pytest.raises(RuntimeError, match="not initialised"):
        await cache.check_redis(timeout_s=0.1)


async def test_engine_is_created_once_and_disposable(env):
    env()
    settings = get_settings()
    first = db.init_engine(settings)
    assert db.init_engine(settings) is first
    assert db.get_engine() is first
    await db.dispose_engine()
    with pytest.raises(RuntimeError):
        db.get_engine()


async def test_redis_client_is_created_once_and_disposable(env):
    env()
    settings = get_settings()
    first = cache.init_redis(settings)
    assert cache.init_redis(settings) is first
    assert cache.get_redis() is first
    await cache.dispose_redis()
    with pytest.raises(RuntimeError):
        cache.get_redis()
