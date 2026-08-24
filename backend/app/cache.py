"""Redis access.

Day 1 owns only the client lifecycle and a PING check. Hot session state, rate
limits and usage counters come later (Phases 5 and 8).
"""

from __future__ import annotations

import asyncio

from redis.asyncio import Redis

from app.config import Settings

_redis: Redis | None = None


def init_redis(settings: Settings) -> Redis:
    global _redis
    if _redis is None:
        _redis = Redis.from_url(
            settings.redis_url,
            decode_responses=True,
            socket_connect_timeout=settings.readiness_timeout_s,
        )
    return _redis


def get_redis() -> Redis:
    if _redis is None:
        raise RuntimeError("Redis client not initialised; call init_redis() first")
    return _redis


async def dispose_redis() -> None:
    global _redis
    if _redis is not None:
        await _redis.aclose()
        _redis = None


async def check_redis(timeout_s: float = 2.0) -> None:
    """Raise if Redis does not answer PING within `timeout_s`."""
    try:
        await asyncio.wait_for(get_redis().ping(), timeout=timeout_s)
    except TimeoutError as exc:
        raise TimeoutError(f"no response within {timeout_s}s") from exc
