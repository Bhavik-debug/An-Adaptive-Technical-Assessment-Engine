"""Redis-backed refresh-token allowlist and login-failure counter.

A signed JWT alone cannot be revoked - that is the trade-off of stateless
tokens. Refresh tokens are long-lived, so for those we accept a lookup and keep
an allowlist: a refresh token is valid only if its ``jti`` is still recorded
here. Rotation then means "delete the old jti, record a new one", and logout
means "delete the jti".

Redis rather than Postgres because every entry has a natural expiry, TTL is
built in, and this is hot-path state that nobody needs to query historically.
"""

from __future__ import annotations

import datetime as dt
import uuid
from collections.abc import Awaitable
from typing import cast

from redis.asyncio import Redis

# redis-py exposes one class for both the sync and async APIs, so several methods
# are typed ``Awaitable[T] | T``. On the async client they are always awaitable;
# these casts tell mypy that without weakening the checking elsewhere.

# Two key shapes:
#   refresh:{jti}        -> user_id      the allowlist entry itself
#   refresh_user:{uid}   -> set of jti   the "family", so we can revoke all at once
_JTI_KEY = "refresh:{jti}"
_USER_KEY = "refresh_user:{user_id}"
_LOCKOUT_KEY = "login_fail:{email}"


def _jti_key(jti: uuid.UUID) -> str:
    return _JTI_KEY.format(jti=jti)


def _user_key(user_id: uuid.UUID) -> str:
    return _USER_KEY.format(user_id=user_id)


def _lockout_key(email: str) -> str:
    return _LOCKOUT_KEY.format(email=email.strip().lower())


async def remember_refresh_token(
    redis: Redis, *, jti: uuid.UUID, user_id: uuid.UUID, ttl: dt.timedelta
) -> None:
    """Record a freshly issued refresh token as valid."""
    seconds = max(int(ttl.total_seconds()), 1)
    pipe = redis.pipeline()
    pipe.set(_jti_key(jti), str(user_id), ex=seconds)
    pipe.sadd(_user_key(user_id), str(jti))
    # The family set outlives any single token; re-stamping the expiry on every
    # issue keeps it alive while the user is active and lets it lapse when not.
    pipe.expire(_user_key(user_id), seconds)
    await pipe.execute()


async def consume_refresh_token(redis: Redis, *, jti: uuid.UUID, user_id: uuid.UUID) -> bool:
    """Atomically spend a refresh token. False means it was not valid.

    DELETE returns the number of keys removed, so the delete IS the check. Doing
    ``exists`` and then ``delete`` would let two concurrent refreshes both see
    the token and both succeed - the single-use guarantee has to be one
    operation.
    """
    removed = await cast("Awaitable[int]", redis.delete(_jti_key(jti)))
    if removed == 0:
        return False
    await cast("Awaitable[int]", redis.srem(_user_key(user_id), str(jti)))
    return True


async def revoke_all_refresh_tokens(redis: Redis, *, user_id: uuid.UUID) -> int:
    """Invalidate every refresh token this user holds. Returns how many.

    Called on suspected theft: presenting a correctly signed refresh token whose
    jti is already spent means either a replay or a stolen copy, and we cannot
    tell which one is the legitimate user. Both sessions are ended and the user
    logs in again - the safe answer.
    """
    key = _user_key(user_id)
    jtis = await cast("Awaitable[set[str]]", redis.smembers(key))
    pipe = redis.pipeline()
    for jti in jtis:
        pipe.delete(_JTI_KEY.format(jti=jti))
    pipe.delete(key)
    await pipe.execute()
    return len(jtis)


async def register_login_failure(
    redis: Redis, *, email: str, lockout_after: int, lockout_for: dt.timedelta
) -> int:
    """Count a failed login. Returns the running total.

    Keyed on the submitted email rather than the IP: an attacker spraying one
    password across many accounts from one address is a different problem
    (rate limiting, Day 30), while this stops a single account being ground down.
    """
    key = _lockout_key(email)
    count = int(await redis.incr(key))
    if count == 1:
        # Start the window on the first failure, so the lockout is "10 failures
        # within N minutes" rather than "10 failures ever".
        await redis.expire(key, max(int(lockout_for.total_seconds()), 1))
    if count >= lockout_after:
        # Once tripped, hold the door shut for the full window.
        await redis.expire(key, max(int(lockout_for.total_seconds()), 1))
    return count


async def is_locked_out(redis: Redis, *, email: str, lockout_after: int) -> bool:
    raw = await redis.get(_lockout_key(email))
    return raw is not None and int(raw) >= lockout_after


async def clear_login_failures(redis: Redis, *, email: str) -> None:
    """A successful login wipes the slate."""
    await redis.delete(_lockout_key(email))
