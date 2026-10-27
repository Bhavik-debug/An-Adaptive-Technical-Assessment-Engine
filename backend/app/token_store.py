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
#   google_jti:{jti}     -> "1"          a Google ID token already exchanged (Day 27)
_GOOGLE_JTI_KEY = "google_jti:{jti}"


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


async def reserve_login_attempt(
    redis: Redis, *, email: str, lockout_after: int, lockout_for: dt.timedelta
) -> bool:
    """Count a login attempt *before* the password is checked. False = locked out.

    Keyed on the submitted email rather than the IP: an attacker spraying one
    password across many accounts from one address is a different problem
    (rate limiting, Day 30), while this stops a single account being ground down.

    The count and the window start are one Redis transaction, and the check is
    on the value ``INCR`` returned, so parallel requests cannot all see "not
    locked yet": exactly ``lockout_after`` attempts per window reach the
    password check. (A separate read-then-increment let a burst of 40
    concurrent guesses get 27 checked against a limit of 10.) A successful
    login clears the count (:func:`clear_login_failures`), so only failures
    accumulate in practice.
    """
    key = _lockout_key(email)
    seconds = max(int(lockout_for.total_seconds()), 1)
    pipe = redis.pipeline(transaction=True)
    # Start the window on the first attempt, so the lockout is "N failures
    # within the window" rather than "N failures ever".
    pipe.set(key, 0, ex=seconds, nx=True)
    pipe.incr(key)
    _, count = await pipe.execute()
    if int(count) == lockout_after:
        # Tripped: hold the door shut for the full window. Only at the trip, so
        # a real user retrying while locked out does not keep extending it.
        await redis.expire(key, seconds)
    return int(count) <= lockout_after


async def clear_login_failures(redis: Redis, *, email: str) -> None:
    """A successful login wipes the slate."""
    await redis.delete(_lockout_key(email))


async def claim_google_token(redis: Redis, *, token_id: str, ttl: dt.timedelta) -> bool:
    """Spend one Google ID token. False means it was already exchanged.

    A Google credential is valid for about an hour; without this, one captured
    from a log, a proxy or a browser extension could be replayed to sign in
    again until it expired. ``SET NX`` is the check and the claim in one
    operation, for the same reason as :func:`consume_refresh_token`. The entry
    lives until the token itself would have expired, and no longer.
    """
    seconds = max(int(ttl.total_seconds()), 1)
    claimed = await redis.set(_GOOGLE_JTI_KEY.format(jti=token_id), "1", ex=seconds, nx=True)
    return bool(claimed)
