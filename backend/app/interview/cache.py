"""The Redis copy of ``fold(events)`` - plan section 11.1's working state.

One Redis hash per session::

    sess:{session_id}   seq   -> "7"              the last event the fold includes
                        state -> "{...json...}"   SessionState.to_dict()
                        (TTL: SESSION_CACHE_TTL_S, 4 h by default - plan 11.1)

**Redis is never the source of truth.**  Postgres is.  The rules that keep it
that way:

1. **Written only after commit.**  A cached state is always the fold of events
   that are durably in ``interview_events``; a rolled-back append never reaches
   Redis.
2. **Monotonic in ``seq``.**  A write lands only if its ``seq`` is *greater*
   than the one already cached, checked and written in one Lua script so two
   writers cannot interleave.  A slow request holding an old fold therefore
   cannot overwrite a newer one (the "stale write" problem).
3. **Trusted only at the database's head.**  A reader compares the cached
   ``seq`` with ``max(seq)`` in Postgres and uses the cached state only when
   they are equal (see :mod:`app.interview.service`).  Anything else is rebuilt
   from the log.
4. **Failure is a miss.**  A timeout, a connection error, an unreadable entry -
   each is logged and degrades to a rebuild from Postgres.  Redis being down
   makes a session slower, never wrong and never unavailable.  Same contract as
   the LLM response cache (:mod:`app.llm.cache`).
"""

from __future__ import annotations

import asyncio
import enum
import json
import logging
import uuid
from collections.abc import Awaitable
from dataclasses import dataclass
from typing import cast

from redis.asyncio import Redis
from redis.exceptions import RedisError, ResponseError

from app.config import Settings
from app.interview.state import SessionState

log = logging.getLogger(__name__)

_KEY = "sess:{session_id}"

# KEYS[1] = the hash, ARGV = (seq, state json, ttl seconds).
# Writes only when nothing is cached or the cached seq is strictly older. A
# cached seq that is not a number is treated as absent, so a damaged entry is
# repaired by the next write rather than blocking every future one.
_STORE_IF_NEWER = """
local cur = tonumber(redis.call('HGET', KEYS[1], 'seq'))
if cur and cur >= tonumber(ARGV[1]) then
  return 0
end
redis.call('DEL', KEYS[1])
redis.call('HSET', KEYS[1], 'seq', ARGV[1], 'state', ARGV[2])
redis.call('EXPIRE', KEYS[1], tonumber(ARGV[3]))
return 1
"""


def cache_key(session_id: uuid.UUID) -> str:
    return _KEY.format(session_id=session_id)


class CacheStatus(enum.StrEnum):
    HIT = "hit"
    MISS = "miss"
    #: Present but unusable: undecodable, another schema version, or its
    #: ``seq`` field disagrees with the state inside it.  The caller rebuilds
    #: and replaces it.
    INVALID = "invalid"
    #: Redis did not answer (timeout, connection refused).  Not the entry's
    #: fault; the caller rebuilds but does not delete anything.
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True, slots=True)
class CacheRead:
    status: CacheStatus
    state: SessionState | None = None


class SessionStateCache:
    """Read and write folded session state, never raising on a Redis failure."""

    def __init__(self, redis: Redis, *, ttl_s: int, timeout_s: float) -> None:
        if ttl_s <= 0:
            # An unbounded working-state cache grows by one key per session
            # forever; plan 11.1 gives it a lifetime.
            raise ValueError(f"ttl_s must be positive, got {ttl_s!r}")
        if timeout_s <= 0:
            raise ValueError(f"timeout_s must be positive, got {timeout_s!r}")
        self._redis = redis
        self._ttl_s = ttl_s
        self._timeout_s = timeout_s

    @classmethod
    def from_settings(cls, redis: Redis, settings: Settings) -> SessionStateCache:
        return cls(
            redis,
            ttl_s=settings.session_cache_ttl_s,
            timeout_s=settings.session_cache_timeout_s,
        )

    async def get(self, session_id: uuid.UUID) -> CacheRead:
        key = cache_key(session_id)
        try:
            raw = await asyncio.wait_for(
                cast("Awaitable[dict[str, str]]", self._redis.hgetall(key)),
                timeout=self._timeout_s,
            )
        except ResponseError as exc:
            # e.g. WRONGTYPE: something that is not our hash sits at our key.
            log.warning("session cache entry %s unreadable (%s); will replace", key, exc)
            return CacheRead(CacheStatus.INVALID)
        except (RedisError, OSError, TimeoutError) as exc:
            log.warning("session cache read failed, rebuilding from the event log: %r", exc)
            return CacheRead(CacheStatus.UNAVAILABLE)

        if not raw:
            return CacheRead(CacheStatus.MISS)
        try:
            seq = int(raw["seq"])
            state = SessionState.from_dict(json.loads(raw["state"]))
        except (KeyError, ValueError, TypeError) as exc:
            log.warning("discarding unreadable session cache entry %s: %s", key, exc)
            return CacheRead(CacheStatus.INVALID)
        if state.seq != seq or state.session_id != session_id:
            log.warning(
                "discarding inconsistent session cache entry %s (field seq %s, state seq %s)",
                key,
                seq,
                state.seq,
            )
            return CacheRead(CacheStatus.INVALID)
        return CacheRead(CacheStatus.HIT, state)

    async def store(self, state: SessionState) -> bool:
        """Cache ``state`` if it is newer than what is there.  True if written.

        False covers both "an equal or newer fold is already cached" (correct,
        nothing to do) and "Redis failed" (logged).  Neither is an error for
        the caller: the durable event is already committed.
        """
        if state.state is None:
            raise ValueError("refusing to cache the empty pre-creation state")
        key = cache_key(state.session_id)
        blob = json.dumps(state.to_dict(), separators=(",", ":"), sort_keys=True)
        try:
            written = await asyncio.wait_for(
                cast(
                    "Awaitable[int]",
                    self._redis.eval(
                        _STORE_IF_NEWER, 1, key, str(state.seq), blob, str(self._ttl_s)
                    ),
                ),
                timeout=self._timeout_s,
            )
        except (RedisError, OSError, TimeoutError) as exc:
            log.warning("session cache write failed for %s at seq %s: %r", key, state.seq, exc)
            return False
        return bool(written)

    async def invalidate(self, session_id: uuid.UUID) -> bool:
        """Drop the entry.  Used when it is unreadable, or ahead of the log."""
        try:
            await asyncio.wait_for(
                cast("Awaitable[int]", self._redis.delete(cache_key(session_id))),
                timeout=self._timeout_s,
            )
        except (RedisError, OSError, TimeoutError) as exc:
            log.warning("session cache invalidation failed for %s: %r", session_id, exc)
            return False
        return True

    async def replace(self, state: SessionState) -> bool:
        """Overwrite whatever is cached, even a higher ``seq``.

        Only for an entry proven wrong - one *ahead* of the database, which
        normal operation cannot produce (rule 1) but a database restore can.
        Delete-then-store: if a legitimate newer write lands in between, the
        compare-and-set keeps it, which is the right outcome.
        """
        if not await self.invalidate(state.session_id):
            return False
        return await self.store(state)


__all__ = [
    "CacheRead",
    "CacheStatus",
    "SessionStateCache",
    "cache_key",
]
