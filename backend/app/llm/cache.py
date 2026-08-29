"""Response cache for the LLM chokepoint.

Plan section 13.3: cache keyed by ``hash(task + prompt_version + inputs)``.

What it buys, in order of how much you notice it:

1. **Development.**  Re-running the same script twenty times costs one call.
2. **Quota.**  The free-tier ceiling is requests per day; a hit is not a request.
3. **Latency.**  A hit is a Redis round trip instead of a two-second generation.

Two rules the implementation exists to enforce:

* **Only deterministic tasks are cached.**  Caching a temperature-0.7 task would
  return the same answer every time, which destroys the variation that the
  temperature was chosen for.  ``TaskSpec.is_cacheable`` decides.
* **A cache failure is a cache miss, never a call failure.**  Redis being down
  must make the system slower and more expensive, not broken.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from typing import Any, Protocol

from redis.asyncio import Redis
from redis.exceptions import RedisError

log = logging.getLogger(__name__)

_KEY_PREFIX = "llm:v1"


@dataclass(frozen=True, slots=True)
class CachedCall:
    """What we keep about a call, so a hit can still be reported honestly."""

    payload: Any
    provider: str
    model: str


def cache_key(
    *,
    task: str,
    prompt_version: str,
    prompt_fingerprint: str,
    schema_fingerprint: str,
    model: str,
    temperature: float,
    top_p: float,
    inputs: dict[str, Any],
) -> str:
    """A key that changes whenever anything that could change the answer changes.

    The plan names task + prompt version + inputs.  Four more components are
    folded in, each for a bug it prevents:

    * ``prompt_fingerprint`` - a prompt edited without bumping its version would
      otherwise keep serving pre-edit answers.  This is the honest key.
    * ``schema_fingerprint`` - renaming a field must not return the old shape.
    * ``model`` - answers from different models are different answers.
    * ``temperature``/``top_p`` - the same, for sampling settings.

    ``sort_keys`` in the input encoding matters: ``{"a":1,"b":2}`` and
    ``{"b":2,"a":1}`` are the same request and must hash the same.
    """
    canonical = json.dumps(
        {
            "task": task,
            "prompt_version": prompt_version,
            "prompt": prompt_fingerprint,
            "schema": schema_fingerprint,
            "model": model,
            "temperature": temperature,
            "top_p": top_p,
            "inputs": inputs,
        },
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return f"{_KEY_PREFIX}:{hashlib.sha256(canonical.encode()).hexdigest()}"


class ResponseCache(Protocol):
    """The seam, so the router does not care whether caching is on."""

    async def get(self, key: str) -> CachedCall | None: ...

    async def set(self, key: str, entry: CachedCall) -> None: ...


class NullCache:
    """Used when caching is disabled, and by tests that want no I/O."""

    async def get(self, key: str) -> CachedCall | None:
        return None

    async def set(self, key: str, entry: CachedCall) -> None:
        return None


class RedisResponseCache:
    """Redis-backed cache with a TTL.

    Redis rather than Postgres because entries have a natural expiry and nobody
    queries them historically - the same reasoning that put refresh tokens
    there on Day 2.
    """

    def __init__(self, redis: Redis, *, ttl_s: int) -> None:
        self._redis = redis
        self._ttl_s = ttl_s

    async def get(self, key: str) -> CachedCall | None:
        try:
            raw = await self._redis.get(key)
        except (RedisError, OSError, TimeoutError) as exc:  # degrade, never fail
            log.warning("llm cache read failed, treating as a miss: %s", exc)
            return None
        if raw is None:
            return None
        try:
            decoded = json.loads(raw)
            return CachedCall(
                payload=decoded["payload"],
                provider=decoded["provider"],
                model=decoded["model"],
            )
        except (ValueError, KeyError, TypeError) as exc:
            # An entry written by an older build. Ignoring it costs one call;
            # trusting it could return a shape the current schema rejects.
            log.warning("discarding unreadable llm cache entry: %s", exc)
            return None

    async def set(self, key: str, entry: CachedCall) -> None:
        blob = json.dumps(
            {"payload": entry.payload, "provider": entry.provider, "model": entry.model},
            separators=(",", ":"),
        )
        try:
            await self._redis.set(key, blob, ex=self._ttl_s or None)
        except (RedisError, OSError, TimeoutError) as exc:
            log.warning("llm cache write failed, continuing uncached: %s", exc)
