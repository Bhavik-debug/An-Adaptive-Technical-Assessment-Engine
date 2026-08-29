"""Process-level lifecycle for the LLM stack.

Same shape as ``app/db.py`` and ``app/cache.py``: build once at boot, hand out a
reference, dispose on shutdown.  Keeping it here rather than in ``client.py``
means ``call_structured()`` can be handed an explicit router in a test and never
touch global state.
"""

from __future__ import annotations

import logging

from redis.exceptions import RedisError

from app.cache import get_redis
from app.config import Settings
from app.llm.cache import NullCache, RedisResponseCache, ResponseCache
from app.llm.router import ProviderHealth, ProviderRouter, build_router

log = logging.getLogger(__name__)

_router: ProviderRouter | None = None
_cache: ResponseCache | None = None


def init_llm(settings: Settings) -> ProviderRouter:
    """Build the router and the response cache. Idempotent."""
    global _router, _cache
    if _router is None:
        _router = build_router(settings)
        log.info(
            "llm router ready: providers=%s cache=%s",
            ",".join(_router.provider_names),
            "on" if settings.llm_cache_enabled else "off",
        )
    if _cache is None:
        _cache = _build_cache(settings)
    return _router


def _build_cache(settings: Settings) -> ResponseCache:
    if not settings.llm_cache_enabled:
        return NullCache()
    try:
        return RedisResponseCache(get_redis(), ttl_s=settings.llm_cache_ttl_s)
    except (RuntimeError, RedisError) as exc:
        # Redis not initialised (a script, a test). Caching is an optimisation;
        # losing it must not stop the LLM layer from working.
        log.warning("llm response cache disabled: %s", exc)
        return NullCache()


def get_router() -> ProviderRouter:
    if _router is None:
        raise RuntimeError("LLM router not initialised; call init_llm() first")
    return _router


def get_response_cache() -> ResponseCache:
    return _cache if _cache is not None else NullCache()


async def dispose_llm() -> None:
    global _router, _cache
    if _router is not None:
        await _router.aclose()
        _router = None
    _cache = None


def llm_provider_health() -> list[ProviderHealth]:
    """What ``/readyz`` reports. Never makes a network call - see router.health()."""
    if _router is None:
        return []
    return _router.health()
