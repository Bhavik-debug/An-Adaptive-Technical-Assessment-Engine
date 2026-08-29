"""Liveness and readiness probes.

The distinction is deliberate and load-bearing for the Phase 6 deploy:

* ``/healthz``  — is this process alive? No dependencies touched. If it fails,
  the orchestrator should restart the container.
* ``/readyz``   — can this process actually serve traffic? Every dependency is
  probed. If it fails, the orchestrator should stop routing to this replica but
  NOT restart it: a down Postgres is not fixed by killing the API.

Rolling a bad build back on a failed ``/readyz`` is exactly what the Phase 6
deploy pipeline keys off, so the checks must be honest — a dependency that is
not wired yet reports ``skipped``, never ``ok``.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Literal

from fastapi import APIRouter, Depends, Response, status

from app.cache import check_redis
from app.config import Settings
from app.db import check_database
from app.deps import get_app_settings
from app.llm import llm_provider_health

router = APIRouter(tags=["health"])

CheckStatus = Literal["ok", "down", "skipped"]


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: CheckStatus
    latency_ms: int
    detail: str | None = None

    @property
    def blocks_readiness(self) -> bool:
        return self.status == "down"

    def as_dict(self) -> dict[str, object]:
        out: dict[str, object] = {"status": self.status, "latency_ms": self.latency_ms}
        if self.detail:
            out["detail"] = self.detail
        return out


async def _run_check(name: str, probe: Callable[[], Awaitable[None]]) -> CheckResult:
    started = time.perf_counter()
    try:
        await probe()
    except Exception as exc:  # noqa: BLE001 - a probe failure is a status, not a crash
        return CheckResult(
            name=name,
            status="down",
            latency_ms=int((time.perf_counter() - started) * 1000),
            detail=f"{type(exc).__name__}: {exc}"[:200],
        )
    return CheckResult(
        name=name,
        status="ok",
        latency_ms=int((time.perf_counter() - started) * 1000),
    )


def _llm_check() -> CheckResult:
    """Is at least one LLM provider in rotation?

    Note what this deliberately does *not* do: send a completion.  ``/readyz``
    is polled every few seconds by the orchestrator, and the free-tier quota
    this project's cost model depends on is measured in requests per day - a
    live probe would spend the whole budget on health checks.  So readiness here
    means "configured, and not currently circuit-broken by real traffic", and
    the honest end-to-end check is the opt-in smoke test.
    """
    providers = llm_provider_health()
    if not providers:
        return CheckResult(
            name="llm_providers",
            status="skipped",
            latency_ms=0,
            detail="router not initialised in this process",
        )
    available = [p.name for p in providers if p.available]
    if not available:
        return CheckResult(
            name="llm_providers",
            status="down",
            latency_ms=0,
            detail="; ".join(f"{p.name}: {p.detail or 'unavailable'}" for p in providers),
        )
    return CheckResult(
        name="llm_providers",
        status="ok",
        latency_ms=0,
        detail=f"{len(available)}/{len(providers)} in rotation: {', '.join(available)}",
    )


@router.get("/healthz", summary="Liveness probe")
async def healthz(settings: Settings = Depends(get_app_settings)) -> dict[str, str]:
    return {"status": "ok", "service": settings.app_name, "env": settings.app_env}


@router.get("/readyz", summary="Readiness probe")
async def readyz(
    response: Response,
    settings: Settings = Depends(get_app_settings),
) -> dict[str, object]:
    timeout = settings.readiness_timeout_s

    # Concurrently: a readiness probe has an orchestrator deadline, and running
    # N dependency checks in series multiplies the worst case by N.
    probed = await asyncio.gather(
        _run_check("postgres", lambda: check_database(timeout)),
        _run_check("redis", lambda: check_redis(timeout)),
    )
    checks = [*probed, _llm_check()]

    ready = not any(c.blocks_readiness for c in checks)
    if not ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return {
        "status": "ready" if ready else "not_ready",
        "checks": {c.name: c.as_dict() for c in checks},
    }
