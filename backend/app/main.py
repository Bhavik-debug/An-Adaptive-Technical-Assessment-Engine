"""FastAPI application entrypoint."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api import auth, health
from app.cache import dispose_redis, init_redis
from app.config import Settings, get_settings
from app.db import dispose_engine, init_engine
from app.llm import dispose_llm, init_llm
from app.obs import (
    RequestContextMiddleware,
    configure_logging,
    init_tracing,
    instrument_app,
    shutdown_tracing,
)

log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: Settings = app.state.settings
    # Clients are constructed lazily-connecting: boot does not fail because a
    # dependency is briefly unavailable — /readyz is what reports that.
    init_engine(settings)
    init_redis(settings)
    # After Redis, because the LLM response cache uses that client. Building the
    # router here rather than lazily means a bad provider configuration stops
    # the process at boot, not at the first interview turn.
    init_llm(settings)
    log.info("%s started in %s mode", settings.app_name, settings.app_env)
    try:
        yield
    finally:
        await dispose_llm()
        await dispose_redis()
        await dispose_engine()
        # Last: spans buffered in the exporter's queue are flushed here, and
        # the most interesting spans in a process's life are usually the ones
        # just before it stopped.
        shutdown_tracing()
        log.info("shutdown complete")


def create_app() -> FastAPI:
    """Build the app. Served via ``uvicorn app.main:create_app --factory``.

    The factory form matters: it keeps the fail-fast config load at server boot
    rather than at module import, so importing ``app.main`` (in a test, a script,
    or a migration) does not require a fully populated environment.
    """
    # Called before the server binds a port: a bad env fails here, loudly.
    settings = get_settings()
    # Logging first, so that anything the rest of this function logs is already
    # structured and already redacted. Tracing second, because the app object
    # has to be instrumented with a provider that exists.
    configure_logging(settings)
    init_tracing(settings)

    app = FastAPI(
        title="Adaptive AI Interviewer API",
        version=settings.app_version,
        docs_url="/docs" if not settings.is_prod else None,
        redoc_url=None,
        openapi_url="/openapi.json" if not settings.is_prod else None,
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.include_router(health.router)
    app.include_router(auth.router)

    # Order matters. Starlette runs the most recently added middleware
    # outermost, and `instrument_app` adds one of its own - so instrumenting
    # last puts the OpenTelemetry server span *around* the correlation
    # middleware, which is what lets that middleware stamp the request id onto
    # the request's root span rather than onto nothing.
    app.add_middleware(RequestContextMiddleware)
    instrument_app(app, settings)
    return app
