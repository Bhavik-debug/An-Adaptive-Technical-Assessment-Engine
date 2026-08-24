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

log = logging.getLogger(__name__)


def _configure_logging(settings: Settings) -> None:
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s %(levelname)-8s %(name)s | %(message)s",
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: Settings = app.state.settings
    # Clients are constructed lazily-connecting: boot does not fail because a
    # dependency is briefly unavailable — /readyz is what reports that.
    init_engine(settings)
    init_redis(settings)
    log.info("%s started in %s mode", settings.app_name, settings.app_env)
    try:
        yield
    finally:
        await dispose_redis()
        await dispose_engine()
        log.info("shutdown complete")


def create_app() -> FastAPI:
    """Build the app. Served via ``uvicorn app.main:create_app --factory``.

    The factory form matters: it keeps the fail-fast config load at server boot
    rather than at module import, so importing ``app.main`` (in a test, a script,
    or a migration) does not require a fully populated environment.
    """
    # Called before the server binds a port: a bad env fails here, loudly.
    settings = get_settings()
    _configure_logging(settings)

    app = FastAPI(
        title="Adaptive AI Interviewer API",
        version="0.1.0",
        docs_url="/docs" if not settings.is_prod else None,
        redoc_url=None,
        openapi_url="/openapi.json" if not settings.is_prod else None,
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.include_router(health.router)
    app.include_router(auth.router)
    return app
