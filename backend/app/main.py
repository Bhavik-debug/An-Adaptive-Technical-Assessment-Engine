"""FastAPI application entrypoint."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api import auth, health, ingestion, sessions, setup
from app.bank.taxonomy import load_taxonomy
from app.cache import dispose_redis, get_redis, init_redis
from app.config import Settings, get_settings
from app.db import dispose_engine, get_sessionmaker, init_engine
from app.ingestion.jobs import extract_with_model, run_once_ingestion
from app.interview.cache import SessionStateCache
from app.interview.grading import RetryPolicy, grade_with_pipeline, run_worker
from app.interview.probing import probe_with_model, run_once_follow_up
from app.llm import dispose_llm, init_llm
from app.llm.providers.stub import StubProvider
from app.obs import (
    RequestContextMiddleware,
    configure_logging,
    init_tracing,
    instrument_app,
    shutdown_tracing,
)
from app.reporting.jobs import narrate_with_model, run_once_report
from app.simulation.scripted_llm import scripted_response

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
    router = init_llm(settings)
    if settings.llm_stub_on_missing == "scripted":
        # Day 25: offline whole-interview runs. The stub answers unrecorded
        # requests from the documented scripted rules (app/simulation/
        # scripted_llm.py); registered here so app/llm never imports domain code.
        for provider in router.providers:
            if isinstance(provider, StubProvider):
                provider.set_responder(
                    scripted_response, delay_s=settings.llm_stub_scripted_delay_s
                )

    # The Day-22 grading worker. It runs in this process for local development,
    # but holds no state of its own: `turns` is the queue (app/interview/
    # grading.py), so a crash loses nothing and a restart resumes the work.
    stop = asyncio.Event()
    worker: asyncio.Task[None] | None = None
    if settings.grading_worker_enabled:
        maker = get_sessionmaker()
        session_cache = SessionStateCache.from_settings(get_redis(), settings)
        policy = RetryPolicy.from_settings(settings)

        async def follow_ups() -> str | None:
            # Day 23: the same worker words decided follow-ups (app/interview/
            # probing.py); Day 24: and extracts resumes and JDs (app/ingestion/
            # jobs.py) - same retry policy, same durability, same fencing.
            worded = await run_once_follow_up(
                maker, session_cache, probe_fn=probe_with_model, policy=policy
            )
            extracted = await run_once_ingestion(
                maker, taxonomy=load_taxonomy(), extract_fn=extract_with_model, policy=policy
            )
            # Day 25: build reports and complete sessions (app/reporting/jobs.py).
            reported = await run_once_report(
                maker, session_cache, narrative_fn=narrate_with_model, policy=policy
            )
            return worded or extracted or reported

        worker = asyncio.create_task(
            run_worker(
                maker,
                session_cache,
                grade_fn=grade_with_pipeline,
                policy=policy,
                poll_interval_s=settings.grading_poll_interval_s,
                stop=stop,
                follow_ups=follow_ups,
            ),
            name="grading-worker",
        )

    log.info("%s started in %s mode", settings.app_name, settings.app_env)
    try:
        yield
    finally:
        if worker is not None:
            stop.set()
            try:
                await asyncio.wait_for(worker, timeout=10)
            except TimeoutError:
                # Mid-grade: cancel. The claim's lease expires and the next
                # worker re-claims the turn; the attempt still counts.
                worker.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await worker
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
    app.include_router(sessions.router)
    app.include_router(ingestion.router)
    app.include_router(setup.router)

    # Order matters. Starlette runs the most recently added middleware
    # outermost, and `instrument_app` adds one of its own - so instrumenting
    # last puts the OpenTelemetry server span *around* the correlation
    # middleware, which is what lets that middleware stamp the request id onto
    # the request's root span rather than onto nothing.
    app.add_middleware(RequestContextMiddleware)
    instrument_app(app, settings)
    return app
