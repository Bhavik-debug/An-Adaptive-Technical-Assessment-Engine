"""The session lifecycle's front door: create, append, load.

This is what a later endpoint calls (plan section 13.8's ``/sessions`` routes
are Day 22 onward).  Every function takes the caller's ``user_id`` and enforces
ownership in SQL, so no caller can reach a session by guessing its id.

Write path - :func:`append_event`::

    BEGIN
      lock session row, fold log, validate, INSERT event, UPDATE projection
    COMMIT                                   <- the event now exists
    cache.store(new state)                   <- best effort, monotonic in seq

Read path - :func:`load_state`::

    head = max(seq) for this (session, user)          one index lookup
    cached = cache.get(session)
    cached.seq == head      -> return cached          hit
    otherwise               -> fold(log), cache it    miss / stale / invalid / Redis down

The consistency boundary sits between COMMIT and the cache write.  A crash
there leaves Redis one event behind Postgres; the next reader sees
``cached.seq < head`` and rebuilds.  The reverse - Redis *ahead* of Postgres -
cannot be produced by this code (nothing is cached before commit) and is
treated as invalid if a database restore ever creates it.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Mapping
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.blueprint import Blueprint
from app.interview.cache import CacheStatus, SessionStateCache
from app.interview.fsm import EventType
from app.interview.state import SessionState, fold
from app.interview.store import (
    SessionNotFound,
    head_seq,
    read_events,
    stage_append,
    stage_new_session,
)

log = logging.getLogger(__name__)


async def create_session(
    db: AsyncSession,
    cache: SessionStateCache | None,
    *,
    user_id: uuid.UUID,
    blueprint: Blueprint,
    resume_id: uuid.UUID | None = None,
    jd_id: uuid.UUID | None = None,
) -> SessionState:
    """Create a session in ``CREATED`` with its blueprint and seq-0 event.  Commits."""
    try:
        state = await stage_new_session(
            db, user_id=user_id, blueprint=blueprint.to_dict(), resume_id=resume_id, jd_id=jd_id
        )
        await db.commit()
    except BaseException:
        await db.rollback()
        raise
    if cache is not None:
        await cache.store(state)
    return state


async def append_event(
    db: AsyncSession,
    cache: SessionStateCache | None,
    *,
    session_id: uuid.UUID,
    user_id: uuid.UUID,
    event_type: EventType,
    payload: Mapping[str, Any] | None = None,
    expected_seq: int | None = None,
) -> SessionState:
    """Validate, persist and commit one event; return the state it produces.

    This function **owns the transaction**: it commits on success and rolls
    back on any failure, so a rejected event leaves no row and no change to
    ``interview_sessions.state``.  A caller that must make the append atomic
    with other writes uses :func:`app.interview.store.stage_append` and
    :func:`commit_and_cache` instead.

    ``expected_seq`` is optimistic concurrency for callers that decided based
    on a state they read earlier: pass that state's ``seq`` and the append is
    refused with :class:`~app.interview.store.SeqConflict` if anything was
    appended in between.
    """
    try:
        state = await stage_append(
            db,
            session_id=session_id,
            user_id=user_id,
            event_type=event_type,
            payload=payload or {},
            expected_seq=expected_seq,
        )
    except BaseException:
        await db.rollback()
        raise
    return await commit_and_cache(db, cache, state)


async def commit_and_cache(
    db: AsyncSession, cache: SessionStateCache | None, state: SessionState
) -> SessionState:
    """Commit the staged append, then - only then - cache the new state."""
    try:
        await db.commit()
    except BaseException:
        await db.rollback()
        raise
    if cache is not None:
        await cache.store(state)
    return state


async def load_state(
    db: AsyncSession,
    cache: SessionStateCache | None,
    *,
    session_id: uuid.UUID,
    user_id: uuid.UUID,
) -> SessionState:
    """The current state of an owned session: from Redis if it is exactly at
    the head of the log, otherwise rebuilt from Postgres (and re-cached).

    Raises :class:`~app.interview.store.SessionNotFound`, or
    :class:`~app.interview.state.EventLogCorrupted` when the log cannot be
    folded - in which case nothing is cached.
    """
    head = await head_seq(db, session_id=session_id, user_id=user_id)
    if head is None:
        raise SessionNotFound(session_id)

    cached_seq: int | None = None
    if cache is not None:
        read = await cache.get(session_id)
        if read.status is CacheStatus.HIT and read.state is not None:
            if read.state.seq == head:
                return read.state
            cached_seq = read.state.seq
            log.info(
                "session cache for %s at seq %s, log at %s; rebuilding",
                session_id,
                cached_seq,
                head,
            )
        elif read.status is CacheStatus.INVALID:
            await cache.invalidate(session_id)

    state = await rebuild_state(db, session_id=session_id)
    if cache is not None:
        if cached_seq is not None and state.seq < cached_seq:
            # The cache is ahead of the durable log: only a restore or a manual
            # edit does that. The database wins.
            log.warning(
                "session cache for %s at seq %s is ahead of the log at %s; replacing",
                session_id,
                cached_seq,
                state.seq,
            )
            await cache.replace(state)
        else:
            await cache.store(state)
    return state


async def rebuild_state(db: AsyncSession, *, session_id: uuid.UUID) -> SessionState:
    """``fold`` over the durable log.  No cache, no ownership check - callers
    of this function have already established both (or are offline tools)."""
    return fold(session_id, await read_events(db, session_id))


__all__ = [
    "append_event",
    "commit_and_cache",
    "create_session",
    "load_state",
    "rebuild_state",
]
