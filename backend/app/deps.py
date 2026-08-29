"""Shared FastAPI dependencies.

A dependency is a function FastAPI runs before your endpoint, passing the result
in as an argument. Endpoints therefore never fetch their own database session or
work out who the caller is - they declare what they need and are handed it,
which is what makes them testable in isolation.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from redis.asyncio import Redis
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.cache import get_redis
from app.config import Settings, get_settings
from app.db import get_sessionmaker
from app.models import User
from app.obs import set_user_id
from app.obs.tracing import set_span_attributes
from app.security import InvalidToken, TokenType, decode_token


def get_app_settings() -> Settings:
    """Injectable settings accessor - overridable in tests."""
    return get_settings()


async def get_db() -> AsyncIterator[AsyncSession]:
    """One database session per request, always closed.

    The session is a unit of work: everything an endpoint does sits in one
    transaction, and it commits or rolls back as a whole. Rolling back on an
    exception matters - without it a failed request could leave half its writes
    behind, which is how "impossible" states end up in a database.
    """
    async with get_sessionmaker()() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise


def get_cache() -> Redis:
    return get_redis()


# auto_error=False: we want to return our own 401 shape rather than FastAPI's,
# and we need the same response whether the header is missing or malformed.
_bearer = HTTPBearer(auto_error=False)


async def get_current_user(
    request: Request,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
    settings: Annotated[Settings, Depends(get_app_settings)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> User:
    """Resolve the caller from their ``Authorization: Bearer <token>`` header.

    Every protected endpoint depends on this, and every future query filters by
    the id it returns - never by an id taken from the URL or request body. That
    single habit is what prevents broken access control (plan section 14.1),
    the most common real-world API vulnerability.
    """
    unauthorised = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Not authenticated",
        headers={"WWW-Authenticate": "Bearer"},
    )
    if credentials is None or not credentials.credentials:
        raise unauthorised

    try:
        claims = decode_token(
            credentials.credentials,
            secret=settings.secret_key,
            expected_type=TokenType.ACCESS,
        )
    except InvalidToken as exc:
        raise unauthorised from exc

    user = await db.scalar(select(User).where(User.id == claims.subject))
    if user is None:
        # Signed correctly, but the account is gone. Same response as a bad
        # token: never confirm which of the two it was.
        raise unauthorised

    request.state.user_id = user.id
    # From here on, every log line and every span for this request says who it
    # belonged to. A user id is an opaque UUID, not personal data - it is the
    # join key that lets "this candidate's interview broke" be investigated
    # without any candidate's email ever reaching a log file.
    set_user_id(str(user.id))
    set_span_attributes({"user.id": str(user.id)})
    return user


CurrentUser = Annotated[User, Depends(get_current_user)]
DbSession = Annotated[AsyncSession, Depends(get_db)]
AppSettings = Annotated[Settings, Depends(get_app_settings)]
CacheClient = Annotated[Redis, Depends(get_cache)]
