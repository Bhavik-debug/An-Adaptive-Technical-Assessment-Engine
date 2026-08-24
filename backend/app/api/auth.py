"""Registration, login, token refresh, logout, and the current-user endpoint.

The token design, and why it is two tokens rather than one:

* An **access token** is a short-lived (15 min) signed JWT sent in the
  ``Authorization`` header on every request. It is fast to check - a signature
  verification, no database round trip - but it cannot be revoked, so its blast
  radius is bounded purely by how soon it expires.
* A **refresh token** is a long-lived (30 day) JWT that does one thing: obtain a
  new access token. It travels in an ``httpOnly`` cookie, so JavaScript on the
  page cannot read it and an XSS bug cannot steal it. Every use *rotates* it -
  the old one is spent and a new one issued.

Rotation is what makes a long-lived credential safe. If a refresh token is
stolen, either the thief or the real user will use it second, and the second use
presents a correctly signed token whose id has already been spent. That is
detectable, and the response is to revoke the whole family and force a fresh
login (plan section 14.1: token theft, session fixation).
"""

from __future__ import annotations

import datetime as dt

from fastapi import APIRouter, Cookie, HTTPException, Response, status
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from app.config import Settings
from app.deps import AppSettings, CacheClient, CurrentUser, DbSession
from app.models import User
from app.security import (
    MAX_PASSWORD_BYTES,
    MIN_PASSWORD_LENGTH,
    InvalidToken,
    TokenType,
    create_token,
    decode_token,
    dummy_hash,
    hash_password,
    password_needs_rehash,
    verify_password,
)
from app.token_store import (
    clear_login_failures,
    consume_refresh_token,
    is_locked_out,
    register_login_failure,
    remember_refresh_token,
    revoke_all_refresh_tokens,
)

router = APIRouter(prefix="/api/auth", tags=["auth"])

REFRESH_COOKIE_NAME = "refresh_token"
# Scoped to the auth routes: the browser then sends this cookie only to the two
# endpoints that need it, instead of attaching it to every API call.
REFRESH_COOKIE_PATH = "/api/auth"


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=MIN_PASSWORD_LENGTH, max_length=MAX_PASSWORD_BYTES)


class LoginRequest(BaseModel):
    email: EmailStr
    # No min_length here on purpose: rejecting a short password at login would
    # tell an attacker their guess was too short to be anyone's real password.
    password: str = Field(max_length=MAX_PASSWORD_BYTES)


class TokenResponse(BaseModel):
    access_token: str
    # noqa on S105: "bearer" is the OAuth token-type name, not a secret.
    token_type: str = "bearer"  # noqa: S105
    expires_in: int  # seconds


class UserResponse(BaseModel):
    id: str
    email: str
    created_at: dt.datetime


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _access_ttl(settings: Settings) -> dt.timedelta:
    return dt.timedelta(minutes=settings.access_token_ttl_minutes)


def _refresh_ttl(settings: Settings) -> dt.timedelta:
    return dt.timedelta(days=settings.refresh_token_ttl_days)


def _set_refresh_cookie(response: Response, token: str, settings: Settings) -> None:
    response.set_cookie(
        key=REFRESH_COOKIE_NAME,
        value=token,
        max_age=int(_refresh_ttl(settings).total_seconds()),
        path=REFRESH_COOKIE_PATH,
        # Unreadable from JavaScript: an XSS bug cannot exfiltrate it.
        httponly=True,
        # HTTPS only outside local development.
        secure=settings.cookie_secure,
        # Not sent on requests originating from another site, which is what
        # makes cross-site request forgery against /refresh a non-event.
        # NOTE for Phase 6: if the frontend is served from a different origin
        # than the API, Strict will suppress the cookie and this becomes
        # SameSite=None + Secure. Revisit when the deploy topology is fixed.
        samesite="strict",
    )


async def _issue_tokens(
    *, user: User, response: Response, settings: Settings, redis: CacheClient
) -> TokenResponse:
    """Mint an access/refresh pair and record the refresh token as live."""
    access = create_token(
        subject=user.id,
        token_type=TokenType.ACCESS,
        ttl=_access_ttl(settings),
        secret=settings.secret_key,
    )
    refresh = create_token(
        subject=user.id,
        token_type=TokenType.REFRESH,
        ttl=_refresh_ttl(settings),
        secret=settings.secret_key,
    )
    await remember_refresh_token(
        redis, jti=refresh.jti, user_id=user.id, ttl=_refresh_ttl(settings)
    )
    _set_refresh_cookie(response, refresh.token, settings)
    return TokenResponse(
        access_token=access.token,
        expires_in=int(_access_ttl(settings).total_seconds()),
    )


def _invalid_credentials() -> HTTPException:
    # One message for "no such account" and for "wrong password". Distinguishing
    # them turns the login form into an account-enumeration oracle.
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid email or password",
        headers={"WWW-Authenticate": "Bearer"},
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/register", status_code=status.HTTP_201_CREATED, summary="Create an account")
async def register(
    payload: RegisterRequest,
    response: Response,
    db: DbSession,
    settings: AppSettings,
    redis: CacheClient,
) -> TokenResponse:
    user = User(email=payload.email, password_hash=hash_password(payload.password))
    db.add(user)
    try:
        await db.commit()
    except IntegrityError as exc:
        await db.rollback()
        # Let the database's unique index decide, rather than SELECT-then-INSERT.
        # Two simultaneous registrations would both pass the SELECT; only one can
        # pass the constraint.
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An account with that email already exists",
        ) from exc

    await db.refresh(user)
    return await _issue_tokens(user=user, response=response, settings=settings, redis=redis)


@router.post("/login", summary="Exchange credentials for tokens")
async def login(
    payload: LoginRequest,
    response: Response,
    db: DbSession,
    settings: AppSettings,
    redis: CacheClient,
) -> TokenResponse:
    if await is_locked_out(redis, email=payload.email, lockout_after=settings.login_max_failures):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=(
                f"Too many failed attempts. Try again in "
                f"{settings.login_lockout_minutes} minutes."
            ),
        )

    user = await db.scalar(select(User).where(User.email == payload.email))

    # Verify against a decoy when the account does not exist, so both branches
    # take the same ~50 ms and the response time leaks nothing.
    stored_hash = user.password_hash if user else dummy_hash()
    password_ok = verify_password(stored_hash, payload.password)

    if user is None or not password_ok:
        await register_login_failure(
            redis,
            email=payload.email,
            lockout_after=settings.login_max_failures,
            lockout_for=dt.timedelta(minutes=settings.login_lockout_minutes),
        )
        raise _invalid_credentials()

    # Transparently upgrade the stored hash if we have since raised the cost
    # parameters. This is the only moment we hold the plaintext, so it is the
    # only moment a rehash is possible.
    if password_needs_rehash(user.password_hash):
        user.password_hash = hash_password(payload.password)
        await db.commit()

    await clear_login_failures(redis, email=payload.email)
    return await _issue_tokens(user=user, response=response, settings=settings, redis=redis)


@router.post("/refresh", summary="Rotate the refresh token for a new access token")
async def refresh(
    response: Response,
    db: DbSession,
    settings: AppSettings,
    redis: CacheClient,
    refresh_token: str | None = Cookie(default=None, alias=REFRESH_COOKIE_NAME),
) -> TokenResponse:
    unauthorised = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired refresh token",
    )
    if not refresh_token:
        raise unauthorised

    try:
        claims = decode_token(
            refresh_token, secret=settings.secret_key, expected_type=TokenType.REFRESH
        )
    except InvalidToken as exc:
        raise unauthorised from exc

    # Single-use. A valid signature whose jti is already spent means this token
    # was replayed - by an attacker or by the legitimate user after a theft, and
    # we cannot tell which. Revoke the entire family and make everyone log in.
    if not await consume_refresh_token(redis, jti=claims.jti, user_id=claims.subject):
        await revoke_all_refresh_tokens(redis, user_id=claims.subject)
        response.delete_cookie(REFRESH_COOKIE_NAME, path=REFRESH_COOKIE_PATH)
        raise unauthorised

    user = await db.scalar(select(User).where(User.id == claims.subject))
    if user is None:
        raise unauthorised

    return await _issue_tokens(user=user, response=response, settings=settings, redis=redis)


@router.post(
    "/logout",
    status_code=status.HTTP_204_NO_CONTENT,
    # A 204 carries no body at all. Without these two, FastAPI infers a response
    # model from the `-> None` annotation and would serialise a literal `null`.
    response_class=Response,
    response_model=None,
    summary="End the session",
)
async def logout(
    response: Response,
    settings: AppSettings,
    redis: CacheClient,
    refresh_token: str | None = Cookie(default=None, alias=REFRESH_COOKIE_NAME),
) -> None:
    # Always clear the cookie and always return 204, even for a token we cannot
    # parse. Logout has no failure mode worth reporting: the user's intent is
    # "end my session", and telling them their token was already invalid helps
    # nobody and leaks state.
    response.delete_cookie(REFRESH_COOKIE_NAME, path=REFRESH_COOKIE_PATH)
    if not refresh_token:
        return
    try:
        claims = decode_token(
            refresh_token, secret=settings.secret_key, expected_type=TokenType.REFRESH
        )
    except InvalidToken:
        return
    await consume_refresh_token(redis, jti=claims.jti, user_id=claims.subject)


@router.get("/me", summary="The authenticated user")
async def me(user: CurrentUser) -> UserResponse:
    return UserResponse(id=str(user.id), email=user.email, created_at=user.created_at)
