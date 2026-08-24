"""The auth endpoints, end to end against a real Postgres and Redis."""

from __future__ import annotations

from httpx import AsyncClient

from app.api.auth import REFRESH_COOKIE_NAME

EMAIL = "ada@example.com"
PASSWORD = "correct-horse-battery"


async def register(client: AsyncClient, email: str = EMAIL, password: str = PASSWORD):
    return await client.post("/api/auth/register", json={"email": email, "password": password})


async def login(client: AsyncClient, email: str = EMAIL, password: str = PASSWORD):
    return await client.post("/api/auth/login", json={"email": email, "password": password})


def auth_header(response) -> dict[str, str]:
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


def present_refresh_cookie(client: AsyncClient, token: str) -> None:
    """Make the client send exactly `token` as its refresh cookie.

    Set on the client jar rather than passed per-request: httpx deprecated
    per-request cookies precisely because merging them with the jar is ambiguous,
    and these tests care about which token is sent.
    """
    client.cookies.clear()
    client.cookies.set(REFRESH_COOKIE_NAME, token)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


async def test_register_returns_tokens_and_sets_a_refresh_cookie(client: AsyncClient):
    response = await register(client)

    assert response.status_code == 201
    body = response.json()
    assert body["token_type"] == "bearer"
    assert body["expires_in"] == 15 * 60
    assert REFRESH_COOKIE_NAME in response.cookies


async def test_the_refresh_cookie_is_httponly_and_scoped(client: AsyncClient):
    response = await register(client)
    cookie = next(
        h for h in response.headers.get_list("set-cookie") if h.startswith(REFRESH_COOKIE_NAME)
    ).lower()

    # httponly: unreadable from JavaScript, so an XSS bug cannot steal it.
    assert "httponly" in cookie
    # samesite=strict: not attached to cross-site requests, which is the CSRF defence.
    assert "samesite=strict" in cookie
    # Path-scoped so it is not sent on every API call.
    assert "path=/api/auth" in cookie


async def test_the_access_token_is_not_in_a_cookie(client: AsyncClient):
    # It belongs in the Authorization header. A cookie would be sent
    # automatically by the browser, which is what makes CSRF possible.
    response = await register(client)
    assert "access_token" not in response.cookies


async def test_duplicate_email_is_rejected(client: AsyncClient):
    await register(client)
    response = await register(client)
    assert response.status_code == 409


async def test_duplicate_email_differing_only_in_case_is_rejected(client: AsyncClient):
    await register(client, email="Ada@Example.com")
    response = await register(client, email="ada@example.com")
    # citext, not application-level lower(): no code path can forget it.
    assert response.status_code == 409


async def test_password_is_not_stored_in_plaintext(client: AsyncClient, db_engine):
    from sqlalchemy import text

    await register(client)
    async with db_engine.connect() as conn:
        stored = await conn.scalar(
            text("SELECT password_hash FROM users WHERE email = :e"), {"e": EMAIL}
        )
    assert PASSWORD not in stored
    assert stored.startswith("$argon2id$")


async def test_short_password_is_rejected(client: AsyncClient):
    response = await register(client, password="short")
    assert response.status_code == 422


async def test_malformed_email_is_rejected(client: AsyncClient):
    response = await register(client, email="not-an-email")
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Login
# ---------------------------------------------------------------------------


async def test_login_succeeds_with_correct_credentials(client: AsyncClient):
    await register(client)
    response = await login(client)
    assert response.status_code == 200
    assert response.json()["access_token"]


async def test_login_works_with_a_differently_cased_email(client: AsyncClient):
    await register(client, email="Ada@Example.com")
    response = await login(client, email="ADA@EXAMPLE.COM")
    assert response.status_code == 200


async def test_wrong_password_and_unknown_email_are_indistinguishable(client: AsyncClient):
    await register(client)
    wrong_password = await login(client, password="not-the-password")
    unknown_email = await login(client, email="nobody@example.com")

    assert wrong_password.status_code == unknown_email.status_code == 401
    # Identical bodies: the login form must not reveal which emails have accounts.
    assert wrong_password.json() == unknown_email.json()


async def test_account_locks_out_after_repeated_failures(client: AsyncClient):
    await register(client)
    for _ in range(10):
        assert (await login(client, password="wrong")).status_code == 401

    # Eleventh attempt is refused before the password is even checked...
    locked = await login(client, password="wrong")
    assert locked.status_code == 429

    # ...and the CORRECT password is refused too, or the lockout would be
    # trivially bypassable by whoever eventually guesses right.
    assert (await login(client)).status_code == 429


async def test_a_successful_login_clears_the_failure_count(client: AsyncClient):
    await register(client)
    for _ in range(5):
        await login(client, password="wrong")
    assert (await login(client)).status_code == 200

    # Counter reset: five more failures must not trip a lockout at 10 total.
    for _ in range(5):
        assert (await login(client, password="wrong")).status_code == 401


# ---------------------------------------------------------------------------
# GET /me  - the Phase 1 exit-gate endpoint
# ---------------------------------------------------------------------------


async def test_me_returns_the_authenticated_user(client: AsyncClient):
    registered = await register(client)
    response = await client.get("/api/auth/me", headers=auth_header(registered))

    assert response.status_code == 200
    assert response.json()["email"] == EMAIL


async def test_me_requires_a_token(client: AsyncClient):
    response = await client.get("/api/auth/me")
    assert response.status_code == 401


async def test_me_rejects_a_garbage_token(client: AsyncClient):
    response = await client.get("/api/auth/me", headers={"Authorization": "Bearer not.a.token"})
    assert response.status_code == 401


async def test_me_rejects_a_token_signed_with_a_different_secret(client: AsyncClient):
    import datetime as dt
    import uuid

    from app.security import TokenType, create_token

    forged = create_token(
        subject=uuid.uuid4(),
        token_type=TokenType.ACCESS,
        ttl=dt.timedelta(minutes=15),
        secret="a-completely-different-secret-key-value",
    )
    response = await client.get("/api/auth/me", headers={"Authorization": f"Bearer {forged.token}"})
    assert response.status_code == 401


async def test_me_rejects_the_refresh_token_as_a_bearer_credential(client: AsyncClient):
    registered = await register(client)
    refresh_token = registered.cookies[REFRESH_COOKIE_NAME]

    response = await client.get(
        "/api/auth/me", headers={"Authorization": f"Bearer {refresh_token}"}
    )
    # A 30-day credential must never work where a 15-minute one is expected.
    assert response.status_code == 401


# ---------------------------------------------------------------------------
# Refresh + rotation
# ---------------------------------------------------------------------------


async def test_refresh_issues_a_new_access_token(client: AsyncClient):
    await register(client)
    response = await client.post("/api/auth/refresh")

    assert response.status_code == 200
    assert response.json()["access_token"]
    me = await client.get("/api/auth/me", headers=auth_header(response))
    assert me.status_code == 200


async def test_refresh_rotates_the_cookie(client: AsyncClient):
    registered = await register(client)
    original = registered.cookies[REFRESH_COOKIE_NAME]

    refreshed = await client.post("/api/auth/refresh")
    rotated = refreshed.cookies[REFRESH_COOKIE_NAME]

    assert rotated != original


async def test_an_old_refresh_token_stops_working(client: AsyncClient):
    """Single-use is the whole point of rotation."""
    registered = await register(client)
    original = registered.cookies[REFRESH_COOKIE_NAME]
    await client.post("/api/auth/refresh")

    present_refresh_cookie(client, original)
    replayed = await client.post("/api/auth/refresh")
    assert replayed.status_code == 401


async def test_replaying_a_spent_token_revokes_the_whole_family(client: AsyncClient):
    """Theft response: we cannot tell thief from victim, so end both sessions."""
    registered = await register(client)
    stolen = registered.cookies[REFRESH_COOKIE_NAME]

    # The legitimate user refreshes; `stolen` is now spent but `current` is live.
    rotated = await client.post("/api/auth/refresh")
    current = rotated.cookies[REFRESH_COOKIE_NAME]

    # The attacker replays the stolen copy. Detected.
    present_refresh_cookie(client, stolen)
    assert (await client.post("/api/auth/refresh")).status_code == 401

    # The legitimate user's still-unused token is now dead too - deliberately.
    present_refresh_cookie(client, current)
    assert (await client.post("/api/auth/refresh")).status_code == 401


async def test_refresh_without_a_cookie_is_rejected(client: AsyncClient):
    response = await client.post("/api/auth/refresh")
    assert response.status_code == 401


async def test_an_access_token_cannot_be_used_to_refresh(client: AsyncClient):
    registered = await register(client)
    access = registered.json()["access_token"]

    present_refresh_cookie(client, access)
    response = await client.post("/api/auth/refresh")
    assert response.status_code == 401


# ---------------------------------------------------------------------------
# Logout
# ---------------------------------------------------------------------------


async def test_logout_invalidates_the_refresh_token(client: AsyncClient):
    await register(client)
    assert (await client.post("/api/auth/logout")).status_code == 204
    assert (await client.post("/api/auth/refresh")).status_code == 401


async def test_logout_is_safe_to_call_twice(client: AsyncClient):
    await register(client)
    assert (await client.post("/api/auth/logout")).status_code == 204
    assert (await client.post("/api/auth/logout")).status_code == 204


async def test_logout_without_a_session_still_succeeds(client: AsyncClient):
    assert (await client.post("/api/auth/logout")).status_code == 204


# ---------------------------------------------------------------------------
# Cross-tenant isolation (plan section 14.1: broken access control)
# ---------------------------------------------------------------------------


async def test_each_users_token_resolves_only_to_themselves(client: AsyncClient):
    first = await register(client, email="one@example.com")
    second = await register(client, email="two@example.com")

    me_one = await client.get("/api/auth/me", headers=auth_header(first))
    me_two = await client.get("/api/auth/me", headers=auth_header(second))

    assert me_one.json()["email"] == "one@example.com"
    assert me_two.json()["email"] == "two@example.com"
    assert me_one.json()["id"] != me_two.json()["id"]
