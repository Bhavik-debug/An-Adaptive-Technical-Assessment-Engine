"""POST /api/auth/google and GET /api/auth/providers, against real Postgres + Redis.

Google's verification library runs for real (app.google_auth); only the fetch
of Google's signing keys is replaced, by a transport that serves a local test
key (tests/google_support.py). No test reaches Google.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator

import pytest
import pytest_asyncio
from asgi_lifespan import LifespanManager
from httpx import ASGITransport, AsyncClient
from sqlalchemy import text

from app.api.auth import REFRESH_COOKIE_NAME
from tests.google_support import CLIENT_ID, FakeCertsTransport, make_token

PASSWORD = "correct-horse-battery"
SUB = "109876543210987654321"


@pytest_asyncio.fixture
async def google_client(
    env, migrated_database_url, redis_url, monkeypatch
) -> AsyncIterator[AsyncClient]:
    """The real app with Google sign-in configured for the test client ID."""
    from app import google_auth
    from app.cache import dispose_redis
    from app.db import dispose_engine
    from app.main import create_app

    await dispose_engine()
    await dispose_redis()
    env(
        {
            "DATABASE_URL": migrated_database_url,
            "REDIS_URL": redis_url,
            "GOOGLE_CLIENT_ID": CLIENT_ID,
        }
    )
    monkeypatch.setattr(google_auth, "_default_transport", FakeCertsTransport)
    app = create_app()
    async with LifespanManager(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as http:
            yield http
    await dispose_engine()
    await dispose_redis()


async def google(http: AsyncClient, credential: str | None = None, **claims):
    return await http.post(
        "/api/auth/google", json={"credential": credential or make_token(**claims)}
    )


async def users(db_engine) -> list[tuple[str, str | None, str | None]]:
    async with db_engine.connect() as conn:
        rows = await conn.execute(
            text("SELECT email, password_hash, google_sub FROM users ORDER BY created_at")
        )
    return [tuple(r) for r in rows]


def bearer(response) -> dict[str, str]:
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


async def test_providers_reports_the_public_client_id(google_client: AsyncClient):
    response = await google_client.get("/api/auth/providers")
    assert response.status_code == 200
    assert response.json() == {"password": True, "google_client_id": CLIENT_ID}


async def test_without_a_client_id_google_sign_in_is_off(client: AsyncClient):
    providers = await client.get("/api/auth/providers")
    assert providers.json()["google_client_id"] is None

    response = await google(client)
    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "google_not_configured"
    assert REFRESH_COOKIE_NAME not in response.cookies


@pytest.mark.parametrize(
    "value",
    [
        "not-a-client-id",
        "abc.apps.googleusercontent.com.evil.com",
        "a b.apps.googleusercontent.com",
    ],
)
def test_a_malformed_client_id_refuses_to_boot(env, value):
    from pydantic import ValidationError

    from app.config import Settings

    env({"GOOGLE_CLIENT_ID": value})
    with pytest.raises(ValidationError, match="google_client_id"):
        Settings()  # type: ignore[call-arg]


def test_an_empty_client_id_means_off(env):
    from app.config import Settings

    env({"GOOGLE_CLIENT_ID": "  "})
    assert Settings().google_client_id is None  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# Sign-up and sign-in
# ---------------------------------------------------------------------------


async def test_a_new_google_account_is_created_with_a_normal_session(
    google_client: AsyncClient, db_engine
):
    response = await google(google_client)

    assert response.status_code == 201
    body = response.json()
    assert body["token_type"] == "bearer" and body["expires_in"] == 15 * 60
    cookie = next(
        h for h in response.headers.get_list("set-cookie") if h.startswith(REFRESH_COOKIE_NAME)
    ).lower()
    assert "httponly" in cookie and "samesite=strict" in cookie and "path=/api/auth" in cookie
    assert "access_token" not in response.cookies

    me = await google_client.get("/api/auth/me", headers=bearer(response))
    assert me.status_code == 200 and me.json()["email"] == "grace@example.com"

    # No password is invented, and Google's token is stored nowhere.
    assert await users(db_engine) == [("grace@example.com", None, SUB)]


async def test_signing_in_again_reuses_the_account(google_client: AsyncClient, db_engine):
    first = await google(google_client)
    second = await google(google_client)  # a fresh credential (new jti)
    assert (first.status_code, second.status_code) == (201, 200)
    a = await google_client.get("/api/auth/me", headers=bearer(first))
    b = await google_client.get("/api/auth/me", headers=bearer(second))
    assert a.json()["id"] == b.json()["id"]
    assert len(await users(db_engine)) == 1


async def test_the_account_is_keyed_by_google_sub_not_email(google_client: AsyncClient, db_engine):
    await google(google_client)
    # The Google account's email changed; it is still the same account.
    again = await google(google_client, email="grace.h@example.com")
    assert again.status_code == 200
    assert await users(db_engine) == [("grace@example.com", None, SUB)]


async def test_a_credential_is_single_use(google_client: AsyncClient):
    credential = make_token()
    assert (await google(google_client, credential)).status_code == 201
    replay = await google(google_client, credential)
    assert replay.status_code == 401
    assert replay.json()["detail"]["code"] == "google_token_invalid"
    assert REFRESH_COOKIE_NAME not in replay.cookies


async def test_a_credential_without_jti_is_refused(google_client: AsyncClient, db_engine):
    response = await google(google_client, jti=None)
    assert response.status_code == 401
    assert await users(db_engine) == []


@pytest.mark.parametrize(
    "claims",
    [
        {"aud": "someone-else.apps.googleusercontent.com"},
        {"iss": "https://evil.example.com"},
        {"exp": 1_000_000_000, "iat": 999_996_400},
    ],
    ids=["wrong-audience", "wrong-issuer", "expired"],
)
async def test_invalid_credentials_get_one_generic_answer(
    google_client: AsyncClient, db_engine, claims
):
    response = await google(google_client, **claims)
    assert response.status_code == 401
    assert response.json()["detail"] == {
        "code": "google_token_invalid",
        "message": "Google sign-in failed.",
    }
    assert await users(db_engine) == []


async def test_a_bad_signature_is_rejected(google_client: AsyncClient):
    from tests.google_support import OTHER_PRIVATE_KEY

    response = await google(google_client, make_token(key=OTHER_PRIVATE_KEY))
    assert response.status_code == 401


async def test_an_unverified_email_is_refused(google_client: AsyncClient, db_engine):
    response = await google(google_client, email_verified=False)
    assert response.status_code == 403
    assert response.json()["detail"]["code"] == "google_email_unverified"
    assert await users(db_engine) == []


@pytest.mark.parametrize(
    "body",
    [{}, {"credential": ""}, {"credential": "x" * 5000}, {"credential": 42}, {"token": "abc"}],
)
async def test_malformed_requests_are_422(google_client: AsyncClient, body):
    response = await google_client.post("/api/auth/google", json=body)
    assert response.status_code == 422


async def test_google_being_unreachable_is_503(google_client: AsyncClient, monkeypatch):
    from app import google_auth

    monkeypatch.setattr(google_auth, "_default_transport", lambda: FakeCertsTransport(status=500))
    response = await google(google_client)
    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "google_unavailable"


async def test_the_credential_is_never_logged(google_client: AsyncClient, caplog):
    caplog.set_level(logging.DEBUG)
    good, bad = make_token(), make_token(aud="other.apps.googleusercontent.com")
    # google-auth quotes malformed tokens in its error text; a truncated real
    # token still carries the email in its payload segment.
    header, payload, _ = make_token().split(".")
    truncated = f"{header}.{payload}"
    for credential in (good, bad, truncated):
        await google(google_client, credential)
    assert caplog.text  # the rejections were logged...
    for credential in (good, bad, truncated):
        assert credential not in caplog.text  # ...without the credential
        assert credential.split(".")[1] not in caplog.text  # or its payload


# ---------------------------------------------------------------------------
# Account-linking policy
# ---------------------------------------------------------------------------


async def test_an_existing_password_account_is_not_linked_by_email(
    google_client: AsyncClient, db_engine
):
    await google_client.post(
        "/api/auth/register", json={"email": "Grace@Example.com", "password": PASSWORD}
    )
    response = await google(google_client)  # grace@example.com: same, case-insensitively

    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "account_exists"
    assert REFRESH_COOKIE_NAME not in response.cookies
    # Untouched: still a password account, not tied to the Google identity.
    [(_, password_hash, google_sub)] = await users(db_engine)
    assert password_hash is not None and google_sub is None
    login = await google_client.post(
        "/api/auth/login", json={"email": "grace@example.com", "password": PASSWORD}
    )
    assert login.status_code == 200


async def test_a_google_only_account_cannot_password_login(google_client: AsyncClient):
    await google(google_client)
    for password in (PASSWORD, "not-a-real-password", "x"):  # the last: the timing decoy's
        response = await google_client.post(
            "/api/auth/login", json={"email": "grace@example.com", "password": password}
        )
        assert response.status_code == 401
        assert response.json()["detail"] == "Invalid email or password"


async def test_registering_a_google_accounts_email_is_a_conflict(google_client: AsyncClient):
    await google(google_client)
    response = await google_client.post(
        "/api/auth/register", json={"email": "grace@example.com", "password": PASSWORD}
    )
    assert response.status_code == 409


async def test_concurrent_first_sign_ins_create_one_account(google_client: AsyncClient, db_engine):
    responses = await asyncio.gather(*(google(google_client) for _ in range(5)))
    statuses = sorted(r.status_code for r in responses)
    assert statuses.count(201) == 1 and statuses.count(200) == 4, statuses
    ids = {
        (await google_client.get("/api/auth/me", headers=bearer(r))).json()["id"] for r in responses
    }
    assert len(ids) == 1
    assert len(await users(db_engine)) == 1


async def test_google_and_password_sign_up_racing_for_one_email(
    google_client: AsyncClient, db_engine
):
    g, p = await asyncio.gather(
        google(google_client),
        google_client.post(
            "/api/auth/register", json={"email": "grace@example.com", "password": PASSWORD}
        ),
    )
    assert sorted((g.status_code, p.status_code)) == [201, 409]
    assert len(await users(db_engine)) == 1


# ---------------------------------------------------------------------------
# The session it creates is an ordinary one
# ---------------------------------------------------------------------------


async def test_refresh_rotation_and_replay_detection_apply(google_client: AsyncClient):
    await google(google_client)
    spent = google_client.cookies.get(REFRESH_COOKIE_NAME)
    assert spent

    rotated = await google_client.post("/api/auth/refresh")
    assert rotated.status_code == 200
    assert google_client.cookies.get(REFRESH_COOKIE_NAME) != spent

    # Presenting the spent token is a replay: the whole family is revoked.
    current = google_client.cookies.get(REFRESH_COOKIE_NAME)
    google_client.cookies.clear()
    google_client.cookies.set(REFRESH_COOKIE_NAME, spent)
    assert (await google_client.post("/api/auth/refresh")).status_code == 401
    google_client.cookies.clear()
    google_client.cookies.set(REFRESH_COOKIE_NAME, current)
    assert (await google_client.post("/api/auth/refresh")).status_code == 401


async def test_logout_ends_a_google_session(google_client: AsyncClient):
    await google(google_client)
    assert (await google_client.post("/api/auth/logout")).status_code == 204
    assert (await google_client.post("/api/auth/refresh")).status_code == 401


async def test_protected_routes_need_the_access_token(google_client: AsyncClient):
    response = await google(google_client)
    assert (await google_client.get("/api/sessions")).status_code == 401
    assert (await google_client.get("/api/sessions", headers=bearer(response))).status_code == 200


async def test_password_auth_is_unchanged_when_google_is_on(google_client: AsyncClient):
    reg = await google_client.post(
        "/api/auth/register", json={"email": "ada@example.com", "password": PASSWORD}
    )
    assert reg.status_code == 201
    login = await google_client.post(
        "/api/auth/login", json={"email": "ada@example.com", "password": PASSWORD}
    )
    assert login.status_code == 200
    wrong = await google_client.post(
        "/api/auth/login", json={"email": "ada@example.com", "password": "wrong-password"}
    )
    assert wrong.status_code == 401
