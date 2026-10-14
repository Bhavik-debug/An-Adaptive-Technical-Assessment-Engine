"""app.google_auth: Google's own library, run for real against a local key.

No network: :class:`FakeCertsTransport` serves the public key google-auth
would otherwise fetch from googleapis.com.
"""

from __future__ import annotations

import time

import jwt
import pytest
from google.auth import exceptions as google_exceptions

from app import google_auth
from app.google_auth import (
    MAX_CREDENTIAL_CHARS,
    GoogleEmailUnverified,
    GoogleTokenInvalid,
    GoogleUnavailable,
    verify_google_credential,
)
from tests.google_support import (
    CLIENT_ID,
    OTHER_PRIVATE_KEY,
    FakeCertsTransport,
    claims,
    make_token,
)


def verify(token: str, **kwargs):
    return verify_google_credential(
        token, client_id=kwargs.pop("client_id", CLIENT_ID), transport=FakeCertsTransport()
    )


def test_a_valid_token_yields_the_identity():
    token = make_token(jti="abc123")
    identity = verify(token)
    assert identity.subject == "109876543210987654321"
    assert identity.email == "grace@example.com"
    assert identity.token_id == "abc123"
    assert identity.expires_at > time.time()


def test_the_issuer_without_https_is_also_google():
    assert verify(make_token(iss="accounts.google.com")).subject


@pytest.mark.parametrize(
    ("label", "token"),
    [
        ("signed by a key Google never published", make_token(key=OTHER_PRIVATE_KEY)),
        ("an unknown key id", make_token(kid="not-a-google-key")),
        ("expired", make_token(iat=int(time.time()) - 7200, exp=int(time.time()) - 3600)),
        ("issued in the future", make_token(iat=int(time.time()) + 600)),
        ("for another app", make_token(aud="someone-else.apps.googleusercontent.com")),
        ("from another issuer", make_token(iss="https://evil.example.com")),
        ("not a JWT at all", "not-a-jwt"),
        ("a truncated JWT", make_token()[:-20]),
    ],
)
def test_invalid_tokens_are_rejected(label, token):
    with pytest.raises(GoogleTokenInvalid) as exc:
        verify(token)
    assert not isinstance(exc.value, GoogleEmailUnverified), label


def test_a_tampered_payload_fails_the_signature():
    header, payload, signature = make_token().split(".")
    other = make_token(sub="attacker", email="attacker@example.com").split(".")[1]
    with pytest.raises(GoogleTokenInvalid):
        verify(f"{header}.{other}.{signature}")


def test_an_alg_none_token_is_rejected():
    unsigned = jwt.encode(claims(), key=None, algorithm="none")
    with pytest.raises(GoogleTokenInvalid):
        verify(unsigned)


@pytest.mark.parametrize("value", [False, None, "true", 1])
def test_email_must_be_verified_exactly_true(value):
    # None removes the claim entirely.
    with pytest.raises(GoogleEmailUnverified):
        verify(make_token(email_verified=value))


def test_a_token_without_email_or_subject_is_rejected():
    with pytest.raises(GoogleTokenInvalid):
        verify(make_token(email=""))
    with pytest.raises(GoogleTokenInvalid):
        verify(make_token(sub=""))


def test_a_token_without_jti_verifies_but_reports_no_token_id():
    # The endpoint refuses these (it cannot make them single-use); the verifier
    # only reports the fact.
    assert verify(make_token(jti=None)).token_id is None


def test_empty_and_oversized_credentials_are_refused_before_parsing():
    transport = FakeCertsTransport()
    for credential in ("", "x" * (MAX_CREDENTIAL_CHARS + 1)):
        with pytest.raises(GoogleTokenInvalid):
            verify_google_credential(credential, client_id=CLIENT_ID, transport=transport)
    assert transport.calls == 0


def test_a_missing_client_id_is_a_programming_error_not_an_unchecked_audience():
    # google-auth skips the audience check when audience is None; never allow it.
    with pytest.raises(ValueError):
        verify_google_credential(make_token(), client_id="", transport=FakeCertsTransport())


def test_unreachable_google_keys_are_unavailable_not_invalid():
    for transport in (
        FakeCertsTransport(status=503),
        FakeCertsTransport(fail=google_exceptions.TransportError("connection refused")),
    ):
        with pytest.raises(GoogleUnavailable):
            verify_google_credential(make_token(), client_id=CLIENT_ID, transport=transport)


def test_the_caching_transport_honours_max_age(monkeypatch):
    upstream = FakeCertsTransport(cache_control="public, max-age=60")
    caching = google_auth._CachingCertsTransport()
    monkeypatch.setattr(caching, "_transport", upstream)
    clock = [1000.0]
    monkeypatch.setattr(google_auth.time, "monotonic", lambda: clock[0])

    for _ in range(3):
        verify_google_credential(make_token(), client_id=CLIENT_ID, transport=caching)
    assert upstream.calls == 1

    clock[0] += 61  # past max-age: Google may have rotated its keys
    verify_google_credential(make_token(), client_id=CLIENT_ID, transport=caching)
    assert upstream.calls == 2


def test_the_caching_transport_does_not_cache_failures(monkeypatch):
    upstream = FakeCertsTransport(status=500)
    caching = google_auth._CachingCertsTransport()
    monkeypatch.setattr(caching, "_transport", upstream)
    for _ in range(2):
        with pytest.raises(GoogleUnavailable):
            verify_google_credential(make_token(), client_id=CLIENT_ID, transport=caching)
    assert upstream.calls == 2


def test_rejection_reasons_never_quote_the_credential():
    # google-auth's MalformedError quotes the token; the reason must not.
    token = make_token()
    header, payload, _ = token.split(".")
    for credential in (f"{header}.{payload}", "abcdefghijk.lmnopqrstuv", token[:-30]):
        with pytest.raises(GoogleTokenInvalid) as exc:
            verify(credential)
        reason = str(exc.value)
        for part in credential.split("."):
            if len(part) >= 8:
                assert part not in reason
