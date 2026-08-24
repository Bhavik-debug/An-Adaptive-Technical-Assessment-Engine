"""Password hashing and token signing. No database, no Redis, no HTTP."""

from __future__ import annotations

import datetime as dt
import uuid

import jwt
import pytest

from app.security import (
    JWT_ALGORITHM,
    InvalidToken,
    TokenType,
    create_token,
    decode_token,
    dummy_hash,
    hash_password,
    password_needs_rehash,
    verify_password,
)

SECRET = "s" * 48
OTHER_SECRET = "o" * 48


# ---------------------------------------------------------------------------
# Passwords
# ---------------------------------------------------------------------------


def test_hash_then_verify_round_trips():
    digest = hash_password("correct horse battery staple")
    assert verify_password(digest, "correct horse battery staple") is True


def test_wrong_password_is_rejected():
    digest = hash_password("correct horse battery staple")
    assert verify_password(digest, "Correct horse battery staple") is False


def test_the_plaintext_never_appears_in_the_digest():
    digest = hash_password("hunter2-and-more")
    assert "hunter2" not in digest
    assert digest.startswith("$argon2id$")


def test_identical_passwords_produce_different_hashes():
    # Proves a random salt is applied. Without one, two users sharing a password
    # would share a hash, which is visible in a leaked dump and makes precomputed
    # rainbow tables effective.
    a = hash_password("same-password-123")
    b = hash_password("same-password-123")
    assert a != b
    assert verify_password(a, "same-password-123")
    assert verify_password(b, "same-password-123")


def test_verify_returns_false_for_a_garbage_hash():
    # A corrupted column must not raise a 500 out of the login endpoint.
    assert verify_password("not-a-hash", "anything") is False


def test_dummy_hash_is_a_real_verifiable_hash():
    # It has to cost the same as a real verification, or the timing-equalisation
    # it exists for does not work.
    assert dummy_hash().startswith("$argon2id$")
    assert verify_password(dummy_hash(), "not-a-real-password") is True


def test_current_parameters_do_not_need_rehashing():
    assert password_needs_rehash(hash_password("whatever-123")) is False


def test_unparseable_hash_is_treated_as_needing_rehash():
    assert password_needs_rehash("garbage") is True


# ---------------------------------------------------------------------------
# Tokens
# ---------------------------------------------------------------------------


def test_access_token_round_trips():
    user_id = uuid.uuid4()
    issued = create_token(
        subject=user_id,
        token_type=TokenType.ACCESS,
        ttl=dt.timedelta(minutes=15),
        secret=SECRET,
    )
    claims = decode_token(issued.token, secret=SECRET, expected_type=TokenType.ACCESS)

    assert claims.subject == user_id
    assert claims.jti == issued.jti
    assert claims.token_type is TokenType.ACCESS


def test_every_token_gets_a_unique_id():
    user_id = uuid.uuid4()
    kwargs = {"token_type": TokenType.REFRESH, "ttl": dt.timedelta(days=1), "secret": SECRET}
    first = create_token(subject=user_id, **kwargs)  # type: ignore[arg-type]
    second = create_token(subject=user_id, **kwargs)  # type: ignore[arg-type]
    # Rotation identifies tokens by jti; duplicates would make "already spent"
    # unanswerable.
    assert first.jti != second.jti


def test_a_refresh_token_is_not_accepted_as_an_access_token():
    # The single most important check in this file. Without the `typ` claim a
    # 30-day refresh token would work as a bearer credential everywhere.
    issued = create_token(
        subject=uuid.uuid4(),
        token_type=TokenType.REFRESH,
        ttl=dt.timedelta(days=30),
        secret=SECRET,
    )
    with pytest.raises(InvalidToken, match="expected a access token"):
        decode_token(issued.token, secret=SECRET, expected_type=TokenType.ACCESS)


def test_an_access_token_is_not_accepted_as_a_refresh_token():
    issued = create_token(
        subject=uuid.uuid4(),
        token_type=TokenType.ACCESS,
        ttl=dt.timedelta(minutes=15),
        secret=SECRET,
    )
    with pytest.raises(InvalidToken):
        decode_token(issued.token, secret=SECRET, expected_type=TokenType.REFRESH)


def test_expired_token_is_rejected():
    issued = create_token(
        subject=uuid.uuid4(),
        token_type=TokenType.ACCESS,
        ttl=dt.timedelta(minutes=15),
        secret=SECRET,
        now=dt.datetime.now(dt.UTC) - dt.timedelta(hours=2),
    )
    with pytest.raises(InvalidToken):
        decode_token(issued.token, secret=SECRET, expected_type=TokenType.ACCESS)


def test_token_signed_with_another_secret_is_rejected():
    issued = create_token(
        subject=uuid.uuid4(),
        token_type=TokenType.ACCESS,
        ttl=dt.timedelta(minutes=15),
        secret=OTHER_SECRET,
    )
    with pytest.raises(InvalidToken):
        decode_token(issued.token, secret=SECRET, expected_type=TokenType.ACCESS)


def test_tampered_payload_is_rejected():
    issued = create_token(
        subject=uuid.uuid4(),
        token_type=TokenType.ACCESS,
        ttl=dt.timedelta(minutes=15),
        secret=SECRET,
    )
    header, payload, signature = issued.token.split(".")
    forged = f"{header}.{payload[:-4]}AAAA.{signature}"
    with pytest.raises(InvalidToken):
        decode_token(forged, secret=SECRET, expected_type=TokenType.ACCESS)


def test_alg_none_token_is_rejected():
    # The classic JWT attack: strip the signature and claim the algorithm is
    # "none". Pinning algorithms=[HS256] on decode is what stops it.
    forged = jwt.encode(
        {
            "sub": str(uuid.uuid4()),
            "typ": "access",
            "jti": str(uuid.uuid4()),
            "iat": int(dt.datetime.now(dt.UTC).timestamp()),
            "exp": int((dt.datetime.now(dt.UTC) + dt.timedelta(hours=1)).timestamp()),
        },
        key="",
        algorithm="none",
    )
    with pytest.raises(InvalidToken):
        decode_token(forged, secret=SECRET, expected_type=TokenType.ACCESS)


def test_token_missing_required_claims_is_rejected():
    naked = jwt.encode({"sub": str(uuid.uuid4()), "typ": "access"}, SECRET, algorithm=JWT_ALGORITHM)
    with pytest.raises(InvalidToken):
        decode_token(naked, secret=SECRET, expected_type=TokenType.ACCESS)


def test_completely_malformed_string_is_rejected():
    with pytest.raises(InvalidToken):
        decode_token("this is not a jwt", secret=SECRET, expected_type=TokenType.ACCESS)
