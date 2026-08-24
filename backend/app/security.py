"""Password hashing and token signing.

Deliberately pure: no database, no Redis, no HTTP. Every function here is a
value in, value out transformation, which makes the security-critical logic
testable without any infrastructure running.
"""

from __future__ import annotations

import datetime as dt
import uuid
from dataclasses import dataclass
from enum import StrEnum
from functools import lru_cache

import jwt
from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError

# ---------------------------------------------------------------------------
# Passwords
# ---------------------------------------------------------------------------
#
# A hash is a one-way function: easy to compute forwards, infeasible to invert.
# We store hash(password) and never the password, so a stolen database does not
# hand the attacker anyone's credentials.
#
# argon2id, not SHA-256, because a general-purpose hash is *designed* to be fast
# and a GPU can try billions per second. argon2 is deliberately slow AND
# memory-hard: each guess needs ~64 MiB of RAM, which is what makes massively
# parallel cracking hardware expensive rather than cheap. It won the 2015
# Password Hashing Competition and is the current OWASP first choice.
#
# argon2 also salts every hash automatically. A salt is random bytes mixed in
# before hashing, stored alongside the result, so two users with the same
# password get different hashes - which defeats precomputed rainbow tables and
# stops "these 400 accounts all share a hash" from being visible at a glance.
_ARGON2 = PasswordHasher(
    time_cost=2,  # passes over memory
    memory_cost=65536,  # 64 MiB per hash
    parallelism=1,  # threads; 1 keeps a login from monopolising the box
    hash_len=32,
    salt_len=16,
)

# Longer than this and the KDF cost becomes a cheap denial-of-service vector.
MAX_PASSWORD_BYTES = 1024
MIN_PASSWORD_LENGTH = 8


def hash_password(password: str) -> str:
    """Return an argon2id digest that embeds its own parameters and salt."""
    return _ARGON2.hash(password)


def verify_password(password_hash: str, password: str) -> bool:
    """Constant-ish time check. Returns False rather than raising."""
    try:
        return _ARGON2.verify(password_hash, password)
    except (VerifyMismatchError, VerificationError, InvalidHashError):
        return False


@lru_cache(maxsize=1)
def dummy_hash() -> str:
    """A throwaway hash to verify against when the account does not exist.

    Without this, "no such user" returns in ~1 ms and "wrong password" takes
    ~50 ms, and anyone can enumerate which emails have accounts by timing the
    responses. Verifying against a decoy costs the same as a real check.
    """
    return _ARGON2.hash("not-a-real-password")


def password_needs_rehash(password_hash: str) -> bool:
    """True when a stored hash used weaker parameters than we now require.

    Lets us silently upgrade a user's hash on their next successful login when
    the cost parameters above are raised.
    """
    try:
        return _ARGON2.check_needs_rehash(password_hash)
    except InvalidHashError:
        return True


# ---------------------------------------------------------------------------
# Tokens
# ---------------------------------------------------------------------------
#
# A JWT is three base64 segments joined by dots: header.payload.signature.
# The payload is *encoded, not encrypted* - anyone holding the token can read
# the claims inside. The signature is an HMAC over the first two segments using
# our SECRET_KEY, so nobody can change a claim without invalidating it.
#
# The value: the server can trust a token without a database lookup, because
# only the server can produce a valid signature. The cost: an issued token
# cannot be un-issued, which is exactly why access tokens are short-lived and
# why refresh tokens are additionally tracked in Redis (see token_store).

JWT_ALGORITHM = "HS256"


class TokenType(StrEnum):
    ACCESS = "access"
    REFRESH = "refresh"


class InvalidToken(Exception):
    """Raised for any token that is expired, tampered with, or the wrong type."""


@dataclass(frozen=True)
class TokenClaims:
    subject: uuid.UUID
    token_type: TokenType
    jti: uuid.UUID
    issued_at: dt.datetime
    expires_at: dt.datetime


@dataclass(frozen=True)
class IssuedToken:
    token: str
    jti: uuid.UUID
    expires_at: dt.datetime


def create_token(
    *,
    subject: uuid.UUID,
    token_type: TokenType,
    ttl: dt.timedelta,
    secret: str,
    now: dt.datetime | None = None,
) -> IssuedToken:
    """Sign a token for ``subject``.

    ``now`` is injectable purely so tests can produce an already-expired token
    without sleeping.
    """
    issued_at = now or dt.datetime.now(dt.UTC)
    expires_at = issued_at + ttl
    jti = uuid.uuid4()
    payload = {
        "sub": str(subject),
        # Without this claim, a refresh token would also be accepted as an
        # access token - a long-lived bearer credential, which is the bug this
        # whole design exists to avoid.
        "typ": token_type.value,
        "jti": str(jti),
        "iat": int(issued_at.timestamp()),
        "exp": int(expires_at.timestamp()),
    }
    token = jwt.encode(payload, secret, algorithm=JWT_ALGORITHM)
    return IssuedToken(token=token, jti=jti, expires_at=expires_at)


def decode_token(token: str, *, secret: str, expected_type: TokenType) -> TokenClaims:
    """Verify signature, expiry and type. Raises InvalidToken on any failure."""
    try:
        payload = jwt.decode(
            token,
            secret,
            # Pinned explicitly. Accepting the token's own `alg` header is the
            # classic JWT vulnerability: an attacker sets alg=none, or swaps
            # HS256 for RS256 so the public key gets used as an HMAC secret.
            algorithms=[JWT_ALGORITHM],
            options={"require": ["exp", "iat", "sub", "jti"]},
        )
    except jwt.PyJWTError as exc:
        raise InvalidToken(str(exc)) from exc

    if payload.get("typ") != expected_type.value:
        raise InvalidToken(f"expected a {expected_type.value} token")

    try:
        return TokenClaims(
            subject=uuid.UUID(payload["sub"]),
            token_type=expected_type,
            jti=uuid.UUID(payload["jti"]),
            issued_at=dt.datetime.fromtimestamp(payload["iat"], tz=dt.UTC),
            expires_at=dt.datetime.fromtimestamp(payload["exp"], tz=dt.UTC),
        )
    except (KeyError, ValueError) as exc:
        raise InvalidToken("malformed claims") from exc
