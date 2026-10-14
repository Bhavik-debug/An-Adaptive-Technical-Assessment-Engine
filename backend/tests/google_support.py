"""Test-only Google ID tokens: a local RSA key standing in for Google's.

``app.google_auth`` verifies tokens with Google's real library; these helpers
let a test exercise that *real* verification without the network. A token is
signed with a throwaway key, and :class:`FakeCertsTransport` serves that key's
public half where google-auth fetches Google's signing keys from. Nothing here
is a credential.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import jwt
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

CLIENT_ID = "1234567890-test.apps.googleusercontent.com"
KEY_ID = "test-signing-key"


def _keypair() -> tuple[str, str]:
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    private = key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    ).decode()
    public = (
        key.public_key()
        .public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)
        .decode()
    )
    return private, public


PRIVATE_KEY, PUBLIC_KEY = _keypair()
#: A second key Google never published: tokens signed with it must fail.
OTHER_PRIVATE_KEY, _ = _keypair()


def claims(**overrides: Any) -> dict[str, Any]:
    """A valid Google ID token's claims, with any field overridden or removed (None)."""
    now = int(time.time())
    base: dict[str, Any] = {
        "iss": "https://accounts.google.com",
        "aud": CLIENT_ID,
        "azp": CLIENT_ID,
        "sub": "109876543210987654321",
        "email": "grace@example.com",
        "email_verified": True,
        "iat": now,
        "exp": now + 3600,
        "jti": uuid.uuid4().hex,
    }
    base.update(overrides)
    return {k: v for k, v in base.items() if v is not None}


def make_token(*, key: str = PRIVATE_KEY, kid: str = KEY_ID, **overrides: Any) -> str:
    return jwt.encode(claims(**overrides), key, algorithm="RS256", headers={"kid": kid})


@dataclass
class _Response:
    status: int
    data: bytes
    headers: dict[str, str] = field(default_factory=dict)


@dataclass
class FakeCertsTransport:
    """Answers google-auth's key fetch with our public key, and counts calls."""

    status: int = 200
    cache_control: str = "public, max-age=3600"
    calls: int = 0
    fail: Exception | None = None

    def __call__(self, url: str, method: str = "GET", **_: Any) -> _Response:
        self.calls += 1
        if self.fail is not None:
            raise self.fail
        body = json.dumps({KEY_ID: PUBLIC_KEY}).encode()
        return _Response(self.status, body, {"cache-control": self.cache_control})
