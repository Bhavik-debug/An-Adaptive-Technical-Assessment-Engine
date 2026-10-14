"""Verifying a Google Identity Services ID token ("Continue with Google").

The browser obtains a *credential* - an ID token, a JWT signed by Google - from
Google's own client library and posts it to ``POST /api/auth/google``.  Nothing
in it is trusted until :func:`verify_google_credential` has checked it with
Google's official library (``google-auth``):

* the **signature**, against Google's published signing keys (fetched over
  HTTPS from ``googleapis.com`` and cached for their ``Cache-Control`` max-age);
* the **issuer**, ``accounts.google.com`` (with or without ``https://``);
* the **expiry** and issue time, with a small clock-skew allowance;
* the **audience**, which must be *our* OAuth client ID - a token minted for any
  other site is refused, however valid its signature;

and then, here, the claims this application depends on: a subject (``sub``,
Google's stable account id), an email, and ``email_verified`` exactly ``true``.

Decoding a JWT without these checks would let anyone write their own "token";
that is why there is no code path that reads the payload any other way.

Deliberately pure apart from the key fetch: no database, no Redis.  The
endpoint (``app.api.auth``) decides what a verified identity is allowed to do.
"""

from __future__ import annotations

import re
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

import requests
from google.auth import exceptions as google_exceptions
from google.auth.transport import Response as TransportResponse
from google.auth.transport.requests import Request as RequestsTransport
from google.oauth2 import id_token

#: Google's ID tokens are ~1 KB.  Anything far larger is not one, and is refused
#: before any parsing work is done.
MAX_CREDENTIAL_CHARS = 4096

#: Seconds of clock disagreement tolerated on ``iat``/``exp``.
CLOCK_SKEW_S = 10

#: The key fetch must not hold a request open for long: Google's default
#: transport timeout is 120 s.
CERTS_TIMEOUT_S = 5.0

#: Used when Google's response carries no usable ``max-age``.
DEFAULT_CERTS_TTL_S = 300.0


class GoogleTokenInvalid(Exception):
    """The credential is not a valid, verified Google ID token for this app.

    The message is for logs and tests only; clients get one generic answer, so
    the endpoint is not an oracle for *which* check failed.
    """


class GoogleEmailUnverified(GoogleTokenInvalid):
    """A genuine token, but Google does not vouch for the email address."""


class GoogleUnavailable(Exception):
    """Google's signing keys could not be fetched; nothing was verified."""


@dataclass(frozen=True, slots=True)
class GoogleIdentity:
    #: Google's stable, never-reused account id.  The identity key - emails
    #: can change, ``sub`` cannot.
    subject: str
    email: str
    #: The token's own id, used to make each credential single-use.
    token_id: str | None
    #: When the token expires (epoch seconds) - how long a replay guard must last.
    expires_at: int


class _CachingCertsTransport:
    """A ``google.auth.transport.Request`` that caches GETs (the signing keys).

    ``google-auth`` re-fetches the keys on every verification otherwise.  The
    response is cached for its own ``Cache-Control: max-age``, which is how
    Google announces key rotation, so a cached copy never outlives Google's
    instruction.  Thread-safe: verification runs in the threadpool.
    """

    _MAX_AGE = re.compile(r"max-age=(\d+)")

    def __init__(self) -> None:
        self._transport = RequestsTransport(session=requests.Session())
        self._lock = threading.Lock()
        self._cache: dict[str, tuple[float, TransportResponse]] = {}

    def __call__(
        self, url: str, method: str = "GET", body: Any = None, headers: Any = None, **kwargs: Any
    ) -> TransportResponse:
        if method != "GET":  # pragma: no cover - verification only ever GETs
            return cast(
                TransportResponse,
                self._transport(url, method=method, body=body, headers=headers, **kwargs),
            )
        now = time.monotonic()
        with self._lock:
            hit = self._cache.get(url)
            if hit is not None and hit[0] > now:
                return hit[1]
        response = cast(
            TransportResponse,
            self._transport(url, method="GET", headers=headers, timeout=CERTS_TIMEOUT_S),
        )
        if response.status == 200:
            ttl = DEFAULT_CERTS_TTL_S
            match = self._MAX_AGE.search(str(response.headers.get("cache-control", "")))
            if match:
                ttl = float(match.group(1))
            with self._lock:
                self._cache[url] = (now + ttl, response)
        return response


_transport_lock = threading.Lock()
_transport: _CachingCertsTransport | None = None


def _default_transport() -> _CachingCertsTransport:
    global _transport
    with _transport_lock:
        if _transport is None:
            _transport = _CachingCertsTransport()
        return _transport


def _scrubbed(exc: BaseException, credential: str) -> str:
    """The library's reason, safe to log.

    google-auth's errors can quote the token itself (``MalformedError: Wrong
    number of segments in token: b'eyJ...'``), and even a truncated token
    carries the user's email in its payload. The credential, and each of its
    segments, is removed before the message goes anywhere.
    """
    message = str(exc)
    for part in sorted({credential, *credential.split(".")}, key=len, reverse=True):
        if len(part) >= 8:
            message = message.replace(part, "<credential>")
    return f"{type(exc).__name__}: {message}"[:200]


def verify_google_credential(
    credential: str, *, client_id: str, transport: Any = None
) -> GoogleIdentity:
    """Verify ``credential`` and return the identity it proves.

    Synchronous (the key fetch is blocking I/O); call it from a threadpool.
    ``transport`` is injectable so tests can serve their own signing keys
    without reaching the network.

    Raises :class:`GoogleTokenInvalid` (or its subclass
    :class:`GoogleEmailUnverified`), or :class:`GoogleUnavailable`.
    """
    if not client_id:
        # audience=None would make google-auth skip the audience check entirely.
        raise ValueError("client_id is required")
    if not credential or len(credential) > MAX_CREDENTIAL_CHARS:
        raise GoogleTokenInvalid("credential is empty or too long")

    try:
        claims: Mapping[str, Any] = id_token.verify_oauth2_token(  # type: ignore[no-untyped-call]
            credential,
            transport if transport is not None else _default_transport(),
            audience=client_id,
            clock_skew_in_seconds=CLOCK_SKEW_S,
        )
    except google_exceptions.TransportError as exc:
        raise GoogleUnavailable("could not fetch Google's signing keys") from exc
    except (google_exceptions.GoogleAuthError, ValueError) as exc:
        # Bad signature, unknown key id, wrong issuer, wrong audience, expired,
        # not yet valid, malformed - all the same to the caller.
        raise GoogleTokenInvalid(_scrubbed(exc, credential)) from exc

    subject = claims.get("sub")
    email = claims.get("email")
    if not isinstance(subject, str) or not subject:
        raise GoogleTokenInvalid("token has no subject")
    if not isinstance(email, str) or not email:
        raise GoogleTokenInvalid("token has no email")
    # Exactly True: a missing claim, or a string, is not Google vouching.
    if claims.get("email_verified") is not True:
        raise GoogleEmailUnverified("Google has not verified this email address")
    token_id = claims.get("jti")
    expires_at = claims.get("exp")
    return GoogleIdentity(
        subject=subject,
        email=email,
        token_id=token_id if isinstance(token_id, str) and token_id else None,
        expires_at=int(expires_at) if isinstance(expires_at, int | float) else 0,
    )


__all__ = [
    "MAX_CREDENTIAL_CHARS",
    "GoogleEmailUnverified",
    "GoogleIdentity",
    "GoogleTokenInvalid",
    "GoogleUnavailable",
    "verify_google_credential",
]
