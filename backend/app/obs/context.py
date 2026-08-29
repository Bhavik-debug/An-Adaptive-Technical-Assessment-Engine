"""The per-request identifiers every log line and span is stamped with.

A ``ContextVar`` is the async equivalent of a thread-local: a value set while
handling one request is visible to everything that request awaits, and invisible
to the request being handled concurrently next to it.  That property is the
whole reason correlation works in an async server - without it, two candidates
answering questions at the same moment would interleave their log lines with no
way to tell them apart.

Two ids, deliberately, because they answer different questions:

* ``request_id`` - one HTTP request.  Cheap, always present, returned to the
  caller in a response header so a bug report can quote it.
* ``user_id``    - who the caller turned out to be.  Unknown until the auth
  dependency has run, so it is bound later in the request than the first one.
"""

from __future__ import annotations

import re
import uuid
from contextvars import ContextVar, Token

#: What we are willing to accept from a client's ``X-Request-ID`` header.
#: An id from outside is untrusted input that ends up in every log line for the
#: request, so it is length-capped and character-restricted: a header carrying
#: a newline could otherwise forge log entries.
_SAFE_REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")

_request_id: ContextVar[str | None] = ContextVar("request_id", default=None)
_user_id: ContextVar[str | None] = ContextVar("user_id", default=None)


def new_request_id() -> str:
    """A fresh id. Hex rather than a dashed UUID: shorter in every log line."""
    return uuid.uuid4().hex


def sanitise_request_id(candidate: str | None) -> str:
    """Accept a client-supplied id, or mint one.

    Honouring an inbound id is what makes a request traceable across a proxy or
    a future frontend.  Honouring it *unvalidated* is a log-injection bug.
    """
    if candidate and _SAFE_REQUEST_ID.match(candidate):
        return candidate
    return new_request_id()


def set_request_id(value: str) -> Token[str | None]:
    return _request_id.set(value)


def reset_request_id(token: Token[str | None]) -> None:
    _request_id.reset(token)


def get_request_id() -> str | None:
    return _request_id.get()


def set_user_id(value: str | None) -> Token[str | None]:
    return _user_id.set(value)


def get_user_id() -> str | None:
    return _user_id.get()


def clear_context() -> None:
    """Drop both ids. For tests, and for workers between jobs."""
    _request_id.set(None)
    _user_id.set(None)
