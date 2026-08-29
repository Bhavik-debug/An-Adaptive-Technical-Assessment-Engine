"""Request correlation: giving every request one id and one latency number.

**The problem this solves.**  An async server handles many requests at once, so
the log is interleaved.  "The grader returned 500" is useless on its own; "the
grader returned 500 *for the request that started with these inputs, 1.2 s
earlier*" is a diagnosis.  A **correlation id** is the thread that ties those
lines together.

**Correlation id vs trace id.**  They overlap and are not the same thing:

* the ``trace_id`` comes from OpenTelemetry and only exists where tracing is
  switched on and a span is open;
* the ``request_id`` is ours, always exists, is returned to the caller in the
  ``X-Request-ID`` response header, and survives being pasted into a bug report.

Both are stamped on every log line, so either one gets you to the same place.

**Written as raw ASGI, not Starlette's ``BaseHTTPMiddleware``.**  That base class
runs the rest of the application in a separate task, and a ``ContextVar`` set in
the middleware is then invisible to the endpoint - which would defeat the entire
purpose.  Raw ASGI stays on one task, so the context propagates.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable
from typing import Any

from starlette.datastructures import MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.obs.context import reset_request_id, sanitise_request_id, set_request_id
from app.obs.tracing import set_span_attributes

log = logging.getLogger("app.request")

REQUEST_ID_HEADER = "x-request-id"

#: Probed continuously by the orchestrator. Logged at DEBUG so a healthy stack
#: does not write a line every five seconds forever.
_QUIET_PATHS = frozenset({"/healthz", "/readyz"})


class RequestContextMiddleware:
    """Bind a request id, then log one line saying what happened and how long."""

    def __init__(self, app: ASGIApp, *, header_name: str = REQUEST_ID_HEADER) -> None:
        self.app = app
        self.header_name = header_name

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_id = sanitise_request_id(_header(scope, self.header_name))
        token = set_request_id(request_id)
        # The auto-instrumented server span is already open by the time this
        # runs, so the id lands on the request's root span and is inherited by
        # every child - including the LLM spans several layers down.
        set_span_attributes({"request.id": request_id})

        started = time.perf_counter()
        status_code = 500

        async def send_wrapper(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = int(message["status"])
                MutableHeaders(scope=message).append(self.header_name, request_id)
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        except Exception:
            # Log, then re-raise: the error belongs to whoever handles it, but a
            # request that dies before producing a response would otherwise leave
            # no latency record at all.
            self._log(scope, status_code, started, failed=True)
            raise
        else:
            self._log(scope, status_code, started, failed=False)
        finally:
            reset_request_id(token)

    def _log(self, scope: Scope, status_code: int, started: float, *, failed: bool) -> None:
        path = str(scope.get("path", ""))
        duration_ms = int((time.perf_counter() - started) * 1000)
        level = logging.DEBUG if path in _QUIET_PATHS and not failed else logging.INFO
        if failed or status_code >= 500:
            level = logging.ERROR
        log.log(
            level,
            "http_request",
            extra={
                "http.method": scope.get("method"),
                # The template (`/api/auth/me`), not the concrete URL: a path
                # parameter can be a user id, and grouping by template is what
                # makes a latency percentile mean anything.
                "http.route": _route(scope, path),
                "http.status_code": status_code,
                "duration_ms": duration_ms,
            },
        )


def _header(scope: Scope, name: str) -> str | None:
    wanted = name.lower().encode("latin-1")
    headers: Iterable[tuple[bytes, bytes]] = scope.get("headers", ())
    for key, value in headers:
        if key.lower() == wanted:
            return value.decode("latin-1", errors="replace")
    return None


def _route(scope: Scope, fallback: str) -> str:
    """The matched route template, when the router has already resolved one."""
    route: Any = scope.get("route")
    path_format = getattr(route, "path_format", None)
    if isinstance(path_format, str) and path_format:
        return path_format
    return fallback
