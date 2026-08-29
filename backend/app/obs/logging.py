"""Structured logging: one JSON object per line, correlated and redacted.

**Why not plain text.**  ``2026-08-29 12:01:04 INFO app.llm.client | llm_call
{...}`` is readable by a human staring at one terminal and useless to everything
else.  A machine cannot answer "p95 latency of grading calls last hour" from
prose.  Emitting one JSON object per line - a *structured* log - means every
field is a queryable column, and the same line still reads fine in a terminal.

**What every line carries, whether the call site asked for it or not:**

* ``ts``, ``level``, ``logger``, ``msg`` - the ordinary things;
* ``request_id`` - which HTTP request this belongs to (``obs/context.py``);
* ``trace_id`` / ``span_id`` - which trace and which span were active, so a log
  line and a span in Langfuse can be joined without guessing;
* ``service`` / ``env`` - which process wrote it.

Anything a call site passes as ``extra={...}`` is merged in as top-level fields,
which is how ``call_structured`` emits the plan section 14.2 attribute set.

**Everything is redacted on the way out** (``obs/redaction.py``).  That is a
deliberate belt-and-braces: call sites are supposed to never log a secret, and
this layer assumes that one day a call site will.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import sys
from typing import Any

from app.config import LoggingSettings, Settings
from app.obs.context import get_request_id, get_user_id
from app.obs.redaction import Redactor

#: Attributes ``logging`` puts on every record itself. Anything else in a
#: record's ``__dict__`` arrived via ``extra=`` and is ours to emit.
_STANDARD_RECORD_ATTRS: frozenset[str] = frozenset(
    {
        "args",
        "asctime",
        # uvicorn attaches its own pre-formatting copy of the message, complete
        # with ANSI colour codes. Useful on a terminal, noise in a JSON field.
        "color_message",
        "created",
        "exc_info",
        "exc_text",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "message",
        "module",
        "msecs",
        "msg",
        "name",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "stack_info",
        "taskName",
        "thread",
        "threadName",
    }
)

#: Marks the handler this module installed, so re-configuring replaces our own
#: handler and leaves anyone else's alone - notably pytest's ``caplog``.
_OWNED = "_obs_owned_handler"

#: Libraries that log **request and response bodies** at DEBUG.
#:
#: This is not noise control, it is a privacy control.  ``openai._base_client``
#: writes the entire request options - prompt included - at DEBUG, and this
#: project's prompts carry resumes, answers and other candidate data.  Setting
#: ``LOG_LEVEL=DEBUG`` to investigate our own code would otherwise silently turn
#: on full-content logging for every model call, which is exactly the leak the
#: chokepoint was built to make impossible.  Found by running the live smoke
#: test with DEBUG on every logger and grepping the output for the prompt.
#:
#: They are floored at INFO, unconditionally.  A developer who genuinely needs
#: the SDK's wire log can raise it by hand after boot, deliberately, on a
#: machine with no real candidate data:
#:
#:     logging.getLogger("openai").setLevel(logging.DEBUG)
_CONTENT_LOGGING_LIBRARIES: tuple[str, ...] = (
    "openai",
    "httpx",
    "httpcore",
    "urllib3",
)

_redactor = Redactor()


def get_redactor() -> Redactor:
    """The process-wide redactor. Seeded with real secrets by ``configure_logging``."""
    return _redactor


class ContextFilter(logging.Filter):
    """Stamps correlation ids onto every record, from wherever it was emitted.

    A filter rather than an adapter: filters run for every record on the
    handler, so a third-party library's log line gets the same correlation ids
    as ours without the library knowing this project exists.
    """

    def __init__(self, *, service: str, env: str) -> None:
        super().__init__()
        self.service = service
        self.env = env

    def filter(self, record: logging.LogRecord) -> bool:
        # Imported here: tracing pulls in the OTel SDK, and logging must work in
        # a process that never configured tracing at all.
        from app.obs.tracing import current_span_id_hex, current_trace_id_hex

        record.service = self.service
        record.env = self.env
        record.request_id = get_request_id()
        record.user_id = get_user_id()
        record.trace_id = current_trace_id_hex()
        record.span_id = current_span_id_hex()
        return True


class RedactingJsonFormatter(logging.Formatter):
    """Renders a record as one redacted JSON object."""

    def __init__(self, redactor: Redactor) -> None:
        super().__init__()
        self._redactor = redactor

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": dt.datetime.fromtimestamp(record.created, tz=dt.UTC).isoformat(
                timespec="milliseconds"
            ),
            "level": record.levelname,
            "logger": record.name,
            "msg": self._redactor.text(record.getMessage()),
        }
        for key, value in record.__dict__.items():
            if key in _STANDARD_RECORD_ATTRS or key.startswith("_") or value is None:
                continue
            payload[key] = self._redactor.value(key, value)

        if record.exc_info:
            exc_type, exc, _tb = record.exc_info
            payload["error.type"] = exc_type.__name__ if exc_type else "Exception"
            payload["error.message"] = self._redactor.text(str(exc)) if exc else ""
            # A traceback names our own files and line numbers and can quote a
            # line of source - never a request body - so it is safe to keep, and
            # it is the single most useful field when something breaks.
            payload["error.stack"] = self._redactor.text(self.formatException(record.exc_info))
        if record.stack_info:
            payload["stack"] = self._redactor.text(self.formatStack(record.stack_info))

        return json.dumps(payload, default=str, ensure_ascii=False)


class RedactingTextFormatter(logging.Formatter):
    """The human-readable format, with the same redaction guarantees.

    Offered because tailing JSON in a terminal during development is miserable.
    It is never the default: what runs in CI and in a container should be the
    format the log pipeline expects.
    """

    _FMT = "%(asctime)s %(levelname)-8s %(name)s [%(request_id)s] | %(message)s"

    def __init__(self, redactor: Redactor) -> None:
        super().__init__(fmt=self._FMT)
        self._redactor = redactor

    def format(self, record: logging.LogRecord) -> str:
        if getattr(record, "request_id", None) is None:
            record.request_id = "-"
        return self._redactor.text(super().format(record))


def build_formatter(settings: LoggingSettings, redactor: Redactor) -> logging.Formatter:
    if settings.log_format == "text":
        return RedactingTextFormatter(redactor)
    return RedactingJsonFormatter(redactor)


def configure_logging(settings: LoggingSettings) -> None:
    """Install the root handler. Safe to call more than once.

    Takes ``LoggingSettings`` rather than ``Settings`` so that *every* process
    configures logging the same way - including the Alembic migration runner,
    which deliberately loads neither an LLM credential nor a database URL. A
    migration that emitted a different log format, unredacted, would be the one
    process in the system whose output nothing could parse.

    When the caller does have the full settings, the redactor is additionally
    seeded with this process's *actual* secret values: from that moment a log
    line containing the signing key or the provider key has it removed, even if
    the line was built by code that had no idea it was handling a secret.
    """
    if isinstance(settings, Settings):
        _redactor.add_secret(settings.secret_key)
        if settings.nvidia_api_key is not None:
            _redactor.add_secret(settings.nvidia_api_key.get_secret_value())

    root = logging.getLogger()
    for existing in list(root.handlers):
        if getattr(existing, _OWNED, False):
            root.removeHandler(existing)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(build_formatter(settings, _redactor))
    handler.addFilter(ContextFilter(service=settings.app_name, env=settings.app_env))
    setattr(handler, _OWNED, True)
    root.addHandler(handler)
    root.setLevel(settings.log_level)

    _adopt_uvicorn_loggers()
    _floor_content_logging_libraries()


def _floor_content_logging_libraries() -> None:
    """Stop ``LOG_LEVEL=DEBUG`` from turning on full-prompt logging.

    See ``_CONTENT_LOGGING_LIBRARIES``. The redactor would still mask an API key
    inside such a line, but it cannot mask a candidate's answer - and the
    correct answer is not to write it down at all.
    """
    for name in _CONTENT_LOGGING_LIBRARIES:
        logging.getLogger(name).setLevel(logging.INFO)


def _adopt_uvicorn_loggers() -> None:
    """Make uvicorn's own logs go through our handler.

    uvicorn installs handlers on ``uvicorn``, ``uvicorn.error`` and
    ``uvicorn.access``.  Left alone, that is a second log format *and* a second
    path to stdout that the redactor never sees - and the access log prints full
    request URLs, which is exactly where a token pasted into a query string
    would surface.  Clearing those handlers and letting the records propagate to
    the root sends them through the filter and the redactor instead.
    """
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logger = logging.getLogger(name)
        logger.handlers.clear()
        logger.propagate = True
