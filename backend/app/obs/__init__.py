"""Observability - plan sections 13.1 (``obs/``) and 14.2.

The question this package exists to answer: **when something goes wrong, what
happened, where, and how long did it take?**

Three kinds of signal, and they are not interchangeable:

* **Logs** - discrete events with detail.  "This grading call failed schema
  validation twice and then gave up."  Best signal-to-effort ratio when you
  already know roughly where to look.
* **Traces** - the causal, timed structure of one operation.  "This request
  spent 1,240 ms of its 1,600 ms inside one model call."  Traces are what tell
  you *where* the time went, which no amount of log-reading gives you.
* **Metrics** - numbers aggregated over time.  "p95 grading latency, hourly."
  Not built here: plan section 12.4 puts the operational dashboard in a later
  phase, and metrics without a system under real load measure nothing.

Module map:

* ``context.py``    the request id and user id carried through async code
* ``redaction.py``  what must never be written down, and how it is masked
* ``logging.py``    the structured, correlated, redacted log formatter
* ``tracing.py``    OpenTelemetry setup and the exporter choice
* ``spans.py``      ``CallMeta`` (Day 3) projected onto a span (Day 4)
* ``middleware.py`` request correlation and per-request latency

The two entry points a process calls at boot are ``configure_logging`` and
``init_tracing``; everything else is used by the code being observed.
"""

from app.obs.context import clear_context, get_request_id, set_user_id
from app.obs.logging import configure_logging, get_redactor
from app.obs.middleware import REQUEST_ID_HEADER, RequestContextMiddleware
from app.obs.spans import llm_call_span, record_error, record_llm_meta
from app.obs.tracing import get_tracer, init_tracing, instrument_app, shutdown_tracing

__all__ = [
    "REQUEST_ID_HEADER",
    "RequestContextMiddleware",
    "clear_context",
    "configure_logging",
    "get_redactor",
    "get_request_id",
    "get_tracer",
    "init_tracing",
    "instrument_app",
    "llm_call_span",
    "record_error",
    "record_llm_meta",
    "set_user_id",
    "shutdown_tracing",
]
