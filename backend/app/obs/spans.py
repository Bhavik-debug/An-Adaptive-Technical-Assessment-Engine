"""Turning a Day-3 ``CallMeta`` into a Day-4 span.

Day 3 already computes everything plan section 14.2 asks for - tokens summed
across retries, cost from the ``PRICES`` table, cache hit, schema retries,
failover count, latency.  Day 4 does **not** recompute any of it.  This module
is a projection: one object in, one span's worth of attributes out.

That split is the whole reason the chokepoint was worth building first.  The
accounting has one owner (``app/llm/client.py``) and the *reporting* has another
(here), so adding a second observability backend later cannot quietly change
what a token costs.

Two attribute vocabularies are emitted on the same span, on purpose:

* ``llm.*``      - the plan's own names, and the contract this project tests
  against.  ``llm.prompt_version`` is the one that matters most: without it you
  cannot attribute a quality regression to a prompt change, and doing that
  attribution is what production AI engineering actually is.
* ``gen_ai.*``   - the OpenTelemetry semantic convention for model calls, plus
  ``langfuse.observation.type``.  These make the span render as a *generation*
  in Langfuse - with token counts in the right boxes - instead of as an
  anonymous span with unfamiliar attribute names.

What is never on a span: the prompt, the model's answer, and its reasoning.
``CallMeta`` does not carry them, which makes that guarantee structural rather
than a rule someone has to remember (see ``app/llm/client.py``).
"""

from __future__ import annotations

import traceback
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from opentelemetry.trace import Span, SpanKind, Status, StatusCode

from app.obs.logging import get_redactor
from app.obs.tracing import get_tracer

if TYPE_CHECKING:  # pragma: no cover - typing only; importing this at runtime
    # would be a cycle: app.llm.client imports this module.
    from app.llm.client import CallMeta

#: ``llm.grade_answer`` rather than a bare ``grade_answer``: the prefix makes
#: every model call filterable in one query, at any depth of the trace.
SPAN_NAME_PREFIX = "llm."


@contextmanager
def llm_call_span(task: str) -> Iterator[Span]:
    """One span around one ``call_structured`` invocation.

    ``SpanKind.CLIENT`` says this process is calling out to something else,
    which is how a trace UI knows to render it as an external dependency and
    how latency here gets attributed to a provider rather than to our code.
    """
    with get_tracer().start_as_current_span(
        f"{SPAN_NAME_PREFIX}{task}", kind=SpanKind.CLIENT
    ) as span:
        span.set_attribute("llm.task", task)
        yield span


def record_llm_meta(span: Span, meta: CallMeta) -> None:
    """Stamp a completed call's metadata onto its span."""
    if not span.is_recording():
        return
    for key, value in meta.as_span_attributes().items():
        span.set_attribute(key, value)
    for key, value in gen_ai_attributes(meta).items():
        span.set_attribute(key, value)
    span.set_status(Status(StatusCode.OK))


def gen_ai_attributes(meta: CallMeta) -> dict[str, Any]:
    """The standard model-call attributes, derived from the same one object.

    Kept small deliberately.  The plan's ``llm.*`` set is the contract; this is
    a compatibility layer so a general-purpose trace viewer shows something
    sensible, and it must never become a second, drifting source of truth.
    """
    return {
        "gen_ai.operation.name": "chat",
        "gen_ai.system": meta.provider,
        "gen_ai.request.model": meta.model,
        "gen_ai.request.temperature": meta.temperature,
        "gen_ai.usage.input_tokens": meta.input_tokens,
        "gen_ai.usage.output_tokens": meta.output_tokens,
        "langfuse.observation.type": "generation",
    }


def record_error(span: Span, exc: BaseException) -> None:
    """Mark a span as failed, and say why - redacted.

    An ``exception`` event answers *what happened*; the error status is what
    makes the span countable in an error-rate query.  Both are set, because they
    answer different questions.

    The event is assembled by hand rather than with OTel's ``record_exception``
    for one reason: a span leaves this process for a third-party service, and an
    exception message is the classic place a credential or a fragment of
    candidate text ends up.  Building the event here means every field goes
    through the same redactor the logs use.  The full, unabridged traceback is
    still written to the structured log, which never leaves the host - and both
    carry the same ``trace_id``, so joining them is a filter, not a hunt.
    """
    if not span.is_recording():
        return
    redactor = get_redactor()
    stack = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    span.add_event(
        "exception",
        {
            "exception.type": type(exc).__name__,
            "exception.message": redactor.text(str(exc)),
            "exception.stacktrace": redactor.text(stack),
        },
    )
    span.set_status(Status(StatusCode.ERROR, type(exc).__name__))
