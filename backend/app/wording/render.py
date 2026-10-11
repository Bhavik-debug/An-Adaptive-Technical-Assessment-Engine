"""Contextual binding of a canonical bank question - plan section 6.2.

"The canonical question text is authoritative and never invented by the model.
The model may only re-word it to reference something the candidate wrote", and
code then verifies "that the rendered text still contains the question's anchor
terms".

**Rendering is cosmetic, so it can never fail an interview.**  Any problem - no
context to bind to, a provider error, an invalid or anchor-dropping rewrite -
returns the canonical text, with the reason recorded.  The canonical text is
always a valid question; the rendered one is only ever a nicer framing of it.

**Today there is no candidate context**: it comes from the resume and job
description, which are Day 24.  With ``context=None`` this returns the
canonical text without calling any model, which is what every session does
until Day 24 supplies a context.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

from app.config import Settings
from app.grading.prompt import isolate_candidate_text
from app.llm.cache import ResponseCache
from app.llm.client import call_structured
from app.llm.router import ProviderRouter
from app.llm.tasks import TaskName
from app.llm.types import ANONYMOUS_TRACE, TraceCtx
from app.wording.prompt import RENDER_MAX_CHARS, RenderedQuestionOut

log = logging.getLogger(__name__)

RENDER_CANONICAL = "canonical"  # rendered == canonical, by design (no context)
RENDER_LLM = "rendered"


@dataclass(frozen=True, slots=True)
class RenderedQuestion:
    text: str
    #: ``rendered`` or ``canonical``; on a fallback, ``canonical`` and a reason.
    source: str
    fallback_reason: str | None = None


def check_rendering(text: str, anchor_terms: Sequence[str]) -> str | None:
    """``None`` if the rewrite is acceptable, else the rule it broke."""
    stripped = text.strip()
    if not stripped or len(stripped) > RENDER_MAX_CHARS:
        return "length"
    lowered = stripped.lower()
    missing = [t for t in anchor_terms if t and t.lower() not in lowered]
    return "anchor_terms_missing" if missing else None


async def render_question(
    canonical: str,
    *,
    anchor_terms: Sequence[str] = (),
    context: str | None = None,
    trace: TraceCtx = ANONYMOUS_TRACE,
    router: ProviderRouter | None = None,
    cache: ResponseCache | None = None,
    settings: Settings | None = None,
) -> RenderedQuestion:
    """The text to show for a bank question.  Never raises on model trouble."""
    if not context or not context.strip():
        return RenderedQuestion(canonical, RENDER_CANONICAL)
    try:
        out, _ = await call_structured(
            TaskName.QUESTION_RENDER,
            {"question": canonical, "context": isolate_candidate_text(context.strip())},
            RenderedQuestionOut,
            trace=trace,
            router=router,
            cache=cache,
            settings=settings,
        )
    except Exception as exc:  # noqa: BLE001 - rendering is optional; canonical always works
        log.warning("question rendering failed, using canonical text: %s", type(exc).__name__)
        return RenderedQuestion(canonical, RENDER_CANONICAL, "provider_error")
    problem = check_rendering(out.question, anchor_terms)
    if problem is not None:
        return RenderedQuestion(canonical, RENDER_CANONICAL, problem)
    return RenderedQuestion(out.question.strip(), RENDER_LLM)


__all__ = [
    "RENDER_CANONICAL",
    "RENDER_LLM",
    "RenderedQuestion",
    "check_rendering",
    "render_question",
]
