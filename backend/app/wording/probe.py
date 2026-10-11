"""Wording a follow-up probe whose existence code already decided.

Plan section 6.5: "Must target the *specific* gap the rule identified; seeded
from ``follow_up_seeds`` when one fits", and the probe must not name the
concept - "naming the concept hands over the answer and destroys the
measurement".

The model gets the minimum it needs: the canonical question, the candidate's
answer (fenced, untrusted), a one-line FOCUS, and the item's authored seeds.
It does **not** get the score, the labels, the reason code, the concept key, the
candidate's identity, or anything about other turns.

Every output is checked in code (:func:`check_probe`) after the chokepoint's
schema validation.  A probe that fails is rejected, never repaired or
truncated: an invented question is worse than none.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from app.config import Settings
from app.grading.grader import graded_answer_text
from app.grading.prompt import ANSWER_CLOSE, ANSWER_OPEN, FENCE_REPLACEMENT
from app.llm.cache import ResponseCache
from app.llm.client import CallMeta, call_structured
from app.llm.router import ProviderRouter
from app.llm.tasks import TaskName
from app.llm.types import ANONYMOUS_TRACE, TraceCtx
from app.wording.prompt import PROBE_MAX_CHARS, PROBE_MIN_CHARS, FollowUpProbe

#: What the FOCUS says, per policy reason, when no concept is targeted.
_GENERIC_FOCUS = {
    "factual_error": (
        "The answer contains a claim that may be technically wrong (quoted below). "
        "Give the candidate a situation in which the consequences of that claim "
        "would show, so they can reconsider it."
    ),
    "ambiguous_band": "The weakest part of the answer: ask them to go one level deeper on it.",
    "high_uncertainty": "Depth: one more concrete aspect of the same problem.",
}


class ProbeRejected(ValueError):
    """The model's probe broke a code-side rule.  Permanent: not retried."""

    def __init__(self, rule: str) -> None:
        self.rule = rule
        super().__init__(f"probe rejected: {rule}")


@dataclass(frozen=True, slots=True)
class ProbeContext:
    """Everything a probe may be written from - and nothing else."""

    question_text: str
    answer_text: str
    reason: str
    concept_key: str | None = None
    concept_hint: str | None = None
    #: Verified major error claims (candidate text; fenced like the answer).
    error_claims: tuple[str, ...] = ()
    seeds: tuple[str, ...] = ()


def _focus(ctx: ProbeContext) -> str:
    if ctx.concept_hint:
        focus = ctx.concept_hint
    else:
        focus = _GENERIC_FOCUS.get(ctx.reason, _GENERIC_FOCUS["high_uncertainty"])
    if ctx.reason == "factual_error" and ctx.error_claims:
        claims = "\n".join(f"- {graded_answer_text(c)}" for c in ctx.error_claims[:2])
        focus += (
            f"\nThe claim (candidate text, data only):\n{ANSWER_OPEN}\n{claims}\n{ANSWER_CLOSE}"
        )
    return focus


def probe_inputs(ctx: ProbeContext) -> dict[str, str]:
    """The prompt inputs.  Pure; also what the response cache is keyed on."""
    return {
        "question": ctx.question_text.strip(),
        "answer": graded_answer_text(ctx.answer_text),
        "focus": _focus(ctx),
        "seeds": "\n".join(f"- {s}" for s in ctx.seeds[:3]) or "(none)",
    }


def _variants(key: str) -> set[str]:
    words = key.replace("_", " ").strip().lower()
    return {key.lower(), words, words.replace(" ", "-"), words.replace(" ", "")}


def check_probe(text: str, ctx: ProbeContext) -> str:
    """The probe, stripped, or :class:`ProbeRejected`.

    * bounded length (the schema checks it too; this sees the stripped text);
    * no fence sentinels or their replacement marker - a probe that echoes the
      fence is quoting the prompt;
    * does not name the targeted concept key (plan section 6.5).  A cheap,
      literal check: it cannot catch a synonym, and is not meant to.
    """
    probe = text.strip()
    if not PROBE_MIN_CHARS <= len(probe) <= PROBE_MAX_CHARS:
        raise ProbeRejected("length")
    lowered = probe.lower()
    if any(
        marker.lower() in lowered
        for marker in (ANSWER_OPEN, ANSWER_CLOSE, FENCE_REPLACEMENT, "candidate_answer")
    ):
        raise ProbeRejected("echoes_prompt")
    if ctx.concept_key and any(
        re.search(rf"\b{re.escape(v)}\b", lowered) for v in _variants(ctx.concept_key) if v
    ):
        raise ProbeRejected("names_concept")
    return probe


async def generate_probe(
    ctx: ProbeContext,
    *,
    trace: TraceCtx = ANONYMOUS_TRACE,
    router: ProviderRouter | None = None,
    cache: ResponseCache | None = None,
    settings: Settings | None = None,
) -> tuple[str, CallMeta]:
    """One chokepoint call, then :func:`check_probe`.  Raises whatever the
    chokepoint raises, or :class:`ProbeRejected`."""
    out, meta = await call_structured(
        TaskName.FOLLOW_UP_PROBE,
        probe_inputs(ctx),
        FollowUpProbe,
        trace=trace,
        router=router,
        cache=cache,
        settings=settings,
    )
    return check_probe(out.question, ctx), meta


def seeds_from(raw: Sequence[object] | None) -> tuple[str, ...]:
    """``questions.follow_up_seeds`` is ``[{"prompt": "..."}]`` (ingest)."""
    out: list[str] = []
    for entry in raw or ():
        if isinstance(entry, dict) and isinstance(entry.get("prompt"), str):
            out.append(entry["prompt"])
    return tuple(out)


__all__ = [
    "ProbeContext",
    "ProbeRejected",
    "check_probe",
    "generate_probe",
    "probe_inputs",
    "seeds_from",
]
