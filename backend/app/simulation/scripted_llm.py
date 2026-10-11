"""A deterministic, *scripted* stand-in for the model (Day 25).

Plan section 3, Day 25: "full 10-question session against a scripted
candidate, with a stubbed LLM".  The repository has a real-model recording for
exactly one bank question, so replay cannot grade a ten-question session, and
the stub's schema synthesis invents concept keys the real pipeline (correctly)
rejects.  This module answers the requests a session makes, **from the prompt
itself, by fixed rules** - so a whole interview runs offline, with no network.

**This is not grading.  Read the rule before reading any score it produces:**

* ``GradeJudgment`` (grade and re-grade): the answer **covers every expected
  concept** if it is at least 120 characters long and does not contain "not
  sure"; otherwise **every concept is absent**.  Evidence for a covered concept
  is the answer's first 60 characters (a verbatim substring, so Day 17's
  evidence check passes honestly); rubric scores are all 4 or all 0; confidence
  0.9; no errors, no misconceptions.
* ``FollowUpProbe``: a fixed probe question.
* ``ReportNarrative``: a summary naming the assessed topics, with no digits
  (so the report's invented-number guard has nothing to reject).
* Resume / JD extraction: empty results.
* Anything else: declined (``None``), which the stub turns into an error.

Everything *downstream* of the model is production code: prompt assembly, the
chokepoint, schema validation, ``check_judgment``, evidence verification,
scoring, the confidence gate, ability updates, the follow-up policy, selection,
stopping, reports.  Calls answered here are marked ``structured_mode =
"stub_scripted"`` in their metadata, so a scripted grade can never be mistaken
for a model's.

Registered on the stub provider by the app's composition root
(``app.main``) only when ``LLM_STUB_ON_MISSING=scripted``.
"""

from __future__ import annotations

import json
import re
from typing import Any

from app.grading.prompt import ANSWER_CLOSE, ANSWER_OPEN
from app.llm.types import CompletionRequest

STRONG_MIN_CHARS = 120
PROBE_TEXT = "Suppose traffic doubles overnight. What would you change first, and why?"

_CONCEPT_LINE = re.compile(r"^- ([a-z][a-z0-9_]{1,63}): ", re.MULTILINE)


def _user(request: CompletionRequest) -> str:
    return "\n".join(m.content for m in request.messages if m.role.value == "user")


def _between(text: str, start: str, end: str) -> str:
    i = text.rfind(start)
    j = text.find(end, i + len(start)) if i >= 0 else -1
    return text[i + len(start) : j].strip("\n") if i >= 0 and j >= 0 else ""


def _judgment(prompt: str) -> dict[str, Any]:
    block = prompt.split("## Expected concepts", 1)[-1].split("## Common misconceptions", 1)[0]
    keys = _CONCEPT_LINE.findall(block)
    answer = _between(prompt, ANSWER_OPEN, ANSWER_CLOSE)
    strong = len(answer.strip()) >= STRONG_MIN_CHARS and "not sure" not in answer.lower()
    evidence = answer.strip()[:60].strip()
    level = 4 if strong else 0
    return {
        "concepts": [
            {
                "key": k,
                "label": "covered" if strong else "absent",
                "evidence": evidence if strong else None,
            }
            for k in keys
        ],
        "errors": [],
        "misconceptions_hit": [],
        "rubric": {
            "structure": level,
            "tradeoff_awareness": level,
            "specificity": level,
            "communication": level,
        },
        "grader_confidence": 0.9,
    }


def _narrative(prompt: str) -> dict[str, Any]:
    try:
        stats = json.loads(_between(prompt, "<stats>", "</stats>"))
        topics = ", ".join(t["topic"] for t in stats.get("topics", [])) or "the planned topics"
    except (ValueError, KeyError, TypeError):
        topics = "the planned topics"
    return {
        "summary": (
            f"This scripted interview covered {topics}. The figures in this report "
            "come from the recorded grades; this summary only describes them."
        ),
        "strengths": [],
        "growth_areas": [],
        "next_steps": ["Review the subtopics listed as needing more assessment."],
    }


def scripted_response(request: CompletionRequest) -> str | None:
    """The scripted answer for ``request``, or ``None`` to decline."""
    name = request.json_schema.name
    prompt = _user(request)
    if name == "GradeJudgment":
        payload: dict[str, Any] = _judgment(prompt)
    elif name == "FollowUpProbe":
        payload = {"question": PROBE_TEXT}
    elif name == "ReportNarrative":
        payload = _narrative(prompt)
    elif name == "ResumeExtraction":
        payload = {"skills": []}
    elif name == "JdExtraction":
        payload = {"requirements": []}
    else:
        return None
    return json.dumps(payload, separators=(",", ":"))


__all__ = ["PROBE_TEXT", "STRONG_MIN_CHARS", "scripted_response"]
