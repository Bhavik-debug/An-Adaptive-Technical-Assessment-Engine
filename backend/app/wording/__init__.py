"""What the model *says*: follow-up probes and question rendering (Day 23).

Plan section 6.5 names three generators; this package is two of them.  Both are
"speak" jobs in the plan's sense (section 2: the model understands, judges and
speaks - it never decides control flow):

* :mod:`app.wording.probe` - the follow-up probe.  **Whether** a follow-up
  happens was already decided in code (:mod:`app.interview.followup`); this
  only words it.
* :mod:`app.wording.render` - binding a canonical bank question to the
  candidate's own context, with the anchor-term guard of plan section 6.2.

Kept out of ``app.interview`` for the reason plan section 13.2 gives: the
interview module must not know about prompts.  The prompts are registered with
the LLM chokepoint the same way ``app.grading.prompt`` registers the grader's.
"""

from __future__ import annotations

from app.wording.probe import (
    ProbeContext,
    ProbeRejected,
    check_probe,
    generate_probe,
    probe_inputs,
)
from app.wording.render import RenderedQuestion, check_rendering, render_question

__all__ = [
    "ProbeContext",
    "ProbeRejected",
    "RenderedQuestion",
    "check_probe",
    "check_rendering",
    "generate_probe",
    "probe_inputs",
    "render_question",
]
