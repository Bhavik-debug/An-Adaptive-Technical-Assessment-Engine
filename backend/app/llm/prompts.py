"""Versioned prompt templates.

A prompt is the text we send the model.  It is not a string literal buried in a
function - it is an input to the system that changes its output, which makes it
*code*, and code gets a version number.

Why the version matters more than it looks: when grading agreement drops next
month, the only question worth asking is "which prompt version caused this?".
That question is answerable if and only if every call records the version it
used.  ``prompt_version`` is therefore a required attribute on every LLM span
(plan section 14.2), and it originates here.

The templates use ``string.Template`` (``$name``) rather than ``str.format``
(``{name}``) on purpose: these prompts contain JSON examples full of braces, and
``str.format`` would try to interpret every one of them.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from string import Template

from app.llm.errors import PromptNotRegisteredError
from app.llm.tasks import TaskName
from app.llm.types import ChatMessage, Role


@dataclass(frozen=True, slots=True)
class PromptTemplate:
    task: TaskName
    #: Bumped by hand whenever the wording changes in a way that could move
    #: outputs.  Appears in traces and in the cache key.
    version: str
    system: str
    user: str
    required_inputs: tuple[str, ...] = ()

    @property
    def fingerprint(self) -> str:
        """Digest of the actual text.

        The hand-maintained ``version`` is the label a human reads; this is the
        machine's check that the label was actually bumped.  A prompt edited
        without a version bump shows up as the same version with a different
        fingerprint, which is exactly the mistake worth catching.
        """
        digest = hashlib.sha256(f"{self.system}\x00{self.user}".encode())
        return digest.hexdigest()[:12]

    def render(self, inputs: dict[str, object]) -> tuple[ChatMessage, ...]:
        """Fill the template, or fail loudly about what is missing."""
        missing = [key for key in self.required_inputs if key not in inputs]
        if missing:
            raise KeyError(
                f"task {self.task.value!r} prompt {self.version} requires " f"{', '.join(missing)}"
            )
        substitutions = {key: str(inputs[key]) for key in self.required_inputs}
        return (
            ChatMessage(role=Role.SYSTEM, content=self.system.strip()),
            # ``safe_substitute`` would silently leave an unknown ``$x`` in the
            # prompt; ``substitute`` raises, which is what we want.
            ChatMessage(
                role=Role.USER,
                content=Template(self.user).substitute(substitutions).strip(),
            ),
        )


# ---------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------
#
# Day 3 ships exactly one template: the connectivity probe.  Every other task in
# ``TaskName`` is declared for routing but has no prompt until the phase that
# owns it - grading prompts are Phase 4 work, not Phase 1 work.

_CONNECTIVITY_PROBE = PromptTemplate(
    task=TaskName.CONNECTIVITY_PROBE,
    version="v1",
    system=(
        "You are a health probe for a backend service. "
        "Answer only with the JSON object requested. Do not add commentary."
    ),
    user=(
        "Echo the token below back in the `echo` field, set `ok` to true, and "
        "name the model you are in the `model_said` field.\n\n"
        "token: $token"
    ),
    required_inputs=("token",),
)

PROMPTS: dict[TaskName, PromptTemplate] = {
    TaskName.CONNECTIVITY_PROBE: _CONNECTIVITY_PROBE,
}


def get_prompt(task: TaskName) -> PromptTemplate:
    try:
        return PROMPTS[task]
    except KeyError as exc:
        raise PromptNotRegisteredError(
            f"task {task.value!r} has no prompt template yet; "
            "it is declared in the routing table but lands in a later phase"
        ) from exc
