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

#: Alternate prompt versions for a task, keyed by ``(task, version)``.
#:
#: **What this is for, and what it is not.** :data:`PROMPTS` holds the one
#: *active* prompt per task - the one production uses, which
#: :func:`register_prompt` refuses to let two modules disagree about. This holds
#: versions that exist only to be **measured against** the active one.
#:
#: Day 19's planned work is "prompt iteration v1 -> v3 against the metric" (plan
#: section 3's Day-19 row), and plan section 7.4 technique 6 wants a pinned
#: prompt version "referenced by content hash in every trace, so a quality
#: regression is attributable to a specific prompt change". Both need more than
#: one version of a task's prompt to be *addressable at once*, while exactly one
#: stays the default.
#:
#: Registering a variant changes nothing about production behaviour: no caller
#: reaches a variant unless it passes it explicitly to ``call_structured``.
PROMPT_VARIANTS: dict[tuple[TaskName, str], PromptTemplate] = {}


def register_prompt(template: PromptTemplate) -> PromptTemplate:
    """Let the package that owns a task's *wording* own its prompt.

    Day 16 added the first domain prompt, and with it a boundary question the
    connectivity probe never raised.  Plan section 13.2 says ``llm/`` must not
    know about domain logic, and a grading prompt is domain logic: it carries
    the anchored rubric, which ``grading/`` owns.  Defining it here would put
    four rubric dimensions and twenty level descriptors inside the provider
    layer; importing ``app.grading`` from here would point the dependency arrow
    the wrong way.

    So the registry stays here and the *content* lives with its owner:
    ``app/grading/prompt.py`` calls this at import time.  That is safe because
    nothing can reach ``TaskName.GRADE_ANSWER`` without going through
    ``app.grading.judge_answer``, which imports the package that registers it -
    and calling ``call_structured(TaskName.GRADE_ANSWER, ...)`` directly, from
    outside the owning package, is the thing the architecture forbids anyway.

    Re-registering identical text is a no-op so that a double import is
    harmless; re-registering *different* text raises, because two modules
    disagreeing about one task's prompt is a bug that would otherwise be decided
    by import order.
    """
    existing = PROMPTS.get(template.task)
    if existing is not None and existing.fingerprint != template.fingerprint:
        raise ValueError(
            f"task {template.task.value!r} already has a prompt "
            f"({existing.version}/{existing.fingerprint}); refusing to replace it with "
            f"{template.version}/{template.fingerprint}"
        )
    PROMPTS[template.task] = template
    return template


def get_prompt(task: TaskName) -> PromptTemplate:
    try:
        return PROMPTS[task]
    except KeyError as exc:
        raise PromptNotRegisteredError(
            f"task {task.value!r} has no prompt template yet; "
            "it is declared in the routing table but lands in a later phase, or its "
            "owning package has not been imported (see register_prompt)"
        ) from exc


def register_prompt_variant(template: PromptTemplate) -> PromptTemplate:
    """Declare an alternate version of a task's prompt, for evaluation only.

    The active prompt (``PROMPTS``) is untouched, so nothing in production
    changes: a variant is reachable only by passing it explicitly to
    ``call_structured(..., prompt=...)``, which the Day-19 harness does and no
    turn path does.

    Registering is still required rather than optional, for the same reason
    :func:`register_prompt` exists: a prompt version that is not in a registry
    cannot be listed, cannot be checked for a duplicated version label, and
    cannot be found a year later by the version string in a trace.

    Re-registering identical text is a no-op; re-registering *different* text
    under a version label that is already taken raises, because two prompts
    claiming to be "v2" of one task makes every measurement attributed to "v2"
    ambiguous.
    """
    key = (template.task, template.version)
    if template.task in PROMPTS and PROMPTS[template.task].version == template.version:
        active = PROMPTS[template.task]
        if active.fingerprint != template.fingerprint:
            raise ValueError(
                f"task {template.task.value!r} version {template.version!r} is already the "
                f"ACTIVE prompt with a different fingerprint ({active.fingerprint} vs "
                f"{template.fingerprint}); pick a new version label for the variant"
            )
        return template
    existing = PROMPT_VARIANTS.get(key)
    if existing is not None and existing.fingerprint != template.fingerprint:
        raise ValueError(
            f"task {template.task.value!r} already has a variant {template.version!r} "
            f"({existing.fingerprint}); refusing to replace it with {template.fingerprint}"
        )
    PROMPT_VARIANTS[key] = template
    return template


def get_prompt_version(task: TaskName, version: str) -> PromptTemplate:
    """One named version of a task's prompt: the active one, or a variant."""
    active = PROMPTS.get(task)
    if active is not None and active.version == version:
        return active
    try:
        return PROMPT_VARIANTS[(task, version)]
    except KeyError as exc:
        known = sorted(prompt_versions(task))
        raise PromptNotRegisteredError(
            f"task {task.value!r} has no prompt version {version!r}; known versions: {known}"
        ) from exc


def prompt_versions(task: TaskName) -> list[str]:
    """Every registered version label for a task, active first."""
    out = []
    active = PROMPTS.get(task)
    if active is not None:
        out.append(active.version)
    out.extend(version for (candidate, version) in sorted(PROMPT_VARIANTS) if candidate == task)
    return out
