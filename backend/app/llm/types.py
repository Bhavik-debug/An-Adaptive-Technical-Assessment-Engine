"""The provider-agnostic vocabulary.

Everything above the provider adapter speaks only in these types.  Nothing here
mentions OpenAI, NVIDIA, HTTP, or JSON-mode flags - which is the whole point:
adding a second provider later must not require editing anything that imports
this module.
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class Role(StrEnum):
    """Who is speaking in a chat-style prompt.

    ``system`` sets standing instructions and is the only place trusted policy
    goes.  ``user`` carries the request.  ``assistant`` carries what the model
    said - we send one back during a schema-repair retry so the model can see
    its own malformed output.
    """

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass(frozen=True, slots=True)
class ChatMessage:
    role: Role
    content: str


@dataclass(frozen=True, slots=True)
class JsonSchemaSpec:
    """A pydantic model reduced to what a provider needs to constrain output."""

    name: str
    schema: dict[str, Any]
    #: Stable digest of the schema. Part of the cache key, so that changing a
    #: field name cannot serve you yesterday's answer for today's model.
    fingerprint: str


@dataclass(frozen=True, slots=True)
class ReasoningPolicy:
    """Whether the model may think before answering, and for how long.

    ``budget_tokens`` is a ceiling on the hidden reasoning, not on the answer.
    A provider that does not support reasoning ignores this entirely.
    """

    enabled: bool = False
    budget_tokens: int | None = None


class ModelTier(StrEnum):
    """Capability class, not a model name.

    The routing table (plan section 13.6) assigns tiers to tasks; each provider
    decides which of *its* models serves a tier.  That indirection is what lets
    a provider swap without touching the routing table.
    """

    SMALL_FAST = "small_fast"
    MID = "mid"
    LARGE = "large"


@dataclass(frozen=True, slots=True)
class CompletionRequest:
    """One provider-agnostic request for one JSON object."""

    messages: tuple[ChatMessage, ...]
    json_schema: JsonSchemaSpec
    tier: ModelTier
    temperature: float
    top_p: float
    max_output_tokens: int
    timeout_s: float
    reasoning: ReasoningPolicy


@dataclass(frozen=True, slots=True)
class CompletionResult:
    """One provider-agnostic answer.

    ``text`` is the final answer only.  Any chain-of-thought the model produced
    is in ``reasoning_text`` and is never merged into ``text`` - see
    ``providers/nvidia.py`` for where the two are separated.
    """

    text: str
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    reasoning_text: str | None = None
    reasoning_tokens: int = 0
    finish_reason: str | None = None
    #: How the provider was asked to enforce the schema on this call, for traces.
    structured_mode: str = "unknown"

    @property
    def truncated(self) -> bool:
        """True when the answer was cut off by the output-token ceiling.

        Worth naming: a truncated answer is always invalid JSON, and "the model
        cannot follow a schema" and "you gave it 40 tokens" deserve different
        error messages.
        """
        return self.finish_reason == "length"


class LLMProvider(ABC):
    """The seam. One implementation per vendor; the router knows only this."""

    #: Stable identifier used in configuration, metrics and trace attributes.
    name: str

    @abstractmethod
    def model_for(self, tier: ModelTier) -> str:
        """The concrete model id this provider uses for a capability tier."""

    @abstractmethod
    async def complete(self, request: CompletionRequest) -> CompletionResult:
        """Run one completion, or raise a ``ProviderError``.

        Contract every implementation owes the router:

        * never raise a vendor SDK exception - map it in ``errors.py`` terms
        * never exceed ``request.timeout_s``
        * never return reasoning text in ``CompletionResult.text``
        """

    async def aclose(self) -> None:
        """Release sockets. Default is a no-op for providers that hold none."""
        return None


@dataclass(frozen=True, slots=True)
class TraceCtx:
    """Who this call belongs to.

    Carried through purely so the metadata a call emits can be joined to a
    session, a turn and a billing plan.  Day 4 turns these into OpenTelemetry
    span attributes; today they land in the structured log line.
    """

    session_id: uuid.UUID | None = None
    turn_id: uuid.UUID | None = None
    user_id: uuid.UUID | None = None
    plan: str | None = None


#: Default for calls that belong to no session yet (scripts, smoke tests).
ANONYMOUS_TRACE = TraceCtx()


@dataclass(frozen=True, slots=True)
class ProviderAttempt:
    """One provider's outcome within a single ``call_structured`` invocation."""

    provider: str
    ok: bool
    error: str | None = None
    skipped_reason: str | None = None


@dataclass(frozen=True, slots=True)
class RouterOutcome:
    """What the router did, so the caller can report failover honestly."""

    result: CompletionResult
    attempts: tuple[ProviderAttempt, ...] = field(default_factory=tuple)

    @property
    def failover_count(self) -> int:
        """How many providers were tried and rejected before the winner."""
        return sum(1 for a in self.attempts if not a.ok)
