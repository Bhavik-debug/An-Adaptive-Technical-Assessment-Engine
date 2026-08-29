"""NVIDIA provider adapter - Nemotron 3.5 Lightning over the NIM endpoint.

**This is the only module in the repository allowed to know that NVIDIA
exists.**  Everything vendor-shaped is contained here: the base URL, the model
id, how the schema is enforced, how thinking is switched on, which HTTP status
means what, and how the response body is taken apart.  The router above receives
a ``CompletionResult`` and cannot tell which vendor produced it.

Four adaptations of the reference snippet NVIDIA publishes, and why each one:

**1. ``stream=False``.**  The reference streams.  Streaming exists so a UI can
paint tokens as they arrive; it is the wrong shape for ``call_structured()``,
whose contract is "a validated object, or an error".  A half-arrived JSON object
cannot be validated, so a streaming implementation would buffer every chunk and
validate at the end - the same outcome as a plain call, plus chunk reassembly,
plus a class of partial-failure states.  Plan section 13.8 also puts SSE at the
*API* boundary, carrying state transitions rather than model tokens, so nothing
downstream ever wanted a token stream from this layer.

**2. ``temperature`` and ``top_p`` come from the task, not the snippet.**  The
snippet's ``temperature=1, top_p=0.95`` are chat-demo defaults.  Extraction and
grading are judged on being repeatable, so they run at temperature 0.

**3. Thinking is off unless the task asks for it.**  See ``_reasoning_body``.

**4. Structured output is negotiated, not assumed.**  See ``StructuredMode``.
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Mapping
from enum import StrEnum
from typing import Any, Protocol

import openai

from app.llm.errors import (
    ProviderAuthError,
    ProviderBadRequestError,
    ProviderError,
    ProviderRateLimitedError,
    ProviderResponseError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)
from app.llm.structured import schema_instruction
from app.llm.types import (
    ChatMessage,
    CompletionRequest,
    CompletionResult,
    LLMProvider,
    ModelTier,
    Role,
)

log = logging.getLogger(__name__)

NVIDIA_DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"
NVIDIA_DEFAULT_MODEL = "nvidia/nemotron-3.5-lightning-30b-a3b"


class StructuredMode(StrEnum):
    """How hard the *provider* is asked to constrain the output.

    NIM deployments differ in which of these a given model build accepts, and
    the endpoint is the only authority on that.  Rather than guess, the adapter
    starts on the strongest rung and steps down once when the endpoint rejects
    it - then remembers, so the rejection is paid once per process rather than
    once per call.

    None of this changes correctness.  Validation happens in ``structured.py``
    either way; the rungs only change how often the first attempt succeeds.
    """

    #: Constrained decoding against our exact schema. Strongest.
    JSON_SCHEMA = "json_schema"
    #: "Emit syntactically valid JSON." The shape is left to the model.
    JSON_OBJECT = "json_object"
    #: Nothing enforced; the schema is in the prompt and validation is the gate.
    PROMPT_ONLY = "prompt_only"


_LADDER: tuple[StructuredMode, ...] = (
    StructuredMode.JSON_SCHEMA,
    StructuredMode.JSON_OBJECT,
    StructuredMode.PROMPT_ONLY,
)

#: A 400 mentioning any of these is the endpoint saying it does not accept that
#: parameter, which is recoverable by stepping down the ladder.  A 400 about
#: context length or a malformed message is not, and must surface as an error.
_UNSUPPORTED_MARKERS = (
    "response_format",
    "json_schema",
    "guided",
    "structured output",
    "not supported",
    "unsupported",
    "unrecognized",
    "extra_body",
)

#: Some Nemotron chat templates emit reasoning inline between these tags rather
#: than in a separate field.  Both shapes are handled; see ``_split_reasoning``.
_THINK_BLOCK = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
_THINK_OPEN = re.compile(r"<think>", re.IGNORECASE)
_THINK_CLOSE = re.compile(r"</think>", re.IGNORECASE)


class CompletionCreator(Protocol):
    """The single SDK call this adapter makes.

    Narrowing the dependency to one callable is what lets a unit test substitute
    a plain function and assert on the exact request body we send - with no
    network, no API key, and no HTTP-level mocking.
    """

    async def __call__(self, **kwargs: Any) -> Any: ...


class NvidiaProvider(LLMProvider):
    name = "nvidia"

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = NVIDIA_DEFAULT_BASE_URL,
        model: str = NVIDIA_DEFAULT_MODEL,
        tier_models: Mapping[ModelTier, str] | None = None,
        completion_creator: CompletionCreator | None = None,
    ) -> None:
        self._default_model = model
        self._tier_models = dict(tier_models or {})
        self._structured_mode = _LADDER[0]
        self._client: openai.AsyncOpenAI | None = None

        if completion_creator is not None:
            self._create: CompletionCreator = completion_creator
        else:
            # max_retries=0 on purpose. The SDK will happily retry 429s and 5xx
            # for us, but then the retry is invisible: it does not appear in our
            # attempt counts, it ignores our failover order, and a trace that
            # says "one call" secretly contains three. Retry policy is the
            # router's job, and a policy split across two layers is not a policy.
            client = openai.AsyncOpenAI(
                base_url=base_url,
                api_key=api_key,
                # Retrying is the router's job (see the note above).
                max_retries=0,
            )
            self._client = client

            async def _create(**kwargs: Any) -> Any:
                # A thin wrapper rather than the bound method directly: the
                # SDK's `create` is an overloaded function whose signature
                # varies with `stream`, and pinning it to one shape here is
                # what keeps the seam a single, checkable callable.
                return await client.chat.completions.create(**kwargs)

            self._create = _create

    # -- LLMProvider -------------------------------------------------------

    def model_for(self, tier: ModelTier) -> str:
        """NVIDIA currently serves every tier from one selected model.
        The mapping exists rather than the model id being hard-coded so that
        the day a second NVIDIA model is chosen for grading, it is a config
        entry rather than a code change.
        """
        return self._tier_models.get(tier, self._default_model)

    async def complete(self, request: CompletionRequest) -> CompletionResult:
        model = self.model_for(request.tier)
        for mode in self._modes_to_try():
            try:
                raw = await self._create_with_deadline(request, model=model, mode=mode)
            except ProviderBadRequestError as exc:
                if mode is not _LADDER[-1] and _looks_unsupported(str(exc)):
                    demoted = _LADDER[_LADDER.index(mode) + 1]
                    log.info(
                        "nvidia endpoint rejected structured mode %s; using %s from now on",
                        mode.value,
                        demoted.value,
                    )
                    self._structured_mode = demoted
                    continue
                raise
            return _parse_completion(raw, provider=self.name, fallback_model=model, mode=mode)

        raise ProviderResponseError(  # pragma: no cover - last rung raises instead
            "exhausted every structured-output mode", provider=self.name
        )

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None

    # -- internals ---------------------------------------------------------

    def _modes_to_try(self) -> list[StructuredMode]:
        return list(_LADDER[_LADDER.index(self._structured_mode) :])

    async def _create_with_deadline(
        self, request: CompletionRequest, *, model: str, mode: StructuredMode
    ) -> Any:
        body = self._build_body(request, model=model, mode=mode)
        try:
            # Two layers of deadline on purpose. The SDK's own timeout covers
            # the HTTP exchange; `wait_for` covers everything else, including a
            # connection that never returns a byte. A turn budget a provider can
            # silently exceed is not a budget.
            return await asyncio.wait_for(self._create(**body), timeout=request.timeout_s)
        except TimeoutError as exc:
            # :g rather than :.0f - a 0.5s deadline reported as "within 0s" is
            # the kind of log line that costs someone twenty minutes.
            raise ProviderTimeoutError(
                f"no response within {request.timeout_s:g}s", provider=self.name
            ) from exc
        except openai.OpenAIError as exc:
            raise _map_sdk_error(exc, provider=self.name) from exc

    def _build_body(
        self, request: CompletionRequest, *, model: str, mode: StructuredMode
    ) -> dict[str, Any]:
        messages = _with_schema_instruction(request.messages, request)
        body: dict[str, Any] = {
            "model": model,
            "messages": [{"role": m.role.value, "content": m.content} for m in messages],
            "temperature": request.temperature,
            "top_p": request.top_p,
            "max_tokens": request.max_output_tokens,
            # See the module docstring: structured calls are never streamed.
            "stream": False,
            "timeout": request.timeout_s,
        }

        response_format = _response_format(request, mode)
        if response_format is not None:
            body["response_format"] = response_format

        body["extra_body"] = self._reasoning_body(request)
        return body

    def _reasoning_body(self, request: CompletionRequest) -> dict[str, Any]:
        """NVIDIA's thinking switches, stated explicitly in both directions.

        ``enable_thinking`` is sent as ``False`` rather than omitted when
        reasoning is off, because the model's chat template - not us - owns the
        default, and a default that changes in a model update would silently
        change our latency and our token bill.  Stating it makes the behaviour
        ours to reason about.
        """
        kwargs: dict[str, Any] = {
            "chat_template_kwargs": {"enable_thinking": request.reasoning.enabled}
        }
        if request.reasoning.enabled and request.reasoning.budget_tokens:
            kwargs["reasoning_budget"] = request.reasoning.budget_tokens
        return kwargs


def _with_schema_instruction(
    messages: tuple[ChatMessage, ...], request: CompletionRequest
) -> tuple[ChatMessage, ...]:
    """Append the schema to the last user message.

    Done on every rung, including the strictest.  Constrained decoding forces
    the *shape*; the written schema, with its field descriptions, is what tells
    the model what to put in each field.
    """
    instruction = schema_instruction(request.json_schema)
    out = list(messages)
    for index in range(len(out) - 1, -1, -1):
        if out[index].role is Role.USER:
            out[index] = ChatMessage(
                role=Role.USER, content=f"{out[index].content}\n\n{instruction}"
            )
            return tuple(out)
    out.append(ChatMessage(role=Role.USER, content=instruction))
    return tuple(out)


def _response_format(request: CompletionRequest, mode: StructuredMode) -> dict[str, Any] | None:
    if mode is StructuredMode.JSON_SCHEMA:
        return {
            "type": "json_schema",
            "json_schema": {
                "name": request.json_schema.name,
                "schema": request.json_schema.schema,
                "strict": True,
            },
        }
    if mode is StructuredMode.JSON_OBJECT:
        return {"type": "json_object"}
    return None


def _looks_unsupported(message: str) -> bool:
    lowered = message.lower()
    return any(marker in lowered for marker in _UNSUPPORTED_MARKERS)


def _map_sdk_error(exc: openai.OpenAIError, *, provider: str) -> ProviderError:
    """Translate one SDK exception into this project's taxonomy.

    Order matters: ``APITimeoutError`` subclasses ``APIConnectionError``, and
    every specific status error subclasses ``APIStatusError``.
    """
    if isinstance(exc, openai.APITimeoutError):
        return ProviderTimeoutError("request timed out", provider=provider)
    if isinstance(exc, openai.APIConnectionError):
        return ProviderUnavailableError(f"connection failed: {exc}", provider=provider)

    if isinstance(exc, openai.APIStatusError):
        status = exc.status_code
        detail = _status_detail(exc)
        if status in (401, 403):
            # Deliberately does not echo the response body: an auth error is the
            # one place a provider might quote part of the credential back.
            return ProviderAuthError(
                f"authentication rejected (HTTP {status})",
                provider=provider,
                status_code=status,
            )
        if status == 429:
            return ProviderRateLimitedError(
                f"rate limited: {detail}",
                provider=provider,
                status_code=status,
                retry_after_s=_retry_after(exc),
            )
        if status >= 500:
            return ProviderUnavailableError(
                f"server error {status}: {detail}", provider=provider, status_code=status
            )
        return ProviderBadRequestError(
            f"rejected with {status}: {detail}", provider=provider, status_code=status
        )

    return ProviderUnavailableError(f"{type(exc).__name__}: {exc}", provider=provider)


def _status_detail(exc: openai.APIStatusError) -> str:
    message = getattr(exc, "message", None) or str(exc)
    return str(message)[:300]


def _retry_after(exc: openai.APIStatusError) -> float | None:
    """Honour ``Retry-After`` when the provider sends one."""
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    raw = headers.get("retry-after")
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        # The header also permits an HTTP date. Rare here, and the router's own
        # backoff is a fine substitute, so we do not parse that form.
        return None


def _parse_completion(
    raw: Any, *, provider: str, fallback_model: str, mode: StructuredMode
) -> CompletionResult:
    """Take the vendor response apart into provider-agnostic fields."""
    choices = getattr(raw, "choices", None)
    if not choices:
        raise ProviderResponseError("response contained no choices", provider=provider)

    choice = choices[0]
    message = getattr(choice, "message", None)
    if message is None:
        raise ProviderResponseError("response choice contained no message", provider=provider)

    content = getattr(message, "content", None) or ""
    # NIM returns chain-of-thought in a dedicated field when the chat template
    # has one; other templates inline it in <think> tags. Handle both.
    field_reasoning = getattr(message, "reasoning_content", None)
    answer, inline_reasoning = _split_reasoning(str(content))
    reasoning = _first_non_empty(field_reasoning, inline_reasoning)

    usage = getattr(raw, "usage", None)
    input_tokens = _int_attr(usage, "prompt_tokens")
    output_tokens = _int_attr(usage, "completion_tokens")
    reasoning_tokens = _int_attr(
        getattr(usage, "completion_tokens_details", None), "reasoning_tokens"
    )

    finish_reason = getattr(choice, "finish_reason", None)

    if not answer.strip():
        # A reasoning model that spends its whole budget thinking returns an
        # empty answer. Saying so beats a downstream "expected object, got ''".
        hint = " (the model spent its entire output budget on reasoning)" if reasoning else ""
        raise ProviderResponseError(
            f"response contained no answer content{hint}", provider=provider
        )

    return CompletionResult(
        text=answer,
        provider=provider,
        model=str(getattr(raw, "model", None) or fallback_model),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        reasoning_text=reasoning,
        reasoning_tokens=reasoning_tokens,
        finish_reason=str(finish_reason) if finish_reason is not None else None,
        structured_mode=mode.value,
    )


def _split_reasoning(content: str) -> tuple[str, str | None]:
    """Separate inline chain-of-thought from the answer.

    Three shapes occur in practice, all handled:

    * ``<think>...</think>answer``  - the normal case
    * ``...</think>answer``         - the template pre-filled the opening tag
    * ``<think>...``  (never closed) - the reasoning budget ran out mid-thought

    In the third case the answer is empty, which is correct and is reported as
    such by the caller.  Returning the truncated reasoning *as* the answer is
    exactly the bug this function exists to prevent.
    """
    if not content:
        return "", None

    blocks = _THINK_BLOCK.findall(content)
    if blocks:
        answer = _THINK_BLOCK.sub("", content).strip()
        return answer, "\n".join(b.strip() for b in blocks) or None

    close = _THINK_CLOSE.search(content)
    if close:
        head = content[: close.start()]
        reasoning = _THINK_OPEN.sub("", head).strip()
        return content[close.end() :].strip(), reasoning or None

    if _THINK_OPEN.search(content):
        reasoning = _THINK_OPEN.sub("", content).strip()
        return "", reasoning or None

    return content.strip(), None


def _first_non_empty(*values: object) -> str | None:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _int_attr(obj: object, name: str) -> int:
    value = getattr(obj, name, None)
    return int(value) if isinstance(value, int) else 0


# complete architecture diagram for the NVIDIA provider adapter

#  ┌─────────────────────────┐
#  │      APPLICATION        │
#  └────────────┬────────────┘
#               │
#               ▼
#  ┌─────────────────────────┐
#  │         ROUTER          │
#  │                         │
#  │ Provider selection      │
#  │ Retry policy            │
#  │ Failover policy         │
#  └────────────┬────────────┘
#               │
#      CompletionRequest
#               │
#               ▼
#  ┌─────────────────────────┐
#  │    NvidiaProvider       │
#  │      THIS FILE          │
#  ├─────────────────────────┤
#  │ model_for()             │
#  │ _build_body()           │
#  │ _response_format()      │
#  │ _reasoning_body()       │
#  │ _create_with_deadline() │
#  │ _map_sdk_error()        │
#  │ _parse_completion()     │
#  └────────────┬────────────┘
#               │
#     OpenAI-compatible API
#               │
#               ▼
#  ┌─────────────────────────┐
#  │       NVIDIA NIM        │
#  │                         │
#  │ Nemotron 3.5 Lightning  │
#  └────────────┬────────────┘
#               │
#        Raw response
#               │
#               ▼
#  ┌─────────────────────────┐
#  │    NvidiaProvider       │
#  │                         │
#  │ Parse + normalize       │
#  └────────────┬────────────┘
#               │
#               ▼
#     CompletionResult
#               │
#               ▼
#  ┌─────────────────────────┐
#  │         ROUTER          │
#  └────────────┬────────────┘
#               │
#               ▼
#  ┌─────────────────────────┐
#  │      APPLICATION        │
#  └─────────────────────────┘
