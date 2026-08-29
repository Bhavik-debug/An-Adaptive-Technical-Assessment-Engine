"""The NVIDIA adapter: what it sends, and how it reads what comes back.

Nothing here touches the network.  The adapter's single SDK call is replaced by
a plain async function, so every assertion is about our request body and our
parsing - not about httpx, and not about NVIDIA's uptime.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import httpx
import openai
import pytest
from pydantic import BaseModel

from app.llm.errors import (
    ProviderAuthError,
    ProviderBadRequestError,
    ProviderRateLimitedError,
    ProviderResponseError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)
from app.llm.providers.nvidia import NvidiaProvider, StructuredMode
from app.llm.structured import schema_spec
from app.llm.types import (
    ChatMessage,
    CompletionRequest,
    ModelTier,
    ReasoningPolicy,
    Role,
)


class Answer(BaseModel):
    ok: bool
    echo: str


def a_request(
    *,
    temperature: float = 0.0,
    top_p: float = 1.0,
    reasoning: ReasoningPolicy | None = None,
    timeout_s: float = 10.0,
    max_output_tokens: int = 256,
) -> CompletionRequest:
    return CompletionRequest(
        messages=(
            ChatMessage(role=Role.SYSTEM, content="You are a health probe."),
            ChatMessage(role=Role.USER, content="echo abc123"),
        ),
        json_schema=schema_spec(Answer),
        tier=ModelTier.SMALL_FAST,
        temperature=temperature,
        top_p=top_p,
        max_output_tokens=max_output_tokens,
        timeout_s=timeout_s,
        reasoning=reasoning or ReasoningPolicy(),
    )


def a_response(
    content: str = '{"ok": true, "echo": "abc123"}',
    *,
    reasoning_content: str | None = None,
    finish_reason: str = "stop",
    prompt_tokens: int = 120,
    completion_tokens: int = 24,
    reasoning_tokens: int | None = None,
    model: str = "nvidia/nemotron-3.5-lightning-30b-a3b",
) -> SimpleNamespace:
    """A duck-typed stand-in for an OpenAI ``ChatCompletion``.

    The adapter reads the response with ``getattr``, so a namespace is a faithful
    substitute and keeps the test readable.
    """
    message = SimpleNamespace(content=content)
    if reasoning_content is not None:
        message.reasoning_content = reasoning_content
    details = (
        SimpleNamespace(reasoning_tokens=reasoning_tokens) if reasoning_tokens is not None else None
    )
    return SimpleNamespace(
        model=model,
        choices=[SimpleNamespace(message=message, finish_reason=finish_reason)],
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            completion_tokens_details=details,
        ),
    )


class Recorder:
    """Captures the request body and returns scripted responses."""

    def __init__(self, *steps: Any) -> None:
        self.steps = list(steps) or [a_response()]
        self.bodies: list[dict[str, Any]] = []

    async def __call__(self, **kwargs: Any) -> Any:
        self.bodies.append(kwargs)
        step = self.steps[min(len(self.bodies) - 1, len(self.steps) - 1)]
        if isinstance(step, BaseException):
            raise step
        return step

    @property
    def body(self) -> dict[str, Any]:
        return self.bodies[0]


def provider(recorder: Recorder, **kwargs: Any) -> NvidiaProvider:
    return NvidiaProvider(api_key="not-a-real-key", completion_creator=recorder, **kwargs)


# --- the request body ------------------------------------------------------


async def test_structured_calls_are_never_streamed():
    """The single most important deviation from NVIDIA's reference snippet."""
    rec = Recorder()
    await provider(rec).complete(a_request())
    assert rec.body["stream"] is False


async def test_sampling_settings_come_from_the_request_not_the_snippet():
    rec = Recorder()
    await provider(rec).complete(a_request(temperature=0.0, top_p=1.0))
    assert rec.body["temperature"] == 0.0
    assert rec.body["top_p"] == 1.0
    assert rec.body["max_tokens"] == 256


async def test_the_configured_model_is_sent():
    rec = Recorder()
    await provider(rec, model="nvidia/nemotron-3.5-lightning-30b-a3b").complete(a_request())
    assert rec.body["model"] == "nvidia/nemotron-3.5-lightning-30b-a3b"


async def test_tier_override_selects_a_different_model():
    rec = Recorder()
    p = provider(rec, model="small-one", tier_models={ModelTier.MID: "big-one"})
    assert p.model_for(ModelTier.SMALL_FAST) == "small-one"
    assert p.model_for(ModelTier.MID) == "big-one"


async def test_schema_is_sent_as_a_strict_response_format():
    rec = Recorder()
    await provider(rec).complete(a_request())
    fmt = rec.body["response_format"]
    assert fmt["type"] == "json_schema"
    assert fmt["json_schema"]["strict"] is True
    assert fmt["json_schema"]["schema"]["properties"].keys() >= {"ok", "echo"}


async def test_schema_is_also_stated_in_the_prompt():
    """Enforcement fixes the shape; the written schema is what fills the fields."""
    rec = Recorder()
    await provider(rec).complete(a_request())
    last_user = [m for m in rec.body["messages"] if m["role"] == "user"][-1]
    assert "echo abc123" in last_user["content"]
    assert "JSON Schema" in last_user["content"]


async def test_thinking_is_explicitly_disabled_by_default():
    rec = Recorder()
    await provider(rec).complete(a_request())
    extra = rec.body["extra_body"]
    assert extra["chat_template_kwargs"]["enable_thinking"] is False
    assert "reasoning_budget" not in extra


async def test_thinking_and_budget_are_sent_when_a_task_asks_for_them():
    rec = Recorder()
    await provider(rec).complete(
        a_request(reasoning=ReasoningPolicy(enabled=True, budget_tokens=4096))
    )
    extra = rec.body["extra_body"]
    assert extra["chat_template_kwargs"]["enable_thinking"] is True
    assert extra["reasoning_budget"] == 4096


# --- structured-output negotiation ----------------------------------------


async def test_endpoint_rejecting_json_schema_steps_down_to_json_object():
    rec = Recorder(_status_error(openai.BadRequestError, 400, "response_format is not supported"))
    rec.steps.append(a_response())
    p = provider(rec)

    result = await p.complete(a_request())

    assert [b.get("response_format", {}).get("type") for b in rec.bodies] == [
        "json_schema",
        "json_object",
    ]
    assert result.structured_mode == StructuredMode.JSON_OBJECT.value


async def test_the_demotion_is_remembered_for_later_calls():
    rec = Recorder(_status_error(openai.BadRequestError, 400, "json_schema unsupported"))
    rec.steps.append(a_response())
    p = provider(rec)

    await p.complete(a_request())
    await p.complete(a_request())

    # First call paid the rejection; the second starts on the working rung.
    assert rec.bodies[2]["response_format"]["type"] == "json_object"


async def test_a_400_that_is_not_about_the_parameter_is_not_swallowed():
    rec = Recorder(_status_error(openai.BadRequestError, 400, "maximum context length exceeded"))
    with pytest.raises(ProviderBadRequestError, match="context length"):
        await provider(rec).complete(a_request())
    assert len(rec.bodies) == 1


async def test_stepping_all_the_way_down_reaches_prompt_only():
    rec = Recorder(
        _status_error(openai.BadRequestError, 400, "response_format is not supported"),
        _status_error(openai.BadRequestError, 400, "response_format is not supported"),
        a_response(),
    )
    result = await provider(rec).complete(a_request())
    assert "response_format" not in rec.bodies[2]
    assert result.structured_mode == StructuredMode.PROMPT_ONLY.value


# --- reasoning vs answer ---------------------------------------------------


async def test_reasoning_in_a_dedicated_field_is_kept_out_of_the_answer():
    rec = Recorder(
        a_response(
            content='{"ok": true, "echo": "abc123"}',
            reasoning_content="Let me think. The token is abc123, so...",
            reasoning_tokens=310,
        )
    )
    result = await provider(rec).complete(a_request())

    assert result.text == '{"ok": true, "echo": "abc123"}'
    assert result.reasoning_text is not None
    assert "Let me think" in result.reasoning_text
    assert result.reasoning_tokens == 310


async def test_inline_think_tags_are_stripped_from_the_answer():
    rec = Recorder(a_response(content='<think>hmm, abc123</think>{"ok": true, "echo": "abc123"}'))
    result = await provider(rec).complete(a_request())

    assert result.text == '{"ok": true, "echo": "abc123"}'
    assert result.reasoning_text == "hmm, abc123"
    assert "<think>" not in result.text


async def test_a_prefilled_opening_think_tag_is_handled():
    """Some chat templates open the tag for the model, so only </think> appears."""
    rec = Recorder(a_response(content='reasoning here</think>{"ok": true, "echo": "abc123"}'))
    result = await provider(rec).complete(a_request())

    assert result.text == '{"ok": true, "echo": "abc123"}'
    assert result.reasoning_text == "reasoning here"


async def test_reasoning_that_never_finishes_is_an_error_not_an_answer():
    """The bug this whole split exists to prevent."""
    rec = Recorder(
        a_response(content="<think>still thinking and the budget ran", finish_reason="length")
    )
    with pytest.raises(ProviderResponseError) as exc:
        await provider(rec).complete(a_request())
    assert "no answer content" in str(exc.value)
    assert "reasoning" in str(exc.value)


# --- response parsing ------------------------------------------------------


async def test_token_counts_and_finish_reason_are_carried_through():
    rec = Recorder(a_response(prompt_tokens=812, completion_tokens=97))
    result = await provider(rec).complete(a_request())

    assert (result.input_tokens, result.output_tokens) == (812, 97)
    assert result.finish_reason == "stop"
    assert result.truncated is False
    assert result.provider == "nvidia"


async def test_a_length_finish_reason_marks_the_result_truncated():
    rec = Recorder(a_response(content='{"ok": true, "echo": "abc', finish_reason="length"))
    result = await provider(rec).complete(a_request())
    assert result.truncated is True


async def test_missing_usage_block_does_not_crash_accounting():
    raw = SimpleNamespace(
        model="m",
        choices=[SimpleNamespace(message=SimpleNamespace(content="{}"), finish_reason="stop")],
    )
    result = await provider(Recorder(raw)).complete(a_request())
    assert (result.input_tokens, result.output_tokens, result.reasoning_tokens) == (0, 0, 0)


@pytest.mark.parametrize(
    "raw",
    [
        SimpleNamespace(model="m", choices=[], usage=None),
        SimpleNamespace(model="m", choices=None, usage=None),
        SimpleNamespace(
            model="m", choices=[SimpleNamespace(message=None, finish_reason="stop")], usage=None
        ),
        SimpleNamespace(
            model="m",
            choices=[SimpleNamespace(message=SimpleNamespace(content=""), finish_reason="stop")],
            usage=None,
        ),
    ],
)
async def test_a_malformed_response_is_a_provider_error(raw):
    with pytest.raises(ProviderResponseError):
        await provider(Recorder(raw)).complete(a_request())


# --- error mapping ---------------------------------------------------------


def _request() -> httpx.Request:
    return httpx.Request("POST", "https://integrate.api.nvidia.com/v1/chat/completions")


def _status_error(
    cls: type[openai.APIStatusError],
    status: int,
    message: str,
    headers: dict[str, str] | None = None,
) -> openai.APIStatusError:
    response = httpx.Response(status, headers=headers or {}, request=_request())
    return cls(message, response=response, body=None)


@pytest.mark.parametrize(
    ("sdk_error", "expected", "retryable"),
    [
        (openai.APITimeoutError(request=_request()), ProviderTimeoutError, True),
        (
            openai.APIConnectionError(message="dns failure", request=_request()),
            ProviderUnavailableError,
            True,
        ),
    ],
)
async def test_transport_errors_are_mapped(sdk_error, expected, retryable):
    with pytest.raises(expected) as exc:
        await provider(Recorder(sdk_error)).complete(a_request())
    assert exc.value.retryable is retryable
    assert exc.value.provider == "nvidia"


@pytest.mark.parametrize(
    ("cls", "status", "expected", "retryable"),
    [
        (openai.AuthenticationError, 401, ProviderAuthError, False),
        (openai.PermissionDeniedError, 403, ProviderAuthError, False),
        (openai.RateLimitError, 429, ProviderRateLimitedError, True),
        (openai.InternalServerError, 500, ProviderUnavailableError, True),
        (openai.InternalServerError, 503, ProviderUnavailableError, True),
        (openai.NotFoundError, 404, ProviderBadRequestError, False),
    ],
)
async def test_status_codes_are_mapped(cls, status, expected, retryable):
    rec = Recorder(_status_error(cls, status, "upstream said no"))
    with pytest.raises(expected) as exc:
        await provider(rec).complete(a_request())
    assert exc.value.retryable is retryable
    assert exc.value.status_code == status


async def test_an_auth_failure_never_echoes_the_response_body():
    """A 401 body is the one place a provider might quote the key back."""
    rec = Recorder(_status_error(openai.AuthenticationError, 401, "invalid key nvapi-abcdef"))
    with pytest.raises(ProviderAuthError) as exc:
        await provider(rec).complete(a_request())
    assert "nvapi-abcdef" not in str(exc.value)


async def test_retry_after_header_is_extracted():
    rec = Recorder(
        _status_error(openai.RateLimitError, 429, "slow down", headers={"retry-after": "7"})
    )
    with pytest.raises(ProviderRateLimitedError) as exc:
        await provider(rec).complete(a_request())
    assert exc.value.retry_after_s == 7.0


async def test_a_non_numeric_retry_after_is_ignored_rather_than_crashing():
    rec = Recorder(
        _status_error(
            openai.RateLimitError,
            429,
            "slow down",
            headers={"retry-after": "Wed, 21 Oct 2026 07:28:00 GMT"},
        )
    )
    with pytest.raises(ProviderRateLimitedError) as exc:
        await provider(rec).complete(a_request())
    assert exc.value.retry_after_s is None


# --- deadline --------------------------------------------------------------


async def test_a_hanging_provider_is_cut_off_at_the_deadline():
    """The SDK timeout is not trusted on its own; the adapter enforces its own."""

    async def never_returns(**_kwargs: Any) -> Any:
        await asyncio.sleep(30)

    p = NvidiaProvider(api_key="not-a-real-key", completion_creator=never_returns)
    with pytest.raises(ProviderTimeoutError) as exc:
        await p.complete(a_request(timeout_s=0.05))
    assert exc.value.retryable is True
    # A sub-second deadline reported as "within 0s" is a log line that costs
    # someone twenty minutes of debugging the wrong thing.
    assert "0.05s" in str(exc.value)


async def test_a_whole_second_deadline_reads_naturally():
    async def never_returns(**_kwargs: Any) -> Any:
        await asyncio.sleep(30)

    p = NvidiaProvider(api_key="not-a-real-key", completion_creator=never_returns)
    with pytest.raises(ProviderTimeoutError, match="within 1s"):
        await p.complete(a_request(timeout_s=1.0))


async def test_the_deadline_is_also_passed_to_the_sdk():
    rec = Recorder()
    await provider(rec).complete(a_request(timeout_s=12.5))
    assert rec.body["timeout"] == 12.5
