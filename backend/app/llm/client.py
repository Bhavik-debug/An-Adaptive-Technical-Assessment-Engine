"""``call_structured()`` - the single door every LLM call walks through.

Plan section 13.3 calls this "the highest-leverage 150 lines in the project",
and the reason is that all of the following are implemented once, here, instead
of being re-implemented badly at each of the thirty call sites a system like
this grows:

* prompt versioning        - which wording produced this answer
* model routing            - task chooses tier, provider chooses model
* schema validation        - a pydantic object or an exception, never a string
* retry on invalid output  - with the model shown its own mistake
* response caching         - deterministic tasks are computed once
* token and cost accounting - summed across every attempt, not just the last
* provider failover        - handled by the router underneath
* trace metadata           - the attribute set Day 4 exports as spans

The contract for every caller in this project, forever:

    answer, meta = await call_structured(TaskName.X, inputs, MySchema)

``answer`` is an instance of ``MySchema``.  Not a string, not a dict, not
"probably JSON".  If the model could not produce one, this raises.
"""

from __future__ import annotations

import logging
import time
from decimal import Decimal
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from app.config import Settings, get_settings
from app.llm.cache import CachedCall, NullCache, ResponseCache, cache_key
from app.llm.errors import SchemaValidationFailedError
from app.llm.pricing import price_call
from app.llm.prompts import get_prompt
from app.llm.router import ProviderRouter
from app.llm.structured import (
    JsonExtractionError,
    extract_json,
    format_validation_error,
    repair_messages,
    schema_spec,
)
from app.llm.tasks import TaskName, get_task_spec
from app.llm.types import (
    ANONYMOUS_TRACE,
    ChatMessage,
    CompletionRequest,
    ModelTier,
    ReasoningPolicy,
    TraceCtx,
)
from app.obs import llm_call_span, record_error, record_llm_meta

log = logging.getLogger(__name__)

SchemaT = TypeVar("SchemaT", bound=BaseModel)


class CallMeta(BaseModel):
    """Everything about a call except the answer itself.

    This is exactly the attribute set plan section 14.2 calls non-negotiable on
    an LLM span.  Day 4 exports it to OpenTelemetry; Day 3 emits it as a log
    line.  Producing it now means Day 4 changes the *exporter*, not every call
    site - which is the point of building the chokepoint first.

    Note what is *not* here: the prompt, the answer, and the model's reasoning.
    Metadata is safe to log at volume; content is not.
    """

    model_config = {"frozen": True}

    task: TaskName
    prompt_version: str
    prompt_fingerprint: str
    schema_fingerprint: str
    provider: str
    model: str
    tier: ModelTier
    temperature: float
    #: Summed across every attempt, including schema repairs. A retry that is
    #: not billed in the trace is a retry that looks free in the cost report.
    input_tokens: int
    output_tokens: int
    reasoning_tokens: int
    cost_usd: Decimal
    #: False when the model is absent from the price table: the cost is 0.0
    #: because it is unknown, not because it is free.
    price_known: bool
    cache_hit: bool
    schema_retry_count: int
    failover_count: int
    structured_mode: str
    reasoning_enabled: bool
    latency_ms: int
    session_id: str | None = None
    turn_id: str | None = None
    user_id: str | None = None
    plan: str | None = None

    def as_span_attributes(self) -> dict[str, Any]:
        """Flat, primitive-valued dict - the shape a tracer wants."""
        data = self.model_dump(mode="json")
        return {f"llm.{key}": value for key, value in data.items() if value is not None}


async def call_structured(
    task: TaskName,
    inputs: dict[str, Any],
    schema: type[SchemaT],
    *,
    temperature: float | None = None,
    trace: TraceCtx = ANONYMOUS_TRACE,
    router: ProviderRouter | None = None,
    cache: ResponseCache | None = None,
    settings: Settings | None = None,
) -> tuple[SchemaT, CallMeta]:
    """Run one task through the chokepoint and return a validated object.

    ``temperature`` differs from the sketch in plan section 13.3, which gives it
    a default of 0.0.  Here ``None`` means "use the task's configured
    temperature" (routing table, section 13.6), so the sampling decision lives
    in one table instead of at every call site; passing a float still overrides
    it for a deliberate one-off.

    Raises ``AllProvidersFailedError`` if no provider answered, or
    ``SchemaValidationFailedError`` if none of the answers validated.

    Day 4 wraps the whole thing - cache lookup included - in one span.  A thin
    wrapper rather than instrumentation threaded through the body below,
    because the accounting and the reporting of that accounting are different
    jobs: this function decides what a call *cost*, ``app/obs/spans.py`` decides
    how that gets written down, and neither can quietly change the other.
    """
    with llm_call_span(task.value) as span:
        try:
            answer, meta = await _run_call(
                task,
                inputs,
                schema,
                temperature=temperature,
                trace=trace,
                router=router,
                cache=cache,
                settings=settings,
            )
        except Exception as exc:
            # Every failure path - no provider answered, nothing validated, a
            # bug in a prompt template - marks the span red before it leaves.
            record_error(span, exc)
            raise
        record_llm_meta(span, meta)
        return answer, meta


async def _run_call(
    task: TaskName,
    inputs: dict[str, Any],
    schema: type[SchemaT],
    *,
    temperature: float | None,
    trace: TraceCtx,
    router: ProviderRouter | None,
    cache: ResponseCache | None,
    settings: Settings | None,
) -> tuple[SchemaT, CallMeta]:
    """The chokepoint itself. See ``call_structured`` for the contract."""
    settings = settings or get_settings()
    if router is None or cache is None:
        # Local import: runtime imports the router, the router does not import
        # this module, and keeping the import here keeps that true.
        from app.llm.runtime import get_response_cache, get_router

        router = get_router() if router is None else router
        cache = get_response_cache() if cache is None else cache

    spec = get_task_spec(task)
    prompt = get_prompt(task)
    schema_json = schema_spec(schema)
    effective_temperature = spec.temperature if temperature is None else temperature

    # Reasoning is opt-in twice over: the task must ask for it and the global
    # switch must allow it. See the Day 3 report for why today's tasks do not.
    reasoning = ReasoningPolicy(
        enabled=spec.reasoning and settings.llm_reasoning_enabled,
        budget_tokens=spec.reasoning_budget_tokens or settings.llm_reasoning_budget_tokens,
    )

    base_messages = prompt.render(inputs)
    # The model id is part of the cache key, and only a provider knows its own
    # model ids - so ask the first provider in the routing order.
    keyed_model = router.providers[0].model_for(spec.tier)
    key = cache_key(
        task=task.value,
        prompt_version=prompt.version,
        prompt_fingerprint=prompt.fingerprint,
        schema_fingerprint=schema_json.fingerprint,
        model=keyed_model,
        temperature=effective_temperature,
        top_p=spec.top_p,
        inputs=inputs,
    )

    use_cache = settings.llm_cache_enabled and spec.is_cacheable
    active_cache: ResponseCache = cache if cache is not None else NullCache()
    started = time.perf_counter()

    if use_cache:
        hit = await active_cache.get(key)
        if hit is not None:
            validated = _validate_cached(hit, schema)
            if validated is not None:
                cached_meta = _meta(
                    task=task,
                    prompt_version=prompt.version,
                    prompt_fingerprint=prompt.fingerprint,
                    schema_fingerprint=schema_json.fingerprint,
                    provider=hit.provider,
                    model=hit.model,
                    tier=spec.tier,
                    temperature=effective_temperature,
                    # A hit costs no tokens. Reporting the tokens it *would*
                    # have cost would double-count them in the session total.
                    input_tokens=0,
                    output_tokens=0,
                    reasoning_tokens=0,
                    cost_usd=Decimal("0"),
                    price_known=True,
                    cache_hit=True,
                    schema_retry_count=0,
                    failover_count=0,
                    structured_mode="cache",
                    reasoning_enabled=reasoning.enabled,
                    started=started,
                    trace=trace,
                )
                # A cache hit is a call that happened, and it is the cheapest
                # one there is. Leaving it unlogged would make the cache hit
                # rate - an operational metric in plan section 12.4 -
                # unmeasurable, and would make a served-from-cache answer
                # invisible when tracing why a session behaved oddly.
                _log_call(cached_meta)
                return validated, cached_meta

    messages: tuple[ChatMessage, ...] = base_messages
    total_input = total_output = total_reasoning = 0
    failover_count = 0
    provider_name = model_name = structured_mode = "unknown"
    last_error = "no attempt was made"
    raw_output = ""

    # attempt 0 is the real call; the rest are schema repairs.
    for schema_attempt in range(settings.llm_schema_max_retries + 1):
        outcome = await router.complete(
            CompletionRequest(
                messages=messages,
                json_schema=schema_json,
                tier=spec.tier,
                temperature=effective_temperature,
                top_p=spec.top_p,
                max_output_tokens=spec.max_output_tokens,
                timeout_s=settings.llm_timeout_s,
                reasoning=reasoning,
            )
        )
        result = outcome.result
        failover_count += outcome.failover_count
        total_input += result.input_tokens
        total_output += result.output_tokens
        total_reasoning += result.reasoning_tokens
        provider_name = result.provider
        model_name = result.model
        structured_mode = result.structured_mode
        raw_output = result.text

        # The model's chain of thought stops here. It is used to explain an
        # empty answer (in the provider) and to count tokens, and it is never
        # attached to the returned object, the metadata, or the log.
        try:
            payload = extract_json(result.text)
            answer = schema.model_validate(payload)
        except JsonExtractionError as exc:
            last_error = _truncation_hint(result.truncated) or str(exc)
        except ValidationError as exc:
            last_error = format_validation_error(exc)
        else:
            cost = price_call(
                model=model_name, input_tokens=total_input, output_tokens=total_output
            )
            if use_cache:
                await active_cache.set(
                    key,
                    CachedCall(
                        payload=answer.model_dump(mode="json"),
                        provider=provider_name,
                        model=model_name,
                    ),
                )
            meta = _meta(
                task=task,
                prompt_version=prompt.version,
                prompt_fingerprint=prompt.fingerprint,
                schema_fingerprint=schema_json.fingerprint,
                provider=provider_name,
                model=model_name,
                tier=spec.tier,
                temperature=effective_temperature,
                input_tokens=total_input,
                output_tokens=total_output,
                reasoning_tokens=total_reasoning,
                cost_usd=cost.usd,
                price_known=cost.price_known,
                cache_hit=False,
                schema_retry_count=schema_attempt,
                failover_count=failover_count,
                structured_mode=structured_mode,
                reasoning_enabled=reasoning.enabled,
                started=started,
                trace=trace,
            )
            _log_call(meta)
            return answer, meta

        if schema_attempt < settings.llm_schema_max_retries:
            log.info(
                "llm task %s returned invalid output (%s); repair attempt %d",
                task.value,
                last_error,
                schema_attempt + 1,
            )
            messages = base_messages + repair_messages(
                raw_output=result.text, problem=last_error, spec=schema_json
            )

    raise SchemaValidationFailedError(
        attempts=settings.llm_schema_max_retries + 1,
        last_error=last_error,
        raw_output=raw_output,
    )


def _truncation_hint(truncated: bool) -> str | None:
    if not truncated:
        return None
    return (
        "the response was cut off by the output-token limit, so it could not be "
        "valid JSON - raise max_output_tokens for this task"
    )


def _validate_cached(hit: CachedCall, schema: type[SchemaT]) -> SchemaT | None:
    """A cached entry still has to validate.

    The schema fingerprint is in the key, so this should never fail - which is
    exactly why it is checked. A stale entry that quietly does not match the
    current model is worse than a cache miss.
    """
    try:
        return schema.model_validate(hit.payload)
    except ValidationError as exc:
        log.warning("cached llm response no longer validates, ignoring it: %s", exc)
        return None


def _meta(
    *,
    task: TaskName,
    prompt_version: str,
    prompt_fingerprint: str,
    schema_fingerprint: str,
    provider: str,
    model: str,
    tier: ModelTier,
    temperature: float,
    input_tokens: int,
    output_tokens: int,
    reasoning_tokens: int,
    cost_usd: Decimal,
    price_known: bool,
    cache_hit: bool,
    schema_retry_count: int,
    failover_count: int,
    structured_mode: str,
    reasoning_enabled: bool,
    started: float,
    trace: TraceCtx,
) -> CallMeta:
    return CallMeta(
        task=task,
        prompt_version=prompt_version,
        prompt_fingerprint=prompt_fingerprint,
        schema_fingerprint=schema_fingerprint,
        provider=provider,
        model=model,
        tier=tier,
        temperature=temperature,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        reasoning_tokens=reasoning_tokens,
        cost_usd=cost_usd,
        price_known=price_known,
        cache_hit=cache_hit,
        schema_retry_count=schema_retry_count,
        failover_count=failover_count,
        structured_mode=structured_mode,
        reasoning_enabled=reasoning_enabled,
        latency_ms=int((time.perf_counter() - started) * 1000),
        session_id=str(trace.session_id) if trace.session_id else None,
        turn_id=str(trace.turn_id) if trace.turn_id else None,
        user_id=str(trace.user_id) if trace.user_id else None,
        plan=trace.plan,
    )


def _log_call(meta: CallMeta) -> None:
    """One structured line per call, whether or not anything is exporting spans.

    Day 3 wrote this as ``json.dumps`` inside the message, which was the plan's
    Phase 1 cut-line.  Day 4 passes the same attributes as ``extra=`` instead,
    so the structured formatter emits them as real top-level fields alongside
    ``trace_id`` and ``request_id`` - queryable rather than a string that
    happens to contain JSON.  The attribute *names* are unchanged, which was the
    condition the cut-line attached to swapping the exporter.

    Metadata only - no prompt, no answer, no reasoning, because those carry
    candidate data and log volume both.
    """
    log.info("llm_call", extra=meta.as_span_attributes())
