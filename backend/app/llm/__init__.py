"""The LLM chokepoint - plan sections 13.2 and 13.3.

The rule this package exists to enforce: **no module outside ``app/llm/`` ever
calls a model API.**  Business logic imports ``call_structured`` and a pydantic
schema, and gets back a validated object.  It never sees a provider, an HTTP
status, a token count, or a retry.

    from app.llm import TaskName, call_structured

    answer, meta = await call_structured(TaskName.GRADE_ANSWER, inputs, Grade)

Module map:

* ``types.py``      the provider-agnostic vocabulary and the provider interface
* ``tasks.py``      the routing table: task -> tier, temperature, limits
* ``prompts.py``    versioned prompt templates
* ``structured.py`` schema derivation, JSON extraction, validation, repair
* ``providers/``    one adapter per vendor - the only vendor-aware code
* ``router.py``     ordering, retry, failover, circuit breaker
* ``cache.py``      response cache for deterministic tasks
* ``pricing.py``    token prices and per-call cost
* ``client.py``     ``call_structured()`` and ``CallMeta``
* ``runtime.py``    process lifecycle, wired into the FastAPI lifespan
"""

from app.llm.client import CallMeta, call_structured
from app.llm.errors import (
    AllProvidersFailedError,
    LLMConfigError,
    LLMError,
    PromptNotRegisteredError,
    ProviderAuthError,
    ProviderError,
    ProviderRateLimitedError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    SchemaValidationFailedError,
)
from app.llm.runtime import dispose_llm, get_router, init_llm, llm_provider_health
from app.llm.tasks import TaskName
from app.llm.types import ModelTier, TraceCtx

__all__ = [
    "AllProvidersFailedError",
    "CallMeta",
    "LLMConfigError",
    "LLMError",
    "ModelTier",
    "PromptNotRegisteredError",
    "ProviderAuthError",
    "ProviderError",
    "ProviderRateLimitedError",
    "ProviderTimeoutError",
    "ProviderUnavailableError",
    "SchemaValidationFailedError",
    "TaskName",
    "TraceCtx",
    "call_structured",
    "dispose_llm",
    "get_router",
    "init_llm",
    "llm_provider_health",
]
