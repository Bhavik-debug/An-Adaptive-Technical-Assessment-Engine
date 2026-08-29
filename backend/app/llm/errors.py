"""The error taxonomy for the LLM chokepoint.

Every failure a provider can produce is mapped onto one of these before it
leaves the provider adapter.  The rest of the system therefore never sees an
``openai.RateLimitError`` or an ``httpx.ConnectError`` - it sees a
``ProviderRateLimitedError``, and the router can make its failover decision from
a single attribute (``retryable``) instead of from a growing ``isinstance``
ladder that has to learn every SDK's exception hierarchy.

The split that matters:

* **retryable**   - the request might succeed if we try again, either on the same
  provider after a pause (429, 503, timeout) or on the next provider.
* **not retryable** - trying again changes nothing, because the fault is in the
  request or the credentials (401, 400).  We still move to the next provider,
  since a bad key for provider A says nothing about provider B, but we do not
  waste attempts hammering the one that just refused us.
"""

from __future__ import annotations


class LLMError(Exception):
    """Base class for everything this package raises."""


class LLMConfigError(LLMError):
    """No usable provider could be built from the current configuration."""


class PromptNotRegisteredError(LLMError):
    """A task was called that has no prompt template yet.

    Tasks are declared in the routing table as soon as the plan names them;
    their prompts land in the phase that owns them.  Calling one early is a
    programming error, and this says so rather than sending an empty prompt.
    """


class ProviderError(LLMError):
    """A single provider failed a single attempt."""

    #: Whether re-sending the identical request could plausibly succeed.
    retryable: bool = False

    def __init__(
        self,
        message: str,
        *,
        provider: str,
        status_code: int | None = None,
        retry_after_s: float | None = None,
    ) -> None:
        super().__init__(f"[{provider}] {message}")
        self.provider = provider
        self.status_code = status_code
        #: Honour the server's own advice when it sends ``Retry-After``.
        self.retry_after_s = retry_after_s


class ProviderTimeoutError(ProviderError):
    """The provider did not answer inside the deadline."""

    retryable = True


class ProviderRateLimitedError(ProviderError):
    """HTTP 429. The reason the plan calls for a multi-provider router at all."""

    retryable = True


class ProviderUnavailableError(ProviderError):
    """A 5xx, a dropped connection, or a DNS failure - the provider's fault."""

    retryable = True


class ProviderAuthError(ProviderError):
    """HTTP 401/403. A missing or rejected API key. Retrying cannot fix it."""

    retryable = False


class ProviderBadRequestError(ProviderError):
    """HTTP 400/404/422. We sent something this provider will not accept."""

    retryable = False


class ProviderResponseError(ProviderError):
    """The call returned 200 but the body was not a usable completion.

    Empty ``choices``, a null message, a content-filter stop - anything where
    the transport succeeded and the payload did not.
    """

    retryable = False


class AllProvidersFailedError(LLMError):
    """Every provider in the routing order was tried and none produced a completion."""

    def __init__(self, failures: dict[str, str]) -> None:
        detail = "; ".join(f"{name}: {reason}" for name, reason in failures.items())
        super().__init__(f"all LLM providers failed ({detail or 'none configured'})")
        self.failures = failures


class SchemaValidationFailedError(LLMError):
    """The model never produced output matching the requested schema.

    Carries the attempt count and the last validation error so a trace can show
    *what* the model got wrong, not merely that it got something wrong.
    """

    def __init__(self, *, attempts: int, last_error: str, raw_output: str) -> None:
        super().__init__(
            f"model output failed schema validation after {attempts} attempt(s): {last_error}"
        )
        self.attempts = attempts
        self.last_error = last_error
        #: Truncated in the router's log; kept in full here for tests and traces.
        self.raw_output = raw_output
