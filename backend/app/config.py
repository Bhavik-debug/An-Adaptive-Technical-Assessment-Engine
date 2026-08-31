"""Application configuration.

One rule: the process must not start in a half-configured state.  A missing or
malformed environment variable is a boot-time failure with a readable message,
never a 500 three hours into a running interview.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, ValidationError, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ConfigError(RuntimeError):
    """Raised at boot when the environment is missing or invalid."""


#: Provider adapters this build knows how to construct.  A name in
#: ``LLM_PROVIDER_ORDER`` that is not here is a typo, and a typo in the routing
#: order should stop the process rather than silently shorten the failover list.
#:
#: ``stub`` is the Day-5 offline provider: it answers from recorded fixtures
#: instead of the network, needs no credential, and is what makes the test suite
#: independent of any vendor's uptime.  It is a real adapter behind the real
#: router, not a test-only code path.
KNOWN_LLM_PROVIDERS = ("nvidia", "stub")

_ENV_CONFIG = SettingsConfigDict(
    env_file=".env",
    env_file_encoding="utf-8",
    extra="ignore",
    case_sensitive=False,
)


class LoggingSettings(BaseSettings):
    """How this process writes things down.

    The smallest settings object in the system, and deliberately the only one
    with **no required fields**.  Logging has to be configurable before anything
    else is validated: if the process is about to die because ``DATABASE_URL``
    is missing, the message saying so still has to come out in the right format.
    A logging configuration that could itself fail to load would be the one
    failure nobody could debug.

    Every process inherits this - the API, the Alembic migration runner, the
    Phase 6 restore drill - so they all emit the same structured, redacted lines.
    """

    model_config = _ENV_CONFIG

    app_name: str = "adaptive-ai-interviewer"
    app_env: Literal["local", "ci", "staging", "prod"] = "local"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    # JSON by default: a log line is data, and a machine has to be able to read
    # it. `text` exists for tailing a terminal during development and changes
    # nothing about what is redacted.
    log_format: Literal["json", "text"] = "json"


class DatabaseSettings(LoggingSettings):
    """Just enough configuration to reach Postgres.

    Exists so that a process which only touches the database - the Alembic
    migration runner, and the Phase 6 restore drill - validates exactly what it
    uses.  Requiring an LLM credential to run a schema migration would be real
    coupling with a real cost: it makes a database restore depend on a provider
    account.

    ``Settings`` inherits from this rather than redeclaring the field, so there
    is still exactly one definition of what a valid ``DATABASE_URL`` looks like.
    """

    model_config = _ENV_CONFIG

    database_url: str

    @field_validator("database_url")
    @classmethod
    def _check_database_url(cls, v: str) -> str:
        if not v.startswith("postgresql+asyncpg://"):
            raise ValueError(
                "must be an async Postgres DSN, e.g. postgresql+asyncpg://user:pass@host:5432/db"
            )
        return v


class Settings(DatabaseSettings):
    model_config = _ENV_CONFIG

    # --- Defaults are fine for these ---------------------------------------
    # (``app_name``, ``app_env``, ``log_level`` and ``log_format`` come from
    # LoggingSettings - every process needs them, including the migrator.)
    app_version: str = "0.1.0"
    debug: bool = False
    api_port: int = 8000

    # How long a readiness probe waits on a dependency before calling it down.
    readiness_timeout_s: float = Field(default=2.0, gt=0, le=30)

    # --- Authentication -----------------------------------------------------
    # Short, because an access token cannot be revoked once issued - the only
    # limit on a stolen one is how soon it expires.
    access_token_ttl_minutes: int = Field(default=15, gt=0, le=1440)
    # Long, because it is single-use: every refresh rotates it (see token_store).
    refresh_token_ttl_days: int = Field(default=30, gt=0, le=365)
    # Plan section 14.1: lockout after 10 failures.
    login_max_failures: int = Field(default=10, gt=0, le=100)
    login_lockout_minutes: int = Field(default=15, gt=0, le=1440)

    # --- LLM chokepoint (plan section 13.3) --------------------------------
    # Comma-separated, highest priority first. A plain string rather than a
    # list because pydantic-settings parses list-typed env vars as JSON, and
    # `LLM_PROVIDER_ORDER=nvidia,groq` is what a person actually writes.
    llm_provider_order: str = "nvidia"
    # Wall-clock ceiling for one provider call. Generous enough for a long
    # report, short enough that a hung provider cannot hold a turn open.
    llm_timeout_s: float = Field(default=60.0, gt=0, le=600)
    # Attempts against one provider before moving to the next one.
    llm_max_attempts_per_provider: int = Field(default=2, ge=1, le=5)
    # Re-prompts allowed when output does not match the schema. 0 disables repair.
    llm_schema_max_retries: int = Field(default=2, ge=0, le=5)
    llm_cache_enabled: bool = True
    llm_cache_ttl_s: int = Field(default=604_800, ge=0)  # 7 days
    # Consecutive failures that take a provider out of rotation, and for how long.
    llm_breaker_failure_threshold: int = Field(default=3, ge=1, le=20)
    llm_breaker_cooldown_s: float = Field(default=30.0, gt=0, le=3600)
    # Global kill switch for model "thinking". A task must ask for reasoning
    # AND this must be true; turning it off is how a cost spike gets contained
    # without a deploy.
    llm_reasoning_enabled: bool = True
    llm_reasoning_budget_tokens: int = Field(default=4096, ge=256, le=65_536)

    # --- NVIDIA provider ----------------------------------------------------
    # The key is a SecretStr: pydantic renders it as `**********` in every repr,
    # log line and exception, so an accidental `print(settings)` cannot leak it.
    nvidia_api_key: SecretStr | None = None
    # Defaults mirror app/llm/providers/nvidia.py; test_llm_config keeps them in
    # step. The literals live here because config must not import the llm
    # package - that would be an import cycle through app/llm/__init__.py.
    nvidia_base_url: str = "https://integrate.api.nvidia.com/v1"
    nvidia_model: str = "nvidia/nemotron-3.5-lightning-30b-a3b"

    # --- Offline stub provider (Day 5) --------------------------------------
    # Selected with LLM_PROVIDER_ORDER=stub. Needs no API key by design.
    #: Where recordings live. Empty means the package default,
    #: ``backend/fixtures/llm`` - resolved from the package so it also works
    #: inside the container image.
    llm_stub_fixture_dir: str | None = None
    #: What to do when no recording matches the request.
    #:   strict     - raise, naming the key. The right default: a suite that
    #:                meant to replay must not silently run on invented data.
    #:   synthesize - derive a shape-correct object from the schema. For
    #:                developing a phase before any recording of it exists.
    #: Either way the answer records which happened, in CallMeta.structured_mode.
    llm_stub_on_missing: Literal["strict", "synthesize"] = "strict"

    # --- Embeddings and retrieval (Day 8) -----------------------------------
    # Embeddings are NOT an LLM concern and deliberately do not go through the
    # `llm/` chokepoint: that layer exists to control cost, routing and failover
    # for a metered remote API. This model runs locally on CPU, costs nothing
    # per call, and has no provider to fail over to. Sharing the abstraction
    # would buy nothing and would make an offline component depend on an
    # online one.
    #:   fastembed - the real model. Needs the optional [embeddings] extra.
    #:   hashing   - a deterministic in-repo stand-in. NOT semantic; it exists
    #:               so the test suite and CI never download a model.
    embedding_backend: Literal["fastembed", "hashing"] = "fastembed"
    #: Changing this invalidates every stored vector: `embedding_model` is
    #: recorded per row, and the embed step re-embeds anything that disagrees.
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    #: Where the downloaded weights live. Empty means `<repo>/.model-cache`,
    #: which is gitignored. fastembed's own default is a temp directory, and a
    #: cache the operating system may delete is a 67 MB download every reboot.
    embedding_cache_dir: str | None = None
    embedding_batch_size: int = Field(default=32, ge=1, le=512)

    # Retrieval K values (plan section 3, Day 8). Two different things:
    # *candidate* K is how many each retriever proposes, *final* K is how many
    # survive fusion. Candidates are deliberately larger - a document ranked
    # 25th by vectors and 3rd lexically should get the chance to be fused.
    retrieval_vector_k: int = Field(default=30, ge=1, le=500)
    retrieval_lexical_k: int = Field(default=30, ge=1, le=500)
    retrieval_final_k: int = Field(default=10, ge=1, le=100)
    #: RRF's damping constant. 60 is the value from Cormack et al. (2009), the
    #: paper the method comes from. See app/retrieval/rrf.py.
    retrieval_rrf_k: float = Field(default=60.0, gt=0, le=1000)

    # --- Cross-encoder reranking (Day 9) ------------------------------------
    # Stage 2. The bi-encoder above narrows the bank to a candidate set; this
    # model reads the query and each candidate TOGETHER and scores the pair.
    # It cannot be an index - one forward pass per candidate - which is exactly
    # why it runs over ~40 candidates and not over the whole bank.
    #: Off means the hybrid (RRF) order is served unchanged, and the result says
    #: so. Reranking is an improvement to the ordering, never a dependency of it.
    rerank_enabled: bool = True
    #:   fastembed - the real cross-encoder. Needs the [embeddings] extra.
    #:   overlap   - a deterministic in-repo stand-in scoring word overlap.
    #:               NOT relevance judgement; it exists so the test suite and CI
    #:               never download a model.
    rerank_backend: Literal["fastembed", "overlap"] = "fastembed"
    rerank_model: str = "BAAI/bge-reranker-base"
    #: Empty means the same `<repo>/.model-cache` the embedder uses - one
    #: gitignored directory for every model this project downloads.
    rerank_cache_dir: str | None = None
    rerank_batch_size: int = Field(default=16, ge=1, le=128)

    # Plan section 5.3: "the bi-encoder to go from 150 to 40 and the
    # cross-encoder to go from 40 to 8. Recall first, precision second."
    #: How many hybrid candidates are scored. This is what `hybrid_search` is
    #: asked for when reranking is on - deliberately far larger than the final
    #: K, because the reranker can only promote what stage 1 handed it.
    rerank_candidate_k: int = Field(default=40, ge=1, le=200)
    #: How many survive reranking. The number a caller actually wants.
    rerank_final_k: int = Field(default=8, ge=1, le=100)

    # --- Observability (plan section 14.2, Day 4) ---------------------------
    # Spans are created whenever this is on, even with no exporter: that is
    # what puts a `trace_id` on every log line, which is most of the value.
    otel_enabled: bool = True
    #: Defaults to ``app_name``. Overridable so two processes from the same
    #: image (api, worker) are distinguishable in one trace backend.
    otel_service_name: str | None = None
    #: Where spans go. ``none`` is the plan's Phase 1 cut-line and the default:
    #: local development must not require an observability stack to be running.
    otel_exporter: Literal["none", "console", "otlp", "langfuse"] = "none"
    #: Full URL of the traces endpoint, e.g. http://localhost:4318/v1/traces.
    #: Deliberately not the OTel base-URL convention - one variable that means
    #: exactly one thing is worth more than compatibility with a spec detail.
    otel_exporter_endpoint: str | None = None
    #: `key=value,key=value`. A SecretStr because this is where an API token
    #: for a hosted collector would live.
    otel_exporter_headers: SecretStr | None = None

    # --- Langfuse (self-hosted; plan section 13.10) -------------------------
    # Langfuse ingests OpenTelemetry directly, so it is an exporter target
    # rather than a second SDK in the call path. See docs/observability.md.
    langfuse_host: str = "http://localhost:3000"
    langfuse_public_key: str | None = None
    langfuse_secret_key: SecretStr | None = None

    # --- No defaults: absence is a boot failure -----------------------------
    # (``database_url`` is inherited from DatabaseSettings.)
    redis_url: str
    secret_key: str = Field(min_length=32)

    @field_validator("redis_url")
    @classmethod
    def _check_redis_url(cls, v: str) -> str:
        if not v.startswith(("redis://", "rediss://")):
            raise ValueError("must start with redis:// or rediss://")
        return v

    @field_validator("llm_provider_order")
    @classmethod
    def _check_provider_order(cls, v: str) -> str:
        names = [part.strip().lower() for part in v.split(",") if part.strip()]
        if not names:
            raise ValueError("must name at least one provider, e.g. 'nvidia'")
        unknown = [n for n in names if n not in KNOWN_LLM_PROVIDERS]
        if unknown:
            raise ValueError(
                f"unknown provider(s) {', '.join(unknown)}; "
                f"known: {', '.join(KNOWN_LLM_PROVIDERS)}"
            )
        if len(set(names)) != len(names):
            raise ValueError("lists the same provider twice")
        return ",".join(names)

    @model_validator(mode="after")
    def _check_provider_credentials(self) -> Settings:
        """A provider in the routing order must have the key it needs.

        This is the cross-field check that ``field_validator`` cannot express,
        and it is the reason a missing ``NVIDIA_API_KEY`` is a boot failure with
        a readable message rather than a 401 in the middle of an interview.
        """
        if "nvidia" in self.llm_providers and not (
            self.nvidia_api_key and self.nvidia_api_key.get_secret_value().strip()
        ):
            raise ValueError(
                "NVIDIA_API_KEY is required because LLM_PROVIDER_ORDER includes 'nvidia'"
            )
        return self

    @model_validator(mode="after")
    def _check_tracing_target(self) -> Settings:
        """An exporter that cannot possibly reach anywhere is a boot failure.

        Same fail-fast contract as the provider-credential check above.  The
        distinction being drawn: a *missing configuration* stops the process,
        because it is a mistake someone made; a *collector that is down* does
        not, because that is an outage somewhere else and an interview must
        survive it.  Only the first is checked here.
        """
        if self.otel_exporter == "otlp" and not (self.otel_exporter_endpoint or "").strip():
            raise ValueError(
                "OTEL_EXPORTER_ENDPOINT is required because OTEL_EXPORTER is 'otlp' "
                "(e.g. http://localhost:4318/v1/traces)"
            )
        if self.otel_exporter == "langfuse":
            missing = [
                name
                for name, value in (
                    ("LANGFUSE_PUBLIC_KEY", self.langfuse_public_key),
                    (
                        "LANGFUSE_SECRET_KEY",
                        self.langfuse_secret_key.get_secret_value()
                        if self.langfuse_secret_key
                        else None,
                    ),
                )
                if not (value or "").strip()
            ]
            if missing:
                raise ValueError(
                    f"{' and '.join(missing)} required because OTEL_EXPORTER is 'langfuse'"
                )
        return self

    @property
    def otel_exporter_headers_value(self) -> str | None:
        """The header list in plain text, read at the point of use only."""
        return self.otel_exporter_headers.get_secret_value() if self.otel_exporter_headers else None

    @property
    def llm_providers(self) -> tuple[str, ...]:
        """The routing order, parsed. Normalised by the validator above."""
        return tuple(part for part in self.llm_provider_order.split(",") if part)

    @property
    def is_prod(self) -> bool:
        return self.app_env == "prod"

    @property
    def cookie_secure(self) -> bool:
        """Send the refresh cookie over HTTPS only.

        Off locally because local development is plain http://localhost and a
        Secure cookie would simply never be stored.
        """
        return self.app_env in ("staging", "prod")


def _format_validation_error(exc: ValidationError, *, what: str = "the API") -> str:
    """Turn pydantic's error list into something readable in a container log."""
    lines = [f"Invalid configuration - {what} will not start.", ""]
    for err in exc.errors():
        field = ".".join(str(p) for p in err["loc"])
        if not field:
            # A whole-model check (e.g. "this provider needs that key"), which
            # names its own variable in the message. Prefixing it with <ROOT>
            # would only get in the way.
            lines.append(f"  - {err['msg'].removeprefix('Value error, ')}")
            continue
        env_var = field.upper()
        if err["type"] == "missing":
            lines.append(f"  - {env_var} is required but was not set")
        else:
            lines.append(f"  - {env_var}: {err['msg']}")
    lines += ["", "Copy .env.example to .env and fill in the values."]
    return "\n".join(lines)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load settings once per process. Raises ConfigError if the env is bad."""
    try:
        return Settings()  # type: ignore[call-arg]  # values come from env
    except ValidationError as exc:
        raise ConfigError(_format_validation_error(exc)) from exc


@lru_cache(maxsize=1)
def get_logging_settings() -> LoggingSettings:
    """Never raises. See ``LoggingSettings`` for why that matters."""
    return LoggingSettings()


@lru_cache(maxsize=1)
def get_database_settings() -> DatabaseSettings:
    """For processes that touch only Postgres - see ``DatabaseSettings``."""
    try:
        return DatabaseSettings()  # type: ignore[call-arg]  # values come from env
    except ValidationError as exc:
        raise ConfigError(_format_validation_error(exc, what="the migration runner")) from exc
