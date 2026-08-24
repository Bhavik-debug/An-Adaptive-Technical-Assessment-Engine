"""Application configuration.

One rule: the process must not start in a half-configured state.  A missing or
malformed environment variable is a boot-time failure with a readable message,
never a 500 three hours into a running interview.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, ValidationError, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ConfigError(RuntimeError):
    """Raised at boot when the environment is missing or invalid."""


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- Defaults are fine for these ---------------------------------------
    app_name: str = "adaptive-ai-interviewer"
    app_env: Literal["local", "ci", "staging", "prod"] = "local"
    debug: bool = False
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
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

    # --- No defaults: absence is a boot failure -----------------------------
    database_url: str
    redis_url: str
    secret_key: str = Field(min_length=32)

    @field_validator("database_url")
    @classmethod
    def _check_database_url(cls, v: str) -> str:
        if not v.startswith("postgresql+asyncpg://"):
            raise ValueError(
                "must be an async Postgres DSN, e.g. " "postgresql+asyncpg://user:pass@host:5432/db"
            )
        return v

    @field_validator("redis_url")
    @classmethod
    def _check_redis_url(cls, v: str) -> str:
        if not v.startswith(("redis://", "rediss://")):
            raise ValueError("must start with redis:// or rediss://")
        return v

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


def _format_validation_error(exc: ValidationError) -> str:
    """Turn pydantic's error list into something readable in a container log."""
    lines = ["Invalid configuration - the API will not start.", ""]
    for err in exc.errors():
        field = ".".join(str(p) for p in err["loc"]) or "<root>"
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
