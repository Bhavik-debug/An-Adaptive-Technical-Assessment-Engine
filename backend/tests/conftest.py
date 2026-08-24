"""Test fixtures.

Every test runs with an explicit, complete environment and no real Postgres or
Redis: dependency probes are patched per-test. Integration tests against real
containers (testcontainers) arrive with the schema on Day 2.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from app.config import Settings, get_settings

# Everything Settings reads, so a test can start from a known-empty environment.
SETTINGS_ENV_KEYS = tuple(name.upper() for name in Settings.model_fields)

VALID_ENV: dict[str, str] = {
    "APP_ENV": "ci",
    "DEBUG": "false",
    "LOG_LEVEL": "WARNING",
    "DATABASE_URL": "postgresql+asyncpg://u:p@localhost:5432/testdb",
    "REDIS_URL": "redis://localhost:6379/0",
    "SECRET_KEY": "x" * 48,
}


@pytest.fixture(autouse=True)
def _reset_settings_cache() -> Iterator[None]:
    """Settings are process-cached; drop the cache around every test."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture(autouse=True)
def _ignore_developer_dotenv(monkeypatch: pytest.MonkeyPatch) -> None:
    """A developer's .env must never leak into a test's expectations."""
    monkeypatch.setitem(Settings.model_config, "env_file", None)


@pytest.fixture
def env(monkeypatch: pytest.MonkeyPatch):
    """Set an explicit environment: VALID_ENV plus/minus the given overrides.

    Pass ``None`` as a value to unset that variable.
    """

    def _set(overrides: dict[str, str | None] | None = None) -> dict[str, str]:
        for key in SETTINGS_ENV_KEYS:
            monkeypatch.delenv(key, raising=False)
        merged = dict(VALID_ENV)
        merged.update({k: v for k, v in (overrides or {}).items() if v is not None})
        for key in overrides or {}:
            if overrides[key] is None:
                merged.pop(key, None)
        for key, value in merged.items():
            monkeypatch.setenv(key, value)
        return merged

    return _set
