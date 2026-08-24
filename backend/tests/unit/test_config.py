"""The config layer must fail at boot, not at request time."""

from __future__ import annotations

import pytest

from app.config import ConfigError, get_settings


def test_valid_environment_loads(env):
    env()
    settings = get_settings()
    assert settings.app_env == "ci"
    assert settings.database_url.startswith("postgresql+asyncpg://")
    assert settings.readiness_timeout_s == 2.0


def test_settings_are_cached_per_process(env):
    env()
    assert get_settings() is get_settings()


@pytest.mark.parametrize("missing", ["DATABASE_URL", "REDIS_URL", "SECRET_KEY"])
def test_missing_required_var_fails_fast(env, missing):
    env({missing: None})
    with pytest.raises(ConfigError) as exc:
        get_settings()
    message = str(exc.value)
    assert missing in message
    assert "required" in message
    assert ".env.example" in message


def test_all_missing_vars_are_reported_at_once(env):
    env({"DATABASE_URL": None, "REDIS_URL": None, "SECRET_KEY": None})
    with pytest.raises(ConfigError) as exc:
        get_settings()
    message = str(exc.value)
    # One boot, one complete list — not three restart-and-discover cycles.
    for key in ("DATABASE_URL", "REDIS_URL", "SECRET_KEY"):
        assert key in message


def test_sync_postgres_dsn_is_rejected(env):
    env({"DATABASE_URL": "postgresql://u:p@localhost:5432/db"})
    with pytest.raises(ConfigError, match="DATABASE_URL"):
        get_settings()


def test_non_redis_url_is_rejected(env):
    env({"REDIS_URL": "http://localhost:6379"})
    with pytest.raises(ConfigError, match="REDIS_URL"):
        get_settings()


def test_short_secret_key_is_rejected(env):
    env({"SECRET_KEY": "too-short"})
    with pytest.raises(ConfigError, match="SECRET_KEY"):
        get_settings()


def test_unknown_app_env_is_rejected(env):
    env({"APP_ENV": "banana"})
    with pytest.raises(ConfigError, match="APP_ENV"):
        get_settings()
