"""The config layer must fail at boot, not at request time."""

from __future__ import annotations

import pytest

from app.config import ConfigError, get_database_settings, get_settings


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


PLACEHOLDER_SECRET = "dev-only-secret-change-me-0123456789abcdef"  # the .env.example value


@pytest.mark.parametrize("app_env", ["staging", "prod"])
def test_deployed_envs_refuse_the_placeholder_secret(env, app_env):
    env({"APP_ENV": app_env, "SECRET_KEY": PLACEHOLDER_SECRET})
    with pytest.raises(ConfigError, match="SECRET_KEY is the documented placeholder"):
        get_settings()


@pytest.mark.parametrize("app_env", ["staging", "prod"])
def test_deployed_envs_refuse_the_stub_provider(env, app_env):
    env({"APP_ENV": app_env, "LLM_PROVIDER_ORDER": "nvidia,stub"})
    with pytest.raises(ConfigError, match="may not include 'stub'"):
        get_settings()


def test_local_and_ci_still_accept_both(env):
    env({"APP_ENV": "local", "SECRET_KEY": PLACEHOLDER_SECRET, "LLM_PROVIDER_ORDER": "stub"})
    assert get_settings().llm_providers == ("stub",)


def test_prod_with_real_values_loads(env):
    env({"APP_ENV": "prod", "SECRET_KEY": "r" * 48})
    assert get_settings().is_prod


def test_unknown_app_env_is_rejected(env):
    env({"APP_ENV": "banana"})
    with pytest.raises(ConfigError, match="APP_ENV"):
        get_settings()


def test_migrations_need_only_a_database_url(env):
    """A schema migration must not depend on an LLM credential.

    Otherwise the Phase 6 restore drill - and every `alembic upgrade head` -
    requires a provider account to be in good standing, which is coupling with
    a real operational cost.
    """
    env({"NVIDIA_API_KEY": None, "REDIS_URL": None, "SECRET_KEY": None})

    settings = get_database_settings()

    assert settings.database_url.startswith("postgresql+asyncpg://")
    # The full application settings still refuse, in the same environment.
    with pytest.raises(ConfigError):
        get_settings()


def test_database_settings_apply_the_same_dsn_rule(env):
    env({"DATABASE_URL": "postgresql://u:p@localhost:5432/db"})
    with pytest.raises(ConfigError) as exc:
        get_database_settings()
    assert "DATABASE_URL" in str(exc.value)
    assert "migration runner" in str(exc.value)
