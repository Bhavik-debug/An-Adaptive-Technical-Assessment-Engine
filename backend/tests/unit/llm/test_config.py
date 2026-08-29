"""Configuration for the LLM layer fails at boot, like everything else."""

from __future__ import annotations

import pytest

from app.config import ConfigError, get_settings
from app.llm.providers.nvidia import NVIDIA_DEFAULT_BASE_URL, NVIDIA_DEFAULT_MODEL


def test_defaults_select_nvidia_nemotron(env):
    env()
    settings = get_settings()
    assert settings.llm_providers == ("nvidia",)
    assert settings.nvidia_model == "nvidia/nemotron-3.5-lightning-30b-a3b"
    assert settings.nvidia_base_url == "https://integrate.api.nvidia.com/v1"


def test_config_defaults_match_the_adapter_defaults(env):
    """The two places the endpoint is written down must agree.

    ``config.py`` cannot import ``app.llm`` (that is an import cycle through the
    package ``__init__``), so the literals are duplicated. This is the guard
    that keeps the duplicate honest.
    """
    env()
    settings = get_settings()
    assert settings.nvidia_base_url == NVIDIA_DEFAULT_BASE_URL
    assert settings.nvidia_model == NVIDIA_DEFAULT_MODEL


def test_missing_nvidia_key_is_a_boot_failure(env):
    env({"NVIDIA_API_KEY": None})
    with pytest.raises(ConfigError) as exc:
        get_settings()
    message = str(exc.value)
    assert "NVIDIA_API_KEY is required" in message
    assert "LLM_PROVIDER_ORDER" in message
    # The whole-model check names its own variable, so it must not be prefixed
    # with the placeholder pydantic uses for a root-level error.
    assert "<ROOT>" not in message


def test_blank_nvidia_key_is_treated_as_missing(env):
    env({"NVIDIA_API_KEY": "   "})
    with pytest.raises(ConfigError, match="NVIDIA_API_KEY is required"):
        get_settings()


def test_api_key_is_never_rendered(env):
    """A SecretStr is what makes an accidental log line harmless."""
    env({"NVIDIA_API_KEY": "nvapi-super-secret-value"})
    settings = get_settings()
    for rendering in (repr(settings), str(settings), repr(settings.nvidia_api_key)):
        assert "super-secret-value" not in rendering
    assert settings.nvidia_api_key is not None
    assert settings.nvidia_api_key.get_secret_value() == "nvapi-super-secret-value"


def test_unknown_provider_name_is_rejected(env):
    env({"LLM_PROVIDER_ORDER": "nvidia,gemini"})
    with pytest.raises(ConfigError) as exc:
        get_settings()
    assert "LLM_PROVIDER_ORDER" in str(exc.value)
    assert "gemini" in str(exc.value)


def test_empty_provider_order_is_rejected(env):
    env({"LLM_PROVIDER_ORDER": " , "})
    with pytest.raises(ConfigError, match="at least one provider"):
        get_settings()


def test_duplicate_provider_is_rejected(env):
    env({"LLM_PROVIDER_ORDER": "nvidia,nvidia"})
    with pytest.raises(ConfigError, match="same provider twice"):
        get_settings()


def test_provider_order_is_normalised(env):
    env({"LLM_PROVIDER_ORDER": " NVIDIA , "})
    assert get_settings().llm_providers == ("nvidia",)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("LLM_TIMEOUT_S", "0"),
        ("LLM_TIMEOUT_S", "-1"),
        ("LLM_MAX_ATTEMPTS_PER_PROVIDER", "0"),
        ("LLM_SCHEMA_MAX_RETRIES", "-1"),
        ("LLM_BREAKER_FAILURE_THRESHOLD", "0"),
        ("LLM_BREAKER_COOLDOWN_S", "0"),
        ("LLM_REASONING_BUDGET_TOKENS", "10"),
    ],
)
def test_nonsensical_llm_tuning_is_rejected(env, key, value):
    env({key: value})
    with pytest.raises(ConfigError, match=key):
        get_settings()
