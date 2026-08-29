"""The security rule Day 4 has to keep: nothing sensitive is ever written down.

Every test here is a leak that would otherwise be possible, written as an
assertion. They are the reason the redactor exists, so if one of them is ever
deleted, the reason has to be written down next to the deletion.
"""

from __future__ import annotations

import pytest

from app.obs.redaction import EMAIL_MASK, REDACTED, Redactor, is_sensitive_key


class TestKeyNames:
    @pytest.mark.parametrize(
        "key",
        [
            "password",
            "PASSWORD",
            "password_hash",
            "secret_key",
            "SECRET_KEY",
            "nvidia_api_key",
            "apikey",
            "authorization",
            "Cookie",
            "set-cookie",
            "refresh_token",
            "access_token",
            "llm.api_key",
        ],
    )
    def test_these_field_names_are_redacted_whatever_they_contain(self, key):
        assert is_sensitive_key(key)
        assert Redactor().value(key, "anything at all") == REDACTED

    @pytest.mark.parametrize(
        "key",
        [
            # The plan section 14.2 attribute set. Redacting any of these would
            # delete the cost model, which is the point of the whole exercise.
            "llm.input_tokens",
            "llm.output_tokens",
            "llm.reasoning_tokens",
            "llm.prompt_version",
            "llm.prompt_fingerprint",
            "llm.model",
            "llm.cost_usd",
            "llm.cache_hit",
            "llm.schema_retry_count",
            "llm.failover_count",
            "max_output_tokens",
            "token_type",
            "request_id",
            "trace_id",
            "duration_ms",
        ],
    )
    def test_these_field_names_survive(self, key):
        assert not is_sensitive_key(key)
        assert Redactor().value(key, "kept") == "kept"


class TestValueShapes:
    """A credential is masked wherever it appears, under any field name."""

    @pytest.mark.parametrize(
        "text",
        [
            "provider rejected key nvapi-Abc123Def456Ghi789Jkl",
            "using sk-proj-0123456789abcdefghijklmnop",
            "Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIn0.c2ln",
            "hash=$argon2id$v=19$m=65536,t=2,p=1$c29tZXNhbHQ$aGFzaA",
        ],
    )
    def test_credential_shapes_are_masked_inside_free_text(self, text):
        out = Redactor().text(text)
        assert REDACTED in out
        for fragment in ("nvapi-Abc123", "sk-proj-0123", "eyJzdWIiOiIxIn0", "aGFzaA"):
            assert fragment not in out

    def test_a_bare_jwt_is_masked_even_without_the_bearer_prefix(self):
        token = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJhYmMiLCJ0eXAiOiJhY2Nlc3MifQ.sIgNaTuRe"
        assert token not in Redactor().text(f"decoded {token} ok")

    def test_candidate_email_addresses_are_masked(self):
        # Plan section 14.1 requires a redaction pass before anything reaches an
        # LLM. A log file is the same exposure with a longer retention.
        out = Redactor().text("registering priya.sharma@example.com")
        assert out == f"registering {EMAIL_MASK}"


class TestKnownLiterals:
    """The backstop: this process's real secrets, masked by exact match.

    This is what catches a secret logged under a field name nobody predicted -
    the failure mode the other two mechanisms cannot cover by construction.
    """

    def test_a_registered_secret_is_removed_from_any_message(self):
        redactor = Redactor(["s3cr3t-signing-key-that-is-long-enough"])
        out = redactor.text("boot ok with key s3cr3t-signing-key-that-is-long-enough here")
        assert "s3cr3t" not in out
        assert REDACTED in out

    def test_a_registered_secret_is_removed_even_under_an_innocent_field_name(self):
        redactor = Redactor(["s3cr3t-signing-key-that-is-long-enough"])
        assert redactor.value("note", "s3cr3t-signing-key-that-is-long-enough") == REDACTED

    def test_short_values_are_not_registered(self):
        # Registering "dev" would mask the word in every log line in the system.
        redactor = Redactor(["dev"])
        assert redactor.text("running in dev mode") == "running in dev mode"

    def test_a_longer_secret_containing_a_shorter_one_is_masked_whole(self):
        redactor = Redactor(["abcdefgh", "abcdefghijklmnop"])
        assert redactor.text("value abcdefghijklmnop end") == f"value {REDACTED} end"


class TestStructures:
    def test_nested_mappings_are_walked(self):
        out = Redactor().mapping({"outer": {"password": "hunter2", "keep": "yes"}})
        assert out["outer"]["password"] == REDACTED
        assert out["outer"]["keep"] == "yes"

    def test_percent_style_log_arguments_are_redacted(self):
        # `log.info("user %s", email)` must be as safe as the f-string version.
        assert Redactor().args(("a@b.com",)) == (EMAIL_MASK,)

    def test_a_cyclic_structure_cannot_hang_a_log_call(self):
        # Observability is not allowed to be the thing that takes the API down.
        node: dict[str, object] = {"name": "loop"}
        node["self"] = node
        assert Redactor().mapping(node)  # terminates
