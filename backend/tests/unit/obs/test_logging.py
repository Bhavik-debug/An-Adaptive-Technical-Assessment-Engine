"""Structured logging: is every line machine-readable, correlated and safe?"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from app.config import Settings, get_settings
from app.obs import configure_logging
from app.obs.context import clear_context, set_request_id
from app.obs.logging import RedactingJsonFormatter, get_redactor
from app.obs.redaction import EMAIL_MASK, REDACTED, Redactor
from app.obs.tracing import build_tracer_provider

log = logging.getLogger("test.obs")


@pytest.fixture(autouse=True)
def _clean_context():
    clear_context()
    yield
    clear_context()


class TestShape:
    def test_every_line_is_one_json_object_with_the_standard_fields(self, env, captured_logs):
        env()
        log.info("something happened")

        record = captured_logs.one("something happened")
        assert record["level"] == "INFO"
        assert record["logger"] == "test.obs"
        assert record["service"] == "adaptive-ai-interviewer"
        assert record["env"] == "ci"
        # ISO 8601, UTC. A log line whose timestamp has no timezone is a log
        # line you cannot line up with anything else.
        assert record["ts"].endswith("+00:00")

    def test_extras_become_top_level_queryable_fields(self, env, captured_logs):
        env()
        log.info("llm_call", extra={"llm.model": "nvidia/x", "llm.input_tokens": 612})

        record = captured_logs.one("llm_call")
        # Not a string that happens to contain JSON: real fields, so a log
        # backend can average `llm.input_tokens` without parsing prose.
        assert record["llm.model"] == "nvidia/x"
        assert record["llm.input_tokens"] == 612

    def test_text_format_is_available_and_still_redacts(self, env):
        env({"LOG_FORMAT": "text"})
        settings = get_settings()
        from app.obs.logging import build_formatter

        formatter = build_formatter(settings, Redactor(["topsecretvalue"]))
        record = logging.LogRecord("n", logging.INFO, "p", 1, "key=topsecretvalue", None, None)
        rendered = formatter.format(record)
        assert "topsecretvalue" not in rendered
        assert REDACTED in rendered


class TestCorrelation:
    def test_the_request_id_is_stamped_on_every_line_without_the_call_site_asking(
        self, env, captured_logs
    ):
        env()
        set_request_id("req-abc")
        log.info("inside a request")
        assert captured_logs.one("inside a request")["request_id"] == "req-abc"

    def test_lines_outside_a_request_simply_have_no_request_id(self, env, captured_logs):
        env()
        log.info("a background job")
        assert "request_id" not in captured_logs.one("a background job")

    def test_the_active_trace_and_span_ids_are_stamped_on(self, env, captured_logs):
        env()
        provider = build_tracer_provider(get_settings())
        with provider.get_tracer("t").start_as_current_span("work"):
            log.info("during a span")

        record = captured_logs.one("during a span")
        # 32 and 16 hex characters: the W3C trace-context wire format, so these
        # ids paste straight into a Langfuse or Jaeger search box.
        assert len(record["trace_id"]) == 32
        assert len(record["span_id"]) == 16

    def test_logging_works_in_a_process_that_never_configured_tracing(self, env, captured_logs):
        env()
        log.info("no tracer here")
        record = captured_logs.one("no tracer here")
        assert "trace_id" not in record


class TestErrors:
    def test_an_exception_is_logged_with_its_type_message_and_stack(self, env, captured_logs):
        env()
        try:
            raise ValueError("the grader exploded")
        except ValueError:
            log.exception("grade failed")

        record = captured_logs.one("grade failed")
        assert record["level"] == "ERROR"
        assert record["error.type"] == "ValueError"
        assert record["error.message"] == "the grader exploded"
        # The stack is what makes an error line actionable rather than a note
        # that something went wrong somewhere.
        assert "test_logging.py" in record["error.stack"]


class TestRedactionIsAppliedByTheFormatter:
    """The formatter is the last thing to touch a line before it leaves."""

    def test_a_secret_in_the_message_never_reaches_the_output(self, env, captured_logs):
        env()
        log.info("calling with nvapi-Abc123Def456Ghi789Jkl")
        assert "nvapi-Abc123" not in captured_logs.text

    def test_a_sensitive_extra_is_redacted_by_its_name(self, env, captured_logs):
        env()
        log.info("login attempt", extra={"password": "hunter2", "email": "a@b.com"})

        record = captured_logs.one("login attempt")
        assert record["password"] == REDACTED
        assert record["email"] == EMAIL_MASK

    def test_a_percent_formatted_argument_is_redacted(self, env, captured_logs):
        env()
        log.info("registering %s", "candidate@example.com")
        assert "candidate@example.com" not in captured_logs.text
        assert EMAIL_MASK in captured_logs.text

    def test_configure_logging_teaches_the_redactor_this_process_secrets(self, env):
        """The backstop, wired at boot.

        After boot, a log line containing the signing key or the provider key
        has it removed even if the code writing that line had no idea it was
        handling a secret.
        """
        env({"SECRET_KEY": "a-very-real-signing-key-0123456789abcdef"})
        configure_logging(get_settings())

        formatter = RedactingJsonFormatter(get_redactor())
        record = logging.LogRecord(
            "n",
            logging.INFO,
            "p",
            1,
            "leaked a-very-real-signing-key-0123456789abcdef and unit-test-placeholder",
            None,
            None,
        )
        rendered = json.loads(formatter.format(record))
        assert "a-very-real-signing-key" not in rendered["msg"]
        assert "unit-test-placeholder" not in rendered["msg"]


class TestConfiguration:
    def test_configuring_twice_does_not_double_every_line(self, env):
        env()
        settings: Settings = get_settings()
        configure_logging(settings)
        first = len(logging.getLogger().handlers)
        configure_logging(settings)
        assert len(logging.getLogger().handlers) == first

    def test_debug_logging_does_not_turn_on_full_prompt_logging(self, env):
        """The leak found by running the live smoke test with DEBUG everywhere.

        ``openai._base_client`` writes the entire request - prompt included - at
        DEBUG, and this project's prompts carry resumes and candidate answers.
        Investigating our own code at DEBUG must not silently start recording
        candidate data. Redaction cannot help here: an answer is not a shape a
        regex can recognise. The only correct behaviour is not to write it down.
        """
        env({"LOG_LEVEL": "DEBUG"})
        sdk = logging.getLogger("openai")
        sdk.setLevel(logging.NOTSET)  # would inherit DEBUG from the root

        configure_logging(get_settings())

        assert logging.getLogger().level == logging.DEBUG
        assert not sdk.isEnabledFor(logging.DEBUG)
        for name in ("httpx", "httpcore", "urllib3"):
            assert not logging.getLogger(name).isEnabledFor(logging.DEBUG)

    def test_a_process_without_llm_settings_can_still_configure_logging(self, env):
        """The migration runner deliberately loads no LLM credential.

        It still has to log, and it has to log in the same format as everything
        else - a migrate container whose output nothing can parse is the one
        process you most want to read when a deploy fails.
        """
        from app.config import get_database_settings

        env()
        configure_logging(get_database_settings())

        root = logging.getLogger()
        assert any(getattr(h, "_obs_owned_handler", False) for h in root.handlers)

    def test_the_migration_environment_never_reconfigures_logging(self):
        """Regression, and the reason it is a source check rather than a
        behavioural one: the damage is done at import time, process-wide.

        `migrations/env.py` used to call `fileConfig()`, Alembic's generated
        default. That disables every existing logger and replaces the root
        handler - so the integration suite, which runs migrations in-process,
        silently destroyed the logging configuration for every test that ran
        after it. The observability tests then captured nothing and failed with
        no hint as to why. Re-adding that call must be caught immediately.
        """
        source = (Path(__file__).resolve().parents[3] / "migrations" / "env.py").read_text(
            encoding="utf-8"
        )
        code = "\n".join(line for line in source.splitlines() if not line.lstrip().startswith("#"))
        assert "fileConfig" not in code
        assert "configure_logging" in code

    def test_uvicorn_loggers_are_routed_through_our_handler(self, env):
        """Otherwise the access log is a second, unredacted path to stdout.

        The access log prints full request URLs, which is exactly where a token
        pasted into a query string would surface.
        """
        env()
        access = logging.getLogger("uvicorn.access")
        access.addHandler(logging.NullHandler())

        configure_logging(get_settings())

        assert access.handlers == []
        assert access.propagate is True
