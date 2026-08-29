"""The bulk fixture recorder - plan section 3, Day 6.

These tests are written against the *whole* recording path - a plan file ->
``call_structured`` -> the router -> a provider -> ``FixtureStore`` - and, for
the headline test, back out again through the shipped ``StubProvider``.  That
round trip is the only assertion that actually proves the point of the day's
work: a fixture this tool writes is one the offline provider can find.  A test
that only checked the file's shape would pass just as happily if the recorder
computed its key with a different algorithm, which is precisely the bug worth
preventing.

The provider is a ``FakeProvider`` from ``conftest`` - a scripted test double,
not a second implementation of the LLM flow.  Everything between the plan and
the provider is the production code.  Nothing here touches the network, needs an
API key, or depends on the shipped recordings staying byte-identical.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from app.config import Settings, get_settings
from app.llm import prompts as prompts_module
from app.llm.client import call_structured
from app.llm.errors import ProviderAuthError, ProviderBadRequestError
from app.llm.fixtures import DEFAULT_FIXTURE_DIR, FixtureStore, fixture_key
from app.llm.probe import ProbeAnswer
from app.llm.prompts import PromptTemplate
from app.llm.providers.stub import (
    MODE_REPLAY,
    STUB_MODEL,
    MissingFixturePolicy,
    StubProvider,
)
from app.llm.recording import (
    RECORDING_PLAN_VERSION,
    SCHEMAS,
    RecordingRequest,
    RecordingStatus,
    assemble_request,
    load_recording_plan,
    parse_recording_plan,
    record_plan,
    record_request,
    recordable_providers,
)
from app.llm.recording import RecordingPlanError as PlanError
from app.llm.router import CircuitBreaker, ProviderRouter
from app.llm.tasks import TaskName
from app.llm.types import CompletionResult, LLMProvider, ModelTier

from .conftest import FakeProvider, json_result, make_router

#: The recording plan shipped in the repository, and the fixture it describes.
PLAN_DIR = Path(__file__).resolve().parents[3] / "fixtures" / "recording_plans"
SHIPPED_PLAN = PLAN_DIR / "connectivity_probe.json"


def written(store: FixtureStore) -> list[str]:
    """The fixture files that actually exist, by name."""
    if not store.directory.is_dir():
        return []
    return sorted(path.name for path in store.directory.glob("*.json"))


def probe_entry(name: str, token: str, **kwargs: Any) -> RecordingRequest:
    """A plan entry for the one task that has a prompt in Phase 1."""
    return RecordingRequest(
        name=name,
        task=TaskName.CONNECTIVITY_PROBE,
        inputs={"token": token},
        schema=ProbeAnswer,
        **kwargs,
    )


def probe_answer(token: str, *, model_said: str = "Nemotron") -> CompletionResult:
    return json_result(
        {"ok": True, "echo": token, "model_said": model_said},
        model="nvidia/nemotron-3.5-lightning-30b-a3b",
        provider="nvidia",
        input_tokens=322,
        output_tokens=19,
    )


def recording_router(*results: CompletionResult | BaseException) -> ProviderRouter:
    """A router holding one scripted provider, with the sleeping removed."""
    return make_router(
        FakeProvider("nvidia", list(results), model="nvidia/nemotron-3.5-lightning-30b-a3b")
    )


@pytest.fixture
def settings(env) -> Settings:
    env()
    return get_settings()


@pytest.fixture
def store(tmp_path: Path) -> FixtureStore:
    return FixtureStore(tmp_path / "llm")


# ---------------------------------------------------------------------------
# Reading a plan: it is untrusted input, and it says so
# ---------------------------------------------------------------------------


class TestReadingAPlan:
    def test_the_shipped_plan_loads(self):
        entries = load_recording_plan(SHIPPED_PLAN)

        assert len(entries) == 1
        assert entries[0].name == "connectivity_probe"
        assert entries[0].task is TaskName.CONNECTIVITY_PROBE
        assert entries[0].schema is ProbeAnswer
        assert entries[0].filename == "connectivity_probe.json"

    async def test_the_shipped_plan_describes_the_shipped_fixture(self, settings):
        """The plan and the Day-5 recording are the same request.

        This is the tie between the two halves of the day: the bulk recorder
        computes a key from a plan entry, and it is the key the shipped fixture
        is already filed under. If the recorder ever grew its own key algorithm,
        this test is where it would be caught.
        """
        entries = load_recording_plan(SHIPPED_PLAN)
        _, key = await assemble_request(entries[0], settings=settings)

        shipped = json.loads(
            (DEFAULT_FIXTURE_DIR / "connectivity_probe.json").read_text(encoding="utf-8")
        )
        assert key == shipped["request_hash"]

    def test_several_entries_are_read_in_order(self):
        entries = parse_recording_plan(
            {
                "format_version": RECORDING_PLAN_VERSION,
                "requests": [
                    {
                        "name": f"probe_{index}",
                        "task": "connectivity_probe",
                        "schema": "ProbeAnswer",
                        "inputs": {"token": f"t{index}"},
                    }
                    for index in range(3)
                ],
            }
        )

        assert [entry.name for entry in entries] == ["probe_0", "probe_1", "probe_2"]
        assert [entry.inputs["token"] for entry in entries] == ["t0", "t1", "t2"]

    @pytest.mark.parametrize(
        ("document", "expected"),
        [
            ([], "expected a JSON object"),
            ({"requests": []}, "format_version"),
            ({"format_version": 99, "requests": []}, "format_version"),
            ({"format_version": 1}, "non-empty list"),
            ({"format_version": 1, "requests": []}, "non-empty list"),
            ({"format_version": 1, "requests": {}}, "non-empty list"),
            ({"format_version": 1, "requests": ["nope"]}, "expected an object"),
            (
                {"format_version": 1, "fixtures": [], "requests": [{}]},
                "unknown top-level key",
            ),
        ],
    )
    def test_a_malformed_document_is_rejected_clearly(self, document, expected):
        with pytest.raises(PlanError, match=expected):
            parse_recording_plan(document, source="plan.json")

    @pytest.mark.parametrize(
        ("entry", "expected"),
        [
            ({"task": "connectivity_probe", "schema": "ProbeAnswer", "inputs": {}}, "'name'"),
            ({"name": "Probe", "task": "x", "schema": "y", "inputs": {}}, "'name'"),
            ({"name": "../evil", "task": "x", "schema": "y", "inputs": {}}, "'name'"),
            ({"name": "a/b", "task": "x", "schema": "y", "inputs": {}}, "'name'"),
            ({"name": "probe", "schema": "ProbeAnswer", "inputs": {}}, "missing required"),
            (
                {"name": "probe", "task": "nope", "schema": "ProbeAnswer", "inputs": {}},
                "unknown task",
            ),
            (
                {"name": "probe", "task": "connectivity_probe", "schema": "Nope", "inputs": {}},
                "unknown schema",
            ),
            (
                {
                    "name": "probe",
                    "task": "connectivity_probe",
                    "schema": "app.llm.probe:ProbeAnswer",
                    "inputs": {},
                },
                "unknown schema",
            ),
            (
                {
                    "name": "probe",
                    "task": "connectivity_probe",
                    "schema": "ProbeAnswer",
                    "inputs": "a string",
                },
                "'inputs'",
            ),
            (
                {
                    "name": "probe",
                    "task": "connectivity_probe",
                    "schema": "ProbeAnswer",
                    "inputs": {},
                    "temperature": 9.0,
                },
                "temperature",
            ),
            (
                {
                    "name": "probe",
                    "task": "connectivity_probe",
                    "schema": "ProbeAnswer",
                    "inputs": {},
                    "note": 7,
                },
                "'note'",
            ),
        ],
    )
    def test_a_malformed_entry_is_rejected_clearly(self, entry, expected):
        with pytest.raises(PlanError, match=expected):
            parse_recording_plan(
                {"format_version": RECORDING_PLAN_VERSION, "requests": [entry]},
                source="plan.json",
            )

    def test_two_entries_cannot_share_a_name(self):
        entry = {
            "name": "probe",
            "task": "connectivity_probe",
            "schema": "ProbeAnswer",
            "inputs": {"token": "a"},
        }
        with pytest.raises(PlanError, match="duplicate name"):
            parse_recording_plan(
                {"format_version": RECORDING_PLAN_VERSION, "requests": [entry, dict(entry)]}
            )

    def test_an_unreadable_file_is_reported_by_path(self, tmp_path):
        with pytest.raises(PlanError, match="cannot read recording plan"):
            load_recording_plan(tmp_path / "not-here.json")

    def test_invalid_json_is_reported_by_name(self, tmp_path):
        path = tmp_path / "broken.json"
        path.write_text("{not json", encoding="utf-8")
        with pytest.raises(PlanError, match="broken.json is not valid JSON"):
            load_recording_plan(path)


class TestAPlanCannotForgeAFixture:
    """Requirement: the *provider* decides provenance, never the input file."""

    @pytest.mark.parametrize(
        "forged",
        [
            {"text": '{"ok":true,"echo":"x","model_said":"me"}'},
            {"recorded_structured_mode": "json_schema"},
            {"request_hash": "0" * 64},
            {"input_tokens": 999},
            {"error": {"kind": "rate_limited", "message": "429"}},
        ],
    )
    def test_a_plan_cannot_set_what_the_provider_decides(self, forged):
        entry = {
            "name": "probe",
            "task": "connectivity_probe",
            "schema": "ProbeAnswer",
            "inputs": {"token": "a"},
            **forged,
        }
        with pytest.raises(PlanError, match="unknown key"):
            parse_recording_plan({"format_version": RECORDING_PLAN_VERSION, "requests": [entry]})

    def test_a_plan_may_only_name_a_registered_schema(self):
        """No import by name: the table is the whole list, like ``_ERROR_KINDS``."""
        assert set(SCHEMAS) == {"ProbeAnswer"}


# ---------------------------------------------------------------------------
# Recording: the headline round trip
# ---------------------------------------------------------------------------


class TestRecordingAndReplaying:
    async def test_a_recorded_fixture_replays_through_the_stub(self, settings, store):
        """Record with the recorder, replay with the shipped offline provider.

        The only assertion that proves the day's work: the recorder files a
        fixture under a key the ``StubProvider`` independently computes from the
        request the chokepoint assembles. Two different code paths, one key.
        """
        outcome = await record_request(
            probe_entry("round_trip", "rt-token"),
            router=recording_router(probe_answer("rt-token")),
            store=store,
            settings=settings,
        )
        assert outcome.status is RecordingStatus.RECORDED

        replay = ProviderRouter(
            providers=[
                StubProvider(
                    store=FixtureStore(store.directory), on_missing=MissingFixturePolicy.STRICT
                )
            ],
            breaker=CircuitBreaker(threshold=3, cooldown_s=30.0),
        )
        answer, meta = await call_structured(
            TaskName.CONNECTIVITY_PROBE,
            {"token": "rt-token"},
            ProbeAnswer,
            router=replay,
            cache=None,
            settings=settings,
        )

        assert answer.echo == "rt-token"
        assert answer.model_said == "Nemotron"
        assert meta.structured_mode == MODE_REPLAY
        # The recorded token counts, so cost arithmetic runs on real numbers.
        assert (meta.input_tokens, meta.output_tokens) == (322, 19)

    async def test_the_written_file_uses_the_day_five_format(self, settings, store):
        outcome = await record_request(
            probe_entry("format_check", "fmt", note="a synthetic example."),
            router=recording_router(probe_answer("fmt")),
            store=store,
            settings=settings,
        )
        assert outcome.path is not None
        written = json.loads(outcome.path.read_text(encoding="utf-8"))

        assert outcome.path.name == "format_check.json"
        assert written["format_version"] == 1
        assert written["request_hash"] == outcome.key
        assert written["text"] == '{"ok": true, "echo": "fmt", "model_said": "Nemotron"}'
        assert written["input_tokens"] == 322
        assert written["output_tokens"] == 19
        assert written["finish_reason"] is None
        # Provenance a human reads, and provenance a machine reads.
        assert "a synthetic example." in written["description"]
        assert "nvidia/nemotron-3.5-lightning-30b-a3b" in written["description"]
        assert written["recorded_structured_mode"] == "json_schema"
        assert written["request_preview"]["task"] == "connectivity_probe"
        assert written["request_preview"]["inputs"] == {"token": "fmt"}
        assert written["request_preview"]["schema"] == "ProbeAnswer"

    async def test_the_key_is_the_one_the_stub_computes(self, settings, store):
        """Not "a hash of something" - the Day-5 ``fixture_key`` of the request."""
        entry = probe_entry("key_check", "kc")
        request, planned_key = await assemble_request(entry, settings=settings)

        assert planned_key == fixture_key(request, model=STUB_MODEL)
        assert planned_key == StubProvider(store=FixtureStore(store.directory)).key_for(request)

        outcome = await record_request(
            entry, router=recording_router(probe_answer("kc")), store=store, settings=settings
        )
        assert outcome.key == planned_key


class TestABatchOfRecordings:
    async def test_many_entries_are_recorded_in_one_invocation(self, settings, store):
        entries = [probe_entry(f"probe_{i}", f"token-{i}") for i in range(3)]
        report = await record_plan(
            entries,
            router=recording_router(*(probe_answer(f"token-{i}") for i in range(3))),
            store=store,
            settings=settings,
        )

        assert report.ok
        assert report.count(RecordingStatus.RECORDED) == 3
        assert written(store) == ["probe_0.json", "probe_1.json", "probe_2.json"]

    async def test_distinct_requests_get_distinct_fixtures(self, settings, store):
        """The property every later phase leans on: a changed request is a
        different key, so it is a clean miss rather than a stale answer."""
        entries = [probe_entry(f"probe_{i}", f"token-{i}") for i in range(3)]
        report = await record_plan(
            entries,
            router=recording_router(*(probe_answer(f"token-{i}") for i in range(3))),
            store=store,
            settings=settings,
        )

        keys = {outcome.key for outcome in report.outcomes}
        assert len(keys) == 3
        assert len(FixtureStore(store.directory)) == 3

    async def test_progress_is_reported_per_entry_as_it_happens(self, settings, store):
        seen: list[str] = []
        await record_plan(
            [probe_entry(f"probe_{i}", f"t{i}") for i in range(2)],
            router=recording_router(*(probe_answer(f"t{i}") for i in range(2))),
            store=store,
            settings=settings,
            on_outcome=lambda outcome: seen.append(outcome.request.name),
        )

        assert seen == ["probe_0", "probe_1"]


# ---------------------------------------------------------------------------
# Idempotency: running it twice
# ---------------------------------------------------------------------------


class TestRunningItTwice:
    async def test_a_second_run_skips_what_is_already_recorded(self, settings, store):
        entry = probe_entry("idem", "idem-token")
        provider = FakeProvider("nvidia", [probe_answer("idem-token")])
        router = make_router(provider)

        first = await record_request(entry, router=router, store=store, settings=settings)
        assert first.status is RecordingStatus.RECORDED
        assert first.path is not None
        original = first.path.read_text(encoding="utf-8")

        second = await record_request(
            entry,
            router=router,
            store=FixtureStore(store.directory),
            settings=settings,
        )

        assert second.status is RecordingStatus.SKIPPED_EXISTING
        assert second.key == first.key
        assert second.path == first.path
        assert first.path.read_text(encoding="utf-8") == original
        # And - the point of computing the key first - no quota was spent.
        assert provider.call_count == 1

    async def test_overwrite_is_explicit_and_replaces_the_same_file(self, settings, store):
        entry = probe_entry("idem", "idem-token")
        provider = FakeProvider(
            "nvidia",
            [probe_answer("idem-token"), probe_answer("idem-token", model_said="Nemotron-2")],
        )
        router = make_router(provider)

        first = await record_request(entry, router=router, store=store, settings=settings)
        second = await record_request(
            entry,
            router=router,
            store=FixtureStore(store.directory),
            settings=settings,
            overwrite=True,
        )

        assert second.status is RecordingStatus.RECORDED
        assert second.key == first.key
        assert second.path == first.path
        assert provider.call_count == 2
        # One key, one file - never a second file claiming the same request.
        assert written(store) == ["idem.json"]
        assert "Nemotron-2" in second.path.read_text(encoding="utf-8")

    async def test_overwrite_replaces_the_existing_file_even_under_a_new_name(
        self, settings, store
    ):
        """Two files with one ``request_hash`` is a duplicate the store would
        have to arbitrate, so an overwrite edits the file that exists."""
        provider = FakeProvider("nvidia", [probe_answer("same"), probe_answer("same")])
        router = make_router(provider)

        first = await record_request(
            probe_entry("original_name", "same"), router=router, store=store, settings=settings
        )
        second = await record_request(
            probe_entry("a_different_name", "same"),
            router=router,
            store=FixtureStore(store.directory),
            settings=settings,
            overwrite=True,
        )

        assert second.path == first.path
        assert written(store) == ["original_name.json"]

    async def test_a_dry_run_computes_keys_and_calls_nothing(self, settings, store):
        provider = FakeProvider("nvidia", [probe_answer("dry")])
        report = await record_plan(
            [probe_entry("dry_a", "dry"), probe_entry("dry_b", "dry-2")],
            router=make_router(provider),
            store=store,
            settings=settings,
            dry_run=True,
        )

        assert [o.status for o in report.outcomes] == [RecordingStatus.WOULD_RECORD] * 2
        assert all(o.key for o in report.outcomes)
        assert provider.call_count == 0
        assert written(store) == []


# ---------------------------------------------------------------------------
# Failure handling
# ---------------------------------------------------------------------------


class TestWhenARecordingFails:
    async def test_one_failure_is_reported_and_the_batch_continues(self, settings, store):
        """A failed entry must not cost the others their recordings - re-running
        would pay for them a second time."""
        provider = FakeProvider(
            "nvidia",
            [
                # Not retryable, so the router makes exactly one attempt.
                ProviderBadRequestError("the model refused this prompt", provider="nvidia"),
                probe_answer("ok-2"),
                probe_answer("ok-3"),
            ],
        )
        report = await record_plan(
            [
                probe_entry("bad", "bad-token"),
                probe_entry("good_a", "ok-2"),
                probe_entry("good_b", "ok-3"),
            ],
            router=make_router(provider),
            store=store,
            settings=settings,
        )

        assert not report.ok
        assert [o.status for o in report.outcomes] == [
            RecordingStatus.FAILED,
            RecordingStatus.RECORDED,
            RecordingStatus.RECORDED,
        ]
        assert [o.request.name for o in report.failures] == ["bad"]
        assert "the model refused this prompt" in (report.outcomes[0].detail or "")
        # The failure wrote nothing.
        assert written(store) == ["good_a.json", "good_b.json"]

    async def test_a_plan_entry_missing_a_template_input_fails_that_entry_only(
        self, settings, store
    ):
        """Not every failure is the provider's. A missing prompt input raises a
        ``KeyError`` deep in rendering, and it is still one entry's problem."""
        entry = RecordingRequest(
            name="no_token",
            task=TaskName.CONNECTIVITY_PROBE,
            inputs={"wrong_key": "x"},
            schema=ProbeAnswer,
        )
        report = await record_plan(
            [entry, probe_entry("fine", "fine")],
            router=recording_router(probe_answer("fine")),
            store=store,
            settings=settings,
        )

        assert report.outcomes[0].status is RecordingStatus.FAILED
        assert "token" in (report.outcomes[0].detail or "")
        assert report.outcomes[1].status is RecordingStatus.RECORDED

    async def test_a_failure_never_prints_a_credential(self, env, store):
        """A provider error can quote the request that carried the key.

        The reported detail goes through the same redactor the log formatter
        uses, seeded with this process's real secret values.
        """
        api_key = "nvapi-abcdefghijklmnopqrstuvwxyz0123456789"  # noqa: S105 - invented
        env({"NVIDIA_API_KEY": api_key})
        settings = get_settings()

        report = await record_plan(
            [probe_entry("leaky", "leak")],
            router=recording_router(
                ProviderAuthError(
                    f"401 rejected Authorization: Bearer {api_key}", provider="nvidia"
                )
            ),
            store=store,
            settings=settings,
        )

        detail = report.outcomes[0].detail or ""
        assert report.outcomes[0].status is RecordingStatus.FAILED
        assert api_key not in detail
        assert "abcdefghijklmnopqrstuvwxyz" not in detail
        assert "[REDACTED]" in detail
        assert "401" in detail, "the diagnosis must survive the redaction"


# ---------------------------------------------------------------------------
# Provenance: a recording is a real answer or it is nothing
# ---------------------------------------------------------------------------


class _UnrealProvider(LLMProvider):
    """Answers a schema-valid object while admitting it is not a real answer.

    Exactly what the stub does on a fixture miss, and what the chokepoint
    reports for a cache hit - the two ways a "successful" call can carry data no
    model just produced.
    """

    def __init__(self, mode: str, *, name: str = "stub") -> None:
        self.name = name
        self._mode = mode

    def model_for(self, tier: ModelTier) -> str:
        return STUB_MODEL

    async def complete(self, request):
        return CompletionResult(
            text='{"ok": true, "echo": "stub:.echo:9f3a", "model_said": "stub"}',
            provider=self.name,
            model=STUB_MODEL,
            input_tokens=10,
            output_tokens=5,
            finish_reason="stop",
            structured_mode=self._mode,
        )

    async def aclose(self) -> None:
        return None


class TestProvenance:
    @pytest.mark.parametrize(
        ("mode", "provider"),
        [
            ("stub_synthesized", "stub"),
            ("stub_replay", "stub"),
            # A cached answer is real, but nobody just gave it - recording one
            # would file today's fixture from last week's call.
            ("cache", "nvidia"),
        ],
    )
    async def test_the_recorder_refuses_to_record_an_unreal_answer(
        self, settings, store, mode, provider
    ):
        """The worst possible outcome of this tool would be a directory where
        invented data is indistinguishable from recorded data."""
        report = await record_plan(
            [probe_entry("fake_source", "x")],
            router=make_router(_UnrealProvider(mode, name=provider)),
            store=store,
            settings=settings,
        )

        assert report.outcomes[0].status is RecordingStatus.FAILED
        assert mode in (report.outcomes[0].detail or "")
        assert written(store) == []

    async def test_the_stub_cannot_be_a_recording_source(self, settings, store):
        """Even claiming a live provenance does not help: the stub answers from
        the very directory being recorded into."""
        report = await record_plan(
            [probe_entry("stub_source", "x")],
            router=make_router(_UnrealProvider("json_schema", name="stub")),
            store=store,
            settings=settings,
        )

        assert report.outcomes[0].status is RecordingStatus.FAILED
        assert "cannot be a recording source" in (report.outcomes[0].detail or "")
        assert written(store) == []

    async def test_recording_never_reads_or_writes_the_response_cache(
        self, settings, store, memory_cache
    ):
        """A recording must come from the wire. Recording the same request twice
        calls the provider twice - the second answer is not served from a cache."""
        entry = probe_entry("uncached", "uncached-token")
        provider = FakeProvider("nvidia", [probe_answer("uncached-token")])
        router = make_router(provider)

        await record_request(entry, router=router, store=store, settings=settings)
        await record_request(
            entry,
            router=router,
            store=FixtureStore(store.directory),
            settings=settings,
            overwrite=True,
        )

        assert provider.call_count == 2
        assert (memory_cache.reads, memory_cache.writes) == (0, 0)

    async def test_a_replayed_fixture_still_reports_stub_replay(self, settings, store):
        """Day-5 semantics, unchanged by Day 6: the *provider* labels a replay,
        and it is not what the recording says about itself."""
        outcome = await record_request(
            probe_entry("provenance", "prov"),
            router=recording_router(probe_answer("prov")),
            store=store,
            settings=settings,
        )
        assert outcome.path is not None
        written = json.loads(outcome.path.read_text(encoding="utf-8"))

        # The recording remembers how NVIDIA was asked to enforce the schema...
        assert written["recorded_structured_mode"] == "json_schema"

        # ...and a replay of it is still labelled a replay.
        _, meta = await call_structured(
            TaskName.CONNECTIVITY_PROBE,
            {"token": "prov"},
            ProbeAnswer,
            router=ProviderRouter(
                providers=[StubProvider(store=FixtureStore(store.directory))],
                breaker=CircuitBreaker(threshold=3, cooldown_s=30.0),
            ),
            cache=None,
            settings=settings,
        )
        assert meta.structured_mode == MODE_REPLAY

    @pytest.mark.parametrize(
        ("order", "expected"),
        [("stub", ()), ("nvidia", ("nvidia",)), ("nvidia,stub", ("nvidia",))],
    )
    def test_the_stub_is_not_a_recordable_provider(self, env, order, expected):
        """Checked before a run starts, so a stub-only configuration is refused
        once rather than discovered one wasted entry at a time."""
        env({"LLM_PROVIDER_ORDER": order, "NVIDIA_API_KEY": None if order == "stub" else "k" * 8})
        assert recordable_providers(get_settings()) == expected


# ---------------------------------------------------------------------------
# Architecture: the recorder uses the production path, not a copy of it
# ---------------------------------------------------------------------------


class TestItUsesTheProductionPath:
    async def test_the_provider_sees_the_request_the_chokepoint_assembles(self, settings, store):
        """Rendered prompt, derived schema, routing-table sampling settings -
        none of it built by the recorder."""
        provider = FakeProvider("nvidia", [probe_answer("assembled")])
        await record_request(
            probe_entry("assembled", "assembled"),
            router=make_router(provider),
            store=store,
            settings=settings,
        )

        request = provider.calls[-1]
        prompt = prompts_module.get_prompt(TaskName.CONNECTIVITY_PROBE)
        assert request.messages == prompt.render({"token": "assembled"})
        assert request.json_schema.name == "ProbeAnswer"
        assert request.tier is ModelTier.SMALL_FAST
        assert request.temperature == 0.0
        assert request.max_output_tokens == 256

    async def test_a_temperature_override_changes_the_key(self, settings):
        """Every field that changes what a model returns is in the key, and the
        recorder does not get its own opinion about which."""
        entry = probe_entry("temp", "same-token")
        _, default_key = await assemble_request(entry, settings=settings)
        _, hot_key = await assemble_request(
            probe_entry("temp", "same-token", temperature=0.9), settings=settings
        )

        assert default_key != hot_key

    async def test_a_repaired_answer_is_recorded_under_the_key_replay_will_ask_for(
        self, settings, store, monkeypatch
    ):
        """The one case where the first request and the last one differ.

        When a model's first answer does not validate, the chokepoint re-asks
        with the failure appended - so the *answer* worth recording is the last
        one and the *key* worth filing it under is the first one, because a
        replay starts at attempt 0 with no repair in it. Getting this backwards
        writes a fixture that no replay can ever find, and this test is the
        round trip that proves it does not.
        """
        monkeypatch.setitem(
            prompts_module.PROMPTS,
            TaskName.RESUME_EXTRACTION,
            PromptTemplate(
                task=TaskName.RESUME_EXTRACTION,
                version="record-v1",
                system="Extract.",
                user="Resume:\n$resume",
                required_inputs=("resume",),
            ),
        )
        entry = RecordingRequest(
            name="repaired",
            task=TaskName.RESUME_EXTRACTION,
            inputs={"resume": "Ada, 7 years"},
            schema=ProbeAnswer,
        )
        provider = FakeProvider(
            "nvidia",
            [
                json_result({"nonsense": True}),
                json_result({"ok": True, "echo": "repaired", "model_said": "Nemotron"}),
            ],
        )

        outcome = await record_request(
            entry, router=make_router(provider), store=store, settings=settings
        )

        assert outcome.status is RecordingStatus.RECORDED
        assert outcome.meta is not None
        assert outcome.meta.schema_retry_count == 1
        # The description carries the repair count, so a reviewer can see that
        # this answer took the model two goes.
        assert (
            "schema_retries=1"
            in json.loads((outcome.path or Path()).read_text(encoding="utf-8"))["description"]
        )

        # And the recording replays: the base request finds it on attempt 0.
        answer, meta = await call_structured(
            entry.task,
            entry.inputs,
            entry.schema,
            router=ProviderRouter(
                providers=[StubProvider(store=FixtureStore(store.directory))],
                breaker=CircuitBreaker(threshold=3, cooldown_s=30.0),
            ),
            cache=None,
            settings=settings,
        )
        assert answer.echo == "repaired"
        assert meta.structured_mode == MODE_REPLAY
        assert meta.schema_retry_count == 0
