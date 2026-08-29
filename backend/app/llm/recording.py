"""Bulk recording: many real provider answers, filed as replayable fixtures.

Plan section 3, Day 6 - the backlog item Day 5 deliberately deferred: *"a bulk
fixture recorder ... Phase 4's grading evals will need dozens, keyed the way plan
section 12.3 describes."*  This is that path, built now so the grading eval set
has somewhere to be recorded into.

**The whole idea in one line:** record a real provider answer once, then replay
it offline forever through the same ``LLMProvider`` interface production uses.

    recording plan (JSON)
        -> call_structured()        the production chokepoint, unmodified
        -> ProviderRouter           the production router, unmodified
        -> NvidiaProvider           the real thing, spending real quota
        -> fixture_key()            the Day-5 key, not a second algorithm
        -> FixtureStore.save()      the Day-5 format, not a second format
        -> StubProvider             offline replay, for every later phase

Nothing in this module knows how to *answer* a request.  It knows how to read a
list of them, push each one through the chokepoint, and write down what came
back.  That distinction is the point: a recorder with its own idea of what a
model call looks like would be recording its own behaviour, not the system's.

**Three safety properties this file is built around.**

1. *A recording is a real answer or it is nothing.*  The provider decides what
   ``structured_mode`` a call had; if that value says the answer was replayed,
   synthesized or served from cache, the entry is **failed**, not written.  A
   fixture directory where invented data is indistinguishable from recorded data
   would poison every eval that reads it.
2. *An existing recording is never silently overwritten.*  The key is computed
   before the call - not after - so a request that is already on disk costs no
   quota at all.  Overwriting is opt-in, per invocation.
3. *A plan file is untrusted input.*  It names a task and a schema from fixed
   tables and supplies inputs; it cannot name an importable object, write
   outside the fixture directory, or set any field that describes provenance.

**And the rule inherited from Day 5, restated because it matters most:** a
fixture is committed to git, which makes it permanent, shared and outside the
reach of the log redactor.  **Never put candidate data in a recording plan.**
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from app.config import Settings
from app.llm.cache import NullCache
from app.llm.client import CallMeta, call_structured
from app.llm.errors import LLMError
from app.llm.fixtures import Fixture, FixtureStore, fixture_key
from app.llm.probe import ProbeAnswer
from app.llm.providers.stub import MODE_REPLAY, MODE_SYNTHESIZED, STUB_MODEL
from app.llm.router import CircuitBreaker, ProviderRouter
from app.llm.tasks import TaskName
from app.llm.types import CompletionRequest, CompletionResult, LLMProvider, ModelTier
from app.obs.redaction import Redactor

log = logging.getLogger(__name__)

#: Bumped if the *plan* format below changes incompatibly. Independent of
#: ``FIXTURE_FORMAT_VERSION``: one describes what to record, the other what was
#: recorded, and they have no reason to move together.
RECORDING_PLAN_VERSION = 1

#: The schemas a plan entry may name, by name.  An explicit table, exactly like
#: ``_ERROR_KINDS`` in the stub provider, and for the same reason: resolving
#: ``"app.llm.probe:ProbeAnswer"`` by import would let a plan file name any
#: importable object in the process.  A phase adds its schema here when it adds
#: its prompt - Phase 4's grading schemas are not this day's work.
SCHEMAS: Mapping[str, type[BaseModel]] = {
    "ProbeAnswer": ProbeAnswer,
}

#: A recording name becomes a filename, so it is restricted to characters that
#: cannot escape the fixture directory or surprise a filesystem. ``../evil`` and
#: ``C:\x`` are rejected here rather than sanitised, because a plan that meant
#: to write outside the directory is a plan whose author should be told.
_NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")

#: Everything a plan entry may say. Anything else is a typo or an attempt to set
#: something the recorder alone decides - both are refusals. In particular
#: ``text``, ``input_tokens`` and ``recorded_structured_mode`` are *outputs* of a
#: recording; a plan that could set them could forge a fixture.
_REQUIRED_ENTRY_KEYS = frozenset({"name", "task", "schema", "inputs"})
_OPTIONAL_ENTRY_KEYS = frozenset({"note", "temperature"})

#: ``structured_mode`` values that mean "this answer did not come from a real
#: provider". Recording one would file invented or already-recorded data as a
#: fresh recording, which is the one thing this tool must never do.
_UNRECORDABLE_MODES = frozenset({MODE_REPLAY, MODE_SYNTHESIZED, "cache"})

#: The provider that cannot be a recording *source*. It answers from the
#: fixture directory, which is where recordings are going.
STUB_PROVIDER_NAME = "stub"


class RecordingPlanError(LLMError):
    """A recording plan file could not be read, or said something invalid."""


class RecordingAbortedError(LLMError):
    """A recording produced no usable real answer, so nothing was written."""


# ---------------------------------------------------------------------------
# The plan: what to record
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RecordingRequest:
    """One thing to record.

    Deliberately *not* a request in the ``CompletionRequest`` sense: it names a
    task and its inputs, and lets the chokepoint assemble the provider request
    from them.  Anything richer here would be a second request builder, and two
    request builders drift.
    """

    #: Identifies the recording to a human, and names its file (``<name>.json``).
    name: str
    task: TaskName
    inputs: dict[str, Any]
    schema: type[BaseModel]
    #: Free text copied into the fixture description. Never parsed.
    note: str = ""
    #: Overrides the task's configured temperature. ``None`` - the normal case -
    #: uses the routing table, so a recording matches what production sends.
    temperature: float | None = None

    @property
    def filename(self) -> str:
        return f"{self.name}.json"


def load_recording_plan(path: Path) -> tuple[RecordingRequest, ...]:
    """Read and validate a plan file. Raises ``RecordingPlanError``."""
    try:
        # ``utf-8-sig`` rather than ``utf-8``: a plan is written by hand, and on
        # Windows a perfectly ordinary editor - or `Out-File -Encoding utf8` -
        # puts a byte-order mark at the front. Rejecting a valid plan over an
        # invisible character is a bad first experience of a tool; the codec
        # strips a BOM if there is one and changes nothing if there is not.
        raw = path.read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise RecordingPlanError(f"cannot read recording plan {path}: {exc}") from exc
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RecordingPlanError(f"{path.name} is not valid JSON: {exc}") from exc
    return parse_recording_plan(data, source=path.name)


def parse_recording_plan(data: Any, *, source: str = "<plan>") -> tuple[RecordingRequest, ...]:
    """Validate a decoded plan document.

    Every rejection names the file, the entry and what was wrong with it: a
    recorder that fails is usually being run by someone who has just written the
    plan by hand, and "entry 3 (grade_easy): unknown task" is the difference
    between a two-second fix and a bisect.
    """
    if not isinstance(data, Mapping):
        raise RecordingPlanError(f"{source}: expected a JSON object at the top level")

    version = data.get("format_version")
    if version != RECORDING_PLAN_VERSION:
        raise RecordingPlanError(
            f"{source}: format_version {version!r}, this build reads {RECORDING_PLAN_VERSION}"
        )

    unknown_top = set(data) - {"format_version", "description", "requests"}
    if unknown_top:
        raise RecordingPlanError(f"{source}: unknown top-level key(s) {_listed(unknown_top)}")

    requests = data.get("requests")
    if not isinstance(requests, list) or not requests:
        raise RecordingPlanError(f"{source}: 'requests' must be a non-empty list")

    entries: list[RecordingRequest] = []
    seen: set[str] = set()
    for index, item in enumerate(requests):
        entry = _parse_entry(item, source=source, index=index)
        if entry.name in seen:
            raise RecordingPlanError(
                f"{source}: entry {index} ({entry.name}): duplicate name; each recording "
                "names its own file, so two entries cannot share one"
            )
        seen.add(entry.name)
        entries.append(entry)
    return tuple(entries)


def _parse_entry(item: Any, *, source: str, index: int) -> RecordingRequest:
    where = f"{source}: entry {index}"
    if not isinstance(item, Mapping):
        raise RecordingPlanError(f"{where}: expected an object")

    name = item.get("name")
    if not isinstance(name, str) or not _NAME_PATTERN.match(name):
        raise RecordingPlanError(
            f"{where}: 'name' must match {_NAME_PATTERN.pattern} - it becomes a "
            f"filename inside the fixture directory (got {name!r})"
        )
    where = f"{source}: entry {index} ({name})"

    missing = _REQUIRED_ENTRY_KEYS - set(item)
    if missing:
        raise RecordingPlanError(f"{where}: missing required key(s) {_listed(missing)}")
    unknown = set(item) - _REQUIRED_ENTRY_KEYS - _OPTIONAL_ENTRY_KEYS
    if unknown:
        raise RecordingPlanError(
            f"{where}: unknown key(s) {_listed(unknown)}; a plan says what to record, "
            "never what was recorded - the answer, its token counts and its "
            "provenance come from the provider"
        )

    raw_task = item["task"]
    try:
        task = TaskName(raw_task)
    except ValueError as exc:
        raise RecordingPlanError(
            f"{where}: unknown task {raw_task!r}; known tasks are "
            f"{_listed(t.value for t in TaskName)}"
        ) from exc

    raw_schema = item["schema"]
    if not isinstance(raw_schema, str) or raw_schema not in SCHEMAS:
        raise RecordingPlanError(
            f"{where}: unknown schema {raw_schema!r}; a plan may only name a schema "
            f"registered in app.llm.recording.SCHEMAS ({_listed(SCHEMAS)})"
        )

    inputs = item["inputs"]
    if not isinstance(inputs, Mapping) or not all(isinstance(k, str) for k in inputs):
        raise RecordingPlanError(f"{where}: 'inputs' must be an object with string keys")

    note = item.get("note", "")
    if not isinstance(note, str):
        raise RecordingPlanError(f"{where}: 'note' must be a string")

    temperature = item.get("temperature")
    if temperature is not None and (
        not isinstance(temperature, int | float)
        or isinstance(temperature, bool)
        or not 0.0 <= float(temperature) <= 2.0
    ):
        raise RecordingPlanError(f"{where}: 'temperature' must be a number in [0, 2]")

    return RecordingRequest(
        name=name,
        task=task,
        inputs=dict(inputs),
        schema=SCHEMAS[raw_schema],
        note=note,
        temperature=None if temperature is None else float(temperature),
    )


def _listed(values: Iterable[Any]) -> str:
    return ", ".join(sorted(str(value) for value in values))


# ---------------------------------------------------------------------------
# The outcome: what happened to each one
# ---------------------------------------------------------------------------


class RecordingStatus(StrEnum):
    """What the recorder did with one plan entry."""

    #: A real answer was captured and written.
    RECORDED = "recorded"
    #: A fixture already exists under this request's key. Nothing was called and
    #: nothing was written - see the module docstring, property 2.
    SKIPPED_EXISTING = "skipped_existing"
    #: ``--dry-run``: the key was computed, no provider was called.
    WOULD_RECORD = "would_record"
    #: Something went wrong. ``detail`` says what, redacted.
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class RecordingOutcome:
    """One line of the report, and one row of a test's assertions."""

    request: RecordingRequest
    status: RecordingStatus
    #: The Day-5 fixture key. Known for everything except an entry that failed
    #: before its request could be assembled.
    key: str | None = None
    path: Path | None = None
    #: Human-readable, and passed through the redactor before it gets here.
    detail: str | None = None
    #: The chokepoint's own metadata for a successful recording.
    meta: CallMeta | None = None

    @property
    def ok(self) -> bool:
        return self.status is not RecordingStatus.FAILED


@dataclass(frozen=True, slots=True)
class RecordingReport:
    """Every outcome, plus the two numbers a caller actually acts on."""

    outcomes: tuple[RecordingOutcome, ...] = field(default_factory=tuple)

    def count(self, status: RecordingStatus) -> int:
        return sum(1 for outcome in self.outcomes if outcome.status is status)

    @property
    def failures(self) -> tuple[RecordingOutcome, ...]:
        return tuple(o for o in self.outcomes if o.status is RecordingStatus.FAILED)

    @property
    def ok(self) -> bool:
        return not self.failures


# ---------------------------------------------------------------------------
# The two providers this module wraps around the real one
# ---------------------------------------------------------------------------


class CapturingProvider(LLMProvider):
    """Delegates to a real provider and remembers what went past.

    A wrapper rather than a reimplementation, for the reason the whole module
    exists: the fixture has to be filed under the key the *stub* will compute at
    replay time, and that key comes from the ``CompletionRequest`` the
    chokepoint assembles.  Rebuilding that request here would create a second
    request builder, and the day the two disagree every recording made by this
    tool silently stops being found.
    """

    def __init__(self, inner: LLMProvider) -> None:
        self.name = inner.name
        self._inner = inner
        self.exchanges: list[tuple[CompletionRequest, CompletionResult]] = []

    def model_for(self, tier: ModelTier) -> str:
        return self._inner.model_for(tier)

    async def complete(self, request: CompletionRequest) -> CompletionResult:
        result = await self._inner.complete(request)
        self.exchanges.append((request, result))
        return result

    async def aclose(self) -> None:
        await self._inner.aclose()


class _RequestCaptured(Exception):  # noqa: N818 - control flow, not a failure
    """The probe below has the request it wanted. Not an error condition.

    Deliberately *not* a ``ProviderError``: the router catches those, counts
    them against the circuit breaker and logs "provider failed permanently",
    all of which would be untrue and, at a hundred entries, two hundred
    misleading warnings.  This unwinds straight out instead.
    """


class _RequestProbe(LLMProvider):
    """Captures the assembled request, then stops.

    This is how the fixture key is known *before* any quota is spent, which is
    what makes "skip what is already recorded" free rather than free-after-the-
    fact.  It goes through the real chokepoint - prompt rendering, schema
    derivation, the routing table's sampling settings - so the request it
    captures is byte-for-byte the one the recording call will send.

    It stops rather than returning an empty answer on purpose: a provider that
    invented a response here would be a second, silent source of fake fixtures.

    It is the one ``LLMProvider`` in the project that breaks the contract in
    ``types.py`` by raising something that is not a ``ProviderError``.  That is
    the point - see ``_RequestCaptured`` - and it is safe because this provider
    is never in a router that serves anything: it is constructed, used once and
    dropped, inside ``assemble_request``.
    """

    name = "recording-key-probe"

    def __init__(self, model: str) -> None:
        self._model = model
        self.request: CompletionRequest | None = None

    def model_for(self, tier: ModelTier) -> str:
        return self._model

    async def complete(self, request: CompletionRequest) -> CompletionResult:
        self.request = request
        raise _RequestCaptured

    async def aclose(self) -> None:
        return None


# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------


async def assemble_request(
    entry: RecordingRequest,
    *,
    settings: Settings,
    model: str = STUB_MODEL,
) -> tuple[CompletionRequest, str]:
    """The provider request this entry would send, and its fixture key.

    Costs nothing: no provider is called.  Raises whatever the chokepoint raises
    for a plan that cannot be turned into a request at all - a task with no
    prompt template yet, or inputs the template needs and the plan omitted.
    """
    probe = _RequestProbe(model)
    router = ProviderRouter(
        providers=[probe],
        breaker=CircuitBreaker(threshold=1, cooldown_s=1.0),
        max_attempts_per_provider=1,
    )
    try:
        await call_structured(
            entry.task,
            entry.inputs,
            entry.schema,
            temperature=entry.temperature,
            router=router,
            cache=NullCache(),
            settings=settings.model_copy(update={"llm_cache_enabled": False}),
        )
    except _RequestCaptured:
        # Expected, and the only way out: the probe always stops here. Anything
        # else - a task with no prompt, a missing template input - propagates,
        # because those are real problems with the plan entry.
        pass
    if probe.request is None:  # pragma: no cover - no path reaches this today
        raise RecordingAbortedError(
            f"{entry.name}: no request reached the provider, so no fixture key exists"
        )
    return probe.request, fixture_key(probe.request, model=model)


async def record_request(
    entry: RecordingRequest,
    *,
    router: ProviderRouter,
    store: FixtureStore,
    settings: Settings,
    overwrite: bool = False,
    dry_run: bool = False,
    redactor: Redactor | None = None,
    today: dt.date | None = None,
) -> RecordingOutcome:
    """Record one entry. Returns an outcome; does not raise for a failed entry.

    Failures are values rather than exceptions because this is the unit a batch
    is made of: one unrecordable entry must not cost the other nineteen their
    recordings, and a caller that wants to stop can still look at ``.ok``.
    """
    redactor = redactor if redactor is not None else redactor_for(settings)
    try:
        return await _record_request(
            entry,
            router=router,
            store=store,
            settings=settings,
            overwrite=overwrite,
            dry_run=dry_run,
            today=today or dt.datetime.now(tz=dt.UTC).date(),
        )
    except Exception as exc:  # noqa: BLE001 - a batch tool reports, it does not abort
        # Deliberately broad. A plan entry can fail for reasons that are not
        # ``LLMError`` at all - a missing template input raises ``KeyError``,
        # an unwritable directory raises ``OSError`` - and every one of them is
        # this entry's problem rather than the batch's. Redacted, because a
        # provider error can quote a request header.
        detail = redactor.text(f"{type(exc).__name__}: {exc}")
        log.warning("recording %s failed: %s", entry.name, detail)
        return RecordingOutcome(request=entry, status=RecordingStatus.FAILED, detail=detail)


async def _record_request(
    entry: RecordingRequest,
    *,
    router: ProviderRouter,
    store: FixtureStore,
    settings: Settings,
    overwrite: bool,
    dry_run: bool,
    today: dt.date,
) -> RecordingOutcome:
    # 1. What key would this request have? Free, and asked first, so an entry
    #    that is already recorded costs nothing at all.
    base_request, key = await assemble_request(entry, settings=settings)

    existing = store.get(key)
    if existing is not None and not overwrite:
        return RecordingOutcome(
            request=entry,
            status=RecordingStatus.SKIPPED_EXISTING,
            key=key,
            path=store.path_for(key),
            detail="a fixture already exists for this exact request; "
            "pass overwrite to replace it",
        )

    if dry_run:
        return RecordingOutcome(
            request=entry,
            status=RecordingStatus.WOULD_RECORD,
            key=key,
            detail="dry run: no provider was called and nothing was written",
        )

    # 2. The real call, through the real chokepoint, spending real quota.
    # Every provider is wrapped, not just the first: if the primary is down
    # and the router fails over, the answer that gets recorded must still be the
    # one that was actually given.
    captured = [CapturingProvider(provider) for provider in router.providers]
    recording_router = ProviderRouter(
        providers=captured,
        breaker=router.breaker,
        max_attempts_per_provider=router.max_attempts_per_provider,
        sleep=router.sleep,
        jitter=router.jitter,
    )
    _, meta = await call_structured(
        entry.task,
        entry.inputs,
        entry.schema,
        temperature=entry.temperature,
        router=recording_router,
        # No cache, in both places it could hide: a recording must come from the
        # wire, and a cache hit would record an answer nobody just gave.
        cache=NullCache(),
        settings=settings.model_copy(update={"llm_cache_enabled": False}),
    )

    # 3. Was that a real answer? The provider decided this, not the plan file
    #    and not this function - see the module docstring, property 1.
    _refuse_unrecordable(entry, meta)

    exchanges = [exchange for provider in captured for exchange in provider.exchanges]
    if not exchanges:  # pragma: no cover - the guard above catches this
        raise RecordingAbortedError(
            f"{entry.name}: no request reached the provider, so there is nothing to record"
        )
    # 4. The LAST answer, filed under the FIRST request's key.
    #
    #    They are the same request in the ordinary case, and they differ in
    #    exactly one: the model's first answer did not validate, so the
    #    chokepoint re-asked with repair messages appended. The answer worth
    #    replaying is the one that validated - the last. The key worth filing it
    #    under is the one production will ask for - the first, because a replay
    #    starts at attempt 0 with no repair in it. Keying on the repair request
    #    instead would file a recording that no replay could ever find: a
    #    fixture that silently is not there.
    _, result = exchanges[-1]

    fixture = Fixture(
        key=key,
        description=_description(entry, meta, today),
        text=result.text,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        reasoning_tokens=result.reasoning_tokens,
        finish_reason=result.finish_reason,
        # Provenance: how the real provider was asked to enforce the schema. The
        # label a *replayed* result reports is decided by the stub, never read
        # from here - see ``Fixture.as_result``.
        recorded_structured_mode=meta.structured_mode,
        request_preview={
            "task": entry.task.value,
            "inputs": entry.inputs,
            "prompt_version": meta.prompt_version,
            "prompt_fingerprint": meta.prompt_fingerprint,
            "schema": base_request.json_schema.name,
            "schema_fingerprint": base_request.json_schema.fingerprint,
            "temperature": base_request.temperature,
        },
    )

    # Overwriting replaces the file the recording is already in, rather than
    # writing a second file under a new name: two files with the same
    # ``request_hash`` is a duplicate the store would have to arbitrate.
    filename = entry.filename
    if existing is not None:
        current = store.path_for(key)
        if current is not None:
            filename = current.name

    path = store.save(fixture, filename=filename)
    return RecordingOutcome(
        request=entry,
        status=RecordingStatus.RECORDED,
        key=key,
        path=path,
        meta=meta,
    )


def _refuse_unrecordable(entry: RecordingRequest, meta: CallMeta) -> None:
    """Fail unless a real provider genuinely just answered.

    Three ways an answer can be unreal, all of them silent without this check:
    the stub replayed a fixture, the stub synthesized one from the schema, or
    the response cache served an earlier call.  Writing any of them as a fresh
    recording would put invented or recycled data in the directory that later
    phases treat as ground truth.
    """
    if meta.cache_hit or meta.structured_mode in _UNRECORDABLE_MODES:
        raise RecordingAbortedError(
            f"{entry.name}: the answer came back as structured_mode="
            f"{meta.structured_mode!r} (cache_hit={meta.cache_hit}), which means no "
            "provider produced it. Recording needs a real provider - check "
            "LLM_PROVIDER_ORDER."
        )
    if meta.provider == STUB_PROVIDER_NAME:
        raise RecordingAbortedError(
            f"{entry.name}: the offline stub answered this call, and the stub cannot be "
            "a recording source - it answers from the very directory being recorded into"
        )


def _description(entry: RecordingRequest, meta: CallMeta, today: dt.date) -> str:
    note = f" {entry.note.strip()}" if entry.note.strip() else ""
    return (
        f"{entry.task.value}:{note} "
        f"Recorded from {meta.model} on {today.isoformat()} "
        f"(structured_mode={meta.structured_mode}, schema_retries={meta.schema_retry_count})."
    )


async def record_plan(
    entries: Sequence[RecordingRequest],
    *,
    router: ProviderRouter,
    store: FixtureStore,
    settings: Settings,
    overwrite: bool = False,
    dry_run: bool = False,
    on_outcome: Callable[[RecordingOutcome], None] | None = None,
) -> RecordingReport:
    """Record every entry, in order, continuing past a failure.

    **Sequential, and continuing on failure.**  Both choices are about the thing
    being spent.  Recording is the only operation in this repository that costs
    money and quota, and it is run by hand a handful of times per phase: there
    is nothing to gain from concurrency except a rate-limit storm and a harder-
    to-read report.  And stopping at the first failure would throw away the
    recordings that already succeeded, forcing a re-run that pays for them a
    second time - so each entry is independent, every failure is reported, and
    the caller decides what a partial batch means.

    One router, one breaker, for the whole batch: a provider that is genuinely
    down trips the circuit after a few entries and the rest fail instantly
    instead of each waiting out its own timeout.

    ``on_outcome`` is called with each ``RecordingOutcome`` as it happens, so a
    long batch prints progress rather than going quiet for a minute.
    """
    redactor = redactor_for(settings)
    outcomes: list[RecordingOutcome] = []
    for entry in entries:
        outcome = await record_request(
            entry,
            router=router,
            store=store,
            settings=settings,
            overwrite=overwrite,
            dry_run=dry_run,
            redactor=redactor,
        )
        outcomes.append(outcome)
        if on_outcome is not None:
            on_outcome(outcome)
    return RecordingReport(outcomes=tuple(outcomes))


def redactor_for(settings: Settings) -> Redactor:
    """A redactor that knows *this* process's real secrets.

    A local instance rather than the process-wide logging one: the recorder
    prints to a terminal as well as logging, and a report is exactly the place a
    provider's error message - which may quote the request that carried the key -
    ends up being read and pasted into an issue.
    """
    redactor = Redactor()
    redactor.add_secret(settings.secret_key)
    if settings.nvidia_api_key is not None:
        redactor.add_secret(settings.nvidia_api_key.get_secret_value())
    return redactor


def recordable_providers(settings: Settings) -> tuple[str, ...]:
    """The configured providers that could actually produce a new recording.

    Everything except the stub, which answers from the fixture directory rather
    than from a model. Used to refuse a recording run before it spends a minute
    discovering the same thing one entry at a time.
    """
    return tuple(name for name in settings.llm_providers if name != STUB_PROVIDER_NAME)
