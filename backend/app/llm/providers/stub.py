"""The deterministic offline provider - plan section 3, Day 5.

**What it is.**  A real provider adapter, implementing the same ``LLMProvider``
interface as ``NvidiaProvider``, selected the same way, sitting behind the same
router.  It simply answers from the repository instead of from the network.

    LLM_PROVIDER_ORDER=stub    # no API key, no quota, no network, no waiting

**Why it exists.**  A test suite whose result depends on somebody else's uptime
is not a test suite, it is a status page.  Every phase after this one - the
retrieval evals, the grader's QWK measurement, the FSM's turn loop, the injection
suite - needs to run hundreds of LLM calls per commit, deterministically, for
free.  That is only possible if there is a provider that always answers the same
way.  Plan section 12.3 makes the same point about CI: *"CI replays from cache;
only CHANGED prompts hit the API."*

**Two ways it can answer, and the difference matters enormously:**

* **Replay** - a recorded fixture exists for this exact request, and its text is
  returned verbatim.  This is a *real* model response, captured once.  Semantics
  are real: if the recording said the candidate covered three concepts, that is
  what a real model actually said.

* **Synthesis** - no recording exists, so a shape-correct object is derived from
  the request's own JSON Schema, seeded by the request hash.  Deterministic, and
  **semantically meaningless**.  It proves plumbing works.  It proves nothing
  about whether an answer is any good.

Confusing those two would be the worst possible outcome of this file, so they
are impossible to confuse from the outside: ``structured_mode`` is
``stub_replay`` or ``stub_synthesized``, which lands in ``CallMeta``, on the
span, and in the log line for every call.  A grading eval that quietly ran on
synthesized labels is then one query away from being spotted.

The default on a miss is configurable, and the default default is to *raise*:
a test that expected a recording and got invented data should fail loudly.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Mapping
from enum import StrEnum
from typing import Any

from app.llm.errors import (
    ProviderAuthError,
    ProviderBadRequestError,
    ProviderError,
    ProviderRateLimitedError,
    ProviderResponseError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)
from app.llm.fixtures import Fixture, FixtureStore, RecordedError, fixture_key
from app.llm.types import (
    CompletionRequest,
    CompletionResult,
    LLMProvider,
    ModelTier,
)

log = logging.getLogger(__name__)

#: One model id for every tier, mirroring ``NvidiaProvider``. It is in the
#: ``PRICES`` table at zero cost with a note, so cost arithmetic still runs and
#: ``price_known`` stays true - "free because it is a stub", not "unknown".
STUB_MODEL = "stub/deterministic-v1"

#: The value that reaches ``CallMeta.structured_mode`` and every span.
MODE_REPLAY = "stub_replay"
MODE_SYNTHESIZED = "stub_synthesized"

#: Prefixed onto every synthesized string so a synthesized value is recognisable
#: on sight, in a log, in a database row, or in a report someone screenshots.
SYNTHETIC_MARKER = "stub"


class MissingFixturePolicy(StrEnum):
    """What to do when no recording matches the request."""

    #: Raise. The right default for a suite that means to replay.
    STRICT = "strict"
    #: Derive a shape-correct object from the schema. The right default for
    #: developing a new phase before any recording of it exists.
    SYNTHESIZE = "synthesize"


class MissingFixtureError(ProviderResponseError):
    """No recording matched, and the policy said not to invent one.

    A ``ProviderError`` subclass rather than a bare exception so the router
    treats it like any other provider failure: non-retryable, fail over to the
    next provider if one is configured. An offline suite with only the stub
    configured then gets ``AllProvidersFailedError`` naming this reason, which
    is a readable way to be told "you need to record that call".
    """


#: Recorded ``error.kind`` -> the exception the router will see. Keeping this
#: explicit, rather than looking classes up by name, means a fixture cannot
#: name an arbitrary importable object.
_ERROR_KINDS: Mapping[str, type[ProviderError]] = {
    "rate_limited": ProviderRateLimitedError,
    "timeout": ProviderTimeoutError,
    "unavailable": ProviderUnavailableError,
    "auth": ProviderAuthError,
    "bad_request": ProviderBadRequestError,
    "response": ProviderResponseError,
}


class StubProvider(LLMProvider):
    """Answers from recordings, or from the schema. Never from the network."""

    name = "stub"

    def __init__(
        self,
        *,
        store: FixtureStore | None = None,
        on_missing: MissingFixturePolicy = MissingFixturePolicy.STRICT,
        model: str = STUB_MODEL,
    ) -> None:
        self._store = store if store is not None else FixtureStore()
        self._on_missing = on_missing
        self._model = model
        #: Counters a test can assert on without reaching into internals.
        self.replayed = 0
        self.synthesized = 0

    # -- LLMProvider -------------------------------------------------------

    def model_for(self, tier: ModelTier) -> str:
        """One model for every tier.

        Deliberately *not* tier-dependent. Making the stub's model id vary by
        tier would change the fixture key per tier, so the same recorded answer
        would have to be recorded three times - and the stub is not modelling
        capability differences, it is removing the network.
        """
        return self._model

    async def complete(self, request: CompletionRequest) -> CompletionResult:
        key = fixture_key(request, model=self._model)
        fixture = self._store.get(key)

        if fixture is not None:
            if fixture.error is not None:
                raise _rebuild_error(fixture.error, provider=self.name)
            self.replayed += 1
            log.debug("llm stub replayed fixture %s", key[:12])
            return fixture.as_result(
                provider=self.name, model=self._model, structured_mode=MODE_REPLAY
            )

        if self._on_missing is MissingFixturePolicy.STRICT:
            raise MissingFixtureError(
                f"no recorded fixture for this request (key {key}); record one with "
                "`python -m scripts.record_llm_fixture`, add one to the fixture "
                "directory, or set LLM_STUB_ON_MISSING=synthesize to accept a "
                "shape-correct placeholder",
                provider=self.name,
            )

        self.synthesized += 1
        text = json.dumps(synthesize(request.json_schema.schema, seed=key), separators=(",", ":"))
        log.debug("llm stub synthesized a response for key %s", key[:12])
        return CompletionResult(
            text=text,
            provider=self.name,
            model=self._model,
            # Deterministic, and roughly proportional to the real thing, so a
            # cost calculation in a test is exercising real arithmetic rather
            # than multiplying by zero. A token is ~4 characters.
            input_tokens=sum(len(m.content) for m in request.messages) // 4,
            output_tokens=len(text) // 4,
            reasoning_tokens=0,
            finish_reason="stop",
            structured_mode=MODE_SYNTHESIZED,
        )

    async def aclose(self) -> None:
        return None

    # -- introspection, for /readyz and for tests --------------------------

    @property
    def fixture_count(self) -> int:
        return len(self._store)

    @property
    def on_missing(self) -> MissingFixturePolicy:
        return self._on_missing

    def add_fixture(self, fixture: Fixture) -> None:
        """Register a recording in memory. For tests and the recorder."""
        self._store.add(fixture)

    def key_for(self, request: CompletionRequest) -> str:
        """The key this request would be filed under. For the recorder."""
        return fixture_key(request, model=self._model)


def _rebuild_error(recorded: RecordedError, *, provider: str) -> ProviderError:
    error_type = _ERROR_KINDS.get(recorded.kind)
    if error_type is None:
        return ProviderResponseError(
            f"fixture recorded an unknown error kind {recorded.kind!r}", provider=provider
        )
    return error_type(
        recorded.message,
        provider=provider,
        status_code=recorded.status_code,
        retry_after_s=recorded.retry_after_s,
    )


# ---------------------------------------------------------------------------
# Deterministic synthesis from a JSON Schema
# ---------------------------------------------------------------------------
#
# The schema handed to a provider has already been through ``_harden`` in
# ``structured.py``: every object is closed, every property is required, and an
# optional field is expressed as ``anyOf: [{...}, {"type": "null"}]`` rather
# than being absent. So the walk below only has to handle the shapes pydantic
# actually emits, which is a much smaller problem than "any JSON Schema".


def synthesize(schema: Mapping[str, Any], *, seed: str) -> Any:
    """A deterministic, schema-valid value.

    Same schema and same seed always give the same value - that is the whole
    point, and it is what makes a test that runs this repeatable.

    It is *shape* correct and *semantically* meaningless. Nothing here knows
    what a field means; a ``score`` field gets a number that satisfies the type
    and nothing more. Any test that cares what the value *says* needs a recorded
    fixture, not this.
    """
    return _build(schema, schema, seed, "", 0)


#: Deep enough for the schemas this project has (grade -> concepts -> evidence)
#: and bounded so a self-referential ``$ref`` cannot spin forever.
_MAX_DEPTH = 12


def _build(
    node: Mapping[str, Any],
    root: Mapping[str, Any],
    seed: str,
    path: str,
    depth: int,
) -> Any:
    if depth > _MAX_DEPTH:
        return None

    resolved = _resolve(node, root)

    if "const" in resolved:
        return resolved["const"]
    enum = resolved.get("enum")
    if isinstance(enum, list) and enum:
        # First value, not a hashed choice: stable across schema edits that only
        # append, and it makes a synthesized object predictable to read.
        return enum[0]

    if "anyOf" in resolved or "oneOf" in resolved:
        branches = resolved.get("anyOf") or resolved.get("oneOf") or []
        chosen = _first_non_null(branches, root)
        if chosen is None:
            return None
        return _build(chosen, root, seed, path, depth + 1)

    kind = resolved.get("type")
    if isinstance(kind, list):
        kind = next((k for k in kind if k != "null"), "null")

    if kind == "object":
        properties = resolved.get("properties")
        if not isinstance(properties, dict):
            return {}
        return {
            name: _build(sub, root, seed, f"{path}.{name}", depth + 1)
            for name, sub in properties.items()
        }
    if kind == "array":
        items = resolved.get("items")
        count = max(int(resolved.get("minItems", 1) or 1), 1)
        if not isinstance(items, dict):
            return []
        return [_build(items, root, seed, f"{path}[{index}]", depth + 1) for index in range(count)]
    if kind == "boolean":
        return True
    if kind == "integer":
        return _bounded_int(resolved, seed, path)
    if kind == "number":
        return _bounded_number(resolved, seed, path)
    if kind == "null":
        return None
    # Unknown or missing type: a string is the safest guess, and pydantic will
    # reject it loudly if the guess was wrong, which is the behaviour we want.
    return _marked_string(resolved, seed, path)


def _resolve(node: Mapping[str, Any], root: Mapping[str, Any]) -> Mapping[str, Any]:
    """Follow ``$ref`` into ``$defs``. Pydantic emits one per nested model."""
    ref = node.get("$ref")
    if not isinstance(ref, str) or not ref.startswith("#/"):
        return node
    target: Any = root
    for part in ref.removeprefix("#/").split("/"):
        if not isinstance(target, Mapping) or part not in target:
            return node
        target = target[part]
    return target if isinstance(target, Mapping) else node


def _first_non_null(branches: Any, root: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """The real branch of an optional field.

    ``str | None`` becomes ``anyOf: [{"type": "string"}, {"type": "null"}]``.
    Choosing the non-null branch means a synthesized object exercises the field
    rather than leaving it empty, which is the more useful default: a test that
    wants to see a null can record a fixture.
    """
    if not isinstance(branches, list):
        return None
    for branch in branches:
        if isinstance(branch, Mapping) and _resolve(branch, root).get("type") != "null":
            return branch
    return None


def _digest(seed: str, path: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}{path}".encode()).digest()[:8], "big")


def _bounded_int(schema: Mapping[str, Any], seed: str, path: str) -> int:
    low = schema.get("minimum", schema.get("exclusiveMinimum"))
    high = schema.get("maximum", schema.get("exclusiveMaximum"))
    low_int = int(low) if isinstance(low, int | float) else 0
    high_int = int(high) if isinstance(high, int | float) else low_int + 100
    if high_int <= low_int:
        return low_int
    return low_int + _digest(seed, path) % (high_int - low_int + 1)


def _bounded_number(schema: Mapping[str, Any], seed: str, path: str) -> float:
    low = schema.get("minimum", schema.get("exclusiveMinimum"))
    high = schema.get("maximum", schema.get("exclusiveMaximum"))
    low_f = float(low) if isinstance(low, int | float) else 0.0
    high_f = float(high) if isinstance(high, int | float) else low_f + 1.0
    if high_f <= low_f:
        return low_f
    fraction = (_digest(seed, path) % 10_000) / 10_000
    return round(low_f + fraction * (high_f - low_f), 4)


def _marked_string(schema: Mapping[str, Any], seed: str, path: str) -> str:
    """A string that announces what it is.

    ``stub:.concepts[0].evidence:9f3a1c`` rather than ``"lorem ipsum"``. If one
    of these ever turns up in a report, a database row or a screenshot, its
    origin is obvious and nobody has to work out whether the system produced
    something real.
    """
    value = f"{SYNTHETIC_MARKER}:{path or 'root'}:{_digest(seed, path):x}"[:64]
    minimum = schema.get("minLength")
    if isinstance(minimum, int) and len(value) < minimum:
        value = value.ljust(minimum, "x")
    maximum = schema.get("maxLength")
    if isinstance(maximum, int) and maximum > 0:
        value = value[:maximum]
    return value
