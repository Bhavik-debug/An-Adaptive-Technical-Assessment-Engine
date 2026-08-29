"""Recorded provider responses, and the key they are filed under.

Plan section 3, Day 5: *"Offline replay mode (stub provider that serves recorded
fixtures) so every future test runs without API calls."*

**What a fixture is.**  One JSON file holding exactly what a provider returned
for one exact request: the answer text, the token counts, the finish reason.
Replaying it puts a real model's output back into the system without the
network, the key, the quota, or the two-second wait.

**Why the key is derived from the request and not chosen by hand.**  A fixture
filed under a name someone typed is a fixture that silently goes stale: edit the
prompt, and the old recording keeps being served for a request that no longer
matches it.  Hashing the request means a changed prompt is a *different key*,
which is a miss - loud, and correct.

This mirrors ``app/llm/cache.py`` deliberately, and the two are not the same
thing:

|                | response cache        | fixture store              |
|----------------|-----------------------|----------------------------|
| lives in       | Redis, with a TTL     | the repository, in git     |
| keyed on       | task + prompt + inputs| the assembled provider request |
| exists to      | avoid paying twice    | make tests deterministic   |
| written by     | every successful call | a human, deliberately      |

The cache key is computed *above* the provider, from the task and its inputs.
The fixture key is computed *at* the provider, from the request that was
actually assembled - because that, and only that, is what a provider sees.

**Fixtures are committed to git.**  That has one hard consequence, stated here
because it is the only place someone will look for it: **never record a call
whose prompt or response contains candidate data.**  A recording is a permanent,
public, un-redactable copy.  Record the connectivity probe and synthetic
examples; never a real resume, a real answer, or a real grade.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.llm.types import CompletionRequest, CompletionResult

log = logging.getLogger(__name__)

#: Bumped if the recorded shape below ever changes incompatibly, so an old
#: recording is a clean miss rather than a confusing parse error.
FIXTURE_FORMAT_VERSION = 1

#: ``backend/fixtures/llm`` in a checkout, ``/app/fixtures/llm`` in the image.
#: Derived from this package's own location so both work with no configuration.
DEFAULT_FIXTURE_DIR = Path(__file__).resolve().parents[2] / "fixtures" / "llm"


def fixture_key(request: CompletionRequest, *, model: str) -> str:
    """A stable digest of everything that determines the answer.

    Every field that would change what a model returns is in here, and nothing
    that would not.  ``timeout_s`` is excluded on purpose: a deadline changes
    whether you *get* an answer, never which answer you get, and including it
    would invalidate every recording the day someone tunes the timeout.
    """
    canonical = json.dumps(
        {
            "v": FIXTURE_FORMAT_VERSION,
            "model": model,
            "tier": request.tier.value,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "max_output_tokens": request.max_output_tokens,
            "reasoning": {
                "enabled": request.reasoning.enabled,
                "budget_tokens": request.reasoning.budget_tokens,
            },
            "schema": request.json_schema.fingerprint,
            "messages": [
                {"role": message.role.value, "content": message.content}
                for message in request.messages
            ],
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class RecordedError:
    """A recorded *failure*. Replaying one is how an offline test reaches a
    retry, a failover or a circuit breaker without inventing a fake provider.

    ``kind`` names an entry in ``app/llm/errors.py``; the stub maps it back.
    """

    kind: str
    message: str
    status_code: int | None = None
    retry_after_s: float | None = None


@dataclass(frozen=True, slots=True)
class Fixture:
    """One recorded exchange."""

    #: SHA-256 of the assembled request. Serialised as ``request_hash``; see
    #: ``_to_json`` for why the on-disk name differs from the attribute name.
    key: str
    #: Free text, for a human reading the directory. Never parsed.
    description: str
    #: What the provider said. ``None`` when this fixture records a failure.
    text: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    finish_reason: str | None = "stop"
    #: PROVENANCE ONLY - how the *original* provider was asked to enforce the
    #: schema when this was recorded. It is deliberately **not** what a replayed
    #: result reports: see ``as_result``.
    recorded_structured_mode: str = "unknown"
    error: RecordedError | None = None
    #: Kept purely so a human can tell which request a file belongs to. Never
    #: used for matching - the key is the only thing that matches.
    request_preview: dict[str, Any] | None = None

    def as_result(self, *, provider: str, model: str, structured_mode: str) -> CompletionResult:
        """Rebuild the ``CompletionResult``, labelled as a replay.

        ``structured_mode`` is a required argument supplied by the *provider*,
        never read from the recording. That is the fix for a real defect: while
        the fixture chose this value, a recording could set it to
        ``json_schema`` and a replayed answer would then be indistinguishable
        from a live one in ``CallMeta``, on the span, and in the log. The whole
        safety story of the stub rests on that field being trustworthy, so the
        code decides it and the data cannot.
        """
        if self.text is None:  # pragma: no cover - guarded by the stub
            raise ValueError(f"fixture {self.key} records an error, not a response")
        return CompletionResult(
            text=self.text,
            provider=provider,
            model=model,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            # Reasoning text is deliberately never recorded: it is the most
            # content-heavy, least useful part of a response, and a recording is
            # permanent. Only the count survives, which is all cost needs.
            reasoning_text=None,
            reasoning_tokens=self.reasoning_tokens,
            finish_reason=self.finish_reason,
            structured_mode=structured_mode,
        )


def _to_json(fixture: Fixture) -> dict[str, Any]:
    # Serialised as ``request_hash`` rather than ``key``. Two reasons, and the
    # second is the one that bites: it describes what the value actually is, and
    # a field literally named "key" holding 64 high-entropy characters is what
    # every secret scanner is built to flag. `gitleaks` did flag it, which would
    # have turned CI red on the first push over a value that is a SHA-256 of a
    # prompt. Renaming beats allowlisting - an allowlist is how a *real* secret
    # eventually gets ignored.
    payload: dict[str, Any] = {
        "format_version": FIXTURE_FORMAT_VERSION,
        "request_hash": fixture.key,
        "description": fixture.description,
        "text": fixture.text,
        "input_tokens": fixture.input_tokens,
        "output_tokens": fixture.output_tokens,
        "reasoning_tokens": fixture.reasoning_tokens,
        "finish_reason": fixture.finish_reason,
        "recorded_structured_mode": fixture.recorded_structured_mode,
    }
    if fixture.error is not None:
        payload["error"] = {
            "kind": fixture.error.kind,
            "message": fixture.error.message,
            "status_code": fixture.error.status_code,
            "retry_after_s": fixture.error.retry_after_s,
        }
    if fixture.request_preview is not None:
        payload["request_preview"] = fixture.request_preview
    return payload


def _from_json(data: dict[str, Any], *, source: Path) -> Fixture:
    version = data.get("format_version")
    if version != FIXTURE_FORMAT_VERSION:
        raise ValueError(
            f"{source.name}: format_version {version!r}, this build reads "
            f"{FIXTURE_FORMAT_VERSION}"
        )
    raw_error = data.get("error")
    error = (
        RecordedError(
            kind=str(raw_error["kind"]),
            message=str(raw_error.get("message", "")),
            status_code=raw_error.get("status_code"),
            retry_after_s=raw_error.get("retry_after_s"),
        )
        if isinstance(raw_error, dict)
        else None
    )
    return Fixture(
        key=str(data["request_hash"]),
        description=str(data.get("description", "")),
        text=data.get("text"),
        input_tokens=int(data.get("input_tokens", 0)),
        output_tokens=int(data.get("output_tokens", 0)),
        reasoning_tokens=int(data.get("reasoning_tokens", 0)),
        finish_reason=data.get("finish_reason"),
        recorded_structured_mode=str(data.get("recorded_structured_mode", "unknown")),
        error=error,
        request_preview=data.get("request_preview"),
    )


class FixtureStore:
    """The recordings on disk, loaded once and held in memory.

    Loaded eagerly at construction rather than per call: the directory is small,
    a lookup on the hot path should not touch the filesystem, and a malformed
    recording should be reported at boot rather than during an interview.
    """

    def __init__(self, directory: Path | None = None) -> None:
        self.directory = Path(directory) if directory is not None else DEFAULT_FIXTURE_DIR
        self._fixtures: dict[str, Fixture] = {}
        self.reload()

    def reload(self) -> None:
        self._fixtures = {}
        if not self.directory.is_dir():
            log.info("llm fixture directory %s does not exist; replay is empty", self.directory)
            return
        for path in sorted(self.directory.glob("*.json")):
            try:
                fixture = _from_json(json.loads(path.read_text(encoding="utf-8")), source=path)
            except (ValueError, KeyError, TypeError, OSError) as exc:
                # Loud, and not fatal. One unreadable recording must not stop
                # the process; it must be impossible to miss in the log.
                log.warning("ignoring unreadable llm fixture %s: %s", path.name, exc)
                continue
            if fixture.key in self._fixtures:
                log.warning("duplicate llm fixture key in %s; keeping the first", path.name)
                continue
            self._fixtures[fixture.key] = fixture
        log.info("loaded %d llm fixture(s) from %s", len(self._fixtures), self.directory)

    def get(self, key: str) -> Fixture | None:
        return self._fixtures.get(key)

    def add(self, fixture: Fixture) -> None:
        """Register a fixture without writing it to disk. For tests."""
        self._fixtures[fixture.key] = fixture

    def __len__(self) -> int:
        return len(self._fixtures)

    @property
    def keys(self) -> tuple[str, ...]:
        return tuple(self._fixtures)

    def save(self, fixture: Fixture, *, filename: str) -> Path:
        """Write a recording. Used by the recorder script, never at runtime."""
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / filename
        path.write_text(
            json.dumps(_to_json(fixture), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        self._fixtures[fixture.key] = fixture
        return path
