"""Record a real provider response so the stub can replay it offline.

    cd backend
    python scripts/record_llm_fixture.py probe          # record the probe
    python scripts/record_llm_fixture.py probe --dry-run # show the key, call nothing

**This is the only thing in the repository that deliberately spends quota.**  It
is a developer tool, run by hand, never by a test and never by CI.  Everything
else replays what it produced.

**How it works, and why it is a wrapper rather than a reimplementation.**  The
fixture has to be filed under the key the *stub* will compute at replay time,
and that key is derived from the ``CompletionRequest`` the chokepoint assembles -
messages, schema fingerprint, temperature, the lot.  Rebuilding that request here
would mean two places that construct requests, which would drift.  So instead a
thin ``LLMProvider`` wraps the real NVIDIA adapter, the call goes through the
real ``call_structured()``, and the wrapper captures the request on its way past.
What gets recorded is therefore exactly what production sends.

**The safety rule, restated because it matters most.**  A fixture is committed to
git: permanent, and outside the reach of the Day-4 redactor. **Never record a
call whose prompt or response contains candidate data** - no real resume, no real
answer, no real grade. Recordings are for the connectivity probe and for
synthetic examples authored for the purpose.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import sys
from pathlib import Path
from typing import Any

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:  # so `python scripts/...` works without install
    sys.path.insert(0, str(BACKEND))

from app.config import Settings  # noqa: E402
from app.llm.client import call_structured  # noqa: E402
from app.llm.fixtures import Fixture, FixtureStore, fixture_key  # noqa: E402
from app.llm.probe import ProbeAnswer  # noqa: E402
from app.llm.providers.stub import STUB_MODEL  # noqa: E402
from app.llm.router import CircuitBreaker, ProviderRouter, _build_provider  # noqa: E402
from app.llm.tasks import TaskName  # noqa: E402
from app.llm.types import (  # noqa: E402
    CompletionRequest,
    CompletionResult,
    LLMProvider,
    ModelTier,
)
from app.obs import configure_logging  # noqa: E402

DOTENV = BACKEND.parent / ".env"

#: Fixed, not random. ``probe_llm`` normally mints a fresh token per call so a
#: cached or replayed answer cannot fake a live one - which is exactly right for
#: the live smoke test and exactly wrong here, because a recording keyed on a
#: random token could never be replayed.
REPLAY_TOKEN = "replay01"  # noqa: S105 - an echo token for a health probe, not a credential


class _CapturingProvider(LLMProvider):
    """Delegates to a real provider and remembers what went past."""

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


RECIPES: dict[str, dict[str, Any]] = {
    "probe": {
        "task": TaskName.CONNECTIVITY_PROBE,
        "inputs": {"token": REPLAY_TOKEN},
        "schema": ProbeAnswer,
        "filename": "connectivity_probe.json",
        "note": "the Day-5 offline replay of the Phase-1 exit-gate call",
    },
}


async def record(recipe_name: str, *, dry_run: bool) -> int:
    recipe = RECIPES[recipe_name]
    settings = Settings(_env_file=DOTENV)  # type: ignore[call-arg]
    configure_logging(settings)

    if "nvidia" not in settings.llm_providers:
        print(
            f"LLM_PROVIDER_ORDER is {settings.llm_provider_order!r}. Recording needs "
            "the real provider - set LLM_PROVIDER_ORDER=nvidia in .env and retry.",
            file=sys.stderr,
        )
        return 2

    provider = _CapturingProvider(_build_provider("nvidia", settings))
    router = ProviderRouter(
        providers=[provider],
        breaker=CircuitBreaker(threshold=1, cooldown_s=1.0),
        max_attempts_per_provider=settings.llm_max_attempts_per_provider,
    )
    store = FixtureStore(
        Path(settings.llm_stub_fixture_dir) if settings.llm_stub_fixture_dir else None
    )

    try:
        answer, meta = await call_structured(
            recipe["task"],
            recipe["inputs"],
            recipe["schema"],
            router=router,
            # No cache: a recording must come from the wire, not from Redis.
            cache=None,
            settings=settings.model_copy(update={"llm_cache_enabled": False}),
        )
    finally:
        await router.aclose()

    if not provider.exchanges:  # pragma: no cover - only if the call was cached
        print("no request reached the provider; nothing recorded", file=sys.stderr)
        return 1

    # The LAST exchange, not the first: if the model needed a schema repair, the
    # answer that validated is the one worth replaying.
    request, result = provider.exchanges[-1]
    # Keyed with the STUB's model id, because the stub is what will look it up.
    # Provenance - which real model actually produced this - goes in the
    # description, where a human reads it.
    key = fixture_key(request, model=STUB_MODEL)
    today = dt.datetime.now(tz=dt.UTC).date().isoformat()

    fixture = Fixture(
        key=key,
        description=(
            f"{recipe['task'].value}: {recipe['note']}. "
            f"Recorded from {meta.model} on {today} "
            f"(structured_mode={meta.structured_mode}, "
            f"schema_retries={meta.schema_retry_count})."
        ),
        text=result.text,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        reasoning_tokens=result.reasoning_tokens,
        finish_reason=result.finish_reason,
        # Provenance: how NVIDIA was asked to enforce the schema. The label a
        # replayed result reports is decided by the stub, not by this file.
        recorded_structured_mode=meta.structured_mode,
        request_preview={
            "task": recipe["task"].value,
            "inputs": recipe["inputs"],
            "prompt_version": meta.prompt_version,
            "prompt_fingerprint": meta.prompt_fingerprint,
            "schema": request.json_schema.name,
            "schema_fingerprint": request.json_schema.fingerprint,
            "temperature": request.temperature,
        },
    )

    print(f"  task           {recipe['task'].value}")
    print(f"  recorded from  {meta.model}")
    print(f"  fixture key    {key}")
    print(f"  tokens         {result.input_tokens} in / {result.output_tokens} out")
    print(f"  answer         {answer!r}")

    if dry_run:
        print("\n  --dry-run: nothing written")
        return 0

    path = store.save(fixture, filename=str(recipe["filename"]))
    print(f"\n  wrote {path}")
    print("  READ IT before committing: a fixture is permanent and un-redactable.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recipe", choices=sorted(RECIPES), help="which call to record")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="make the call and print the key, but write nothing",
    )
    args = parser.parse_args()
    return asyncio.run(record(args.recipe, dry_run=args.dry_run))


if __name__ == "__main__":
    raise SystemExit(main())
