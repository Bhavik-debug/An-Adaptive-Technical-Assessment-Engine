"""Record one named real provider response so the stub can replay it offline.

    cd backend
    python scripts/record_llm_fixture.py probe             # record the probe
    python scripts/record_llm_fixture.py probe --dry-run   # show the key, call nothing
    python scripts/record_llm_fixture.py probe --overwrite # replace an existing recording

**This and ``record_llm_fixtures.py`` (plural - a whole plan file at a time) are
the only things in the repository that deliberately spend quota.**  Developer
tools, run by hand, never by a test and never by CI.  Everything else replays
what they produced.

**Why this is a wrapper rather than a reimplementation.**  The fixture has to be
filed under the key the *stub* will compute at replay time, and that key is
derived from the ``CompletionRequest`` the chokepoint assembles - messages,
schema fingerprint, temperature, the lot.  Rebuilding that request here would
mean two places that construct requests, which would drift.  So the call goes
through the real ``call_structured()``, a thin ``LLMProvider`` wraps the real
NVIDIA adapter and captures the request on its way past, and what gets recorded
is exactly what production sends.

Day 6 moved all of that into ``app/llm/recording.py``, where it is importable,
type-checked and unit-tested, and where the bulk recorder uses the identical
path.  This file is now the single-recipe command line over the same engine:
one implementation, two front doors.

**One behaviour changed on Day 6**, deliberately: a recipe whose fixture already
exists is *skipped* rather than silently re-recorded.  A real recording is
permanent and expensive; replacing one is now something you ask for with
``--overwrite``.

**The safety rule, restated because it matters most.**  A fixture is committed to
git: permanent, and outside the reach of the Day-4 redactor. **Never record a
call whose prompt or response contains candidate data** - no real resume, no real
answer, no real grade. Recordings are for the connectivity probe and for
synthetic examples authored for the purpose.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:  # so `python scripts/...` works without install
    sys.path.insert(0, str(BACKEND))

from app.config import Settings  # noqa: E402
from app.llm.fixtures import FixtureStore  # noqa: E402
from app.llm.probe import ProbeAnswer  # noqa: E402
from app.llm.recording import (  # noqa: E402
    RecordingRequest,
    RecordingStatus,
    record_request,
    recordable_providers,
)
from app.llm.router import build_router  # noqa: E402
from app.llm.tasks import TaskName  # noqa: E402
from app.obs import configure_logging  # noqa: E402

DOTENV = BACKEND.parent / ".env"

#: Usage or configuration is wrong - nothing was attempted.
EXIT_USAGE = 2

#: Fixed, not random. ``probe_llm`` normally mints a fresh token per call so a
#: cached or replayed answer cannot fake a live one - which is exactly right for
#: the live smoke test and exactly wrong here, because a recording keyed on a
#: random token could never be replayed.
REPLAY_TOKEN = "replay01"  # noqa: S105 - an echo token for a health probe, not a credential

#: The named recipes. The same thing a recording plan expresses as JSON - see
#: ``fixtures/recording_plans/connectivity_probe.json``, which records this exact
#: call through the same engine and therefore under the same key.
RECIPES: dict[str, RecordingRequest] = {
    "probe": RecordingRequest(
        name="connectivity_probe",
        task=TaskName.CONNECTIVITY_PROBE,
        inputs={"token": REPLAY_TOKEN},
        schema=ProbeAnswer,
        note="the Day-5 offline replay of the Phase-1 exit-gate call.",
    ),
}


async def record(recipe_name: str, *, dry_run: bool, overwrite: bool) -> int:
    entry = RECIPES[recipe_name]
    settings = Settings(_env_file=DOTENV)  # type: ignore[call-arg]
    configure_logging(settings)

    if not dry_run and not recordable_providers(settings):
        print(
            f"LLM_PROVIDER_ORDER is {settings.llm_provider_order!r}. Recording needs a real "
            "provider - the offline stub answers from the very directory being recorded "
            "into. Set LLM_PROVIDER_ORDER=nvidia in .env and retry.",
            file=sys.stderr,
        )
        return EXIT_USAGE

    store = FixtureStore(
        Path(settings.llm_stub_fixture_dir) if settings.llm_stub_fixture_dir else None
    )
    router = build_router(settings)
    try:
        outcome = await record_request(
            entry,
            router=router,
            store=store,
            settings=settings,
            overwrite=overwrite,
            dry_run=dry_run,
        )
    finally:
        await router.aclose()

    print(f"  recipe         {recipe_name}")
    print(f"  task           {entry.task.value}")
    print(f"  fixture key    {outcome.key}")
    print(f"  status         {outcome.status.value}")
    if outcome.meta is not None:
        print(f"  recorded from  {outcome.meta.model}")
        print(f"  tokens         {outcome.meta.input_tokens} in / {outcome.meta.output_tokens} out")
    if outcome.detail:
        print(f"  detail         {outcome.detail}")

    if outcome.status is RecordingStatus.RECORDED:
        print(f"\n  wrote {outcome.path}")
        print("  READ IT before committing: a fixture is permanent and un-redactable.")
    if outcome.status is RecordingStatus.SKIPPED_EXISTING:
        print(f"\n  left {outcome.path} alone. Pass --overwrite to replace it.")
    return 0 if outcome.ok else 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("recipe", choices=sorted(RECIPES), help="which call to record")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="compute the fixture key and report what would happen; call nothing",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="re-record even if a fixture for this exact request already exists",
    )
    args = parser.parse_args()
    return asyncio.run(record(args.recipe, dry_run=args.dry_run, overwrite=args.overwrite))


if __name__ == "__main__":
    raise SystemExit(main())
