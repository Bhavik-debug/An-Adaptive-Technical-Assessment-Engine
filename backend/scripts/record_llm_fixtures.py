"""Record many real provider responses in one go, from a recording plan.

    cd backend
    python scripts/record_llm_fixtures.py fixtures/recording_plans/connectivity_probe.json --dry-run
    python scripts/record_llm_fixtures.py fixtures/recording_plans/connectivity_probe.json

**This and ``record_llm_fixture.py`` are the only things in the repository that
deliberately spend quota.**  Developer tools, run by hand, never by a test and
never by CI.  Everything else replays what they produced.

All of the thinking lives in ``app/llm/recording.py``; this file is the command
line around it - argument parsing, a progress line per entry, an exit code.  That
split is deliberate: the engine is importable, type-checked and unit-tested,
while a script under ``scripts/`` is none of those things.

**What it does with what it finds.**

* *Already recorded* - skipped, and no provider is called.  The fixture key is
  computed before the call, so re-running a plan is free rather than free after
  the fact.  ``--overwrite`` replaces the existing file, deliberately.
* *Failed* - reported, redacted, and the batch continues.  One unrecordable
  entry must not cost the others their recordings.  The exit code is 1 if any
  entry failed.
* *Not a real answer* - a replayed, synthesized or cached response is refused
  rather than written.  Recording needs a real provider; the offline stub cannot
  be a recording source.

**The safety rule, restated because it matters most.**  A fixture is committed to
git: permanent, and outside the reach of the Day-4 redactor.  **Never record a
call whose prompt or response contains candidate data** - no real resume, no real
answer, no real grade.  Recordings are for the connectivity probe and for
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
from app.llm.recording import (  # noqa: E402
    RecordingOutcome,
    RecordingPlanError,
    RecordingStatus,
    load_recording_plan,
    record_plan,
    recordable_providers,
)
from app.llm.router import build_router  # noqa: E402
from app.obs import configure_logging  # noqa: E402

DOTENV = BACKEND.parent / ".env"

#: Usage or configuration is wrong - nothing was attempted. Distinct from 1,
#: which means the tool ran and some entry could not be recorded.
EXIT_USAGE = 2


def _print_outcome(index: int, total: int, outcome: RecordingOutcome) -> None:
    key = outcome.key[:12] if outcome.key else "-"
    print(f"  [{index}/{total}] {outcome.request.name:<28} {outcome.status.value:<17} {key}")
    if outcome.status is RecordingStatus.RECORDED and outcome.path is not None:
        meta = outcome.meta
        tokens = f"{meta.input_tokens} in / {meta.output_tokens} out" if meta else ""
        model = meta.model if meta else "?"
        print(f"        from {model}, {tokens}")
        print(f"        wrote {outcome.path}")
    elif outcome.detail:
        print(f"        {outcome.detail}")


async def run(
    plan_path: Path,
    *,
    only: tuple[str, ...],
    dry_run: bool,
    overwrite: bool,
    fixture_dir: Path | None,
) -> int:
    settings = Settings(_env_file=DOTENV)  # type: ignore[call-arg]
    configure_logging(settings)

    try:
        entries = load_recording_plan(plan_path)
    except RecordingPlanError as exc:
        print(exc, file=sys.stderr)
        return EXIT_USAGE

    if only:
        by_name = {entry.name: entry for entry in entries}
        unknown = [name for name in only if name not in by_name]
        if unknown:
            print(
                f"{plan_path.name} has no entry named {', '.join(unknown)}; it has "
                f"{', '.join(by_name)}",
                file=sys.stderr,
            )
            return EXIT_USAGE
        entries = tuple(by_name[name] for name in only)

    # A dry run never calls a provider, so it is allowed with any configuration -
    # including a checkout with no credential at all, which is how the plan
    # format gets checked without an account.
    if not dry_run and not recordable_providers(settings):
        print(
            f"LLM_PROVIDER_ORDER is {settings.llm_provider_order!r}. Recording needs a real "
            "provider - the offline stub answers from the very directory being recorded "
            "into. Set LLM_PROVIDER_ORDER=nvidia in .env and retry.",
            file=sys.stderr,
        )
        return EXIT_USAGE

    directory = fixture_dir or (
        Path(settings.llm_stub_fixture_dir) if settings.llm_stub_fixture_dir else None
    )
    store = FixtureStore(directory)
    router = build_router(settings)

    print(f"  plan            {plan_path}")
    print(f"  entries         {len(entries)}")
    print(f"  provider order  {settings.llm_provider_order}")
    print(f"  fixture dir     {store.directory}")
    print(f"  mode            {'dry run' if dry_run else 'recording'}")
    print(f"  existing        {'OVERWRITE' if overwrite else 'skip'}")
    print()

    counter = {"n": 0}

    def report(outcome: RecordingOutcome) -> None:
        counter["n"] += 1
        _print_outcome(counter["n"], len(entries), outcome)

    try:
        result = await record_plan(
            entries,
            router=router,
            store=store,
            settings=settings,
            overwrite=overwrite,
            dry_run=dry_run,
            on_outcome=report,
        )
    finally:
        await router.aclose()

    print()
    for status in RecordingStatus:
        count = result.count(status)
        if count:
            print(f"  {status.value:<17} {count}")

    if result.count(RecordingStatus.RECORDED):
        print("\n  READ what was written before committing: a fixture is permanent")
        print("  and un-redactable, and it lands in git exactly as recorded.")
    if not result.ok:
        names = ", ".join(outcome.request.name for outcome in result.failures)
        print(f"\n  {len(result.failures)} entr(y/ies) failed: {names}", file=sys.stderr)
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("plan", type=Path, help="a recording plan JSON file")
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        metavar="NAME",
        help="record just this entry; repeatable",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="compute every fixture key and report what would happen; call nothing",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="re-record entries that already have a fixture, replacing the file",
    )
    parser.add_argument(
        "--fixture-dir",
        type=Path,
        default=None,
        help="write into this directory instead of the configured one",
    )
    args = parser.parse_args()
    return asyncio.run(
        run(
            args.plan,
            only=tuple(args.only),
            dry_run=args.dry_run,
            overwrite=args.overwrite,
            fixture_dir=args.fixture_dir,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
