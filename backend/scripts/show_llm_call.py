"""Make one LLM call and print everything the system knows about it.

    cd <repo root>
    .venv\\Scripts\\python.exe backend/scripts/show_llm_call.py           # uses .env
    .venv\\Scripts\\python.exe backend/scripts/show_llm_call.py --stub    # force offline
    .venv\\Scripts\\python.exe backend/scripts/show_llm_call.py --nvidia  # force live

A hand-verification tool, not a test. It exists because "show me one LLM call
working" is the single most useful thing to be able to do by hand, and the
answer involves a validated object *plus* a metadata record - which is more
than fits on one command line.

``--stub`` needs no API key and makes no network call. ``--nvidia`` makes a real
call and spends quota, so it uses a fresh random token: a cached or replayed
answer could not possibly satisfy it.

Run from the repository root, not from ``backend/`` - ``.env`` lives at the root
and pydantic-settings resolves it relative to the working directory.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app.config import Settings, get_settings  # noqa: E402
from app.llm.client import call_structured  # noqa: E402
from app.llm.probe import ProbeAnswer, new_probe_token  # noqa: E402
from app.llm.router import build_router  # noqa: E402
from app.llm.tasks import TaskName  # noqa: E402

#: The token the shipped recording was made with. Replay only works for this one.
REPLAY_TOKEN = "replay01"  # noqa: S105 - an echo token for a health probe

#: The plan section 14.2 attribute set, in the order a human wants to read it.
FIELDS = (
    "task",
    "provider",
    "model",
    "tier",
    "prompt_version",
    "structured_mode",
    "input_tokens",
    "output_tokens",
    "reasoning_tokens",
    "cost_usd",
    "price_known",
    "cache_hit",
    "schema_retry_count",
    "failover_count",
    "latency_ms",
)

MODE_MEANING = {
    "stub_replay": "a REAL recorded model answer, replayed offline",
    "stub_synthesized": "INVENTED from the schema - shape-correct, meaningless",
    "cache": "served from the Redis response cache",
    "json_schema": "a live call the endpoint schema-constrained",
    "json_object": "a live call constrained only to 'valid JSON'",
    "prompt_only": "a live call with nothing enforced but our own validation",
}


async def main(force: str | None, quiet: bool) -> int:
    if quiet:
        logging.disable(logging.CRITICAL)

    settings = get_settings()
    if force:
        settings = Settings(**{**settings.model_dump(), "llm_provider_order": force})

    router = build_router(settings)
    provider = router.provider_names[0]
    offline = provider == "stub"

    # A fresh token against a live provider: nothing cached or recorded could
    # answer it. The fixed one against the stub: it is what was recorded.
    token = REPLAY_TOKEN if offline else new_probe_token()

    print(f"  provider order   {settings.llm_provider_order}")
    print(f"  provider class   {type(router.providers[0]).__name__}")
    print(f"  api key present  {settings.nvidia_api_key is not None}")
    print(f"  token sent       {token!r}" + ("  (the recorded one)" if offline else "  (fresh)"))
    print()

    try:
        answer, meta = await call_structured(
            TaskName.CONNECTIVITY_PROBE,
            {"token": token},
            ProbeAnswer,
            router=router,
            cache=None,
            settings=settings,
        )
    finally:
        await router.aclose()

    print(f"  STRUCTURED RESPONSE  (a validated {type(answer).__name__}, not a string)")
    print(f"    ok           {answer.ok!r}")
    echoed = "   <- the model echoed it back" if answer.echo == token else "   <- MISMATCH"
    print(f"    echo         {answer.echo!r}{echoed}")
    print(f"    model_said   {answer.model_said!r}")
    print()
    print("  CallMeta  (plan section 14.2)")
    for field in FIELDS:
        print(f"    {field:<20} {getattr(meta, field)}")

    meaning = MODE_MEANING.get(meta.structured_mode, "unknown mode")
    print()
    print(f"  structured_mode={meta.structured_mode!r} means: {meaning}")
    return 0


def cli() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--stub", action="store_true", help="force the offline provider (no key, no network)"
    )
    group.add_argument("--nvidia", action="store_true", help="force a real call (spends quota)")
    parser.add_argument("--verbose", action="store_true", help="show the JSON log lines too")
    args = parser.parse_args()

    force = "stub" if args.stub else "nvidia" if args.nvidia else None
    return asyncio.run(main(force, quiet=not args.verbose))


if __name__ == "__main__":
    raise SystemExit(cli())
