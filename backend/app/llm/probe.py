"""The smallest possible real call through the chokepoint.

Phase 1's exit gate is "one LLM call succeeds through ``call_structured()``".
This is that call: a fixed token goes out, the same token must come back inside
a validated object.  It exercises every layer - config, router, provider, HTTP,
reasoning stripping, JSON extraction, schema validation, cost accounting -
while asking the model to do essentially nothing, so a failure points at the
plumbing rather than at the prompt.

It is infrastructure, not interview logic.  Nothing in ``interview/`` or
``grading/`` will ever call it.
"""

from __future__ import annotations

import secrets

from pydantic import BaseModel, Field

from app.config import Settings
from app.llm.cache import ResponseCache
from app.llm.client import CallMeta, call_structured
from app.llm.router import ProviderRouter
from app.llm.tasks import TaskName


class ProbeAnswer(BaseModel):
    """Deliberately tiny, and deliberately checkable.

    ``echo`` is the point: a model that returns a well-formed object with the
    wrong token has not read its input, and a probe that cannot tell those apart
    is not a probe.
    """

    ok: bool = Field(description="Always true.")
    echo: str = Field(description="The token from the request, copied exactly.")
    model_said: str = Field(description="The name of the model answering.")


def new_probe_token() -> str:
    """A fresh token per probe, so a cached or replayed answer cannot pass."""
    return secrets.token_hex(4)


async def probe_llm(
    token: str | None = None,
    *,
    settings: Settings | None = None,
    router: ProviderRouter | None = None,
    cache: ResponseCache | None = None,
) -> tuple[ProbeAnswer, CallMeta]:
    """Run the probe. Raises whatever the chokepoint raises.

    ``settings`` is explicit rather than taken from the process cache so the
    probe can be run by a script or a smoke test that loaded configuration from
    a specific ``.env``, without reaching into ``get_settings``' global cache.

    ``router`` and ``cache`` pass straight through to ``call_structured`` for
    the same reason. Day 5 uses them to point the probe at the offline stub, so
    that "does the whole chokepoint work?" can be answered without a network -
    the same question the live smoke test answers with one.
    """
    return await call_structured(
        TaskName.CONNECTIVITY_PROBE,
        {"token": token or new_probe_token()},
        ProbeAnswer,
        settings=settings,
        router=router,
        cache=cache,
    )
