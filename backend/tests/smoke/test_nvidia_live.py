"""The one test that talks to NVIDIA. Opt-in, and never part of `pytest`.

    pytest -m smoke tests/smoke -q -s

Deselected by default (see ``addopts`` in pyproject) because a test suite whose
result depends on somebody else's uptime is not a test suite - it is a status
page. Every deterministic assertion about the LLM layer lives in
``tests/unit/llm/``; this file answers a different question: *is the wire
actually connected?*

Configuration comes from the repository's real ``.env``, loaded explicitly by
absolute path - the same file the API reads, so a passing smoke test says
something about the configuration you actually deploy with. That is also why
the skip decision below asks the settings object rather than ``os.environ``:
the key lives in ``.env``, and a check against the process environment would
silently skip on a correctly configured machine.

The key is never printed - not by this test, not by the settings object
(``SecretStr``), and not by the adapter's error messages.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from app.config import Settings
from app.llm.probe import ProbeAnswer, new_probe_token, probe_llm
from app.llm.runtime import dispose_llm, init_llm

# backend/tests/smoke/this_file.py -> repository root
_DOTENV = Path(__file__).resolve().parents[3] / ".env"


def _live_settings() -> Settings | None:
    """The developer's real configuration, or None if the LLM is not set up.

    ``_env_file`` is passed at construction rather than patched onto
    ``model_config``, so this is unaffected by the autouse fixture in
    ``tests/conftest.py`` that hides the developer's .env from unit tests.
    """
    try:
        return Settings(_env_file=_DOTENV)  # type: ignore[call-arg]
    except ValidationError:
        return None


SETTINGS = _live_settings()

pytestmark = [
    pytest.mark.smoke,
    pytest.mark.skipif(
        SETTINGS is None,
        reason=f"live smoke test: set NVIDIA_API_KEY in {_DOTENV}",
    ),
]


async def test_one_real_call_survives_the_whole_chokepoint():
    """Phase 1 exit gate: one LLM call succeeds through ``call_structured()``.

    A fresh token per run, so neither a cache nor a replay can make this pass.
    """
    assert SETTINGS is not None  # narrowed by the skipif above
    init_llm(SETTINGS)
    token = new_probe_token()
    try:
        answer, meta = await probe_llm(token, settings=SETTINGS)
    finally:
        await dispose_llm()

    # 1. Structured output validated into the requested type.
    assert isinstance(answer, ProbeAnswer)
    assert answer.ok is True

    # 2. The model actually read its input. A well-formed object carrying the
    #    wrong token would mean the plumbing works and the call does not.
    assert answer.echo.strip() == token

    # 3. The response came through the router and the NVIDIA provider.
    assert meta.provider == "nvidia"
    assert meta.model.startswith("nvidia/")
    assert meta.failover_count == 0

    # 4. Real tokens were spent, and accounting saw them.
    assert meta.input_tokens > 0
    assert meta.output_tokens > 0
    assert meta.price_known is True
    assert meta.cache_hit is False

    # 5. Reasoning is off for this task, and no chain of thought leaked out.
    assert meta.reasoning_enabled is False

    # The numbers the plan cares about, for a human to read. The key is not
    # among them, and neither is any model output beyond the echoed token.
    print(
        f"\n  nvidia live probe"
        f"\n    model            {meta.model}"
        f"\n    structured_mode  {meta.structured_mode}"
        f"\n    tokens           {meta.input_tokens} in / {meta.output_tokens} out"
        f"\n    reasoning_tokens {meta.reasoning_tokens}"
        f"\n    schema_retries   {meta.schema_retry_count}"
        f"\n    cost_usd         {meta.cost_usd} (price_known={meta.price_known})"
        f"\n    latency          {meta.latency_ms} ms"
    )
