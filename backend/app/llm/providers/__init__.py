"""Provider adapters. One module per vendor; nothing else imports them directly.

``stub`` is not a vendor - it is the offline provider from Day 5, which answers
from recordings in the repository. It lives here rather than in ``tests/``
because it is a shipped, configuration-selectable component that sits behind the
same router as everything else.
"""

from app.llm.providers.nvidia import NvidiaProvider, StructuredMode
from app.llm.providers.stub import (
    MissingFixtureError,
    MissingFixturePolicy,
    StubProvider,
)

__all__ = [
    "MissingFixtureError",
    "MissingFixturePolicy",
    "NvidiaProvider",
    "StructuredMode",
    "StubProvider",
]
