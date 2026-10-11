"""The interview orchestration layer: FSM, event log, session state.

Plan section 13.2 gives this module the FSM, session state, item selection,
theta updates and follow-up policy, and forbids it from knowing about LLM
providers, HTTP or prompts.  Day 21 builds the first part::

    fsm.py       the states and the one transition table          (pure)
    events.py    what each event may carry, validated both ways   (pure)
    state.py     SessionState and fold(events)                    (pure)
    store.py     append under a row lock, read in seq order       (Postgres)
    cache.py     the fold in Redis, monotonic in seq              (Redis)
    service.py   create / append / load, owning the transaction   (both)

Item selection (``app.selection``), ability arithmetic (``app.ability``) and
grading (``app.grading``) are *not* re-implemented here.  The log records their
results - the question chosen, the score, the ability after the update - and
the fold replays those results without re-running the algorithms.

See ``docs/interview-fsm.md`` for the design, invariants and failure behaviour.
"""

from __future__ import annotations

from app.interview.cache import CacheRead, CacheStatus, SessionStateCache, cache_key
from app.interview.events import (
    PAYLOAD_MODELS,
    InvalidEventPayload,
    StoredEvent,
    validate_payload,
)
from app.interview.fsm import (
    TERMINAL_STATES,
    TRANSITIONS,
    EventType,
    IllegalTransition,
    InterviewState,
    allowed_events,
    is_terminal,
    next_state,
)
from app.interview.service import (
    append_event,
    commit_and_cache,
    create_session,
    load_state,
    rebuild_state,
)
from app.interview.state import (
    STATE_SCHEMA_VERSION,
    EventLogCorrupted,
    SessionState,
    apply,
    fold,
)
from app.interview.store import SeqConflict, SessionNotFound, stage_append

__all__ = [
    "PAYLOAD_MODELS",
    "STATE_SCHEMA_VERSION",
    "TERMINAL_STATES",
    "TRANSITIONS",
    "CacheRead",
    "CacheStatus",
    "EventLogCorrupted",
    "EventType",
    "IllegalTransition",
    "InterviewState",
    "InvalidEventPayload",
    "SeqConflict",
    "SessionNotFound",
    "SessionState",
    "SessionStateCache",
    "StoredEvent",
    "allowed_events",
    "append_event",
    "apply",
    "cache_key",
    "commit_and_cache",
    "create_session",
    "fold",
    "is_terminal",
    "load_state",
    "next_state",
    "rebuild_state",
    "stage_append",
    "validate_payload",
]
