# The interview FSM and its event log (Day 21)

Plan §3, Day 21: *"The interview FSM (`CREATED → PREPARING → ASKING → GRADING →
PROBING/SELECTING → REPORTING → COMPLETED`) over an append-only event log.
Session state = `fold(events)`, cached in Redis keyed by event `seq`."*

This document explains what was built, why each rule exists, and how it fails.
Code: `backend/app/interview/`. Migration: `backend/migrations/versions/0003_interview_events_append_only.py`.

---

## 1. The idea in one paragraph

A session is **not** a row that gets edited. It is a list of facts that only
ever grows: *created, preparing, asked dsa-001, answer submitted, graded 0.4…*.
Each fact is an **event**, stored in Postgres with a per-session position `seq`
(0, 1, 2, …). The *current* state is computed by starting from nothing and
applying every event in order: `state = fold(events)`. Because the events are
never changed, the same events always give the same state. That is what makes
crash recovery, replaying a "weird" interview, and re-grading old answers
possible (plan §11.2). Redis keeps a copy of the computed state so most reads
do not have to replay, tagged with the `seq` it was computed at so a stale copy
can always be recognised.

```
               write path                                   read path
  request ──► lock session row (Postgres)        request ──► max(seq) for (session, owner)
              fold events  ──► current state                 cache.get(session)
              FSM + invariants check                         cached.seq == max(seq) ? ──► return it
              INSERT event seq = head+1                      else fold events ──► cache.store ──► return
              UPDATE sessions.state (projection)
              COMMIT
              cache.store(new state)   (best effort, monotonic in seq)
```

---

## 2. States and transitions

### 2.1 The states

`app/interview/fsm.py::InterviewState` — eight values, stored as text in
`interview_sessions.state`:

| State | Meaning |
|---|---|
| `CREATED` | The session row and blueprint exist. Nothing has happened yet. |
| `PREPARING` | Blueprint/priors being prepared (resume/JD work lands here on Day 24). |
| `ASKING` | A question is in front of the candidate. |
| `GRADING` | An answer was submitted and is being (or has been) graded. |
| `PROBING` | A follow-up on the *same* item was decided (Day 23 decides *whether*). |
| `SELECTING` | A *new* bank item is being chosen (Days 12–15 decide *which*). |
| `REPORTING` | The interview stopped asking; the report is being produced. |
| `COMPLETED` | Terminal. |

**How `PROBING/SELECTING` was interpreted.** Plan §3 writes it as one stage;
plan §13.4 draws two sibling states that both return to `ASKING`. They are
implemented as two states because they are different decisions: a follow-up
has no bank id and stays on the current item, a selection picks a new item that
must never repeat. Merging them would move that difference into a payload flag.

### 2.2 The transition table

`fsm.TRANSITIONS` is the **only** definition of what is legal. It is keyed by
`(current state, event type)`:

| From | Event | To | Source |
|---|---|---|---|
| *(empty log)* | `SESSION_CREATED` | `CREATED` | §13.4 |
| `CREATED` | `PREPARATION_STARTED` | `PREPARING` | §13.4 |
| `PREPARING` | `QUESTION_ASKED` | `ASKING` | §13.4 |
| `ASKING` | `ANSWER_SUBMITTED` | `GRADING` | §13.4 |
| `GRADING` | `ANSWER_GRADED` | `GRADING` (self-loop) | §11.2 — every grade is in the log |
| `GRADING` | `GRADING_FAILED` | `GRADING` (self-loop) | Day 22 — a grade that could not be produced, recorded ([turn-api.md](turn-api.md) §4.5) |
| `GRADING` | `FOLLOW_UP_STARTED` | `PROBING` | §13.4 |
| `GRADING` | `SELECTION_STARTED` | `SELECTING` | §13.4 |
| `GRADING` | `REPORTING_STARTED` | `REPORTING` | §13.4 "stop rule" |
| `PROBING` | `QUESTION_ASKED` | `ASKING` | §13.4 |
| `PROBING` | `FOLLOW_UP_ABANDONED` | `SELECTING` | Day 23 — the decided follow-up could not be worded; continue normally ([follow-ups.md](follow-ups.md) §4) |
| `SELECTING` | `QUESTION_ASKED` | `ASKING` | §13.4 |
| `SELECTING` | `REPORTING_STARTED` | `REPORTING` | §8.7 pool exhaustion — "never crash" |
| `REPORTING` | `SESSION_COMPLETED` | `COMPLETED` | §13.4 |

Two edges go beyond the §13.4 drawing, each for a stated reason:

* **`ANSWER_GRADED` is a self-loop in `GRADING`.** The grade is a fact worth
  recording on its own (deterministic re-grading needs it), but it does not
  change the phase of the interview.
* **`SELECTING → REPORTING`.** If the stop rule said "continue" but no eligible
  item exists (§8.7), the session would otherwise be stuck in `SELECTING`.

**Day 23 added a fourth:** `PROBING → SELECTING` on `FOLLOW_UP_ABANDONED`.
This is recorded rather than silent, and no question is fabricated.

**Day 22 added a third:** `GRADING_FAILED`, a self-loop in `GRADING`. When no
valid grade can be produced (permanent error or retries used up), the failure
becomes a fact in the log, with no terminal `FAILED` state: the session stays in
`GRADING` with the turn ungraded, which is exactly true. A later `ANSWER_GRADED`
(after an operator re-queue) is still legal. `GRADING_FAILED` has the same
"current turn, not yet graded" rule as `ANSWER_GRADED`, and leaving `GRADING`
still requires a grade.

Everything else is rejected with `IllegalTransition`, which carries the state
and the event type. A unit test checks all 9 × 11 `(state, event)` pairs (as of
Day 23) against
a hand-written copy of this table, and another proves every state can still
reach `COMPLETED` (no dead ends).

### 2.3 Terminal behaviour

`COMPLETED` has no outgoing edge. **Every** event offered to a completed session
is refused — including a second `SESSION_COMPLETED`. The error says
`"(terminal state)"`. Legality is checked *before* the payload, so a completed
session gives that answer even when the payload is also wrong.

### 2.4 Rules the table cannot express

Some rules depend on earlier events, so they live in `state.apply`:

| Rule | Why |
|---|---|
| `seq` is exactly `previous + 1`, starting at 0 | No gaps, duplicates or reordering. |
| A fresh question has a bank `question_id` never asked before in this session | Plan §8.3's "no repeats" hard constraint. |
| A follow-up (`is_follow_up: true`) is asked only from `PROBING` and has no `question_id` | Its wording is generated (Day 23); it has no bank id. |
| `ANSWER_GRADED.turn_id` / `GRADING_FAILED.turn_id` must be the turn being graded, and a turn is graded once | A grade cannot land on the wrong answer or twice. |
| `GRADING` cannot be left (`FOLLOW_UP_STARTED`, `SELECTION_STARTED`, `REPORTING_STARTED`) until the grade is recorded | Every decision after an answer is based on its grade. |

### 2.5 What is deliberately not here

No `ABANDONED`/`FAILED` state and no early-end edge for `POST /sessions/{id}/end`
(plan §13.8). Day 22 settled a failed grade as the `GRADING_FAILED` self-loop
above. Cancellation was **explicitly deferred** on Day 22 (Decision B) until it
is designed.

---

## 3. Events

### 3.1 Storage

The Day-2 table, unchanged in shape:

```sql
interview_events(id bigserial pk, session_id uuid fk → interview_sessions ON DELETE CASCADE,
                 seq int, type text, payload jsonb, created_at timestamptz default now(),
                 unique(session_id, seq))               -- uq_interview_events_session_id_seq
```

`unique(session_id, seq)` is both the correctness guarantee (one event per
position) and the index the only read uses: "all events of one session, in
order" — one ordered range scan.

**Migration 0003 adds:**

* `ck_interview_events_seq_non_negative` — `CHECK (seq >= 0)`.
* `trg_interview_events_append_only` — a `BEFORE UPDATE OR DELETE` row trigger
  that raises `interview_events is append-only` (SQLSTATE `restrict_violation`).
  `DELETE` is still allowed when it comes from the `ON DELETE CASCADE` of the
  session (and so of the user). The trigger tells the two apart with
  `pg_trigger_depth()`: depth 1 for a direct statement, depth ≥ 2 inside the
  foreign key's cascade trigger. This was checked on pg16 before relying on it,
  and an integration test pins it.

`TRUNCATE` does not fire row triggers. The application never truncates; the
test suite does so to reset state between tests.

### 3.2 Event types and payloads

`app/interview/events.py` — one pydantic model per type, all with
`extra="forbid"`:

| Type | Payload |
|---|---|
| `SESSION_CREATED` | `{}` — always seq 0 |
| `PREPARATION_STARTED` | `{}` until Day 23; Day 24: `{priors: {subtopic: {theta ∈ [−3,3], rd ∈ [1.0, 1.3]}}, prior_source: none\|no_profile\|resume_profile}` ([ingestion.md](ingestion.md) §5) |
| `QUESTION_ASKED` | `{question_id: str \| null, is_follow_up: bool}` |
| `ANSWER_SUBMITTED` | `{turn_id: uuid}` |
| `ANSWER_GRADED` | `{turn_id: uuid, score: float in [0,1], ability?: {subtopic_key, theta, rd > 0, n_observations ≥ 0}}` |
| `GRADING_FAILED` | `{turn_id: uuid, error_code: reason, attempts: int ≥ 1, retries_exhausted: bool}` (Day 22) |
| `FOLLOW_UP_ABANDONED` | `{error_code: reason, attempts: int ≥ 0, retries_exhausted: bool}` (Day 23) |

**Day 23 optional fields:** `QUESTION_ASKED.text` (≤ 2000 characters, the
wording shown), `ANSWER_GRADED.delta_theta` (the θ step), and
`FOLLOW_UP_STARTED.concept_key`. Payloads are now stored without unset
optional fields (`exclude_none`), and the round trip is unchanged.
**The follow-up caps (2 per item, 6 per session) are invariants of the log**:
an event that would exceed one is illegal. `SessionState` became schema
version 2 on Day 23 ([follow-ups.md](follow-ups.md) §7) and **3 on Day 24**,
which adds `priors`. Priors are kept apart from `ability` (observed only), so
the stop rule and reports never mistake a claim for a measurement.
| `FOLLOW_UP_STARTED` | `{reasons: [reason, …]}` (may be empty) |
| `SELECTION_STARTED` | `{}` |
| `REPORTING_STARTED` | `{reasons: [reason, …]}` — at least one, no repeats |
| `SESSION_COMPLETED` | `{}`; Day 25: `{report_id}` — committed in the same transaction as the persisted report ([reporting.md](reporting.md) §1) |

A `reason` is a short key matching `^[a-z][a-z0-9_]{0,63}$`, such as
`item_budget_reached` (`app.selection.stopping.STOP_*`) or `pool_exhausted`.

**What is *not* in an event, on purpose.** No answer text, no resume, no
email. An answer is referenced by `turn_id`; its text lives in `turns`. Since
unknown fields are rejected, they *cannot* be added by accident — a test
confirms that `{"turn_id": …, "answer_text": "secret"}` is refused.

**Results, not inputs.** `ANSWER_GRADED.ability` records the ability *after*
`app.ability.update_ability` ran, not the inputs to it. Replay therefore never
re-runs the K-factor arithmetic, and a later change to that arithmetic cannot
silently rewrite what a past session concluded. Re-grading under new arithmetic
is an explicit job over the log (§11.2), not a side effect of reading it.

Payloads are validated when written **and again** when read back, so a row
edited by hand (or by a buggy tool) is detected on replay.

---

## 4. `fold(events)`

`app/interview/state.py`:

```python
state = SessionState.initial(session_id)      # seq = -1, state = None
for event in events:                          # in seq order
    state = apply(state, event)               # FSM + invariants + payload
return state                                  # raises EventLogCorrupted on any problem
```

`SessionState` is a frozen dataclass holding plan §11.1's *working state*:

| Field | Meaning |
|---|---|
| `session_id`, `seq`, `state` | identity, last applied event, FSM state |
| `asked_question_ids` | bank items asked, in order (feeds `SelectionState.asked`) |
| `follow_ups_asked` | follow-up count (Day 23's caps will read it) |
| `current_question_id`, `current_is_follow_up` | what is in front of the candidate |
| `current_turn_id`, `current_graded` | the answer being graded, and whether its grade is in |
| `ability` | `subtopic → AbilityState`, as last recorded by `ANSWER_GRADED` |
| `stop_reasons` | why `REPORTING` was entered |

Budgets are not copied in: they live on the blueprint (session row), and "items
used" is `len(asked_question_ids)`. Never store what you can derive.

**Determinism.** `apply` performs no I/O, reads no clock, uses no randomness and
calls no model. Tests confirm that five folds of the same log are equal and
serialise identically, and that folding all at once equals folding any prefix
and then applying the rest one at a time.

**Corruption is an error, never "best effort".** `fold` raises
`EventLogCorrupted(session_id, seq, reason)` on the first event it cannot apply
— a gap or duplicate `seq`, a non-integer `seq`, an unknown type, an invalid
payload, an illegal transition, or an empty log. It never skips the bad event
and returns the rest, because that would produce a state no real history
produced. A session with a corrupted log can be neither read nor appended to
until someone repairs it deliberately.

---

## 5. Appending: transactions and concurrency

`app/interview/store.py::stage_append` (no commit) and
`app/interview/service.py::append_event` (commits):

1. `SELECT … FROM interview_sessions WHERE id = :sid AND user_id = :uid FOR UPDATE`
   — ownership check and **per-session row lock**. Not this user's session →
   `SessionNotFound` (the same error as "does not exist", so ids cannot be probed).
2. Read the session's events in `seq` order and fold them — **from Postgres,
   never from Redis**. Validation of a durable write must not depend on a cache.
3. If the caller passed `expected_seq` and it is not the head → `SeqConflict`
   (optimistic concurrency for "I decided based on the state I read earlier").
4. Check the transition, validate the payload, `apply` (all pure).
5. `INSERT` the event at `head + 1`; `UPDATE interview_sessions.state`
   (+ `ended_at = now()` on `COMPLETED`).
6. `COMMIT`. Only then, `cache.store(new state)`.

Any exception before the commit rolls back, so a rejected event leaves no event
row and no change to `interview_sessions.state`.

**Why the row lock is enough.** Two concurrent appends would otherwise both
fold the same head, both pass validation, and both write `head + 1`. With the
lock, the second waits until the first commits, then folds a log that already
contains the first event and validates against *that*. The lock is per session,
so unrelated sessions never contend. It is an ordinary Postgres lock inside a
transaction we need anyway — no distributed lock, no new infrastructure.

**The backstop.** If a future code path ever appends without the lock,
`unique(session_id, seq)` refuses the second insert. The store maps that
`IntegrityError` to `SeqConflict` instead of a 500 or a forked history. An
integration test removes the lock on purpose and confirms this.

**`interview_sessions.state` is a projection.** It is written in the same
transaction as the event, so after commit it cannot disagree with the log. It
serves the Day-2 partial index of unfinished sessions and lets sessions be
listed without folding each one. `load_state` never reads it; the log stays
authoritative.

**Composing with other writes (for Day 22).** `stage_append` does not commit, so
a caller can insert a `turns` row and stage `ANSWER_SUBMITTED` in one
transaction, then call `service.commit_and_cache`.

---

## 6. The Redis cache

`app/interview/cache.py::SessionStateCache`.

### 6.1 Key and value

```
sess:{session_id}      (Redis hash, plan §11.1)
    seq    "7"                          last event the fold includes
    state  {"v":1,"session_id":…,…}      SessionState.to_dict(), canonical JSON
TTL: SESSION_CACHE_TTL_S (default 14400 s = 4 h, plan §11.1)
```

`v` is `STATE_SCHEMA_VERSION`. An entry with another version is treated as
invalid and rebuilt, so a deploy that changes `SessionState` can never read an
old shape out of Redis.

### 6.2 The four rules

1. **Written only after commit.** A cached state is always the fold of events
   that are durably in Postgres.
2. **Monotonic in `seq`.** One Lua script checks the cached `seq` and writes
   only if the new `seq` is strictly greater. Because the check and the write
   happen in one atomic script, a slow request holding an old fold cannot
   overwrite a newer one. An integration test shuffles 15 writes and runs them
   concurrently; the newest always survives.
3. **Trusted only at the head.** A reader uses the cached state only if its
   `seq` equals `max(seq)` in Postgres.
4. **Failure is a miss.** Timeouts, connection errors and unreadable entries are
   logged and fall back to a rebuild. Redis being down makes a session slower,
   never wrong, and never unavailable.

### 6.3 Read outcomes (`service.load_state`)

| Situation | Detected by | Result |
|---|---|---|
| Hit at head | `cached.seq == max(seq)` | returned as-is |
| Miss (expired, flushed, never written) | empty hash | fold, `store` |
| Stale (Redis missed a write) | `cached.seq < max(seq)` | fold, `store` (CAS accepts the higher seq) |
| Ahead of the log (only after a DB restore) | rebuilt `seq` < `cached.seq` | fold, `replace` (DEL + store) |
| Invalid (garbage JSON, wrong version, wrong type, `seq` field ≠ state's `seq`, another session's state) | decode/consistency check, `WRONGTYPE` | `invalidate`, fold, `store` |
| Unavailable (timeout, refused) | `RedisError`/`OSError`/`TimeoutError` | fold; cache untouched |
| Log corrupted | `EventLogCorrupted` | raised; **nothing cached** |

Ownership is checked in the `max(seq)` query, which filters on `user_id`.
A cached state is therefore never returned to a user who does not own the
session — a test confirms this.

### 6.4 The consistency boundary

The only window where Postgres and Redis differ is between `COMMIT` and
`cache.store`. A crash there leaves Redis one event behind, and the next reader
sees `cached.seq < head` and rebuilds. The reverse case (Redis ahead of
Postgres) cannot come from this code, because nothing is cached before commit.

### 6.5 Configuration

| Setting | Default | Bounds | Why |
|---|---|---|---|
| `SESSION_CACHE_TTL_S` | 14400 | 1 … 604800 | plan §11.1 "TTL 4 h"; bounds growth to recently active sessions |
| `SESSION_CACHE_TIMEOUT_S` | 0.5 | (0, 10] | a cache is only worth waiting for while it is faster than the fold it replaces |

FastAPI dependency: `app.deps.get_session_cache` / `SessionCache`.

---

## 7. Interfaces (for Days 22–25)

```python
from app.interview import (
    create_session,     # (db, cache, *, user_id, blueprint) -> SessionState        commits
    append_event,       # (db, cache, *, session_id, user_id, event_type, payload, expected_seq=None)
                        #   -> SessionState                                          commits
    stage_append,       # same, no commit — compose with other writes in one transaction
    commit_and_cache,   # (db, cache, state) -> SessionState
    load_state,         # (db, cache, *, session_id, user_id) -> SessionState
    rebuild_state,      # (db, *, session_id) -> SessionState   no cache, no ownership check
    fold, apply,        # pure
)
# errors: SessionNotFound, SeqConflict, IllegalTransition, InvalidEventPayload, EventLogCorrupted
```

HTTP mapping, implemented on Day 22 in `app/api/sessions.py` (see
[turn-api.md](turn-api.md)): `SessionNotFound → 404`,
`IllegalTransition`/`SeqConflict → 409`, `InvalidEventPayload → 422`,
`EventLogCorrupted → 500` plus an error log.

---

## 8. Running it

```bash
docker compose up -d postgres redis           # local only
cd backend
alembic upgrade head                          # applies 0003
pytest tests/unit/interview                   # 234 pure tests, no services
pytest tests/integration/test_interview_event_log.py tests/integration/test_session_cache.py
REQUIRE_INTEGRATION=1 pytest                  # the full suite, failing instead of skipping
ruff check . && ruff format --check . && mypy --strict app
```

---

## 9. Limitations and open questions

* **HTTP surface (Day 22 update).** `POST /turns`, `GET /turns/{id}` and
  `GET /stream` exist now ([turn-api.md](turn-api.md)). `POST /sessions`,
  `/start` and `/end` still do not.
* **Turn idempotency (resolved on Day 22).** A duplicate `turn_id` with the same
  answer returns the original turn; a different answer gets 409.
* **No ending outside the stop rule.** There is no edge for abandonment or the
  user's `/end`. On Day 22 this was explicitly deferred until it is designed
  (Decision B). A permanently failed grade is now `GRADING_FAILED`.
* **Writes fold the whole log.** That costs O(events) per append. With tens of
  events per session this is a few milliseconds; snapshots would be the remedy
  if sessions ever grew to thousands of events.
* **`pg_trigger_depth() > 1` also lets through a `DELETE` issued from inside
  some *other* trigger.** No such trigger exists. If one is ever added, this
  rule must be revisited.
* **A cached state at the head is trusted without re-validation.** Anyone with
  write access to Redis could alter it with a matching `seq`. Redis is treated
  as internal infrastructure, the same trust model as the refresh-token
  allowlist. Durable writes never depend on the cache, so this can affect a
  read, never the log.
* **`ANSWER_GRADED` stores results, not inputs.** Re-grading under a new prompt
  needs the answer text from `turns` plus the bank item; the event alone is not
  enough to re-derive a grade.
