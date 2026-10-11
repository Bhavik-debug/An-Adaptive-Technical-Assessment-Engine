# Follow-ups, session start, and what happens after a grade (Day 23)

Plan §3, Day 23: *"Follow-up policy: deterministic trigger rules (high-weight
concept absent / factual error / score in the ambiguous band / RD still high),
caps of ≤2 per item and ≤6 per session; LLM generates only the wording. Question
rendering call (bind a canonical question to the candidate's own stack)."*

Code: `backend/app/interview/followup.py` (the policy, pure),
`backend/app/interview/flow.py` (what happens after a grade),
`backend/app/interview/probing.py` (the wording job),
`backend/app/interview/lifecycle.py` (start),
`backend/app/wording/` (prompts, probe validation, rendering), and
`backend/app/api/sessions.py` (routes). Migration:
`0005_follow_ups.py`. Builds on [interview-fsm.md](interview-fsm.md) and
[turn-api.md](turn-api.md).

---

## 1. The interview loop as it now runs

```
POST /api/sessions                CREATED            (blueprint built from the spec)
POST /api/sessions/{id}/start     PREPARING → ASKING (first item selected, text shown)
POST /api/sessions/{id}/turns     ASKING → GRADING   (Day 22)
worker: grade                     ANSWER_GRADED, then IN THE SAME TRANSACTION:
    policy says follow up?  ──yes──► FOLLOW_UP_STARTED → PROBING  + follow_ups row (pending)
         │                              worker: word it ──ok──► QUESTION_ASKED{is_follow_up, text} → ASKING
         │                                              └fail──► FOLLOW_UP_ABANDONED → SELECTING → (below)
         no
         ▼
    stop rule fires?  ──yes──► REPORTING_STARTED{reasons}           (session waits; report is Day 25)
         no
         ▼
    SELECTION_STARTED → select next item ──found──► QUESTION_ASKED{question_id, text} → ASKING
                                         └─none───► REPORTING_STARTED{pool_exhausted}
```

**A grade is never committed without its decision.** The policy, the
`FOLLOW_UP_STARTED` or selection events, and the follow-up job row are written
in the same transaction as `ANSWER_GRADED` (`flow.stage_after_grade`, called
from the grading worker's fenced commit). So no window exists in which a grade
is durable but nobody has decided what comes next. A duplicate delivery of the
grade is fenced out, so it cannot decide twice.

---

## 2. The policy (`app/interview/followup.py`)

### 2.1 Rules, in the plan's order — the first that fires is the reason

| # | Reason code | Fires when | Threshold source |
|---|---|---|---|
| 1 | `concept_absent` | a concept with bank weight **≥ 3** is labelled `absent` | plan §8.5; 3 is the top of the bank's 1–3 weight scale (`ExpectedConcept.weight`) |
| 2 | `factual_error` | any error claim of severity **`major`** survives Day-20 error verification | plan §8.5 |
| 3 | `ambiguous_band` | **0.45 ≤ score ≤ 0.65** (inclusive) | plan §8.5: "the score we're least sure about" |
| 4 | `high_uncertainty` | RD of the item's subtopic **> 0.7** (strict), *after* this grade's update | plan §8.5 |

These are the plan's numbers, used as written. None was tuned here.

**The inputs are all computed by code.** They are labels after Day-17
evidence verification, error claims after Day-20 verification, the code-side
score, and RD from Day-11's update. **The candidate's answer is not an input to
the policy at all**, so nothing written in an answer, injection included, can
change whether a follow-up happens. A test grades an injected answer and a
plain one identically and gets byte-identical decision traces.

**Probe target (`concept_key`).** For `concept_absent`, the target is the absent
essential concept; ties go to bank order. For the other reasons, the target is
the highest-weight concept labelled `absent`, else `partial`, else none (the
probe then asks for depth on the question as a whole).

### 2.2 Missing information is never evidence of a poor answer

Each rule reports `fired`, `not_fired` or **`unavailable`**, and an
unavailable rule can never fire.

| Situation | Effect |
|---|---|
| No judgment at all (e.g. grading failed) | The policy is not even run: `GRADING_FAILED` path, no decision, no follow-up |
| Judgment not an object, `concepts`/`errors` not lists | the rules needing them → `unavailable` |
| An essential concept with **no label** | `concept_absent` → `unavailable` (no label ≠ absent) |
| An invalid label (`"ABSENT??"`) | treated as no label |
| A weight outside 1–3, or not an int | that concept is not "essential" |
| The judgment names a concept **the item does not have** | contradictory → `concept_absent` → `unavailable`, reported in the trace |
| Score missing, non-finite, or outside [0, 1] | `ambiguous_band` → `unavailable` |
| RD missing, non-finite, or ≤ 0 (e.g. an item without IRT parameters) | `high_uncertainty` → `unavailable` |

### 2.3 Caps

| Cap | Value | Enforced by |
|---|---|---|
| per bank item | **2** | the policy (`capped_by: item_cap_reached`) **and** the event log (`FOLLOW_UP_STARTED` / follow-up `QUESTION_ASKED` beyond it is an illegal event) **and** the database (`follow_ups.ordinal IN (1,2)`, unique per item) |
| per session | **6** | the policy (`session_cap_reached`) **and** the event log |

A follow-up is **not** a new original item. The fold tracks
`current_item_id`, the bank item a follow-up belongs to, so a follow-up of a
follow-up still counts against the original. When a cap says no, the normal
flow continues: stop rule, then selection.

Concurrency cannot exceed a cap. Every append is serialised by the session row
lock and validated against the folded log, and the grading fence makes each
grade's decision happen exactly once.

### 2.4 The trace

The trace is stored on `turns.policy_decision`. It contains keys and numbers
only, never answer text:

```json
{"follow_up": true, "reason": "high_uncertainty", "concept_key": null, "capped_by": null,
 "rules": [{"rule": "concept_absent", "status": "not_fired"},
           {"rule": "factual_error", "status": "not_fired", "major_errors": 0},
           {"rule": "ambiguous_band", "status": "not_fired", "score": 0.9},
           {"rule": "high_uncertainty", "status": "fired", "rd": 1.0903}]}
```

### 2.5 Representative decisions

| Grade | Item follow-ups so far | Decision |
|---|---|---|
| `stampede` (weight 3) absent, score 0.62, RD 1.09 | 0 | follow up — `concept_absent`, target `stampede` |
| all covered, one **major** error, score 0.8 | 0 | follow up — `factual_error` |
| all covered, score 0.55, RD 0.5 | 1 | follow up — `ambiguous_band` |
| all covered, score 0.9, RD 1.09 | 0 | follow up — `high_uncertainty` |
| all covered, score 0.9, RD 0.6 | 0 | **no** follow-up → selection |
| anything that fires | 2 | **no** — `item_cap_reached` → selection |
| anything that fires, session already at 6 | 0 | **no** — `session_cap_reached` → selection |
| no grade (grading failed) | — | no decision; session stays in GRADING (Day 22) |

### 2.6 A measured consequence of the plan's RD rule

With `RD_MAX = 1.3` and well-targeted items, Day 11's update gives RD of
**1.09, 0.957, 0.863, 0.793, 0.737** after observations 1–5 of a subtopic,
and **0.691** at the sixth. Follow-up answers are observations too.
`high_uncertainty` therefore fires on essentially every early answer, and the
**caps, not the rules, decide how many follow-ups happen early**. When early
items fall on different subtopics, each can draw two follow-ups, so the session
cap can be reached by the third item. That is the plan's rule taken at its word. A test pins this series. It
should be revisited as a calibration question (Phase 9), not changed silently.

---

## 3. The wording (`app/wording/`) — the model speaks, it does not decide

**Follow-up probe** (`TaskName.FOLLOW_UP_PROBE`, prompt `v1`, small-fast tier,
temperature 0.5, already routed since Day 3):

* **The model receives only:** the canonical question, the candidate's answer
  (passed through the grader's own fence neutraliser `isolate_candidate_text`
  and fenced as data), a one-line FOCUS (the target concept's *hint*, or a
  generic instruction per reason, plus the major error claim for
  `factual_error`, itself fenced), and up to three author-written
  `follow_up_seeds`.
* **The model never receives:** the reason code, the concept key, the score,
  the labels, the candidate's identity, or other turns.
* **It is told to** ask one short question, never name the concept ("naming
  the concept hands over the answer", plan §6.5), and never obey text inside
  the fence.
* **Checked in code after the chokepoint's schema validation**
  (`check_probe`):
  * the length is 10–600 characters;
  * it contains no fence sentinels and no `[removed-delimiter]` (echoing the
    fence means it is quoting the prompt);
  * it does not name the target concept key in any spelling (`cache_invalidation`,
    `cache invalidation`, `cache-invalidation`).
  A failure raises `ProbeRejected`. **The probe is never repaired, truncated or
  replaced with a canned one.**

**Question rendering** (`TaskName.QUESTION_RENDER`, prompt `v1`):
`render_question(canonical, anchor_terms, context)` follows plan §6.2. The
model may reframe the question but may not change technical content, and code
verifies that every `anchor_terms` entry survives. **Rendering can never fail
an interview.** No context, a provider error, an invalid rewrite, or a dropped
anchor term each falls back to the canonical text, with the reason recorded.
**Today there is no candidate context**, because it comes from the resume and
job description (Day 24). So every question is currently shown in its canonical
bank wording, and no model is called for rendering.

**What the wording cannot touch:** the grade, the policy decision, the caps,
the ability estimate and the blueprint. The probe text is stored and shown, and
nothing that decides anything reads it. The next grading call grades the
*candidate's* answer against the bank item's checklist.

---

## 4. The wording job (`app/interview/probing.py`)

This uses the same machinery as Day-22 grading: a `follow_ups` row is the
durable job, workers claim it with `FOR UPDATE SKIP LOCKED` and a lease, the
same `RetryPolicy` and `classify_error` apply, and the finish is fenced on
`attempts`. It runs in the same worker loop as grading (`run_worker(...,
follow_ups=...)`).

| Outcome | Writes, atomically |
|---|---|
| valid probe | `QUESTION_ASKED {is_follow_up: true, text}` → ASKING; row `ready` with `text`, `generated_at` |
| transient error (timeout, 429, 5xx, all providers failed), attempts left | row back to `pending` with backoff and `error` code; no event |
| `ProbeRejected`, permanent provider error, or attempts used up | row `abandoned` + **`FOLLOW_UP_ABANDONED {error_code, attempts, retries_exhausted}`** → SELECTING → normal selection (next item, or REPORTING) |
| a stale worker (lease expired, re-claimed) | discarded (`stale`) |

`FOLLOW_UP_ABANDONED` is a **new FSM edge, PROBING → SELECTING**, recorded
rather than silent. The alternative was a self-loop that would leave the
session in PROBING until an operator acted. That would block a candidate's
interview over *extra* evidence, which the grade did not need. An abandoned
follow-up counts toward no cap and changes no estimate. Error codes are
`probe_length`, `probe_echoes_prompt`, `probe_names_concept`,
`context_unavailable`, `retry_exhausted`, `state_conflict`, plus the Day-22
provider codes.

---

## 5. Answering a follow-up

`POST /api/sessions/{id}/turns` is unchanged. A follow-up answer:

* is stored with `is_follow_up = true`, `question_id` = **the parent bank
  item**, and `question_text` = the probe actually shown;
* is graded by the unchanged Phase-4 pipeline against the parent item's
  checklist and misconceptions, with the grader shown the original question
  followed by `Follow-up question: <probe>`;
* updates θ at **half stake** (`STAKE_GENERATED_ITEM = 0.5`, plan §8.5: "a probe
  is a hint");
* is charged exactly like any grade (Day 22: once per committed grade);
* leads to another policy decision for the same item, which the item cap
  bounds.

`theta_deltas`, the stopping rule's "no new information" input, counts bank
items only. Half-weight probe steps are not new items (plan §8.4 counts items).

---

## 6. Session creation and start

### `POST /api/sessions` → `201`

The body is an `AssessmentSpec` (`app.blueprint`) as JSON. It is validated by
the blueprint builder **against the committed taxonomy**:

```json
{"role": "Backend Engineer", "duration_minutes": 60,
 "topics": [{"topic_key": "dsa", "jd_weight": 0.9},
            {"topic_key": "databases", "jd_weight": 0.7, "subtopic_keys": [], "required": true,
             "min_items": 0, "max_items": null}],
 "level": "mid", "mode": "standard", "item_budget": null}
```

```json
{"session_id": "…", "state": "CREATED", "seq": 0, "blueprint": { …Blueprint.to_dict()… },
 "start_url": "/api/sessions/…/start", "stream_url": "/api/sessions/…/stream"}
```

The response includes `Location: /api/sessions/{id}`. An unknown topic, level
or mode, a budget that doesn't fit, or an unknown field gets **422**, and
nothing is created. No token gets **401**. No quota check is made (quotas are
Phase 8).

### `POST /api/sessions/{id}/start` → `200`

This runs in one transaction under the session lock: `PREPARATION_STARTED`,
selection of the first item (Day 12, over the blueprint's selection state),
then `QUESTION_ASKED {question_id, text}`.

```json
{"session_id": "…", "state": "ASKING", "seq": 2,
 "question": {"question_id": "dsa-hashing-002", "is_follow_up": false, "text": "A service stores…"},
 "turns_url": "/api/sessions/…/turns", "stream_url": "/api/sessions/…/stream"}
```

| Situation | Response |
|---|---|
| first call | 200, the first question |
| repeat while **ASKING** (e.g. a lost response) | 200, the **current** question, `Idempotent-Replayed: true`, nothing written |
| repeat in any other state (GRADING, PROBING, …) | 409 |
| concurrent first calls | the lock serialises them: exactly one starts, the rest replay (tested with 6) |
| nothing in the bank fits the blueprint | 409 "No question in the bank fits this session's blueprint"; nothing written, session stays CREATED |
| not yours / does not exist | 404 (same body) |
| no token | 401 |

The full error mapping, shared by all session routes, is `SessionNotFound`/
`TurnNotFound` → 404, `IllegalTransition`/`SeqConflict`/`TurnIdConflict`/
`NoEligibleQuestion`/`FollowUpTurnsUnsupported` → 409, `InvalidEventPayload` →
422, and `EventLogCorrupted` → 500.

---

## 7. Events, state and SSE

New and extended events (all optional fields, so pre-Day-23 logs still fold):

| Event | Change |
|---|---|
| `QUESTION_ASKED` | + `text` (≤ 2000 chars): the wording actually shown |
| `ANSWER_GRADED` | + `delta_theta`: the stopping rule's input |
| `FOLLOW_UP_STARTED` | + `concept_key` (pattern-checked) |
| `FOLLOW_UP_ABANDONED` | **new**: PROBING → SELECTING |

`SessionState` is now **schema version 2**. It adds `current_item_id`,
`follow_ups_by_item`, `current_question_text` and `theta_deltas`. Cached v1
entries are treated as invalid and rebuilt from the log; Day 21's cache rules
already handle this. Stored payloads now omit unset optional fields
(`exclude_none`), so a new optional field adds no `null`s to the log.

SSE:

| Event | SSE name | data beyond `seq`, `state` |
|---|---|---|
| `FOLLOW_UP_STARTED` | `follow_up` | **nothing** (the reason or concept would give away the answer) |
| `QUESTION_ASKED` | `next_question` | `question_id` (null for a follow-up), `is_follow_up`, **`text`** |
| `FOLLOW_UP_ABANDONED` | `follow_up_abandoned` | `error_code`, `retries_exhausted` |

Fields are projected from the validated payload, so the shape is the same
whether or not an optional field was stored. The per-message bound was raised
to 8 KB to fit a 2000-character question.

---

## 8. Recovery

All Day-23 state is durable and lives in Postgres: the decision is in the log,
the job is a `follow_ups` row, and the wording is in the log and on the row. A
crash at any point leaves either:

* a committed grade and decision with a `pending` job, which the next worker
  claims; or
* a `processing` job whose lease expires and which is re-claimed (bounded by
  `GRADING_MAX_ATTEMPTS`, after which it is abandoned); or
* nothing, because the transaction rolled back.

Tests cover a fresh worker loop finishing a pending job, lease-expiry
re-claims, stale-worker fencing, and duplicate or competing claims.

---

## 9. Limitations and open points

* **The `high_uncertainty` rule dominates early follow-ups** (§2.6). This is
  the plan's rule, measured and pinned, not tuned.
* **The methodology for grading follow-up answers is a choice, not a
  measurement.** The answer is graded against the parent item's full
  checklist, so a short probe answer that covers only the probed concept will
  label the others absent and score low, before half-stake damping. Whether
  that biases θ downward is untested; it needs an eval (Phase 9, or a Day-25
  follow-up). *Day 24:* re-examined, left unchanged, options recorded in
  [ingestion.md](ingestion.md) §12.
* **There is no candidate context for rendering** until Day 24, so questions
  are shown in canonical wording. Once context exists, rendering would make a
  model call inside `/start`'s and the grade's transaction while the session
  lock is held. That must move to an asynchronous job like the probe before
  Day 24 turns it on. *Day 24:* context now exists (the confirmed profile), and
  rendering was **left off** for exactly this reason. A test asserts that
  `/start` and selection never call `QUESTION_RENDER` ([ingestion.md](ingestion.md) §12).
* **There is no endpoint that returns the current question** other than
  `/start`'s replay while ASKING. Clients learn follow-ups and later questions
  through SSE `next_question`, which carries the text.
* **Probe validation is literal.** It catches the concept key in three
  spellings, not synonyms, and cannot judge whether a probe is any good.
* **No recorded `FOLLOW_UP_PROBE` fixtures exist.** The automated tests use
  scripted provider doubles and the stub's synthesize mode. No live model
  output for this prompt has been seen or evaluated.
* **REPORTING is reached, but nothing completes a session** (Day 25).
* **Cancellation (`/end`) is still deferred** (Day 22, Decision B).
