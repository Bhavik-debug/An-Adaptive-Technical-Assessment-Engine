# The interview screen (Phase 6, Day 27)

`/interviews/:sessionId`: the screen where a candidate answers questions.
It shows what the server has recorded, and nothing else. The backend's engine
chooses every question; the screen never selects, skips, scores or advances on
its own.

Code: `frontend/src/pages/interview/` (`InterviewPage`, `useInterview`,
`interviewModel`, `AnswerForm`, `QuestionText`, `content`), `src/api/stream.ts`,
`src/lib/sse.ts`. The report route `/interviews/:id/report` is Day 28's
report screen ([report-screen.md](report-screen.md)).

---

## 1. Backend contracts used

| Call | Use |
|---|---|
| `GET /api/sessions/{id}` (**new, Day 27**) | The snapshot: state, `seq`, the current question, honest counts. |
| `GET /api/sessions/{id}/stream` with `Last-Event-ID: <seq>` | Live state transitions, from the snapshot's seq. |
| `POST /api/sessions/{id}/turns` `{turn_id, answer}` | Submit an answer (202 new, 200 replay). |
| `GET /api/sessions/{id}/turns/{turn_id}` | Once, after loading a snapshot in `GRADING`: has this answer's review already failed for good? |
| `POST /api/sessions/{id}/start` | Only for a session still in `CREATED` (idempotent). |
| `GET /api/setup-options` | Display names for the topic chips. |

Every call carries the bearer token; ownership is enforced server-side ("not
yours" is the same 404 as "does not exist").

### Why a new endpoint

Without it, a page refresh mid-interview had no supported way to learn the
current question. `/start` answers 409 once the first answer is in, and
replaying the whole stream from `-1` would put the full event history on the
client just to find one question. `GET /api/sessions/{id}` is read-only and
additive: `load_state` (Redis cache or replay of the log) plus the session row.

```json
{
  "session_id": "…", "state": "ASKING", "seq": 2,
  "role": "Backend engineer", "level": "mid", "mode": "standard",
  "topic_keys": ["dsa", "databases"],
  "started_at": "…", "ended_at": null, "report_status": null,
  "question": {"question_id": "…", "is_follow_up": false, "text": "…"},
  "questions_asked": 1, "follow_ups_asked": 0,
  "current_turn_id": null,
  "stream_url": "/api/sessions/…/stream", "turns_url": "/api/sessions/…/turns"
}
```

`question` is present in `ASKING` (being answered) and in `GRADING`, `PROBING`
and `SELECTING` (just answered), and `null` before the first question and after
the end. `current_turn_id` is set only in `GRADING`. It deliberately carries no
grades, ability estimates, item budget, stop reasons, difficulty window or other
selection internals; a test asserts that none of those keys appears.

### The SSE contract (unchanged; `backend/app/interview/stream.py`)

Each message is `id: <seq>`, `event: <name>`, `data: <JSON with seq and
state>`, plus `: keepalive` comments. Names: `grading_started` (turn_id),
`grade_ready` (turn_id, score), `grading_failed` (turn_id, error_code,
retries_exhausted), `follow_up`, `follow_up_abandoned`, `next_question`
(question_id, is_follow_up, text), `session_complete` (report_id), and
`state_changed` (event_type) for the rest. `event: error` (no id) means the log
could not be read. `Last-Event-ID: n` replays every event with `seq > n`
exactly once. The server ends the stream after `session_complete`.

## 2. How the screen keeps state honest

* **Snapshot, then stream from its seq.** Nothing is missed and nothing is
  seen twice.
* **Strict seq order.** An event with `seq ≤` the applied seq is ignored (a
  replay after reconnecting). One that skips a seq means something was
  missed, so the snapshot is re-read rather than guessed at.
* **Reconnects** use the last *applied* seq as `Last-Event-ID`, with backoff
  of 1, 2, 4… up to 15 s. A 401 or 404 stops the stream (sign-out, or the
  session is gone), and so does `event: error`, with a message and a Refresh
  button.
* **`EventSource` is not used**: it cannot send the bearer token. The stream is
  read with `fetch` through the same client as every other call (bearer token,
  one refresh on 401), and parsed by `lib/sse.ts`, which handles chunks split
  anywhere and CR/LF/CRLF line ends.
* **Cleanup:** the stream is aborted on unmount, so leaving the screen,
  signing out or losing the session (which unmounts the protected route) closes
  it, and the reader is cancelled. At most one stream is open per screen.
* **Scores are dropped.** `grade_ready.score` and `TurnStatus.score` are in
  the contract but never shown during the interview.

## 3. Answering

* A labelled textarea, up to **8000 characters, counted as the backend counts
  them** (Python code points, so emoji are not over-counted). Blank answers are
  refused before sending. The 422 message is shown if the server disagrees.
* Each question gets a **client-generated `turn_id`** (`crypto.randomUUID`),
  created when its form mounts. The form is re-keyed per question, which also
  clears the draft.
* **Submission outcomes** (`AnswerForm.tsx`):

  | Outcome | What the screen does |
  |---|---|
  | 202, or 200 (replay) | "Answer received". It does **not** advance; it waits for the stream (`grading_started`, then `next_question`, …). |
  | 422 | Nothing was written. The message is shown and the draft stays editable. |
  | 409 / 404 | Not accepting answers (for example another tab answered). The snapshot is re-read and the server's state is shown. |
  | Network error, 30 s timeout, 5xx | **Outcome unknown.** The text is frozen (read-only) and only **"Resend the same answer"** is offered: the same `turn_id` with byte-identical text, which the backend treats as a replay, never as a second answer. If the lost request did land, the stream's `grading_started` moves the screen on by itself. |

* Double submission: the button is disabled while sending, and an in-flight
  guard drops repeat submits (a double click sends one request).
* The draft is kept **in memory only** (not in browser storage), and leaving
  the page with an unsent draft triggers the browser's "leave site?" prompt.

## 4. What each state shows

| Server state | Screen |
|---|---|
| `CREATED` | "This interview has not started yet", with **Start interview** (idempotent `/start`) |
| `PREPARING` | "Preparing your first question…" |
| `ASKING` | The question (verbatim; line breaks and indentation kept; ``` fences as code), the answer form |
| `GRADING` | The answered question, "Answer received", "Reviewing your answer…"; "being retried" when a reloaded turn is `pending` with an `error_code` (a retry is scheduled); after **any** `grading_failed` (it is always final, Day 29 fix), "We could not review this answer… the interview is paused", because the backend keeps the session in `GRADING` until an operator re-queues the turn |
| `PROBING` | "Preparing a follow-up question…" |
| `SELECTING` | "Choosing your next question…" |
| `REPORTING` / `COMPLETED` | "Interview complete", with **View report** (the Day 28 report screen) and **Back to dashboard** |

Plus: a loading skeleton, "Interview not found" (404 or a malformed id), and a
retryable load error.

**Progress** is "Question *n*" or "Follow-up on question *n*", and "*n*
questions asked so far". There is **no total and no percentage**, because the
stopping rule decides the length. **Topic chips** are the session's configured
topics. **There is no timer:** the backend's time budget is computed from the
asked items' *estimated* durations, not wall-clock time, so a countdown would be
invented. The level and mode the candidate chose are shown as text.

Focus moves to the new question's heading when a question replaces the
previous one. Status changes use `role="status"`, and errors use
`role="alert"` (through the shared `Alert`).

## 5. Tests

* `pages/interview/interview.test.tsx` (through the real routes, with a
  controllable mocked stream):
  * code formatting kept, a count and no progress bar, topic chips;
  * the stream starting from the snapshot seq, and the bearer token and
    `Accept` header on it;
  * not found (404 and malformed id), retry after a load error, starting a
    `CREATED` session;
  * the finished state and its report link;
  * a refresh mid-review showing the failed review and no editor;
  * validation with no request sent;
  * one POST on a double click with no local advance, then the next question
    via the stream, a cleared draft and a new turn id, with the grade never in
    the DOM;
  * an unknown outcome freezing the text, then an identical resend;
  * 422 keeping the draft editable, and 409 re-reading the snapshot;
  * replayed events ignored and a seq gap re-reading the snapshot;
  * reconnecting from the last applied seq, and the stream aborted when
    leaving.
* `pages/interview/interviewModel.test.ts`: the reducer (a full turn,
  replays, gaps, follow-up counting, retrying vs given-up reviews, the end),
  snapshot details, code-point counting, fenced code.
* `lib/sse.test.ts`: the parser.
* `backend/tests/integration/test_session_detail_api.py`: auth, the same 404
  for missing and foreign sessions, 422 for a malformed id, `CREATED`,
  `ASKING` (verbatim text), `GRADING` (turn id, answer never echoed), no
  internals leaked, read-only.

## 6. Not done / limitations

* **No real-browser check** of this screen at any width, with a keyboard, a
  screen reader or axe. Everything above is jsdom and unit or integration
  tests (see PROGRESS.md, Day 27).
* A session whose review failed permanently stays paused: the backend has no
  candidate-facing retry, and none was added.
* Questions logged before Day 23 have no stored wording. Such a session shows a
  notice instead of a form; none exist in a fresh database.
* The draft is not persisted across a page reload, deliberately (no answer
  text in browser storage).
