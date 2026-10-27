# Frontend (Phase 6, Days 26–27)

The candidate-facing single-page app: authentication (email/password and,
from Day 27, "Continue with Google"), a dashboard, the three-step interview
setup wizard and, from Day 27, the interview screen. It lives in `frontend/`
and talks only to the existing FastAPI backend. **Nothing here is deployed** —
see "Local-only status" at the end.

Day 27 additions have their own documents:
[`google-sign-in.md`](google-sign-in.md) and
[`interview-screen.md`](interview-screen.md). Day 28's report screen:
[`report-screen.md`](report-screen.md).

Sections below describe Day 26 unless they say otherwise; Day 26's results are
kept as recorded.

---

## 1. Stack

| Concern | Choice | Notes |
|---|---|---|
| Framework | React 18.3 + TypeScript 5.6 (strict) | as planned (plan §3, Day 26) |
| Build/dev server | Vite 6 | dev proxy makes the API same-origin (see §4) |
| Styling | Tailwind CSS 4 + shadcn/ui conventions | `components.json` is present; components are hand-authored in shadcn style (cva + `cn`), no Radix dependency was needed |
| Routing | react-router 7 (library mode) | v7 rather than v6: v6 carries an open-redirect advisory fixed only in 7.18 |
| Icons | lucide-react | shadcn's default icon set |
| Tests | Vitest 4 + Testing Library + jsdom | |
| Lint/format | ESLint 9 (typescript-eslint, react-hooks, **jsx-a11y**), Prettier 3 | `no-console` is an error: tokens and resume data must never reach the console |

No chart library was added. The dashboard's skill overview is plain markup
(see §6). Day 28's report screen did not need one either: its bars are the
same plain-markup `SkillRow` (see `report-screen.md` §3 for the live skill
graph, which was not built).

## 2. Running it locally

Prerequisites: Node ≥ 20.19 (developed on Node 24), and the backend running
locally (README "Quick start" / "Running the backend without Docker").

```bash
cd frontend
npm ci                     # or npm install on first checkout
cp .env.example .env.local # optional; the defaults work for a backend on :8000
npm run dev                # http://localhost:5173
```

| Command | What it does |
|---|---|
| `npm run dev` | Vite dev server on :5173, proxying `/api` to the backend |
| `npm run build` | `tsc -b` then a production bundle in `dist/` (local only — not deployed) |
| `npm run preview` | Serves `dist/` on :4173 with the same `/api` proxy |
| `npm run typecheck` | TypeScript, no emit |
| `npm run lint` | ESLint (incl. accessibility rules) |
| `npm run format:check` | Prettier |
| `npm test` | Vitest, once |

### Environment variables

| Variable | Exposed to browser? | Default | Purpose |
|---|---|---|---|
| `VITE_API_BASE_URL` | **yes** (compiled into the bundle) | empty | API origin. Empty = same origin via the proxy. Must never hold a secret. |
| `API_PROXY_TARGET` | no (dev/preview server only) | `http://localhost:<API_HOST_PORT>`: the port in the repo-root `.env` that `docker compose` publishes the API on, else `8000` | Where the Vite proxy forwards `/api`. Only `API_HOST_PORT` is read from the root `.env`. |

There are no secrets in the frontend and none may be added: every `VITE_*`
value is public.

## 3. Structure and routing

```
frontend/src/
├── api/            client.ts (the only fetch), auth/sessions/ingestion/setup endpoints, stream.ts (SSE, Day 27), types.ts
├── auth/           AuthProvider (session state), guards (RequireAuth, PublicOnly)
├── components/     AppLayout, AuthLayout, FormField, EmptyState, StatusBadge, GoogleSignIn (Day 27), …
│   └── ui/         shadcn-style primitives: button, input, textarea, select, label, card, badge, alert, skeleton
├── hooks/          useApiData (abortable load), usePolling (job status), useSetupOptions
├── lib/            format, validation, redirect, session, setup helpers; sse, googleIdentity (Day 27)
├── pages/          LoginPage, RegisterPage, dashboard/, setup/, interview/ (Day 27), report/ (Day 28), placeholders (404 only)
└── test/           Vitest setup and the fetch-routing mock
```

| Route | Access | Screen |
|---|---|---|
| `/` | — | redirects to `/dashboard` |
| `/login`, `/register` | signed-out only | sign in / create account (email, or Google when configured — Day 27) |
| `/dashboard` | signed-in | welcome, start button, past interviews, skill overview |
| `/interviews/new` | signed-in | the three-step setup wizard |
| `/interviews/:id` | signed-in | the interview screen (Day 27; was a placeholder on Day 26) |
| `/interviews/:id/report` | signed-in | the report screen (Day 28; was a placeholder until then), see `report-screen.md` |
| `*` | — | not found |

A signed-out visit to a protected route goes to `/login` and, after signing in,
returns to the requested path. Only internal paths are honoured
(`lib/redirect.ts` rejects `//host`, `https://…`, and backslash tricks).

## 4. Authentication and the API client

The backend's design (`backend/app/api/auth.py`) is used as-is:

* **Access token** (15 min JWT) — held **in memory only** in `api/client.ts`,
  sent as `Authorization: Bearer`. Never written to `localStorage` /
  `sessionStorage`, never logged.
* **Refresh token** (30 days) — an `httpOnly`, `SameSite=Strict` cookie scoped
  to `Path=/api/auth`. JavaScript cannot read it; the client only asks the
  browser to send it to `POST /api/auth/refresh` (`credentials: 'include'`).

**Why the app must be same-origin with the API.** The cookie is
`SameSite=Strict` and the backend has no CORS middleware, so the SPA has to be
served from the API's origin. Locally the Vite proxy does this; the backend
comment in `_set_refresh_cookie` already flags the cross-origin case for the
deploy decision.

**Session restore.** On load, `AuthProvider` calls `/api/auth/refresh` and then
`/api/auth/me`. A 401 means "not signed in". A network failure is reported on
the sign-in page with a retry, not misreported as "logged out".

**Refresh is single-use with replay detection** — presenting a spent refresh
token revokes the whole family. So refreshes must never overlap:

* within a tab, every caller shares one in-flight refresh promise (this also
  covers React StrictMode's double-mounted effects in development);
* across tabs, the Web Locks API (`navigator.locks`) serialises refreshes, so a
  waiting tab sends the cookie the previous refresh just set.

**Expiry and revocation.** Any authenticated request that gets a 401 triggers
one refresh and one retry. If the refresh is rejected, the session is cleared
and the user is returned to `/login` with "Your session has ended".

**Errors.** `ApiError` carries the HTTP status, the backend's `code` (for
`{code, message}` details), and per-field messages parsed from FastAPI 422
lists; components show the backend's message rather than a generic one.

**Not implemented, because the backend has no endpoint for it:** password
reset, email verification, OAuth / social login, account deletion. The sign-in
page says password reset is not available rather than linking to nothing.

### Backend contracts used

| Endpoint | Used by | Notes |
|---|---|---|
| `POST /api/auth/register` | sign-up | 201, 409 duplicate email, 422 validation |
| `POST /api/auth/login` | sign-in | 401 one message for both failure kinds, 429 lockout |
| `POST /api/auth/refresh` | restore + expired token | cookie only; single-use |
| `POST /api/auth/logout` | sign-out | 204 always; local state cleared regardless |
| `GET /api/auth/me` | restore, after sign-in | |
| `GET /api/sessions` | dashboard | **new on Day 26**, see §8 |
| `GET /api/setup-options` | wizard, display names | **new on Day 26**, see §8 |
| `POST /api/resumes` | wizard step 1 | raw body, `Content-Type: application/pdf` or `text/plain`, 202 |
| `GET /api/resumes/{id}` | wizard step 1 | polled until `ready`/`failed` |
| `PATCH /api/resumes/{id}/profile` | wizard step 1 | body is exactly `CandidateProfile` (extra fields are rejected) |
| `POST /api/job-descriptions` | wizard step 2 | `{text}`, 202 |
| `GET /api/job-descriptions/{id}` | wizard step 2 | polled until `ready`/`failed` |
| `POST /api/sessions` | wizard step 3 | 201; **not idempotent** |
| `POST /api/sessions/{id}/start` | wizard step 3, dashboard | idempotent replay |
| `GET /api/sessions/{id}/report` | dashboard skill overview | read only when the list says `report_status: ready` |

## 5. The setup wizard

Three steps with a progress indicator; focus moves to each step's heading.
Resume and JD IDs are kept in the wizard's state and carried to step 3.

**Step 1 — Resume (optional).** PDF or `.txt`, ≤ 5 MB (the backend default;
the server re-checks and sniffs the bytes). Upload → poll the resume until it
is `ready` or `failed` (90 s soft timeout, then "Check again"). The parser's
*suggestion* is shown in an editable profile (headline, years, skills with
experience level and related areas from `/api/setup-options`). Only
**Confirm profile** (`PATCH …/profile`) makes it count, and the UI says so: an
unconfirmed profile seeds nothing. If extraction fails or times out, the
candidate can type skills in by hand. "Skip" continues with no resume.

**Step 2 — Job description (optional).** Paste text (client check: 40–20 000
characters, matching backend defaults). Submit → poll until `ready`/`failed`.
A ready JD shows its title, the topics it emphasises (by display name) and the
requirements quoted verbatim. If the JD fails, is still processing, or maps to
no topic, the candidate is told they will pick topics in step 3. "Skip"
continues with no JD.

**Step 3 — Interview settings.** Only backend-supported values are offered,
from `/api/setup-options`: role (required, ≤ 120), seniority (`levels`), style
(`modes`), length (15/30/45/60/90 min within the backend's bounds) and topics.
If the JD is ready and has topics, the default is "Use topics from the job
description" — the request then omits `topics` and the backend derives them
from the JD's weights. Otherwise at least one topic must be chosen.
`item_budget`, per-topic weights, min/max items and subtopic restrictions are
supported by the API but deliberately not exposed: they are tuning knobs, not
candidate choices.

**Create, then start — and never claim more than the server confirmed.**

1. `POST /api/sessions` is not idempotent, so the submit is guarded by a ref
   (a double click sends one request) and the form locks once a session exists.
2. `POST /{id}/start` follows. Only its success navigates to the hand-off page,
   which says "Your interview has started".
3. If start fails, the created session id is kept; **Try starting again** calls
   `/start` only (idempotent), never `POST /api/sessions` again. The session
   also appears on the dashboard as "Not started" with a **Start** button.
4. If the JD stopped being usable between steps (409), the form switches to
   manual topics and explains why.

## 6. The dashboard

* **Past interviews** — `GET /api/sessions`, newest first: role, seniority,
  style, topics, created/finished dates and a status. The FSM's internal states
  (`GRADING`, `PROBING`, `SELECTING`, …) are shown as "In progress"; they
  describe the engine, not anything the candidate acts on. Loading skeletons,
  an empty state with a start button, and a retryable error.
* **Skill overview** — only from real, persisted data: the newest session that
  is `COMPLETED` **and** whose report is `ready`. It shows each topic's 0–100
  display score with its 95 % interval (the report's own numbers), and topics
  with insufficient evidence are labelled as such rather than given a number.
  With no such report, it explains that nothing has been measured yet. **No
  sample or estimated data is ever shown.** A note states that the grader is
  still being validated (Phase 4 is acceptance-pending), so the numbers are
  practice feedback.

A radar chart was not used: blueprints often cover one to three topics, where
a radar is misleading, and it would have needed a new dependency a day before
Day 28 adds Recharts for the report.

Nothing on any screen exposes difficulty, θ/RD, selection policy, follow-up
decisions or model reasoning. The `blueprint` returned by `POST /api/sessions`
is never rendered.

## 7. Design and accessibility

* Light neutral background, white cards, one restrained blue accent; system
  font stack; 0.5 rem radii; no gradients, glass effects or decorative motion.
  Spinners and skeleton pulses only, and both are disabled under
  `prefers-reduced-motion`.
* Written mobile-first: single column by default, two-column dashboard from
  Tailwind's `lg` breakpoint, wrapping button rows, and navigation that drops
  below the brand on small screens. **This layout has not yet been checked in
  a real browser at any width** (see §10).
* Contrast: every text/background token pair the UI uses computes to at least
  4.5:1 (WCAG AA), from 5.3:1 for the warning badge to 17.7:1 for body text.
  This was calculated from the OKLCH tokens, not measured on rendered pages.
* Every input has a `<label>`; errors are linked with `aria-describedby` and
  `aria-invalid`, and focus moves to the first invalid field. Error alerts use
  `role="alert"`, progress uses `role="status"`. Visible focus rings
  everywhere, a "Skip to content" link, native `<select>`/radio/checkbox
  controls for free keyboard support, and an accessible label on the skill
  bars.
* `eslint-plugin-jsx-a11y` runs in `npm run lint`.

## 8. Backend changes made for Day 26

Two **read-only, additive** endpoints. No existing route, schema or behaviour
changed.

| Endpoint | Why it was needed |
|---|---|
| `GET /api/sessions?limit=1..100` (default 50) | The dashboard must show real past sessions, and no endpoint listed a user's sessions. Returns `{sessions: [{session_id, state, role, level, mode, topic_keys, resume_id, jd_id, started_at, ended_at, report_status}]}`, filtered by the caller's id in SQL. It deliberately leaves out grades, θ and the blueprint's selection internals. |
| `GET /api/setup-options` | The wizard may only offer options the backend accepts, and `PATCH …/profile` rejects unknown subtopic keys. The taxonomy and the level/mode/duration rules existed only server-side. This returns them from `data/question-bank/taxonomy.json` and `app.blueprint.spec`, with no per-level θ, seconds-per-item or difficulty window. Authenticated like every other `/api` route. |

Tests: `backend/tests/integration/test_dashboard_api.py` (9 tests, real
Postgres + Redis): auth required, empty list, newest-first, no internals
leaked, cross-user isolation, `limit` bounds, the `/{id}/report` route still
resolving, options matching the taxonomy and spec, and every offered topic
creating a session.

## 9. Tests

**Day 27 results (local):** `npm test` 96 passed, 0 failed (9 files);
typecheck, lint, format:check and build exit 0; `npm audit` 0 vulnerabilities;
backend full suite 3134 passed, 0 failed, 0 skipped, 22 deselected. The new
test files are listed in `interview-screen.md` §5 and `google-sign-in.md` §6,
and the full evidence is in PROGRESS.md, Day 27. The Day 26 record below is
unchanged.

**Final results (Day 26, local):**

| Check | Result |
|---|---|
| `npm test` | 53 passed, 0 failed (5 files) |
| `npm run typecheck` / `lint` / `format:check` / `build` | all exit 0 |
| `npm audit` | 0 vulnerabilities |
| Backend full suite (`pytest --ignore=tests/smoke`, quiet machine) | **3067 passed, 0 failed, 0 skipped, 22 deselected**, exit 0 (11 min 57 s; JUnit XML agrees) |
| Backend `ruff check`, `ruff format --check`, `mypy --strict app` | clean |

The 22 deselected are the `not smoke and not embeddings` marker filter.
`tests/smoke` was not collected at all, which is why Day 25's run shows 23.
`test_live_client_disconnect_releases_the_server` failed in 2 earlier full
runs and passed in the final one. It is a **pre-existing intermittent
failure**, not caused by Day 26 (see §10).

`npm test` — 53 tests in 5 files:

| File | Covers |
|---|---|
| `api/client.test.ts` | error parsing (string / 422 list / `{code,message}` / network), bearer header, one refresh + retry on 401, **concurrent 401s share one refresh**, lost-session signal, no refresh for login |
| `lib/lib.test.ts` | email/password rules, open-redirect guard, status mapping, JD topic ordering, duration bounds, PATCH body shape |
| `auth/auth.test.tsx` | protected routes redirect, session restore, signed-in users kept off `/login`, revocation mid-use → "session has ended", field validation (no API call), wrong password, lockout, return-to-requested-page, sign-up validation / 409 / success, sign-out |
| `pages/dashboard/dashboard.test.tsx` | empty state, retryable error, real rows with candidate-facing statuses, skills only from a ready report with insufficient evidence flagged, starting a not-started session |
| `pages/setup/wizard.test.tsx` | backend-only options, validation, exact create spec, backend rejection message, **failed start retried with `/start` only (create called once)**, double-click guard, file-type rejection, upload → poll → review → exact PATCH body → resume id carried to create, manual skills after failed extraction, server upload rejection, JD length check, JD topics used (no `topics` sent), processing JD → manual topics |

## 10. Known limitations and deferred work

* **Day 28 built the report screen** (`report-screen.md`). Not built from
  the plan's Day 28 row: an overall band, a 2-week study plan, grader evidence
  on question cards, and the live skill graph. None of them has a backend
  field to show (`report-screen.md` §3).
* **Same-origin only.** Cross-origin hosting would need a backend decision on
  CORS and `SameSite=None; Secure` (already noted in `auth.py`).
* **Wizard state is in memory.** A page reload loses the step and the
  resume/JD ids (the uploaded resume and JD remain on the server; there is no
  endpoint to list them).
* ~~No session detail endpoint~~ — added on Day 27 (`GET /api/sessions/{id}`).
* **The skill overview uses the latest completed report only**; trend over time
  is not shown.
* **Client limits mirror backend defaults** (5 MB, 40–20 000 chars); if those
  settings are changed server-side, the server's message is still shown.
* **No end-to-end browser test suite yet** (Playwright is planned for Day 30).
* **Pre-existing intermittent backend test failure**, not introduced on Day 26:
  `tests/integration/test_session_stream.py::test_live_client_disconnect_releases_the_server`
  asserts that no connection in the test DB is `idle in transaction`. It
  failed in 2 of the 3 Day-26 full runs whose outcome was captured, and
  passes on its own. A diagnostic run against the **unchanged Day-25 code**
  (commit `dffa584`, in a throwaway git worktree) reproduced the condition in
  1 of 3 attempts. One connection stayed `idle in transaction` for all 2000
  samples (about 10 s), and its last statement was `;`, which looks like a
  pool liveness ping. Day 26 does not touch the stream, the worker or the
  engine/pool, so it is recorded here and left for a separate fix.
* Accessibility was checked with jsx-a11y lint and Testing Library role/label
  queries only. **No visual, responsive, keyboard-only, axe or screen-reader
  pass in a real browser has been performed yet** (the browser automation
  extension was unavailable on Day 26); it is in the owner's manual checklist.

## 11. Local-only status

Nothing was deployed, published or provisioned for Day 26, and no deploy
pipeline was run. The Day-26 agent made **no git commit or push**. The project
owner committed and pushed the work themselves as `9cf1d33` ("Phase 6 …").
`npm run build` produces a local `dist/` (git-ignored) only. Deployment (plan
Day 29) is on hold until the project owner explicitly approves it.
