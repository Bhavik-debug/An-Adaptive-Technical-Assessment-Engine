# Running and testing locally (Windows)

The shortest reliable path from a clone to a working app on this machine, and
how to tell what is wrong when sign-in fails. Commands are for **PowerShell**
unless marked otherwise. Nothing here deploys or exposes anything.

## 1. Requirements

| Tool | Version used | Notes |
|---|---|---|
| Docker Desktop | 4.x (engine 27+) | Must be **running** before `docker compose` |
| Node.js | ≥ 20.19 (24.15 used) | Frontend (Vite) |
| Python | 3.12 | Only for running tests or tools outside Docker |

## 2. Configuration (once)

1. `Copy-Item .env.example .env` (if `.env` does not exist yet), then fill in
   `SECRET_KEY` and `NVIDIA_API_KEY` (or set `LLM_PROVIDER_ORDER=stub` with
   `LLM_STUB_ON_MISSING=scripted` for a fully offline run whose grades are
   scripted, not a model's judgement).
2. **Each variable once.** `.env` is read top to bottom and the *last* value
   wins. A duplicated `API_HOST_PORT` was one cause of the Day-27 sign-in
   failure (see §6).
3. `API_HOST_PORT`: the host port for the API. If something else already uses
   8000 (on this machine a DynamoDB-local container from another project does),
   set `API_HOST_PORT=8080`. The Vite dev proxy reads this same value, so
   nothing else needs changing.
4. Optional, for "Continue with Google": `GOOGLE_CLIENT_ID=<id>.apps.googleusercontent.com`.
   See [google-sign-in.md](google-sign-in.md) §4 for creating the client.
   It is a public ID, not a secret, and there is no client secret in this flow.

## 3. Start

```powershell
# from the repository root
docker compose up -d --build        # Postgres :5433, Redis :6380, migrations, API
docker compose ps                   # api should become "healthy"; migrate "Exited (0)"
curl.exe http://localhost:8080/readyz   # use your API_HOST_PORT

cd frontend
npm ci                              # first time, or after package-lock.json changes
npm run dev                         # http://localhost:5173
```

**Always pass `--build` after pulling changes.** Only `backend/app` is mounted
into the container. Python dependencies and `backend/migrations` are baked into
the image, so a stale image means missing packages (the API crash-loops) and
migrations that never run.

Migrations: the `migrate` service runs `alembic upgrade head` against the
compose database (`interviewer` in the `adaptive-ai-interviewer_postgres_data`
volume) before the API starts. To see which revision that database is at:

```powershell
docker compose exec postgres psql -U interviewer -d interviewer -c "select version_num from alembic_version"
```

Back it up before a migration if it holds data you care about:

```powershell
docker compose exec -T postgres pg_dump -U interviewer -d interviewer -Fc > interviewer_backup.dump
```

URLs:

| What | URL |
|---|---|
| App | http://localhost:5173 (use `localhost`, not `127.0.0.1`: Google's origin check is exact) |
| API docs (Swagger) | http://localhost:8080/docs |
| Liveness / readiness | http://localhost:8080/healthz, http://localhost:8080/readyz |

## 4. Stop (keeps all data)

```powershell
# Ctrl+C in the Vite terminal, then:
docker compose stop          # or `docker compose down`, which also keeps the volumes
```

Never use `docker compose down -v` unless you mean to **delete** the database
and Redis volumes.

## 5. Tests

```powershell
# Backend (from backend/). Integration tests use the running compose
# Postgres/Redis but their OWN database (interviewer_test) and Redis DB 15;
# your `interviewer` database is never touched. They skip if Docker is down.
..\.venv\Scripts\python -m pytest
..\.venv\Scripts\ruff check . ; ..\.venv\Scripts\ruff format --check . ; ..\.venv\Scripts\mypy --strict app

# Frontend (from frontend/)
npm run typecheck ; npm run lint ; npm run format:check ; npm test ; npm run build
```

Manual walk-through: register at `/register` → dashboard → **New interview** →
skip resume and job description → enter a role, tick a topic → **Start
interview** → answer → the next question or a follow-up arrives (over SSE) →
reload the page mid-interview (same question) → finish → **View report**
("Preparing your report…", then the report; reload keeps it) → dashboard
**View report** → **Sign out** → the report URL redirects to sign-in → sign in
again (password or Google) and you return to the report.

## 6. Troubleshooting sign-in

| Symptom | Cause | Fix |
|---|---|---|
| Every sign-in fails; `docker compose ps` shows `api` restarting or `unhealthy`; `docker compose logs api` ends in `ModuleNotFoundError` (`pgvector`, `google.auth`) | The image predates a dependency change | `docker compose up -d --build` |
| API logs `UndefinedColumn ... users.google_sub`, or `alembic_version` is below the newest file in `backend/migrations/versions` | Migrations not applied (stale image, or the stack was started without `migrate`) | `docker compose up -d --build` (back up first, see §3) |
| Sign-in shows "The request was not valid." or "Something went wrong on the server", and `curl.exe http://localhost:5173/api/auth/providers` does **not** return `{"password":true,...}` | The Vite proxy points at the wrong port (another service, or nothing) | Make `API_HOST_PORT` in `.env` match `docker compose ps`, appearing once; restart `npm run dev`. Or set `API_PROXY_TARGET` in `frontend/.env.local`. |
| New interview shows "We could not load the setup options" (no resume upload); `docker compose logs api` shows `no taxonomy at /data/question-bank/taxonomy.json` | The API container cannot see `data/`: it is outside the backend build context and is provided by the `./data:/data:ro` mount in `docker-compose.yml` | `docker compose up -d api` (re-creates the container with the mount) |
| No Google button, no message | `GOOGLE_CLIENT_ID` is not set: the API logs `google sign-in disabled (GOOGLE_CLIENT_ID is not set)` and `/api/auth/providers` returns `"google_client_id": null` | Set it in `.env`, then `docker compose up -d` (re-creates the API with the new environment) |
| Same, although the ID **is** in `.env` | The line is still commented out (`# GOOGLE_CLIENT_ID=…`). That was the Day-28 cause. Check what the container received: `docker compose exec api python -c "import os;print(bool(os.environ.get('GOOGLE_CLIENT_ID')))"` | Remove the leading `# `, then `docker compose up -d api` |
| "We could not check whether Google sign-in is available…" | `/api/auth/providers` failed (API down or proxy wrong) | As for the rows above |
| "Google sign-in is unavailable right now…" | Google's script was blocked (extension, network filter) | Allow `accounts.google.com` |
| Button renders; console shows `[GSI_LOGGER]: The given client ID is not found` | The client ID is wrong or was deleted | Copy the ID again from Google Cloud Console |
| Console shows `The given origin is not allowed for the given client ID` | The page's exact origin is not an Authorized JavaScript origin | Add `http://localhost:5173` and `http://localhost` to the client (it can take a few minutes to apply) |
| Google popup completes but "Google sign-in did not complete" | Server returned 401 `google_token_invalid`: usually `GOOGLE_CLIENT_ID` differs from the client the button used (audience), or the machine clock is badly off | Check the API log line `google sign-in: credential rejected` (its `reason` never contains the token) |
| "An account with this email already exists…" | That email has a password account; there is no silent linking, by design | Sign in with the password |
| "Google sign-in is temporarily unavailable" (503 `google_unavailable`) | The API could not fetch Google's signing keys (no internet from the container) | Check connectivity: `docker compose exec api python -c "import urllib.request;print(urllib.request.urlopen('https://www.googleapis.com/oauth2/v1/certs',timeout=5).status)"` |

### A stuck interview or report

"We could not review this answer" (a failed grade) or a report that cannot be
prepared needs an operator once the cause (provider outage, bad key) is fixed:

```powershell
cd backend
$URL = "postgresql+asyncpg://interviewer:interviewer@localhost:5433/interviewer"
..\.venv\Scripts\python scripts\requeue_failed.py --database-url $URL --list
..\.venv\Scripts\python scripts\requeue_failed.py --database-url $URL turn <session_id> <turn_id>
..\.venv\Scripts\python scripts\requeue_failed.py --database-url $URL report <session_id>
```

The running API's worker then retries it. Note that restarting the API (or
saving a backend file, since the dev compose runs `--reload`) while an answer is
being graded leaves that answer "Reviewing…" for up to about 6 minutes: the
job's lease must expire before it is re-claimed (by design; see PROGRESS.md
Day 29).

Logs: `docker compose logs -f api` (JSON lines; tokens, cookies, passwords and
answer text are redacted or never logged).
