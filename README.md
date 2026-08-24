# Adaptive AI Interviewer

An **adaptive assessment engine** for technical interviews: candidate ability is a latent
parameter (θ) estimated with an item-response-theory model, question difficulty (`b`) is
calibrated from response data, and the next question is chosen to maximise information
gained about θ under coverage constraints from the job description.

The LLM does exactly three jobs — **understand** (parse resume/JD), **judge** (classify an
answer against a concept checklist), and **speak** (phrase questions and follow-ups). It
never decides control flow and it never emits a score. All arithmetic happens in code.

> **Status: Phase 1, Day 2 of 45.** Skeleton + schema + auth. No interview logic yet.
> Results table (convergence chart, grader QWK, cost/latency) lands here as the
> measurements exist. See `PROGRESS.md` for the phase gates.

---

## Quick start

```bash
cp .env.example .env          # the API refuses to boot without these
docker compose up --build     # migrations run automatically before the API starts

curl localhost:8000/healthz   # {"status":"ok", ...}
curl localhost:8000/readyz    # per-dependency roll-up
open http://localhost:8000/docs
```

The `migrate` service runs `alembic upgrade head` and exits; the `api` service
waits for it to succeed, so the API can never serve traffic against an
un-migrated database.

Host ports are offset so a locally installed Postgres/Redis does not clash:
Postgres on **5433**, Redis on **6380**, API on **8000**. If 8000 is already taken,
set `API_HOST_PORT` in `.env` — the container always listens on 8000 internally.

### Running the backend without Docker

```bash
python -m venv .venv && .venv/Scripts/activate     # Python 3.12
pip install -e "backend/[dev]"
cd backend && pytest && ruff check . && mypy app
```

Point `DATABASE_URL`/`REDIS_URL` at `localhost:5433` / `localhost:6380` and run
`uvicorn app.main:create_app --factory --reload` from `backend/`.

---

## Health endpoints, and why there are two

| Endpoint | Question it answers | On failure the orchestrator should |
|---|---|---|
| `GET /healthz` | Is the process alive? Touches no dependency. | **Restart** the container |
| `GET /readyz` | Can it serve traffic? Probes Postgres, Redis, (soon) LLM providers. | **Stop routing** to it — but not restart; a down database is not fixed by killing the API |

`/readyz` probes run concurrently and each has a deadline (`READINESS_TIMEOUT_S`), and a
dependency that is not wired yet reports `skipped` rather than `ok`. The Phase 6 deploy
pipeline rolls a release back on a failed `/readyz`, so these checks have to be honest.

---

## Authentication

| Endpoint | Does |
|---|---|
| `POST /api/auth/register` | Create an account; returns an access token, sets the refresh cookie |
| `POST /api/auth/login` | Same, from credentials |
| `POST /api/auth/refresh` | Rotate the refresh cookie, return a new access token |
| `POST /api/auth/logout` | Revoke the refresh token, clear the cookie |
| `GET /api/auth/me` | The authenticated user |

**Two tokens, on purpose.** The *access token* is a 15-minute JWT sent in the
`Authorization` header — cheap to verify (a signature check, no database round
trip) but impossible to revoke, so it is kept short. The *refresh token* is a
30-day JWT in an `httpOnly; SameSite=Strict` cookie that JavaScript cannot read,
and it is **single-use**: every refresh spends the old one and issues a new one.

Rotation is what makes the long-lived half safe. A stolen refresh token gets used
twice — once by the thief and once by the real user — and the second use presents
a correctly signed token whose id is already spent. That is detectable, and the
response is to revoke every refresh token the user holds and force a fresh login.

Passwords are hashed with **argon2id** (64 MiB, t=2), which is memory-hard and
therefore expensive to crack in parallel on a GPU. Ten failed logins lock an
account for 15 minutes.

## Database & migrations

Schema changes are Alembic migrations, forward-only, one revision per change.

```bash
# apply everything (what the migrate container does)
cd backend && alembic upgrade head

# after editing a model: generate a draft, then READ AND EDIT IT
alembic revision --autogenerate -m "what changed"

# is the schema in sync with the models?
alembic check
```

Extensions (`vector`, `citext`) are **not** in migrations — they need superuser
rights and are a property of the server, so they live in `infra/postgres/init/`
and run once when the database volume is first created.

---

## Repository layout

```
backend/app/          FastAPI modular monolith
  config.py           pydantic-settings; fails fast at boot on a bad environment
  db.py / cache.py    Postgres + Redis lifecycle and probes
  deps.py             shared dependencies: db session, current user, settings
  security.py         argon2 hashing + JWT encode/decode (pure, no I/O)
  token_store.py      Redis refresh-token allowlist + login lockout
  models/             SQLAlchemy table definitions
  api/health.py       liveness + readiness
  api/auth.py         register / login / refresh / logout / me
backend/migrations/   Alembic environment and versions
backend/tests/        pytest (unit = no infrastructure, integration = real PG+Redis)
infra/postgres/init/  pgvector + citext extensions, run once on an empty volume
docs/adr/             architecture decision records
docker-compose.yml    api + postgres(pgvector) + redis
plan.md               the 45-day blueprint this repo is built against
```

Modules land as their phase arrives (`interview/`, `grading/`, `retrieval/`,
`ingestion/`, `billing/`, `llm/`, `sandbox/`, `obs/`) — see `plan.md` §13.9.

## Stack

Python 3.12 · FastAPI · pydantic v2 · PostgreSQL 16 + pgvector · Redis 7 · Docker.
No LLM framework: retrieval, prompt assembly, structured output and orchestration are
written directly, because those are exactly the four things worth being able to explain.
