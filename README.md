# Adaptive AI Interviewer

An **adaptive assessment engine** for technical interviews: candidate ability is a latent
parameter (θ) estimated with an item-response-theory model, question difficulty (`b`) is
calibrated from response data, and the next question is chosen to maximise information
gained about θ under coverage constraints from the job description.

The LLM does exactly three jobs — **understand** (parse resume/JD), **judge** (classify an
answer against a concept checklist), and **speak** (phrase questions and follow-ups). It
never decides control flow and it never emits a score. All arithmetic happens in code.

> **Status: Phase 1 complete — Day 5 of 45.** Skeleton + schema + auth + the LLM
> chokepoint + CI + observability + a deterministic offline provider.
> No interview logic yet.
> Results table (convergence chart, grader QWK, cost/latency) lands here as the
> measurements exist. See `PROGRESS.md` for the phase gates.

---

## Quick start

```bash
cp .env.example .env          # the API refuses to boot without these
                              # NVIDIA_API_KEY is required — see "The LLM chokepoint"
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
cd backend && pytest && ruff check . && ruff format --check . && mypy --strict app
```

Those four commands are exactly what CI runs — see "Continuous integration".

Point `DATABASE_URL`/`REDIS_URL` at `localhost:5433` / `localhost:6380` and run
`uvicorn app.main:create_app --factory --reload` from `backend/`.

---

## Health endpoints, and why there are two

| Endpoint | Question it answers | On failure the orchestrator should |
|---|---|---|
| `GET /healthz` | Is the process alive? Touches no dependency. | **Restart** the container |
| `GET /readyz` | Can it serve traffic? Probes Postgres and Redis, and reports whether an LLM provider is in rotation. | **Stop routing** to it — but not restart; a down database is not fixed by killing the API |

`/readyz` probes run concurrently and each has a deadline (`READINESS_TIMEOUT_S`), and a
dependency that is not wired yet reports `skipped` rather than `ok`. The LLM check
deliberately makes **no model call** — a probe polled every few seconds would spend
the daily free-tier quota on health checks — so it reports "configured, and not
currently circuit-broken by real traffic". The honest end-to-end check is the
opt-in smoke test. The Phase 6 deploy
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

---

## The LLM chokepoint

Every call to a language model in this project goes through **one function**:

```python
from app.llm import TaskName, call_structured

answer, meta = await call_structured(TaskName.GRADE_ANSWER, inputs, Grade)
```

`answer` is an instance of the pydantic class you passed in — not a string, not
a dict, not "probably JSON". If the model could not produce one, it raises.

**No module outside `app/llm/` may call a provider API.** That single rule buys
prompt versioning, model routing, retry-on-invalid-output, caching, cost
accounting and failover once, instead of thirty times badly.

```
application code
      |  call_structured(task, inputs, Schema)
      v
router          ordering - retry - failover - circuit breaker
      v
provider        NVIDIA adapter: the only file that knows the vendor exists
      v
Nemotron 3.5 Lightning   (OpenAI-compatible endpoint)
      v
validation      JSON recovered -> pydantic -> repair-retry if it fails
      v
application code   (answer, meta)
```

### Provider and model

| | |
|---|---|
| Provider | NVIDIA, `https://integrate.api.nvidia.com/v1` |
| Model | `nvidia/nemotron-3.5-lightning-30b-a3b` |
| Client | the **OpenAI Python SDK**, pointed at NVIDIA's base URL |

The endpoint is *OpenAI-compatible*: same paths, same request and response
shapes. The OpenAI SDK is used purely as a transport for a wire format several
vendors implement — no OpenAI account, no OpenAI model, no data sent to OpenAI.

### Design decisions worth knowing

**Structured calls are never streamed.** Streaming exists so a UI can paint
tokens as they arrive. `call_structured()` promises a validated object, and a
half-arrived JSON object cannot be validated — so streaming would mean
buffering every chunk and validating at the end, which is a plain call with
extra failure modes. (The SSE at the API boundary carries state transitions,
not model tokens — `plan.md` §13.8.)

**Structured output is negotiated, then remembered.** The adapter asks for
strict `json_schema` decoding first; if the endpoint rejects that parameter it
steps down to `json_object`, then to prompt-only, and remembers which rung
worked. None of this affects correctness: **pydantic validation is the
guarantee**, and provider-side enforcement only makes the first attempt succeed
more often.

> **Measured:** `integrate.api.nvidia.com` accepts the top rung. A live probe
> reports `structured_mode=json_schema` with `schema_retry_count=0`, so the
> ladder is a safety net rather than a routine cost.

**Reasoning is off by default and never leaves the adapter.** Nemotron can emit
chain-of-thought (`extra_body.chat_template_kwargs.enable_thinking`,
`extra_body.reasoning_budget`). Today's tasks are schema-constrained extraction
at temperature 0, where thinking buys nothing and costs latency and output
tokens, so it is explicitly disabled. Whether or not it is on, reasoning
content is separated from the answer inside the provider — from the dedicated
`reasoning_content` field, or from inline `<think>` tags — and only its **token
count** travels onward. Reasoning text never reaches the caller, the metadata,
or the logs.

**Temperature comes from the task, not the call site.** `app/llm/tasks.py` is
`plan.md` §13.6 in code: grading is 0.0 because the same answer must be graded
the same way twice; question phrasing is 0.4 because variety is the point.

### Caching, cost and failover

* **Cache** — Redis, keyed on task + prompt version + prompt fingerprint +
  schema fingerprint + model + sampling settings + inputs. Only deterministic
  (temperature 0) tasks are cached; caching a sampled task would destroy the
  variation it was sampled for. A dead Redis is a cache miss, never a failure.
* **Cost** — a `PRICES` table in `app/llm/pricing.py`, `Decimal` arithmetic,
  summed across every attempt including schema repairs. A model missing from
  the table reports `price_known=False` rather than a silent $0.
* **Failover** — providers are tried in `LLM_PROVIDER_ORDER`. Retryable
  failures (429, 5xx, timeout) get another attempt with jittered backoff,
  honouring `Retry-After`; a bad key or a 400 moves straight to the next
  provider. Repeated failures open a circuit breaker, so a dead provider costs
  one failed call per cooldown rather than one per request.

Today the order has a single entry. The mechanism is what makes adding a second
provider a change to `.env`.

Every call emits the metadata `plan.md` §14.2 calls non-negotiable —
`prompt_version, model, input_tokens, output_tokens, cost_usd, cache_hit,
schema_retry_count, session_id` — as a JSON log line. Day 4 changes the
exporter, not the call sites.

### The API key

`NVIDIA_API_KEY` lives in your local `.env` and nowhere else. `.env` is
gitignored; the value is held as a pydantic `SecretStr`, so it renders as
`**********` in every repr, log line and traceback; and the adapter never
echoes a 401 response body, because that is the one place a provider might
quote the key back. If it is missing, the API refuses to boot:

```
Invalid configuration - the API will not start.

  - NVIDIA_API_KEY is required because LLM_PROVIDER_ORDER includes 'nvidia'
```

Migrations do **not** need it: `alembic upgrade head` validates only
`DATABASE_URL`, so a schema change or a restore drill never depends on a
provider account being in good standing.

### Verifying it

```bash
cd backend
pytest                              # 414 tests; the LLM ones never touch a network
pytest -m smoke tests/smoke -q -s   # one real call to NVIDIA; needs NVIDIA_API_KEY
```

The deterministic suite never touches a provider: unit tests substitute a fake
provider (for the router, the cache and the chokepoint) or the adapter's single
SDK call (for the NVIDIA-specific request building and response parsing). The
live smoke test is excluded from `pytest` by default and skips if the key is
unset. It loads the repository's real `.env` by absolute path, so a pass says
something about the configuration you actually deploy with.

A passing run looks like this:

```
  nvidia live probe
    model            nvidia/nemotron-3.5-lightning-30b-a3b
    structured_mode  json_schema
    tokens           328 in / 23 out
    reasoning_tokens 0
    schema_retries   0
    cost_usd         0.000000 (price_known=True)
    latency          1265 ms
```

## Continuous integration

`.github/workflows/ci.yml` runs on every push to `main` and every pull request.
Plan §3 (Day 4) specifies lint → mypy → pytest; §14.3 adds `gitleaks`.

| Job | Gate | Why |
|---|---|---|
| `static` | `ruff check .` | Lint: unused imports, shadowed names, `async` misuse, security patterns |
| | `ruff format --check .` | Formatting is not an opinion anyone has to hold in review |
| | `mypy --strict app` | Types: the API-contract bugs that a test suite finds late, or never |
| `tests` | `pytest` | Unit tests offline; integration tests against real Postgres + Redis service containers |
| `secrets` | `.env` not tracked | Plan §14.3: never in the image or the repo |
| | `gitleaks detect --redact` | A credential committed and later deleted is still leaked; full history is scanned |

**CI never uses a real provider key.** `pytest` is configured with
`-m "not smoke"`, so the one test that calls NVIDIA is deselected, and the unit
suite supplies a placeholder from a fixture. There is no `NVIDIA_API_KEY` and no
`${{ secrets.* }}` reference anywhere in the workflow. gitleaks runs with
`--redact`, so a finding cannot turn the CI log itself into the leak.

`REQUIRE_INTEGRATION=1` is set in CI. Integration tests *skip* when Postgres or
Redis is unreachable, which keeps `pytest` green on a laptop with Docker closed;
that flag turns those skips into failures, so "CI is green" cannot quietly mean
"CI ran 37 fewer tests".

### Running the same gates locally

```bash
cd backend
ruff check .            # lint
ruff format --check .   # formatting
mypy --strict app       # types
pytest                  # tests (integration ones skip if the stack is down)

docker compose up -d && pytest        # …and now they do not skip
```

`.github/workflows/ci.yml` is itself asserted on by
`backend/tests/unit/test_ci_workflow.py`: the file has to parse, run each gate,
provide its own Postgres and Redis, ask for no credential, and contain no deploy
step. That catches the class of mistake you would otherwise find only after a
push.

---

## Observability

Full guide, including every term explained from scratch:
**[`docs/observability.md`](docs/observability.md)**.

The question it answers: *when something goes wrong, what happened, where, and
how long did it take?*

- **Structured logs.** One JSON object per line. Every line carries
  `request_id`, `trace_id`, `span_id`, `service` and `env` whether or not the
  call site asked. uvicorn's own logs are routed through the same formatter.
- **Request correlation.** `X-Request-ID` is accepted from the caller (validated
  — an unvalidated id in a log line is a log-injection bug) or generated, and
  returned in the response. One line per request records route, status and
  `duration_ms`.
- **Tracing.** OpenTelemetry. One span per request (auto-instrumented) and one
  span per model call, `llm.<task>`, carrying the plan §14.2 attribute set:
  `prompt_version`, `model`, tokens, `cost_usd`, `cache_hit`,
  `schema_retry_count`, `failover_count`, `session_id`, `plan`, `latency_ms`.
- **Redaction.** Three mechanisms — sensitive field names, credential shapes
  (JWT, `nvapi-…`, `Bearer …`, argon2), and this process's actual secrets
  registered as literals at boot. Prompts, model answers and reasoning are never
  logged or traced at all: `CallMeta` does not contain them.

```bash
docker compose logs -f api                    # the default: JSON to stdout, nothing to run

# see spans without any infrastructure
echo 'OTEL_EXPORTER=console' >> .env && docker compose restart api

# or the real thing (plan §13.10) — opt-in, six containers, several GB
docker compose -f infra/langfuse/docker-compose.langfuse.yml up -d
# then set OTEL_EXPORTER=langfuse + LANGFUSE_* in .env
```

Follow one request end to end:

```bash
curl -i -H 'X-Request-ID: my-trace-1' localhost:8000/readyz
docker compose logs api | grep my-trace-1
```

---

## Offline mode — the stub provider

> Plan §3, Day 5: *"Offline replay mode (stub provider that serves recorded
> fixtures) so every future test runs without API calls."*

```bash
LLM_PROVIDER_ORDER=stub     # no API key, no network, no quota, no waiting
```

**What it is.** A real provider adapter implementing the same `LLMProvider`
interface as the NVIDIA one, selected by the same configuration, sitting behind
the same router. It answers from recordings in `backend/fixtures/llm/` instead
of from the network. It is **not** a test double — `tests/` contains those
separately; this is a shipped component you can point the whole application at.

**Why it exists.** A test suite whose result depends on somebody else's uptime is
not a test suite, it is a status page. Every phase after this one — the retrieval
evals, the grader's QWK measurement, the FSM turn loop, the injection suite —
needs to run hundreds of LLM calls per commit, deterministically and for free.

### The architecture does not change

```
application  ──►  call_structured()  ──►  ProviderRouter  ──►  provider  ──►  validated object
                                                                  │
                                                    ┌─────────────┴─────────────┐
                                                 NvidiaProvider           StubProvider
                                                 (real HTTP)              (recordings)
```

Only which provider the router built differs, and only because of one
environment variable. There is no separate offline code path — if there were, an
offline test would be exercising a path production never runs, and would prove
nothing.

### Replay vs synthesis — the distinction that matters

| | Where the answer comes from | Semantics | `structured_mode` |
|---|---|---|---|
| **Replay** | A recorded real model response | **Real** — a model genuinely said this | `stub_replay` |
| **Synthesis** | Derived from the request's JSON Schema | **Meaningless** — shape-correct only | `stub_synthesized` |

Synthesis exists so a new phase can be developed before any recording of it
exists. It proves plumbing works; it proves nothing about answer quality.

Confusing the two would be the worst possible outcome, so they are impossible to
confuse from outside: `structured_mode` lands in `CallMeta`, on the span, and in
the log line for **every** call. A grading eval that quietly ran on synthesized
labels is one query away from being spotted. `LLM_STUB_ON_MISSING` defaults to
`strict`, which *raises* rather than inventing anything.

### Recording fixtures

```bash
cd backend
# many at a time, from a recording plan (Day 6)
python scripts/record_llm_fixtures.py fixtures/recording_plans/connectivity_probe.json --dry-run
python scripts/record_llm_fixtures.py fixtures/recording_plans/connectivity_probe.json

# one named recipe at a time (Day 5)
python scripts/record_llm_fixture.py probe --dry-run   # print the key, write nothing
python scripts/record_llm_fixture.py probe             # make the call, write the file
```

The only things in the repository that deliberately spend quota. Run by hand,
never by a test, never by CI. Both are front doors onto one engine
(`app/llm/recording.py`) that calls the real `call_structured()` — so what gets
recorded is exactly what production sends, and neither tool has its own idea of
what an LLM call looks like.

A fixture is keyed by a SHA-256 of the assembled provider request, so **a changed
prompt is a different key** — a stale recording becomes a clean miss instead of a
silently wrong answer. That same key is computed *before* the call, which is what
makes re-running a plan free: **an existing recording is skipped, never silently
overwritten**; `--overwrite` replaces it deliberately. One failed entry is
reported (with any credential redacted) and the batch carries on, because
recording is the expensive operation and a failure should not cost the entries
that already succeeded.

A recording is a real answer or it is nothing: a `stub_replay`, `stub_synthesized`
or cached response is refused rather than written, and the **provider** decides
that label — a plan file cannot claim it. Recording needs
`LLM_PROVIDER_ORDER=nvidia`; a stub-only order is refused up front. Plan format,
flags and re-run behaviour: `backend/fixtures/llm/README.md`.

> **Never record a call whose prompt or response contains candidate data.** A
> fixture is committed to git: permanent, shared, and outside the reach of the
> log redactor. See `backend/fixtures/llm/README.md`.

### Checking it by hand

```powershell
# from the REPOSITORY ROOT (that is where .env lives)
.venv\Scripts\python.exe backend/scripts/show_llm_call.py --stub     # offline, no key
.venv\Scripts\python.exe backend/scripts/show_llm_call.py --nvidia   # live, spends quota
```

Prints the validated object *and* the full `CallMeta`, and says in words what
`structured_mode` means for that call. `docs/manual-verification.md` walks
through every Phase-1 check the same way.

### Which tests need what

| Command | Network | API key | What it proves |
|---|:--:|:--:|---|
| `pytest` | no | no | Everything below except the live wire. 470 tests. |
| `pytest tests/unit` | no | no | Logic, offline. Runs anywhere. |
| `pytest tests/integration` | localhost | no | Real Postgres + Redis; migrations really apply. Skips if the stack is down. |
| `pytest tests/unit/llm/test_offline_replay.py` | **blocked** | no | A real recorded Nemotron answer replays end to end. A fixture actively makes any non-loopback socket raise. |
| `pytest tests/unit/llm/test_bulk_recording.py` | no | no | The bulk recorder: plan validation, fixture keys, idempotency, failure handling, provenance — and a record→replay round trip. |
| `pytest -m smoke tests/smoke` | **yes** | **yes** | Is the wire actually connected? Opt-in; deselected by default. |

---

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
  obs/                observability — see docs/observability.md
    context.py        request id / user id carried through async code
    redaction.py      what must never be written down
    logging.py        structured JSON logs, correlated and redacted
    tracing.py        OpenTelemetry setup and the exporter choice
    spans.py          CallMeta (Day 3) projected onto a span (Day 4)
    middleware.py     request correlation and per-request latency
  llm/                the single LLM chokepoint — see below
    types.py          provider-agnostic vocabulary + the provider interface
    tasks.py          routing table: task → tier, temperature, output ceiling
    prompts.py        versioned prompt templates
    structured.py     schema derivation, JSON recovery, validation, repair
    providers/        one adapter per vendor, plus the offline stub
      nvidia.py       the only module that knows NVIDIA exists
      stub.py         deterministic offline provider (Day 5)
    fixtures.py       recorded responses, keyed by a hash of the request
    router.py         ordering, retry, failover, circuit breaker
    cache.py          response cache for deterministic tasks
    pricing.py        token prices and per-call cost
    client.py         call_structured() and CallMeta
    runtime.py        process lifecycle, wired into the FastAPI lifespan
    recording.py      the bulk fixture recorder (Day 6): plan -> chokepoint -> fixture
backend/fixtures/llm/ recorded model responses the stub replays
backend/fixtures/recording_plans/  what to record, as JSON — input, never output
backend/scripts/      record_llm_fixture{,s}.py — the only things that spend quota
backend/migrations/   Alembic environment and versions
backend/tests/        pytest (unit = offline, integration = real PG+Redis,
                      smoke = opt-in, hits a real provider)
infra/postgres/init/  pgvector + citext extensions, run once on an empty volume
infra/langfuse/       opt-in self-hosted Langfuse stack (not part of `docker compose up`)
docs/observability.md logging, tracing, redaction, and how to run Langfuse locally
docs/adr/             architecture decision records
.github/workflows/    ci.yml — lint, format, types, tests, secret scan
.gitleaks.toml        secret-scanner allowlist (documented placeholders only)
docker-compose.yml    api + postgres(pgvector) + redis
plan.md               the 45-day blueprint this repo is built against
```

Modules land as their phase arrives (`interview/`, `grading/`, `retrieval/`,
`ingestion/`, `billing/`, `sandbox/`) — see `plan.md` §13.9.

## Stack

Python 3.12 · FastAPI · pydantic v2 · PostgreSQL 16 + pgvector · Redis 7 · Docker ·
NVIDIA Nemotron 3.5 Lightning via an OpenAI-compatible endpoint ·
OpenTelemetry (Langfuse-compatible) · GitHub Actions.
Offline runs need none of it: `LLM_PROVIDER_ORDER=stub`.
No LLM framework: retrieval, prompt assembly, structured output and orchestration are
written directly, because those are exactly the four things worth being able to explain.
