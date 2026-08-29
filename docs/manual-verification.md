# Checking Phase 1 by hand

Every command below was run on this machine (Windows 11, PowerShell 5.1) before
being written down. Where a command has a Git Bash equivalent it is labelled.

**Two things that will trip you up if nobody tells you:**

1. **The API is on port `8080`, not `8000`.** `.env` sets `API_HOST_PORT` twice
   (line 13 = 8000, line 34 = 8080) and the later one wins. Docker Desktop also
   occupies 8000 on this machine. Check yours with `docker compose port api 8000`.
2. **Run Python commands from the repository root, not from `backend/`.**
   `.env` lives at the root and pydantic-settings resolves it relative to the
   working directory. From `backend/` you get
   `ConfigError: DATABASE_URL is required but was not set` — which is the config
   layer working correctly, not a bug.

---

## A. Start the system

```powershell
cd C:\Users\ACER\Desktop\main\Projects\02_AccessmentAgent
docker compose up -d --build
```

**Expect:** `migrate` runs and exits 0, then `api` starts.

## B. Are the containers healthy?

```powershell
docker compose ps
```

**Expect:**

```
SERVICE    STATUS
api        Up 2 hours (healthy)
postgres   Up 2 hours (healthy)
redis      Up 2 hours (healthy)
```

All three must say **healthy**. If `api` is unhealthy, `docker compose logs api`.

## C. Liveness

```powershell
Invoke-RestMethod http://localhost:8080/healthz
```

**Expect:** `status: ok`, `service: adaptive-ai-interviewer`, `env: local`.

`/healthz` touches no dependency. If it fails, the process itself is broken.

## D. Readiness

```powershell
Invoke-RestMethod http://localhost:8080/readyz | ConvertTo-Json -Depth 5
```

**Expect:**

```json
{ "status": "ready",
  "checks": { "postgres":      {"status": "ok", "latency_ms": 37},
              "redis":         {"status": "ok", "latency_ms": 8},
              "llm_providers": {"status": "ok", "detail": "1/1 in rotation: nvidia"} } }
```

`/readyz` probes every dependency. The difference from `/healthz` matters: a
down database should stop traffic being routed here, not restart the container.

## E + F. Register, log in, and the protected endpoint

Paste this whole block — it does all four checks and prints each result:

```powershell
$API = "http://localhost:8080"
$email = "manual-check-$(Get-Random)@example.com"
$body = @{ email = $email; password = "a-very-strong-password-123" } | ConvertTo-Json

Write-Output "1. REGISTER"
Invoke-RestMethod -Method Post -Uri "$API/api/auth/register" -ContentType application/json -Body $body | ConvertTo-Json -Compress

Write-Output "`n2. LOGIN"
$login = Invoke-RestMethod -Method Post -Uri "$API/api/auth/login" -ContentType application/json -Body $body
Write-Output "   token issued, $($login.access_token.Length) chars"

Write-Output "`n3. /me WITH token"
Invoke-RestMethod "$API/api/auth/me" -Headers @{ Authorization = "Bearer $($login.access_token)" } | ConvertTo-Json -Compress

Write-Output "`n4. /me WITHOUT token  (expect 401)"
try   { Invoke-RestMethod "$API/api/auth/me" | Out-Null; Write-Output "   UNEXPECTED: no error" }
catch { Write-Output "   HTTP $($_.Exception.Response.StatusCode.value__)  <- correct" }
```

**Expect:** a token on register, a 268-character JWT on login, your user object
on `/me`, and **HTTP 401** without the header.

> Use `Invoke-RestMethod`, not `Invoke-WebRequest` — the latter needs the
> Internet Explorer engine on PowerShell 5.1 and fails in some environments.

## G. Migrations and tables

```powershell
docker compose exec postgres psql -U interviewer -d interviewer -c "SELECT version_num FROM alembic_version;"
docker compose exec postgres psql -U interviewer -d interviewer -c "\dt"
docker compose exec postgres psql -U interviewer -d interviewer -c "SELECT extname FROM pg_extension;"
```

**Expect:** revision `0001`; **8 tables** (`users`, `topics`, `questions`,
`interview_sessions`, `interview_events`, `turns`, `skill_states`, plus
`alembic_version`); extensions `citext`, `plpgsql`, `vector`.

Prove the API really wrote to Postgres:

```powershell
docker compose exec postgres psql -U interviewer -d interviewer -c "SELECT email, created_at FROM users ORDER BY created_at DESC LIMIT 3;"
```

**Expect:** the account you just registered.

## H. The LLM — live NVIDIA (**needs `NVIDIA_API_KEY`, spends quota**)

```powershell
.venv\Scripts\python.exe backend/scripts/show_llm_call.py --nvidia
```

**Expect:** `provider nvidia`, `structured_mode json_schema`, a **fresh random
token echoed back** (so nothing cached could have faked it), real token counts,
latency around 1–2 s.

The formal version, as a test:

```powershell
cd backend
..\.venv\Scripts\python.exe -m pytest -m smoke tests/smoke -q -s
cd ..
```

**Expect:** `1 passed`. **This is the only command in the repo that spends quota.**

## I. The LLM — offline stub (**no key, no network**)

```powershell
.venv\Scripts\python.exe backend/scripts/show_llm_call.py --stub
```

**Expect:** `provider stub`, `model stub/deterministic-v1`,
`structured_mode stub_replay`, and the last line spelling out what that means:

```
structured_mode='stub_replay' means: a REAL recorded model answer, replayed offline
```

Confirm the router really picked it and found the recording:

```powershell
$env:LLM_PROVIDER_ORDER = "stub"
.venv\Scripts\python.exe -c "from app.config import get_settings; from app.llm.router import build_router; r = build_router(get_settings()); print('providers:', r.provider_names); print('class    :', type(r.providers[0]).__name__); print('fixtures :', r.providers[0].fixture_count)"
Remove-Item Env:\LLM_PROVIDER_ORDER
```

**Expect:** `('stub',)`, `StubProvider`, `1`.

## J. Offline replay, with the network blocked

```powershell
cd backend
..\.venv\Scripts\python.exe -m pytest tests/unit/llm/test_offline_replay.py tests/unit/llm/test_stub_provider.py -q
cd ..
```

**Expect:** `49 passed`.

These run with connections to non-loopback addresses **actively blocked**. If
anything tried to reach the network the test fails saying so — this is enforced,
not asserted.

To see determinism yourself, run the `--stub` command from step I three times.
The `echo`, `model_said` and token counts must be byte-identical every time.

## K. Observability

```powershell
Invoke-RestMethod -Uri "http://localhost:8080/readyz" -Headers @{ "X-Request-ID" = "manual-check-99" } | Out-Null
Start-Sleep -Seconds 1
docker compose logs api --since 30s | Select-String "manual-check-99"
```

**Expect:** log lines carrying **your** id:

```json
{"ts":"...","level":"INFO","logger":"uvicorn.access",
 "msg":"172.24.0.1:51884 - \"GET /readyz HTTP/1.1\" 200",
 "service":"adaptive-ai-interviewer","env":"local","request_id":"manual-check-99"}
```

That is the correlation id: you chose it, the API honoured it, and every line
for that request carries it.

To see a full OTel span printed, add `OTEL_EXPORTER=console` to `.env`, then
`docker compose restart api`.

### Langfuse (optional, six containers, several GB)

```powershell
docker compose -f infra/langfuse/docker-compose.langfuse.yml up -d
```

Wait ~60 s, then open **http://localhost:3000** — login `dev@example.com` /
`local-dev-password`. Point the app at it in `.env`:

```
OTEL_EXPORTER=langfuse
LANGFUSE_HOST=http://host.docker.internal:3000
LANGFUSE_PUBLIC_KEY=pk-lf-local-dev
LANGFUSE_SECRET_KEY=sk-lf-local-dev
```

Stop it with `docker compose -f infra/langfuse/docker-compose.langfuse.yml down -v`.

## L. The complete suite (what CI runs)

```powershell
cd backend
..\.venv\Scripts\python.exe -m ruff check .
..\.venv\Scripts\python.exe -m ruff format --check .
..\.venv\Scripts\python.exe -m mypy --strict app
..\.venv\Scripts\python.exe -m pytest
cd ..
```

**Expect:** `All checks passed!` · `79 files already formatted` ·
`Success: no issues found in 41 source files` · `414 passed, 1 deselected`.

The 1 deselected is the live smoke test — excluded by default so the suite never
depends on a vendor being up.

Secret scan (needs Docker):

```powershell
docker run --rm -v "${PWD}:/repo" ghcr.io/gitleaks/gitleaks:latest detect --source=/repo --config=/repo/.gitleaks.toml --redact --no-banner
```

**Expect:** `no leaks found`.

## M. Clean-start verification

This is the Phase-1 gate *"`docker compose up` from a clean clone gives a
working API"*. **It destroys the database volume** — every account you created
above is deleted.

```powershell
docker compose down -v
docker compose up -d --build
Start-Sleep -Seconds 20
Invoke-RestMethod http://localhost:8080/readyz | ConvertTo-Json -Depth 5
```

**Expect:** migrations re-run from empty, all three services healthy, and
`readyz` returns `ready`. Re-run step G — the tables exist again and `users` is
empty.

---

## Stopping everything

```powershell
docker compose down
docker compose -f infra/langfuse/docker-compose.langfuse.yml down -v
```

## If something fails

| Symptom | Likely cause |
|---|---|
| `curl`/`Invoke-RestMethod` refuses to connect | Wrong port. Run `docker compose port api 8000`. |
| `ConfigError: DATABASE_URL is required` | You ran Python from `backend/`. Run from the repository root. |
| `api` unhealthy | `docker compose logs api`. Usually a bad `.env` — the error names the variable. |
| Integration tests skip | Postgres/Redis are down. `docker compose up -d`, then re-run. |
| Smoke test skips | No `NVIDIA_API_KEY` in `.env`. That is fine — everything else still runs. |
