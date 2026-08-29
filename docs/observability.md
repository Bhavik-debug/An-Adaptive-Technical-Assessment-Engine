# Observability

> Plan §14.2 (the trace shape and the non-negotiable LLM span attributes),
> §13.1 (`obs/` as a module), §13.10 (OpenTelemetry + self-hosted Langfuse).
> Built on Day 4.

The one question this exists to answer:

> **When something goes wrong, what happened, where, and how long did it take?**

---

## The vocabulary, in plain language

Read this once. Every term below is used in the code and in the rest of this
document.

| Term | What it means | In this project |
|---|---|---|
| **Observability** | Being able to work out what a running system did, from the outside, without attaching a debugger. | The difference between "grading is slow" and "grading spent 1,240 ms of 1,600 ms inside one NVIDIA call, on prompt version v3." |
| **Logging** | Writing down discrete events as they happen. | `app.request` writes one line per HTTP request; `app.llm.client` writes one per model call. |
| **Structured logging** | Each line is a machine-readable object with named fields, not a sentence. | Every line is one JSON object. `llm.cost_usd` is a field you can sum, not text you have to parse. |
| **Metrics** | Numbers aggregated over time — counters, histograms. | **Not built yet.** Plan §12.4 puts the operational dashboard in a later phase; metrics before there is real traffic measure nothing. |
| **Tracing** | Recording the causal, timed structure of one operation as it moves through the system. | One HTTP request → one model call → the answer, as a tree with durations. |
| **Span** | One timed operation: a name, a start, an end, attributes, and a status. | `llm.grade_answer`, 1,240 ms, `llm.cost_usd=0.00062`, OK. |
| **Trace** | A tree of spans that belong to the same operation. | The whole request, root span plus every child. |
| **Trace ID** | The 32-hex-character id every span in one trace shares. | Printed on every log line as `trace_id`, so a log line and a span find each other. |
| **Span ID** | The 16-hex-character id of one span. | `span_id` on the log lines emitted while that span was open. |
| **Correlation ID** | An id that ties together everything belonging to one request. Ours is a **request ID**. | `X-Request-ID`. Always present, returned to the caller, quotable in a bug report. |
| **OpenTelemetry (OTel)** | A vendor-neutral standard and set of libraries for producing traces. | We call the OTel API; where the data goes is a config value. |
| **Exporter** | The component that ships finished spans somewhere. | `none` / `console` / `otlp` / `langfuse`. |
| **Instrumentation** | Code that creates spans. *Auto*-instrumentation does it for a library without you writing anything. | `opentelemetry-instrumentation-fastapi` creates the per-request span; we hand-write the LLM span. |
| **Distributed tracing** | Following one operation across process boundaries, by passing the trace id in a header (`traceparent`). | Not needed yet — one process. The wiring is already correct, so the Phase 6 frontend and the arq worker join the same trace for free. |
| **Langfuse** | An open-source observability product built specifically for LLM applications. Self-hosted here. | An OTLP *destination*. It reads the same spans and renders them as generations with token and cost roll-ups. |
| **LLM tracing** | Tracing where the interesting spans are model calls, and the interesting attributes are prompt version, tokens and cost. | Exactly what `app/obs/spans.py` produces. |

### Logs vs metrics vs traces — when to reach for which

- A **log** answers *"what happened?"* — with detail, for one event.
- A **trace** answers *"where did the time go, and what called what?"*
- A **metric** answers *"how often, and how bad, over time?"*

They are not interchangeable. You cannot get p95 latency by reading logs, and
you cannot debug a single broken interview from a p95.

---

## Architecture

```
HTTP request  ──►  X-Request-ID assigned or accepted   (obs/middleware.py)
      │            OTel server span opened             (auto-instrumentation)
      ▼
  endpoint     ──►  user id bound after auth            (deps.py)
      │
      ▼
call_structured()  ──►  span  llm.<task>                (obs/spans.py)
      │                 attributes taken from CallMeta — never recomputed
      ▼
  provider router ──►  NVIDIA ──►  Nemotron
      │
      ▼
  validated answer + CallMeta
      │
      ├──►  one JSON log line   "llm_call"   with the same trace_id
      └──►  span attributes + OK/ERROR status
                    │
                    ▼
            exporter (none | console | otlp | langfuse)
```

Two independent outputs carry the same facts, on purpose:

1. **The structured log** never leaves the host and is always produced. This is
   the plan's own Phase 1 cut-line: *"log spans as structlog JSON; keep the span
   attributes identical so you can swap the exporter."*
2. **The span** goes wherever the exporter points. It carries the same
   attribute names.

Both stamp `trace_id`, so switching between them is a filter rather than a hunt.

### Files

| File | Owns |
|---|---|
| `app/obs/context.py` | `request_id` / `user_id` context vars |
| `app/obs/redaction.py` | what must never be written down |
| `app/obs/logging.py` | the JSON formatter, the context filter, `configure_logging` |
| `app/obs/tracing.py` | OTel setup, exporter choice, `instrument_app` |
| `app/obs/spans.py` | `CallMeta` → span attributes |
| `app/obs/middleware.py` | request id, per-request latency line |

---

## What is on an LLM span

Plan §14.2 calls these **non-negotiable**, and there is a test asserting each
one is present (`tests/unit/obs/test_llm_spans.py`):

`llm.task` · `llm.prompt_version` · `llm.prompt_fingerprint` ·
`llm.schema_fingerprint` · `llm.provider` · `llm.model` · `llm.tier` ·
`llm.temperature` · `llm.input_tokens` · `llm.output_tokens` ·
`llm.reasoning_tokens` · `llm.cost_usd` · `llm.price_known` · `llm.cache_hit` ·
`llm.schema_retry_count` · `llm.failover_count` · `llm.structured_mode` ·
`llm.reasoning_enabled` · `llm.latency_ms` · `llm.session_id` · `llm.turn_id` ·
`llm.user_id` · `llm.plan`

Plus a small `gen_ai.*` set (`gen_ai.system`, `gen_ai.request.model`,
`gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens`) and
`langfuse.observation.type=generation`, so a general-purpose trace viewer knows
this span is a model call.

**Why `prompt_version` is the one that matters most.** Without it you cannot
attribute a quality regression to a prompt change — and doing that attribution
is what production AI engineering *is*.

**Why `structured_mode` is the one that matters second (Day 5).** It says where
the answer came from:

| Value | Meaning |
|---|---|
| `json_schema` / `json_object` / `prompt_only` | A live provider call, and how hard the endpoint was asked to constrain it |
| `cache` | Served from the Redis response cache |
| `stub_replay` | A **recorded real** model response, replayed offline |
| `stub_synthesized` | **Invented** from the JSON Schema — shape-correct, semantically meaningless |

That last row is the important one. From Phase 2 onwards this project measures
LLM quality — retrieval nDCG, grading QWK, injection ASR — and every one of those
numbers is worthless if the answers underneath were synthesized rather than real.
Because the field is on every span and every log line, "were any of these
synthesized?" is a filter, not an investigation. The provider sets it; a fixture
file cannot influence it (see `Fixture.as_result`).

**Where these numbers come from.** All of them are computed by Day 3's
`call_structured()` and read off `CallMeta`. Day 4 adds no accounting of its
own; if it did, a cost report and a trace could disagree about the same call.

---

## What is never written down

| Never logged or traced | Why |
|---|---|
| `NVIDIA_API_KEY`, `SECRET_KEY` | Registered with the redactor at boot as literal strings, so any line containing one has it removed |
| Passwords and argon2 hashes | Field-name rule (`password`, `hash`) and value-shape rule (`$argon2...`) |
| Access and refresh tokens, `Authorization`, cookies | Field-name rule, plus a JWT and `Bearer …` shape rule |
| Prompts, model answers, model reasoning | Structural: `CallMeta` does not contain them, so there is nothing to leak |
| Candidate email addresses | Masked as `[EMAIL]`. Plan §14.1 requires a redaction pass before anything reaches an LLM; a log file is the same exposure with a longer retention |

Three mechanisms, because each catches what the others miss — a **field-name**
rule, a **value-shape** rule, and a **known-literal** backstop for a secret
logged under a name nobody predicted. All three live in
`app/obs/redaction.py` and every one of them has a test.

Exception events on spans are assembled by hand rather than with OTel's
`record_exception`, so the message and stack go through the redactor before a
span leaves the process for a third-party service.

### `LOG_LEVEL=DEBUG` does not turn on prompt logging

`openai._base_client` writes the **entire request options object — prompt
included — at DEBUG**. Setting `LOG_LEVEL=DEBUG` to investigate our own code
would otherwise silently start recording every resume and every candidate answer
that passes through the chokepoint.

`configure_logging()` therefore floors `openai`, `httpx`, `httpcore` and
`urllib3` at INFO, unconditionally. Redaction cannot help here — a candidate's
answer is not a shape a regular expression can recognise — so the only correct
behaviour is not to write it down.

Measured against the live endpoint, with `LOG_LEVEL=DEBUG`:

| | before the floor | after |
|---|---:|---:|
| characters logged for one call | 10,483 | 1,732 |
| contains the API key | no | no |
| contains the prompt / answer | **yes** | no |

If you genuinely need the SDK's wire log, raise it by hand, deliberately, on a
machine with no real candidate data:

```python
logging.getLogger("openai").setLevel(logging.DEBUG)
```

---

## Running it locally

### The default: nothing to run

```bash
docker compose up
```

`OTEL_EXPORTER=none`. Spans are created, log lines carry `trace_id`, nothing is
shipped anywhere. This is the normal development mode.

```bash
docker compose logs -f api            # one JSON object per line
```

Prefer prose while developing? `LOG_FORMAT=text` in `.env`. Redaction is
identical in both.

### Seeing spans without any infrastructure

```bash
# .env
OTEL_EXPORTER=console
```

Every finished span is printed to stdout as JSON, attributes and all. Good for
answering "what does the instrumentation actually produce?" in ten seconds.

### Self-hosted Langfuse

```bash
docker compose -f infra/langfuse/docker-compose.langfuse.yml up -d
# wait for http://localhost:3000 to answer; first boot runs migrations
```

Log in with `dev@example.com` / `local-dev-password`. The stack creates a
project with these keys on first boot:

```bash
# .env
OTEL_EXPORTER=langfuse
LANGFUSE_HOST=http://host.docker.internal:3000   # http://localhost:3000 if the API runs on the host
LANGFUSE_PUBLIC_KEY=pk-lf-local-dev
LANGFUSE_SECRET_KEY=sk-lf-local-dev
```

Restart the API, make a call, and the trace appears under Tracing in the UI with
tokens and cost attached.

> **Verified on 2026-08-29 (Day 5).** The stack was booted, a real LLM span was
> exported over OTLP, and it was read back through the Langfuse API. Getting
> there fixed four bugs in the compose file, each now marked with a comment
> where it was: an unquoted `ENCRYPTION_KEY` that YAML turned into the integer
> `0`; a Redis with no password that rejected Langfuse's `AUTH`;
> `LANGFUSE_INIT_*` on the worker instead of the web container (the stack came
> up *healthy* and every API call returned 401, because no project had ever been
> created); and `dev@localhost` as the init email, which Langfuse rejects for
> having no TLD.

What Langfuse shows for one traced call:

```
trace  interview_turn
└── llm.connectivity_probe        type=GENERATION   model=stub/deterministic-v1
                                  usage: in=322  out=19  total=341
```

The `llm.*` attribute set survives the round trip intact, and the token counts
land in Langfuse's own `usage` field — that is what `gen_ai.usage.*` and
`langfuse.observation.type=generation` are for.

**And the privacy guarantee holds across the boundary.** The stored observation
was checked for the prompt text, the answer text, and the model's own words:
none of them are present. Only metadata crosses into the third-party service,
which is the property `CallMeta` was designed to make structural.

To stop it, and reclaim the disk:

```bash
docker compose -f infra/langfuse/docker-compose.langfuse.yml down -v
```

### Any other OTLP backend

Langfuse is not load-bearing. Jaeger, Grafana Tempo, or an OpenTelemetry
Collector all work:

```bash
OTEL_EXPORTER=otlp
OTEL_EXPORTER_ENDPOINT=http://localhost:4318/v1/traces
```

That substitutability is the reason to instrument with OpenTelemetry rather than
a vendor SDK.

---

## Following one request end to end

```bash
curl -i -H 'X-Request-ID: my-trace-1' localhost:8000/readyz
# → X-Request-ID: my-trace-1  in the response headers

docker compose logs api | grep my-trace-1
```

Every line for that request — the request line with its `duration_ms`, any
`llm_call` lines, any errors with their stack — carries `request_id` and the
same `trace_id`. Paste the `trace_id` into Langfuse to see the same thing as a
timed tree.

If a request produced no id of your choosing, one was generated; it is in the
`X-Request-ID` response header either way.

---

## Deliberate omissions

| Not done | Why |
|---|---|
| Metrics (Prometheus, counters, histograms) | Plan §12.4 puts the operational dashboard in a later phase. Metrics before real traffic measure nothing. |
| Database and Redis auto-instrumentation | Two more dependencies and a lot of span volume, for questions nothing is asking yet. The `obs/` seam makes adding them a one-line change. |
| Log shipping (Loki, ELK) | Days 1–29 are local-only. `docker compose logs` is the log backend until Phase 6. |
| Sampling | One developer's laptop produces every span it should keep. |
| `structlog` | The plan's cut-line names it, but the codebase already uses `logging` everywhere. A JSON formatter on stdlib `logging` gives the same structured output without rewriting Day 1–3 call sites or adding a dependency. |
