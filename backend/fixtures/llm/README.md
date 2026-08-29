# Recorded LLM responses

Replayed offline by the stub provider (`app/llm/providers/stub.py`,
`LLM_PROVIDER_ORDER=stub`). Plan §3, Day 5.

## The safety rule

**Never record a call whose prompt or response contains candidate data.**

A fixture is committed to git. That makes it permanent, shared, and completely
outside the reach of the Day-4 log redactor — which only ever sees log lines,
never these files. No real resume, no real answer, no real grade, ever.
Recordings are for the connectivity probe and for synthetic examples authored
for the purpose.

## The file format

One JSON object per recording. `request_hash` is a SHA-256 digest of the
assembled provider request — model, tier, sampling settings, schema fingerprint, and every
message. It is computed by `app.llm.fixtures.fixture_key`, never typed by hand,
so a changed prompt is a different key and therefore a clean miss rather than a
stale answer served silently.

```jsonc
{
  "format_version": 1,
  "request_hash": "798692b4…",  // derived from the request; do not edit
  "description": "…",           // provenance, for humans; never parsed
  "text": "{\"ok\":true,…}",    // exactly what the provider returned
  "input_tokens": 322,
  "output_tokens": 19,
  "reasoning_tokens": 0,
  "finish_reason": "stop",
  "structured_mode": "stub_replay",
  "request_preview": { … }      // which request this belongs to; never matched on
}
```

A fixture may instead record a **failure**, which is how an offline test reaches
a retry, a failover or the circuit breaker without inventing a fake provider:

```jsonc
{ "format_version": 1, "request_hash": "…", "description": "…",
  "error": {"kind": "rate_limited", "message": "429 from upstream",
            "status_code": 429, "retry_after_s": 0.5} }
```

`kind` is one of `rate_limited`, `timeout`, `unavailable`, `auth`,
`bad_request`, `response` — mapped to a real exception from `app/llm/errors.py`
by an explicit table, so a fixture can never name an arbitrary importable class.

## Recording new ones

Two front doors onto one engine (`app/llm/recording.py`). Both push the call
through the real `call_structured()` and file the answer under the real
`fixture_key()`, so a recording made either way is the same recording.

**Many at a time — a recording plan (Day 6):**

```bash
cd backend
python scripts/record_llm_fixtures.py fixtures/recording_plans/connectivity_probe.json --dry-run
python scripts/record_llm_fixtures.py fixtures/recording_plans/connectivity_probe.json
python scripts/record_llm_fixtures.py <plan> --only grade_easy   # just one entry
python scripts/record_llm_fixtures.py <plan> --overwrite         # replace what exists
python scripts/record_llm_fixtures.py <plan> --fixture-dir /tmp/rec   # write elsewhere
```

**One named recipe at a time (Day 5):**

```bash
python scripts/record_llm_fixture.py probe --dry-run   # print the key, write nothing
python scripts/record_llm_fixture.py probe             # make the call and write it
```

These are the only things in the repository that deliberately spend quota. They
are run by hand, never by a test and never by CI. Read what they wrote before you
commit it.

Recording needs a real provider: set `LLM_PROVIDER_ORDER=nvidia` in `.env`. A
stub-only order is refused before anything runs, because the stub answers from
this very directory. `--dry-run` works with any configuration and needs no key —
it computes every fixture key and reports what *would* happen, calling nothing.

### The recording plan format

`backend/fixtures/recording_plans/*.json`. Deliberately not in this directory:
the store loads every `*.json` it finds here, and a plan is an input, not a
recording.

```jsonc
{
  "format_version": 1,
  "description": "optional, for humans",
  "requests": [
    {
      "name": "connectivity_probe",   // becomes <name>.json; [a-z0-9][a-z0-9_-]*
      "task": "connectivity_probe",   // a TaskName
      "schema": "ProbeAnswer",        // a key of app.llm.recording.SCHEMAS
      "inputs": {"token": "replay01"},// what the prompt template needs
      "note": "why this exists.",     // optional; copied into the description
      "temperature": 0.0              // optional; default = the routing table
    }
  ]
}
```

**A plan says what to record, never what was recorded.** `schema` names an entry
in an explicit table rather than an importable object — the same rule
`error.kind` follows — and any key outside the list above is rejected, so a plan
cannot supply `text`, `input_tokens`, `recorded_structured_mode` or
`request_hash`. Those come from the provider, which is the only thing that knows
them. A malformed plan is refused entirely, naming the file, the entry index and
the entry, before any call is made.

### What happens on a re-run

| Situation | What happens | Quota spent |
|---|---|:--:|
| No fixture for this request | recorded, written as `<name>.json` | yes |
| A fixture already exists | **skipped**, file untouched | **no** |
| A fixture exists, `--overwrite` | re-recorded into *the existing file* | yes |
| `--dry-run` | key computed and reported, nothing called | no |

The key is computed **before** the call, by pushing the entry through the
chokepoint with a probe that captures the assembled request and stops. So a
re-run of a fully-recorded plan costs nothing at all, and a real recording is
never replaced by accident — only on request.

### When one entry fails

The batch **continues**, and the exit code is 1. Recording is the expensive
operation in this repository; stopping at the first failure would throw away the
recordings that already succeeded and make you pay for them again. Each entry is
independent, each failure is reported with its name and its (redacted) reason,
and nothing is written for a failed entry. One router and one circuit breaker
serve the whole batch, so a provider that is genuinely down trips the breaker
after a few entries and the rest fail instantly rather than each waiting out a
timeout.

Error text is passed through the same redactor the log formatter uses, seeded
with this process's real `SECRET_KEY` and `NVIDIA_API_KEY`, so a provider error
that quotes an `Authorization` header is reported with the credential masked.

### A recording is a real answer or it is nothing

The recorder refuses to write a fixture when the answer did not come from a real
provider — `stub_replay`, `stub_synthesized` and `cache` are all rejected, and
the **provider** decides that label, not the plan file. A directory in which
invented data were indistinguishable from recorded data would poison every eval
that reads it.

## What is here

| File | What it is |
|---|---|
| `connectivity_probe.json` | A real Nemotron response to the Phase-1 exit-gate probe, recorded from `integrate.api.nvidia.com`. Replaying it exercises the whole chokepoint — router, provider, JSON extraction, schema validation, cost accounting, tracing — with no network. |
