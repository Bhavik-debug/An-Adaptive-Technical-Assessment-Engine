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

## Recording a new one

```bash
cd backend
python scripts/record_llm_fixture.py probe --dry-run   # print the key, write nothing
python scripts/record_llm_fixture.py probe             # make the call and write it
```

This is the only thing in the repository that deliberately spends quota. It is
run by hand, never by a test and never by CI. Read what it wrote before you
commit it.

## What is here

| File | What it is |
|---|---|
| `connectivity_probe.json` | A real Nemotron response to the Phase-1 exit-gate probe, recorded from `integrate.api.nvidia.com`. Replaying it exercises the whole chokepoint — router, provider, JSON extraction, schema validation, cost accounting, tracing — with no network. |
