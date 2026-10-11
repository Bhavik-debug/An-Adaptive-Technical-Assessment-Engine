"""Drive a complete interview through the PUBLIC API and print the report.

Phase 5 exit gate (plan section 3): "A shell script drives a complete
10-question interview through the public API and prints a report".  This is a
client: it uses only HTTP - register, create, start, the SSE stream, turn
submission, the report - exactly as a frontend would.  It touches no database
and no server internals.

The scripted candidate answers primary questions alternately well and poorly
(odd: a detailed answer; even: "I'm not sure"), and answers every follow-up
well.  Run it against a server started with the scripted stub
(``LLM_PROVIDER_ORDER=stub LLM_STUB_ON_MISSING=scripted``) to stay offline -
see ``docs/reporting.md`` - or against any server, at the cost of real calls.

On its first turn it also exercises gate 2 through the API: the same turn_id is
re-sent sequentially and concurrently (must replay, 200), and once with a
different answer (must be refused, 409).

Usage::

    python scripts/scripted_interview.py --base-url http://127.0.0.1:8000
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import uuid
from collections.abc import AsyncIterator
from typing import Any

import httpx

STRONG = (
    "I would start from the access pattern and the data volume, pick the structure "
    "that makes the hot query cheap, explain the trade-offs in memory and write cost, "
    "and say how I would measure it in production before and after the change."
)
WEAK = "I'm not sure about this one, honestly."
SPEC = {
    "role": "Backend Engineer",
    "level": "mid",
    "mode": "standard",
    "duration_minutes": 90,
    "item_budget": 10,
    "topics": [
        {"topic_key": "dsa", "jd_weight": 0.9},
        {"topic_key": "databases", "jd_weight": 0.7},
    ],
}


async def _events(
    http: httpx.AsyncClient, url: str, headers: dict[str, str], after: int
) -> AsyncIterator[dict[str, Any]]:
    """Parse the SSE stream into dicts.  ``Last-Event-ID`` resumes after ``after``."""
    async with http.stream("GET", url, headers={**headers, "Last-Event-ID": str(after)}) as r:
        r.raise_for_status()
        block: list[str] = []
        async for line in r.aiter_lines():
            if line:
                block.append(line)
                continue
            fields = dict(item.split(": ", 1) for item in block if not item.startswith(":"))
            block = []
            if "event" in fields:
                yield {
                    "id": int(fields["id"]) if "id" in fields else None,
                    "event": fields["event"],
                    "data": json.loads(fields["data"]),
                }


async def run_interview(
    base_url: str,
    *,
    email: str | None = None,
    password: str = "scripted-candidate-pass-9",
    timeout_s: float = 180.0,
) -> dict[str, Any]:
    """Run the whole interview; return the evidence (counts, checks, report)."""
    email = email or f"scripted-{uuid.uuid4().hex[:10]}@example.com"
    evidence: dict[str, Any] = {"email": email, "turns": [], "gate2": {}}
    async with httpx.AsyncClient(base_url=base_url, timeout=60) as http:
        reg = await http.post("/api/auth/register", json={"email": email, "password": password})
        if reg.status_code == 409:
            reg = await http.post("/api/auth/login", json={"email": email, "password": password})
        reg.raise_for_status()
        headers = {"Authorization": f"Bearer {reg.json()['access_token']}"}

        created = await http.post("/api/sessions", json=SPEC, headers=headers)
        created.raise_for_status()
        sid = created.json()["session_id"]
        evidence["session_id"] = sid
        started = await http.post(f"/api/sessions/{sid}/start", headers=headers)
        started.raise_for_status()
        question = started.json()["question"]
        primary = 0
        events = _events(http, f"/api/sessions/{sid}/stream", headers, started.json()["seq"])

        async with asyncio.timeout(timeout_s):
            while True:
                if not question["is_follow_up"]:
                    primary += 1
                answer = STRONG if question["is_follow_up"] or primary % 2 == 1 else WEAK
                turn_id = str(uuid.uuid4())
                body = {"turn_id": turn_id, "answer": answer}
                url = f"/api/sessions/{sid}/turns"
                first = await http.post(url, json=body, headers=headers)
                if first.status_code != 202:
                    raise RuntimeError(f"turn refused: {first.status_code} {first.text}")
                if not evidence["gate2"]:
                    again = await http.post(url, json=body, headers=headers)
                    burst = await asyncio.gather(
                        *(http.post(url, json=body, headers=headers) for _ in range(4))
                    )
                    other = await http.post(
                        url,
                        json={"turn_id": turn_id, "answer": answer + " (edited)"},
                        headers=headers,
                    )
                    evidence["gate2"] = {
                        "turn_id": turn_id,
                        "sequential_retry": again.status_code,
                        "concurrent_retries": sorted(r.status_code for r in burst),
                        "different_answer_same_key": other.status_code,
                    }
                evidence["turns"].append(
                    {
                        "turn_id": turn_id,
                        "follow_up": question["is_follow_up"],
                        "question_id": question["question_id"],
                    }
                )
                async for event in events:
                    if event["event"] == "next_question":
                        question = event["data"]
                        break
                    if event["event"] == "grading_failed":
                        raise RuntimeError(f"grading failed: {event['data']}")
                    if event["event"] == "session_complete":
                        question = None
                        break
                if question is None:
                    break

            report = None
            for _ in range(200):
                r = await http.get(f"/api/sessions/{sid}/report", headers=headers)
                if r.status_code == 200:
                    report = r.json()
                    break
                if r.status_code >= 400:
                    raise RuntimeError(f"report failed: {r.status_code} {r.text}")
                await asyncio.sleep(0.1)
        await events.aclose()

        after = await http.post(
            f"/api/sessions/{sid}/turns",
            json={"turn_id": str(uuid.uuid4()), "answer": STRONG},
            headers=headers,
        )
        evidence["answer_after_completion"] = after.status_code
        evidence["primary_questions"] = sum(not t["follow_up"] for t in evidence["turns"])
        evidence["follow_ups"] = sum(t["follow_up"] for t in evidence["turns"])
        evidence["report"] = report
    return evidence


def _print(evidence: dict[str, Any]) -> None:
    report = evidence["report"] or {}
    stats = report.get("stats") or {}
    print(f"session {evidence['session_id']}  ({evidence['email']})")
    print(
        f"primary questions: {evidence['primary_questions']}   "
        f"follow-ups: {evidence['follow_ups']}   "
        f"answer after completion -> HTTP {evidence['answer_after_completion']}"
    )
    print(f"gate 2 (same turn_id): {evidence['gate2']}")
    print(
        f"report: status={report.get('status')}  narrative_source={report.get('narrative_source')}"
    )
    print(f"stop reasons: {stats.get('stop_reasons')}   counts: {stats.get('counts')}")
    for topic in stats.get("topics", []):
        o = topic["observed"]
        shown = (
            f"{o['display']} [{o['interval'][0]}-{o['interval'][1]}]"
            + (" (insufficient evidence)" if o["insufficient_evidence"] else "")
            if o
            else "not assessed"
        )
        print(f"  {topic['topic_key']:<12} {shown}")
    print(f"gaps: {stats.get('gaps')}   strengths: {stats.get('strengths')}")
    print(f"needs more assessment: {stats.get('needs_more_assessment')}")
    narrative = report.get("narrative") or {}
    print(f"narrative: {narrative.get('summary')}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--email")
    parser.add_argument("--json", action="store_true", help="print the evidence as JSON")
    args = parser.parse_args()
    evidence = asyncio.run(run_interview(args.base_url, email=args.email))
    if args.json:
        print(json.dumps(evidence, indent=2, default=str))
    else:
        _print(evidence)
    ok = (
        evidence["primary_questions"] == SPEC["item_budget"]
        and evidence["report"] is not None
        and evidence["report"]["status"] == "ready"
        and evidence["answer_after_completion"] == 409
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
