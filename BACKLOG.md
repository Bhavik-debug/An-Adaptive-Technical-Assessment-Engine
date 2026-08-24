# Backlog — deliberately cut, not forgotten

Scoping discipline is a hiring signal; silently missing features are not.
Everything here was considered and cut on purpose, with the reason recorded.

## Cut from the project entirely

| Item | Why |
|---|---|
| **Voice / STT / TTS** | ~2,000 lines of real-time plumbing for zero AI-engineering credibility, and it makes grading *worse*: ASR errors get graded as candidate errors. The honest answer is a latency budget showing it would not have been usable. |
| **Team / recruiter screening mode** | Changes the compliance surface entirely (candidate consent, adverse-impact analysis, data processing agreements). Future work only. |
| **Company-specific question packs** | V3 stretch at best; the bank is already the project's biggest time sink. |

## Deferred by version

- **V1 (Day 30) ships without**: coding round, billing, agents, PDF export.
  Deliberate — the adaptive engine and grading validity determine whether every
  other feature is measuring anything real.
- **V2 (Day 40) ships without**: the agent round and difficulty calibration.
  Deliberate — calibration depends on having real response data to calibrate against.

## Pre-agreed cut order if the schedule slips

Cut in exactly this sequence, never out of order:

1. Interviewer personas + fairness measurement (Phase 9)
2. Load test + Grafana dashboard (Phase 9)
3. The agent-vs-FSM A/B (keep the agent, drop the measurement)
4. PDF export (Phase 8)
5. Real payment gateway → simulated checkout (Phase 8)
6. C++ in the sandbox → Python only (Phase 7)
7. The whole coding round → "disabled in this deployment"

**Never cut below this line:** the adaptive engine · the grader · the eval suite ·
the deploy · the docs.

## Noted during the build

_(items discovered mid-build that are out of scope go here rather than into the sprint)_

- Reduced item count vs. the original blueprint: 150 items for V1, 250 by V3
  (was 300/1500). 110 items is an acceptable floor; **unreviewed items never are**.
- Labelled eval set: 120 answers for V1, 200 by V3 (was 300).
