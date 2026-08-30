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

**Day 7 — deferred deliberately.**

- **Human review of the 60 items.** Not deferred so much as *not yet done*: all
  60 are `review_status: "drafted"`. The Phase 2 exit gate does not close until
  a person has read each one and `validate_question_bank.py --require-reviewed`
  exits 0. ~90 minutes of work at §6.4's 90-seconds-per-item budget.
- **A concept-key vocabulary file.** 220 keys across 60 items, only 13 reused.
  The validator warns on near-miss keys, which catches typos, but nothing stops
  a genuinely new synonym being minted. A committed vocabulary with an
  allowlist check is the fix; doing it now, before sprints 2 and 3 reveal which
  keys actually recur, would be guessing at the vocabulary rather than
  observing it.
- **Difficulty calibration.** Every `b` is a comparative human estimate and
  every `discrimination_a` is 1.0, because calibration (§5.11) needs real
  response data that does not exist yet. Deliberate: a fitted `a` from zero
  observations is a fabricated number.
- **Deleting orphaned questions on ingest.** Ingest reports rows that are in the
  database but no longer in the files, and leaves them alone. `turns.question_id`
  references `questions.id`, so a cascade would erase interview history to tidy
  up a dataset edit. Revisit with a real retirement workflow (`retired_at`)
  in Phase 5, when there is history worth protecting.

**Day 5 — deferred deliberately.**

- **A bulk fixture recorder.** ~~`scripts/record_llm_fixture.py` records one named
  call at a time, which is right for Phase 1's single probe. Phase 4's grading
  evals will need dozens, keyed the way plan §12.3 describes.~~ **Delivered on
  Day 6**: `app/llm/recording.py` + `scripts/record_llm_fixtures.py` take a JSON
  recording plan and record every entry through the same `call_structured()`
  path, under the same `fixture_key()`, into the same `FixtureStore`. The
  single-recipe script is now a front door onto the same engine. Phase 4 adds its
  schemas to `recording.SCHEMAS` and its dataset as a plan file; no new machinery.
- **A SQLite fixture store.** §12.3 suggests SQLite; this uses one JSON file per
  recording. JSON is reviewable in a pull request and diffs legibly, which
  matters more at one fixture than lookup speed does. Revisit if the directory
  grows past a few hundred files — the `FixtureStore` interface is the only
  thing that would change.
- **Recording provider *failures* from real life.** The stub can replay a
  recorded 429, but every such fixture today is authored rather than captured:
  you cannot ask a provider to rate-limit you on demand. Capturing real ones
  needs a passive recorder left running, which is Phase 6 work at the earliest.
- **Semantic fidelity of synthesized answers.** Synthesis is shape-correct and
  meaningless by design. Making it *plausible* (a grade that looks like a grade)
  would be actively harmful — it would make a bad eval look like a good one.
  Deliberately not doing this, ever.

**Day 4 — deferred deliberately, all to Phase 6 unless noted.**

- **Docker image build in CI.** Plan §14.3 puts `build → GHCR → SSH deploy` in
  the Phase 6 pipeline. Day 4's brief is lint → mypy → pytest; adding a 3-minute
  image build now buys a check that `docker compose up` already performs daily.
- **Pinning CI action and container versions by digest.** §14.3 asks for pinned
  base digests. `gitleaks` and `minio` currently float, because a version pinned
  without ever running the pipeline is a guess that fails at pull time with a
  confusing error. Pin every one of them on the first green CI run.
- **Metrics** (Prometheus/OTel meters: p50/p95 latency, cache hit rate, error
  rate by task, cost per interview). §12.4 is explicit that operational metrics
  are a separate thing from evals and belong with a dashboard. Metrics computed
  before there is real traffic measure nothing.
- **Auto-instrumenting SQLAlchemy and Redis.** Two more dependencies and a large
  increase in span volume, for questions nothing is asking yet. The `obs/` seam
  makes it a one-line change when a slow query actually needs finding.
- **Log shipping** (Loki/ELK). Days 1–29 are local-only; `docker compose logs`
  is the log backend until there is somewhere to ship to.
- **Trace sampling.** One developer's laptop produces every span it should keep.
  Revisit if the Phase 6 VM's disk says otherwise.
- ~~**Booting the Langfuse stack.**~~ **Done on Day 5.** Booted, a span exported
  over OTLP and read back through the API. Four bugs in the compose file found
  and fixed in the process; see `docs/observability.md`.
