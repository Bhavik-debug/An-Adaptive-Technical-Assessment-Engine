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

**Day 10 — deferred deliberately.**

- **Deciding what to do about the reranker.** Measured on this set it makes
  every metric worse for 42x the latency, but it is *perfect* on exact-term
  queries and the set is far too small to justify removing a component. The
  decision needs: a larger bank (sprints 2 and 3), a larger query set, and
  confidence intervals. Options when that data exists — run it only for
  keyword-shaped queries, run it only when hybrid's top RRF scores are close,
  swap to a smaller cross-encoder, or drop it. **Deliberately not decided now:**
  changing a default because of 40 queries is tuning against the evaluation set.
- **Growing the evaluation set toward the plan's 100 queries.** 40 gives error
  bars wide enough that differences under ~0.05 are noise; the hybrid-vs-vector
  MRR gap of 0.013 is smaller than one query is worth. Also needed: a second
  annotator on a subset, so label noise can be quantified instead of assumed.
- **A held-out query set.** This set has now been used to *form* hypotheses (the
  `search_document` experiment), so it is no longer fully independent of the
  system's design. Confirming any future change needs queries the design has
  never seen.
- **Confidence intervals and significance testing.** Bootstrap over queries
  would take a few lines and turn "hybrid is better" into "hybrid is better,
  probably". Not done because with n=40 the honest answer is already "these
  differences are small", and a CI would mostly confirm that.
- **A CI regression gate on retrieval metrics.** Plan section 12 wants
  `evals/` running in CI and failing the build on a quality regression. The
  numbers now exist to gate on, but the run needs ~2.5 minutes and a ~1 GB model
  download, so it belongs behind the same opt-in marker as the other real-model
  tests, or on a nightly rather than per-commit schedule. Phase 6 work.
- **Relevance thresholds.** Explicitly out of scope for Day 10 and still are.
  Any "score below X is irrelevant" rule needs calibration data this set is too
  small to provide, and adding one now would be tuning.
- **The `MVCC` query (eq-003) is a permanent miss for all four modes.** Kept in
  the set on purpose. The acronym appears nowhere in `db-conc-003` - not in its
  text, not in its concept keys - so lexical cannot match an absent string and
  vector has almost nothing to work with. **This is a question-bank content gap,
  not a retrieval bug**, and the fix is authoring: it belongs with the pending
  human review, not with the search code.
- **Lexical search returns nothing for 28 of 40 queries.** `websearch_to_tsquery`
  joins terms with AND, so any conversational query fails to match. An OR
  fallback when the AND query returns nothing would raise lexical recall
  substantially. Not done today for the same reason as everything else here: it
  is a retrieval change, and Day 10 measures rather than changes.

**Day 9 — deferred deliberately.**

- **Bank sprint 2: 50 items (OS, networks, API design).** Plan Day 9 pairs the
  reranker with a second authoring sprint. This session was scoped to the
  reranker only, so the bank is still 60 items — and still 0 reviewed. The
  taxonomy has no `os`, `networks` or `api_design` branches yet either; those
  arrive with the items.
- **Measuring whether reranking actually helps.** Day 9 makes reranking possible
  and observable; it does not show that it improves anything on *this* bank. The
  four-row ablation (vector / BM25 / hybrid / hybrid+rerank) with Recall@10, MRR
  and nDCG@10 is Day 10's exit gate, and `PipelineResult` deliberately keeps the
  pre-rerank ordering so both rows can come from one run.
- **A relevance threshold.** Nothing filters on the reranker score, on purpose.
  The score is an unbounded logit, so "below X is irrelevant" needs labelled data
  to calibrate against — Day 10's. A guessed constant would look principled and
  be arbitrary.
- **Tuning `RERANK_CANDIDATE_K`.** Fixed at the plan's 40. More candidates means
  strictly better recall into stage 2 and linearly more model time; where that
  curve flattens is an empirical question Day 10 can answer and today cannot.
- **Making reranking fast enough to be interactive.** Measured at **~100 ms per
  candidate** on this laptop, so 40 candidates is ~4 s — about 16x plan §5.3's
  ~250 ms estimate. The gap is document length, not the model: one-sentence
  documents cost ~27 ms each, real `search_document` texts are 150-200 tokens.
  Four levers exist, each a trade against ranking quality: fewer candidates,
  truncating the document shown to the reranker, an INT8-quantised ONNX build
  (fastembed ships none for this model), or a smaller cross-encoder such as
  `ms-marco-MiniLM-L-6-v2`, which fastembed does support at ~1/13th the size.
  Deliberately not chosen today - Day 9's goal was a correct, observable
  architecture, and picking among these without Day 10's Recall/nDCG numbers
  would be trading accuracy for speed blind.
- **Caching rerank scores.** A (query, question, model) score is deterministic
  and cacheable, and the same question will be scored repeatedly across a
  session. Worth doing when there is traffic to measure; today it would be a
  cache with no hit rate to justify its invalidation rules.
- **Guarding against an embedding-backend switch without re-ingest.** Found while
  running Day 9's demo: setting `EMBEDDING_BACKEND=hashing` against a bank
  embedded with `bge-small` returns nonsense from vector search, silently,
  because query and stored vectors come from different models. Ingest records
  `embedding_model` per row precisely to detect this, but *search* never checks
  it. A one-line comparison at query time would turn a silent wrong answer into a
  loud error. It is a Day 8 concern discovered on Day 9, and deliberately not
  fixed here rather than edited into a day it does not belong to.

**Day 8 — deferred deliberately.**

- **Caching the query embedding.** Embedding the query is ~33 ms of a 35 ms
  hybrid search — over 90% of it — while both SQL queries together are ~3 ms.
  A small LRU or Redis cache keyed on the normalised query text is the obvious
  first optimisation. Not done now because there is no traffic to measure
  against, and a cache added before a latency budget exists is a cache nobody
  can size. The measurement is recorded so the decision has evidence when it is
  taken.
- **Tuning the HNSW parameters** (`m`, `ef_construction`, `ef_search`). Left at
  pgvector's defaults. At 60 rows the planner does not use the index at all, so
  any value would be as good as any other and "tuned" would mean "guessed".
  Day 10's retrieval evaluation produces the recall numbers to tune against.
- **Tuning the RRF constant.** k = 60 is the published default and is
  configurable. Same reason: Day 10 can measure whether another value wins on
  this bank; today there is nothing to compare against.
- **The bge query instruction prefix.** The model card offers an optional
  "Represent this sentence for searching relevant passages: " prefix for short
  queries and says v1.5 generally does not need it; fastembed applies none, which
  was checked rather than assumed and is pinned by a test. Whether it helps *this*
  bank is a Day 10 measurement, not a guess.
- **Weighting the two retrievers.** RRF currently treats vector and lexical as
  equally credible. A per-source weight is a one-line change to `rrf.py`, and
  choosing the weights needs the ablation table Day 10 produces.
- **Acronyms that appear in no question's text.** Searching "MVCC" ranks nothing
  useful: the acronym is in `db-conc-003`'s *concept key*
  (`multiversion_concurrency_control`), which humanises to "multiversion
  concurrency control", and nowhere in its prose. Vector search cannot bridge an
  acronym it has little context for, and lexical search cannot match a string
  that is absent. This is a **question-bank content gap, not a retrieval bug** —
  the fix is authoring, and it belongs with the sprint-2 items and the pending
  human review, not with the search code.
- **Concurrent vector and lexical queries.** Run sequentially. Two round trips
  to a local Postgres are ~3 ms combined, and one `AsyncSession` cannot serve
  concurrent statements — doing it properly needs two sessions, for no
  measurable gain at this size.

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
