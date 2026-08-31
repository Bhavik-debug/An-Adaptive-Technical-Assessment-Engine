# Reranking — putting the best candidate first

*Phase 2, Day 9. Written for someone who has never studied machine learning.
It follows on from [`docs/retrieval.md`](retrieval.md), which explains
embeddings, vector search, lexical search and RRF from scratch — read that
first if those are new.*

---

## The problem Day 8 left behind

Day 8 can find, out of 60 questions, the ~40 that have anything to do with your
query. That is **recall**: the right answer is somewhere in the list.

What it is less good at is **precision**: putting the right answer *first*.

The reason is structural, not a bug. Day 8's embedding model turns each question
into 384 numbers **before any query exists**. It has to summarise the question
into a fixed list of numbers while guessing what someone might one day ask. Then
a query arrives, becomes its own 384 numbers, and the two are compared. At no
point did the model look at the question and the query *together*.

That is fine for narrowing 60 (later 150, later still 100,000) down to 40. It is
not enough to reliably pick the single best one.

---

## 1. What a cross-encoder is

A model that reads the query and one candidate **at the same time**, as a single
piece of text, and outputs one number: how relevant this candidate is to this
query.

```
Query:      "How can database reads be made faster?"
Candidate:  "Explain why a B-tree index on a column makes lookups faster."
                              │
                              ▼
                    ┌───────────────────┐
                    │  cross-encoder    │   reads BOTH, together
                    └─────────┬─────────┘
                              ▼
                         -0.63          ← one relevance score
                                        (negative is normal — see §5)
```

Because it sees both texts at once, it can notice things two separate summaries
cannot: that the query asks about cache *invalidation* while this candidate only
discusses *eviction*; that "faster reads" and "index lookups" are the same idea
here even though the words differ; that a candidate mentions the right topic but
answers a different question about it.

---

## 2. How it differs from the Day 8 embedding model

Day 8's model is a **bi-encoder** — "bi" because each side goes through the
encoder separately.

```
BI-ENCODER  (Day 8, bge-small)                CROSS-ENCODER  (Day 9, bge-reranker)

  query    ──► [encoder] ──► vector ┐          [ query  SEP  candidate ] ──► [encoder] ──► score
                                    ├─► cosine
  question ──► [encoder] ──► vector ┘
             (computed once, offline)          (one model run per candidate, every query)
```

| | Bi-encoder | Cross-encoder |
|---|---|---|
| Sees the query and doc together? | ❌ never | ✅ always |
| Can precompute the document side? | ✅ once, at ingest | ❌ nothing can be precomputed |
| Can an index help? | ✅ HNSW over the vectors | ❌ there is nothing to index |
| Cost per query | one encode + an index lookup | **one model run per candidate** |
| Good at | recall — finding the right region | precision — picking the winner |

The last row is the whole point. **They are not competing; they are two halves
of one design.**

---

## 3. Why retrieval comes first, and why we rerank only a few

A cross-encoder has no index, so scoring the whole bank means running the model
once per question. Compare:

```
   ❌ one stage                            ✅ two stages

   100,000 questions                       100,000 questions
          │                                       │
          ▼                                       ▼  vector + lexical + RRF (indexed)
   100,000 cross-encoder runs                 40 candidates
          │                                       │
          ▼                                       ▼  cross-encoder, 40 model runs
     ~3 hours (measured rate)                  best 8
                                                  │
                                                  ▼
                                              ~4 seconds
```

At our current 60 questions the difference is small enough not to matter. At
150 it starts to. At any real size the one-stage version is simply impossible,
and the architecture that works at 100,000 is the one worth building at 60.

So:

* **Stage 1 (Day 8) optimises recall.** Get the right answer *somewhere* in the
  40. It is allowed to be sloppy about the order.
* **Stage 2 (Day 9) optimises precision.** Get it to the top. It is allowed to
  be slow, because it only ever sees 40 things.

**Stage 2 can only reorder what stage 1 handed it.** If the right question was
ranked 60th and only 40 candidates were generated, no reranker can rescue it.
That is why the candidate count is deliberately much larger than the number of
results returned — and why the plan's phrasing is *"recall first, precision
second"*.

---

## 4. What `BAAI/bge-reranker-base` is

The model the plan specifies (§3 Day 9, §5.3).

| | |
|---|---|
| What it is | a cross-encoder trained to score (query, passage) relevance |
| Size | ~110M parameters; ~1 GB downloaded, once |
| Runs on | your CPU, locally — no API, no key, no network after the download |
| Speed | ~100 ms per candidate on this laptop's CPU — see §13 |
| Output | one number per pair |

**It added no new dependency.** It runs through `fastembed`'s
`TextCrossEncoder`, the same ONNX-based library that already serves Day 8's
embedding model, from the same optional extra and the same cache directory:

```bash
pip install -e "./backend[embeddings]"
```

The alternatives (`FlagEmbedding`, `sentence-transformers`) both need PyTorch —
about 2.5 GB — to run a model fastembed already supports.

---

## 5. What the score means

**It is not a probability.** Not a percentage, not a confidence, not bounded to
0–1. It is a raw model output (a *logit*): an unbounded number that is routinely
negative for an irrelevant pair.

Everything in this codebase relies on exactly one property:

> **Higher means more relevant, within a single query.**

Three consequences worth internalising:

1. **No transformation is applied.** No sigmoid, no rescaling to 0–1, no
   normalisation. Squashing a logit through a sigmoid would produce a
   number that *looks* like a probability and is not one — inventing precision
   the model never offered.
2. **No thresholds.** There is no "score below X means irrelevant" rule
   anywhere. Choosing such a number honestly requires labelled data measuring
   what a given score implies, which is Day 10's retrieval evaluation. Guessing
   one today would be a made-up constant with a confident-looking name.
3. **Scores are not comparable across queries.** A −0.6 for one query and a
   −7.1 for another say nothing about which query was better served. The model
   ranks candidates *within* one query, and that is all we use it for.

Measured on this bank, for the query *"How can database reads be made faster?"*:

```
 -7.06   Explain why a B-tree index on a column makes lookups faster.   ← best
-10.19   Describe Floyd's cycle-detection algorithm for a linked list.
-10.20   Compare READ COMMITTED with REPEATABLE READ isolation levels.
```

Every score is negative, and the *best* one is −7. Any code applying a
`score > 0.5` rule to this number would reject everything.

---

## 6. What the model is shown

For each candidate, the reranker receives the query plus the candidate's
**`search_document`** — the exact same text Day 8 indexed:

```
Explain, in terms of what the database actually does at read time, why a B-tree
index on a column makes lookups on that column faster...
Concepts: btree index, index read path, index maintenance cost, storage overhead
Topic: databases / indexing
Tags: databases, indexing
```

**Why the same text and not just the question prose.** Day 8 found and fixed a
bug where the vector arm searched "question + concepts" and the lexical arm
searched "question" alone — fusing two rankings over two different corpora ranks
apples against oranges. Showing the reranker *less* than the retrievers saw
would reintroduce that same mismatch one stage later: a candidate retrieved
*because* of a concept key could then be demoted by a reranker that cannot see
it.

That is not a hypothetical. Measured against the real model, for the query
**"thundering herd"**, comparing the right item against a plausible distractor
(another caching question):

| document shown to the reranker | target | distractor | margin |
|---|--:|--:|--:|
| question prose only | −10.195 | −10.192 | **−0.003** |
| prose + concepts + topic + tags | **−0.634** | −10.193 | **+9.559** |

With prose alone the reranker cannot separate them *at all* — it ranks the
correct item marginally **below** the distractor. With the concept line present
the separation is decisive. An opt-in test pins this
(`test_concepts_in_the_document_are_visible_to_the_model`), because it is the
entire justification for the choice.

Still excluded, for the same reasons as Day 8: the question id (an opaque
label), `difficulty_b` (a number whose text means nothing to a language model —
it is a *filter*, applied in SQL), the reference answer (it describes the
*answer*, and is far longer than the question), and all review metadata and
timestamps.

---

## 7. How candidates are ordered

Sorted by three keys, in order:

1. **relevance score, descending** — the reranker's judgement;
2. **then the retrieval rank** — where the reranker is indifferent between two
   candidates, Day 8's opinion decides. This is better than sorting by id
   because it degrades toward the ranking that would have been served anyway;
3. **then the question id** — so identical evidence always produces identical
   output.

The ordering is **total and deterministic**: the same candidates and the same
scores always produce the same list. A ranking that shuffled on ties would make
every test asserting on it worthless.

Each result keeps both positions, so you can always ask what changed:

```
 #  id                    score  was  move
 1  db-index-001        -0.8677    3    +2     ← the reranker promoted this
 2  db-index-002        -2.3801    1    -1
 3  db-part-001         -6.9104    2    -1
```

---

## 8. Fallback — what happens when the model will not load

Reranking **improves** an ordering. It is never a dependency of having one. If
the model cannot load — not installed, download interrupted, corrupt file,
out of memory — the pipeline returns **stage 1's ranking**, which is a perfectly
good ranking, rather than failing the request.

But it never *pretends*:

```python
result.reranked          # False
result.fallback_reason   # "RuntimeError: onnxruntime session failed to initialise"
result.results[0].rerank_score   # None — not 0.0
```

`rerank_score` is `None` rather than `0.0` because a zero reads like a
measurement and `None` does not. A warning is logged as well, because a reranker
that silently stops working looks exactly like one that is working — the worst
possible failure for a component whose entire job is ordering quality.

The same path handles three cases, distinguishable by the reason string:

| Situation | `reranked` | `fallback_reason` |
|---|:--:|---|
| `RERANK_ENABLED=false`, or no reranker passed | `False` | `"reranking disabled"` |
| model failed to load or run | `False` | the exception type and message |
| model returned the wrong number of scores | `False` | `"… returned 1 scores for 3 candidates"` |
| nothing to rerank (empty candidate list) | `True` | `None` — nothing failed |

An **empty candidate list returns immediately without touching the model**:
loading a gigabyte to score nothing is pure waste.

---

## 9. Model loading and caching

Loading takes seconds; scoring takes milliseconds. Reloading per query would
make the whole two-stage design pointless.

```
first rerank ──► load model (seconds) ──► score ──► cached in the process
next rerank  ──► reuse ────────────────► score
```

* **Lazy** — the model loads on first use, so constructing a reranker is free
  and importing the module never pulls in onnxruntime.
* **Cached per process** — `get_reranker(settings)` memoises the loaded
  instance. It holds only weights and no per-caller state, so sharing one is
  safe. (`build_reranker()` skips the cache, which is what tests want.)
* **Weights on disk** — `.model-cache/` in the repo root, gitignored, shared
  with Day 8's embedding model. fastembed's own default is a temporary
  directory, and a cache the operating system may delete means downloading a
  gigabyte again after a reboot.

---

## 10. The complete pipeline

```
                              QUERY
                                │
        ┌───────────────────────┴────────────────────────┐
        │  STAGE 1 — candidate generation (Day 8)        │
        │                                                │
        │    ┌──────────────┐      ┌──────────────┐       │
        │    │ vector search│      │lexical search│       │
        │    │ bge-small +  │      │ tsvector +   │       │
        │    │ pgvector HNSW│      │ GIN, ts_rank │       │
        │    └──────┬───────┘      └──────┬───────┘       │
        │           │  30 each            │               │
        │           └──────────┬──────────┘               │
        │                      ▼                          │
        │           Reciprocal Rank Fusion (k=60)         │
        └───────────────────────┬────────────────────────┘
                                │  40 candidates
        ┌───────────────────────┴────────────────────────┐
        │  STAGE 2 — reranking (Day 9)                   │
        │                                                │
        │    for each candidate:                         │
        │      [query SEP search_document] ──► score      │
        │                                                │
        │    sort by score, then retrieval rank, then id │
        └───────────────────────┬────────────────────────┘
                                │
                          top 8 results
```

### The four K values

Four different numbers that are easy to confuse:

| Setting | Default | Meaning |
|---|--:|---|
| `RETRIEVAL_VECTOR_K` | 30 | candidates the **vector** arm proposes |
| `RETRIEVAL_LEXICAL_K` | 30 | candidates the **lexical** arm proposes |
| `RERANK_CANDIDATE_K` | 40 | fused candidates the **cross-encoder scores** |
| `RERANK_FINAL_K` | 8 | results **returned** |

40 → 8 is the plan's own ratio (§5.3). `RERANK_CANDIDATE_K` replaces Day 8's
`RETRIEVAL_FINAL_K` when reranking is on — the hybrid stage is simply asked for
more, because stage 2 can only reorder what it is given.

Other settings:

| Setting | Default | Meaning |
|---|---|---|
| `RERANK_ENABLED` | `true` | off serves the RRF order, and says so |
| `RERANK_BACKEND` | `fastembed` | or `overlap` for the deterministic stand-in |
| `RERANK_MODEL` | `BAAI/bge-reranker-base` | |
| `RERANK_BATCH_SIZE` | 16 | pairs per model call |
| `RERANK_CACHE_DIR` | *(unset)* | defaults to `.model-cache/` |

### The code

| File | Job |
|---|---|
| `app/retrieval/rerank.py` | the `Reranker` interface, ordering, fallback — no I/O |
| `app/retrieval/rerankers.py` | the real cross-encoder and the deterministic stand-in |
| `app/retrieval/pipeline.py` | composes stage 1 and stage 2 |
| `scripts/search_questions.py` | drive the whole thing by hand |

**The two stages stay separate on purpose.** `search.py` does not import
`rerank`; `rerank` does not run queries; `pipeline.py` is the only module that
knows both exist. Phase 3 will change how candidates are selected — adding
difficulty and coverage constraints — and none of that should require touching
the reranker, or the reverse.

### What Day 9 is *not*

* **No HTTP endpoint.** `GET /questions/search` is later.
* **No evaluation.** Recall@10, MRR, nDCG@10 and the four-row ablation table
  are Day 10. Day 9 makes reranking *possible and observable*; Day 10 measures
  whether it actually helps on this bank.
* **No thresholds or tuning.** Both need Day 10's labelled data.

---

## 11. How to run it locally

```bash
pip install -e "./backend[embeddings]"       # once; ~1 GB on first rerank
cd backend

# the bank must be ingested and embedded first (Day 8)
python scripts/ingest_question_bank.py

# both stages (the default)
python scripts/search_questions.py "how do database indexes improve performance"

# stage 1 only, to see what reranking changed
python scripts/search_questions.py "how do database indexes improve performance" --no-rerank

# no model at all — the deterministic stand-in
RERANK_BACKEND=overlap python scripts/search_questions.py "cache stampede"
```

The output shows both rankings at once. `was` is the position after stage 1;
`move` is what stage 2 did. Real output from this bank:

```
$ python scripts/search_questions.py "thundering herd" --final-k 3

bank: 60 questions, 60 with an embedding
embedder: BAAI/bge-small-en-v1.5 (384 dimensions)
reranker: BAAI/bge-reranker-base
stage 1: 30 vector, 1 lexical, 30 candidates
stage 2: 30 pairs scored by the cross-encoder
timings: hybrid=611.4ms  rerank=6134.5ms  total=6745.9ms

 #  id                     score  was  move      rrf  vec  lex  question
 1  sys-cache-003        -1.6384    1     -  0.03279    1    1  A popular cache entry with …
 2  db-part-001         -10.1935    4    +2  0.01562    4    -  An events table has grown …
 3  db-model-002        -10.1935   28   +25  0.01136   28    -  Students enrol in courses, …
```

Three things in that output are worth reading carefully.

* **Every score is negative**, and that is normal (§5). The *best* one is −1.64.
* **The reranker is decisive where it should be.** The one genuinely relevant
  question scores −1.64; everything else sits at −10.19. For this query exactly
  one item is relevant and the model says so with an 8.5-point gap.
* **The rest are effectively tied.** Ranks 2 and 3 differ in the fourth decimal
  place — the model is saying "these are all equally irrelevant", not ranking
  them. Their order comes from the tie-break (§7), which is why that has to be
  deterministic rather than arbitrary.

The timings here are a *cold* single-shot run: `hybrid` includes loading
bge-small and `rerank` includes the cross-encoder's first-call warm-up. Warm
figures are in §13.

---

## 13. What it actually costs

Measured on this laptop's CPU, against the 60-question bank. **These are local
numbers from a tiny dataset and are not a production benchmark.**

| | |
|---|--:|
| download, first run only | ~21 min (~1.1 GB) |
| model load from disk, once per process | ~3 s |
| scoring, per candidate | **~100 ms** |
| stage 1 (hybrid retrieval), whole query | ~65–100 ms |
| stage 2 over 30 candidates | ~3.0 s |

Reranking is roughly **30× the cost of the entire retrieval stage** — which is
the two-stage argument made concrete, and exactly why the cross-encoder never
sees more than a few dozen candidates.

Cost is linear in the number of candidates, and strongly dependent on how long
each document is:

| candidates | rerank time | per candidate |
|--:|--:|--:|
| 5 | 610 ms | 122 ms |
| 10 | 1113 ms | 111 ms |
| 20 | 2032 ms | 102 ms |
| 30 | 2969 ms | 99 ms |

**A discrepancy worth recording.** Plan §5.3 budgets *"40 pairs through
`bge-reranker-base` on CPU ≈ 250 ms"*. Measured here it is roughly 4 s — about
16× that. The difference is document length, not the model: scoring 40
*one-sentence* documents takes ~1.1 s (~27 ms each), while a real
`search_document` is 150–200 tokens and costs ~100 ms. Attention cost grows with
sequence length, so the plan's figure was for much shorter passages than this
bank actually has.

Nothing was optimised in response, deliberately — Day 9's goal is a correct,
observable architecture, and every available lever (fewer candidates, truncated
documents, a quantised ONNX build) is a trade against ranking quality that
should be chosen with Day 10's evaluation numbers rather than guessed at now.
The options are recorded in `BACKLOG.md`.

---

## 12. How to run the tests

```bash
cd backend

# everything (needs `docker compose up` for the integration tests)
pytest

# just the reranking layer
pytest tests/unit/retrieval/test_rerank.py tests/integration/test_rerank_pipeline.py

# ordering logic only — pure functions, no database, instant
pytest tests/unit/retrieval/test_rerank.py

# the real cross-encoder: opt-in, needs the [embeddings] extra and ~1 GB
pytest -m embeddings
```

### The test seam, and the line it must not cross

The default suite drives the entire two-stage pipeline with
**`LexicalOverlapReranker`**: a scorer that counts how many of the query's words
appear in the candidate, divided by the square root of its length. No model, no
download, no randomness.

It is enough to test every *mechanism*: candidates are reordered by score, ties
break deterministically, top-K truncates, an empty list skips the model, a
failure falls back, the model receives the right text.

It is **not** relevance judgement — it counts shared words. So no test using it
may claim otherwise. That claim lives in
`tests/unit/retrieval/test_real_reranker.py`, against the real model, behind
`-m embeddings`, or it does not get made. Those assertions are deliberately
*relative* ("this pair scores above that pair"), never absolute thresholds, for
the reason in §5.

| File | Covers |
|---|---|
| `unit/retrieval/test_rerank.py` | ordering, ties, top-K, empty input, fallback, the stand-in |
| `integration/test_rerank_pipeline.py` | the two stages composing against real Postgres |
| `unit/retrieval/test_real_reranker.py` | the real model's behaviour *(opt-in)* |
