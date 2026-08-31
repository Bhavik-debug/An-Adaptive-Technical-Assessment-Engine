# Retrieval — finding the right question in the bank

*Phase 2, Day 8. Written for someone who has never studied machine learning.
Nothing here assumes you know what an embedding, a vector, or cosine similarity
is.*

*This is **stage 1** of a two-stage system. Stage 2 — the cross-encoder that
reorders these results — is [`docs/reranking.md`](reranking.md).*

---

## The problem

The bank holds 60 questions (150 by the end of Phase 2). Something has to answer
"which questions are about *this*?" — for the adaptive engine picking the next
item, and later for a search endpoint.

The obvious approach is keyword matching. It works until it doesn't:

| Query | A question in the bank | Same words? | Same meaning? |
|---|---|:--:|:--:|
| "database indexes" | "…why a **B-tree index** on a column makes **lookups** faster" | ✅ | ✅ |
| "How can a relational database speed up slow lookups?" | *(the same question)* | ❌ | ✅ |

The second row is the whole problem. A person asking about slow lookups wants
the index question, but they share almost no words with it — "database" and
"lookups", against a question built around "B-tree", "index" and "read time".
Keyword search scores that near zero.

Day 8 adds a second way of searching that handles the second row, keeps the
first, and combines them.

---

## 1. What an embedding is

An **embedding model** reads a piece of text and returns a fixed-length list of
numbers. Here, 384 of them:

```
"Why do database indexes improve query performance?"
        │
        ▼  embedding model
[ 0.041, -0.118,  0.007,  0.052, … ]      ← 384 numbers

"What is the purpose of an index in a relational database?"
        │
        ▼  embedding model
[ 0.038, -0.104,  0.011,  0.049, … ]      ← 384 numbers, and very close to the first
```

That list of numbers is called a **vector**. The model is built so that texts
which *mean* similar things get similar numbers — even when they share no
words. That is the entire trick, and it is what keyword matching cannot do.

You do not need to know how the model produces the numbers. What matters is the
property it guarantees: **close numbers ⇒ close meaning.**

### What "close" means

Think of each vector as an arrow pointing somewhere from the origin. Two texts
about database indexes point in almost the same direction; a text about linked
lists points somewhere else. **Cosine similarity** measures the angle between
two arrows:

```
   similarity  1.0  ── same direction        ── same meaning
               0.0  ── at right angles       ── unrelated
              −1.0  ── opposite directions   ── (rare in practice)
```

The model we use returns vectors that are all exactly length 1 ("L2-normalised",
verified in `tests/unit/retrieval/test_real_model.py`). That has one convenient
consequence: cosine similarity is just the sum of the products of matching
numbers, and **cosine distance** — what the database actually computes — is
`1 − similarity`. Distance 0 = identical direction; distance 1 = unrelated.

> **A warning that will save you confusion.** These similarity numbers are not
> percentages, and they do not start at zero. With this model, two questions
> from *completely different domains* still score around 0.55–0.60. That is
> normal — the model compresses everything into a narrow high band. So never
> write a rule like "similarity above 0.7 means relevant". **Only the ranking
> means anything**, never the absolute value. There is no such threshold
> anywhere in this codebase, deliberately.

---

## 2. Why this project uses embeddings

Because the alternative fails on exactly the queries that matter.

The adaptive engine will eventually need to find questions related to a concept
a candidate just got wrong. That request will not arrive as a keyword list — it
arrives as a topic, a gap, a description. Keyword search would only find items
that happen to reuse the same vocabulary, which is not the same thing as finding
items about the same idea.

It also costs nothing to run. The model is small enough to run on the CPU of a
laptop, in a few milliseconds, with no API and no per-call charge.

---

## 3. What `BAAI/bge-small-en-v1.5` is

The specific embedding model this project uses (chosen in the plan, §3 Day 8).

| | |
|---|---|
| Full name | BAAI General Embedding, small, English, version 1.5 |
| Output | 384 numbers per text |
| Size | ~67 MB, downloaded once |
| Runs on | your CPU, locally — no API, no key, no network after the download |
| Speed | ~5 ms per question when embedding a batch; ~35 ms for a single query |

"Small" is the size class — there are `base` and `large` versions that are more
accurate and much slower. Small is the right trade for 60–150 questions on a
laptop.

**How we run it.** Through [`fastembed`](https://github.com/qdrant/fastembed),
which runs the model with ONNX Runtime. The common alternative,
`sentence-transformers`, needs PyTorch — about 2.5 GB installed, for a 67 MB
model. fastembed needs ~200 MB and treats bge-small-en-v1.5 as its default
model, so the pooling and normalisation are handled exactly as the model's
authors specify.

It is an **optional dependency**, because CI never needs it (see
[§12](#12-how-to-run-the-tests)):

```bash
pip install -e "./backend[embeddings]"
```

The first run downloads the weights to `.model-cache/` in the repository root
(gitignored). After that it is entirely offline.

---

## 4. What text we actually embed, and why

Not the whole database row. Embedding a field just because it exists is
embedding noise.

The recipe lives in **one function**,
`app/retrieval/embedding.py::document_text()`, and produces something like:

```
Explain, in terms of what the database actually does at read time, why a B-tree
index on a column makes lookups on that column faster. Then describe the cost the
same index imposes on INSERT, UPDATE and DELETE...
Concepts: btree index, index read path, index maintenance cost, storage overhead
Topic: databases / indexing
Tags: databases, indexing
```

| Included | Why |
|---|---|
| the question text | the thing being searched for |
| `expected_concepts` keys | curated, high-signal vocabulary — often the words a searcher will use |
| topic / subtopic | "caching", "binary search" are strong topical signals |
| tags | cheap, curated, already there |

| Excluded | Why |
|---|---|
| `id` | an opaque label; `db-index-001` means nothing to a language model |
| `difficulty_b`, `discrimination_a` | numbers whose *text* is meaningless. These are **filters**, applied in SQL |
| `reference_answer` | it describes the *answer*. Embedding it makes questions match queries about answers — and it is ~10× longer than the question, so it would drown it |
| review metadata, timestamps | not content |

**Snake_case keys are converted to words.** `cache_invalidation` becomes
`cache invalidation`. The model has seen those two English words many times and
the identifier almost never, so feeding it identifiers throws away most of what
the concept tags are worth.

**The same recipe, one place.** A vector is only comparable with vectors built
the same way. If indexing embedded one thing and search embedded another, every
similarity would be measured against a slightly different space — and the system
would be quietly wrong rather than loudly broken. So there is one function, and
`text_fingerprint()` records which recipe produced each stored vector.

**Queries are treated differently, on purpose.** A query gets the same
whitespace normalisation and nothing else. Wrapping a user's search in
`Concepts: … Tags: …` would invent metadata they never supplied and push the
query away from the documents it should match. Retrieval models are trained on
exactly this asymmetry: a short query against a longer passage.

---

## 5. What pgvector does

PostgreSQL cannot store a list of 384 numbers usefully, or compare two of them,
on its own. **pgvector** is an extension that adds:

* a column type — `vector(384)`;
* distance operators, of which we use `<=>`, **cosine distance**;
* index types for making those comparisons fast (see §6).

It is already part of this project's stack: the compose file runs the
`pgvector/pgvector:pg16` image, and `infra/postgres/init/001-extensions.sql`
enables it. Day 8 added no new database and no second vector store — the vectors
live in a column on the existing `questions` table.

```sql
-- what vector search actually runs
SELECT id, embedding <=> :query_vector AS distance
FROM questions
WHERE embedding IS NOT NULL
ORDER BY embedding <=> :query_vector
LIMIT 30;
```

Three columns were added to `questions`:

| Column | Purpose |
|---|---|
| `embedding vector(384)` | the numbers |
| `embedding_model text` | which model produced them |
| `embedding_text_sha256 text` | a hash of the exact text that was embedded |

The last two are what make re-running ingest cheap and correct — see §11.

---

## 6. What HNSW does

Without an index, finding the nearest vectors means comparing the query against
**every** row:

```
query ──► compare with row 1
      ──► compare with row 2
      ──► …
      ──► compare with row 60          ── fine at 60. Not fine at 500,000.
```

**HNSW** — Hierarchical Navigable Small World — builds a graph over the vectors
where each one is linked to its neighbours, with a few "express lane" layers on
top. A search enters at the top, takes big jumps toward the right region, then
drops to finer layers to refine:

```
layer 2   ●───────────────────────●          few nodes, long jumps
              ╲                 ╱
layer 1   ●─────●───────●─────●───●          more nodes, shorter jumps
            ╲  ╱ ╲     ╱ ╲   ╱ ╲ ╱
layer 0   ●─●─●─●─●─●─●─●─●─●─●─●─●          every vector
                    ▲
                 answer
```

Instead of 500,000 comparisons you follow a few hundred edges. You do **not**
implement any of this — pgvector provides it. The migration just says:

```sql
CREATE INDEX ix_questions_embedding_hnsw
    ON questions USING hnsw (embedding vector_cosine_ops);
```

`vector_cosine_ops` matters: it fixes *which distance* the index can answer for.
An index built for one operator is simply not used by a query written with
another — it silently falls back to scanning everything. The operator class and
the query have to be chosen together.

Parameters (`m`, `ef_construction`) are left at their defaults. Tuning them
without a measured recall number would be guessing; Day 10's retrieval
evaluation is where numbers to tune against come from.

> **HNSW is approximate.** It can miss a true nearest neighbour in exchange for
> speed. That is the deal, and it is a good one at scale.

> **At 60 rows, PostgreSQL will not use this index — and that is correct.** A
> sequential scan over 60 rows is genuinely cheaper than traversing a graph, so
> the planner picks it. Running `EXPLAIN` and seeing `Seq Scan` is not a bug.
> The test `test_the_hnsw_index_is_usable_by_the_planner` forces the choice with
> `enable_seqscan = off` to prove the index, the operator class and the query are
> compatible — which is the part that could actually be wrong.

---

## 7. What lexical search is

Plain keyword search, done properly by PostgreSQL's full-text search.

```
"database index performance"
        │
        ▼  to_tsquery: split into terms, reduce each to its stem
   'databas' & 'index' & 'perform'
        │
        ▼  match against a pre-computed tsvector per question, via a GIN index
   ranked by how well and how closely the terms occur
```

* **`tsvector`** is a question's text pre-processed into searchable terms:
  lower-cased, stemmed (so "indexes", "indexing" and "index" all become
  `index`), with stop words like "the" and "how" dropped.
* **GIN index** — a "which rows contain this term" index, the inverse of a
  B-tree's "where does this value sort". It is what makes term lookup fast.
* **`ts_rank_cd`** scores each match. The `cd` is "cover density": it also
  accounts for how *close together* the matched terms are, so a question using
  the words together outranks one that mentions them in unrelated sentences.
  This is the plan's "BM25-ish" ranking — PostgreSQL has no true BM25.

Two deliberate choices in `app/retrieval/search.py`:

* **`websearch_to_tsquery`**, not `to_tsquery`. It accepts whatever a person
  types — quotes, `OR`, stray punctuation — and never raises. `to_tsquery
  ('database index')` is a hard syntax error, which would make a search endpoint
  fail on an ordinary two-word query.
* Note that it joins terms with **AND**. A long conversational query therefore
  often matches nothing lexically. That is expected, and it is what the vector
  arm is for.

**The `tsv` column indexes the same `search_document` the embedder sees.** This
was a bug found while building Day 8 and it is worth understanding: the column
originally covered only the question prose. A search for "cache stampede
thundering herd" returned *nothing* lexically, because those words live in the
item's `expected_concepts`, not in its text — while the embedding *did* cover
them. Fusing a ranking over "question + concepts" with a ranking over "question
only" ranks two different corpora against each other. Both arms now search the
same document.

---

## 8. Why combine the two

They fail in opposite directions:

| | finds | misses |
|---|---|---|
| **Lexical** | exact words, rare terms, acronyms, identifiers | anything phrased differently |
| **Vector** | paraphrases, related concepts, different vocabulary | exact rare tokens — embeddings *blur* precisely the detail you asked it to be precise about |

Each covers the other's blind spot. And when both independently rank something
highly, that agreement is real evidence — stronger than either one alone.

This is called **hybrid retrieval**.

---

## 9. What RRF does

We now have two ranked lists, and a problem: **their scores are not
comparable.** A cosine similarity of 0.78 and a `ts_rank_cd` of 0.043 are
numbers on unrelated scales. Averaging them means inventing a conversion rate
that does not exist. Normalising each into 0–1 first is worse: the normalisation
depends on which documents happened to be returned, so adding one document
reorders the others.

**Reciprocal Rank Fusion** throws the scores away and keeps only the **ranks**,
which *are* comparable between any two retrievers by construction.

```
score(d)  =   sum over each list containing d of   1 / (k + rank of d in that list)
```

Ranks are 1-based; `k` is a constant, 60 by default. A worked example:

```
vector : A B C D                 lexical: C A E F

A: 1/(60+1) + 1/(60+2) = 0.03252   ← 1st in one list, 2nd in the other
C: 1/(60+3) + 1/(60+1) = 0.03225   ← 3rd and 1st
B: 1/(60+2)            = 0.01613
E: 1/(60+2)            = 0.01613   ← ties with B; the tie-break decides
D: 1/(60+3)            = 0.01587
F: 1/(60+3)            = 0.01587
```

Notice that **appearing in both lists is worth roughly twice being first in
one** — which is exactly the behaviour we want.

### Why k = 60

It is the value from the paper the method comes from (Cormack, Clarke &
Buettcher, 2009) and the near-universal default. Its job is **damping**:

* With `k = 0`, rank 1 scores 1.0 and rank 2 scores 0.5 — whichever list ranked
  something first would dominate everything else.
* With `k = 60`, rank 1 and rank 2 differ by about 1.6% — so a document needs
  *support* from both retrievers to win, rather than one lucky first place.

`test_k_decides_whether_one_first_place_beats_two_third_places` demonstrates
exactly this: with the same input, `k = 60` puts the doubly-found document first
and `k = 0.5` puts the single first-place document first. Configurable via
`RETRIEVAL_RRF_K`; tuning it belongs to Day 10's evaluation, which can measure
the effect.

### Determinism

A ranking is worthless to test if ties resolve differently between runs. The
order is total, by three keys:

1. higher fused score;
2. then better (lower) best rank across the sources — a document some retriever
   put near the top outranks one everybody put in the middle;
3. then the id, ascending, so identical evidence always gives identical output.

Fusion also **discards nothing**: its output is the union of both lists.
Truncation to the final K happens afterwards and is *reported* in
`HybridSearchResult.truncated`, never silent.

---

## 10. The Day 8 architecture

```
                          "how do indexes speed up queries"
                                        │
                        ┌───────────────┴───────────────┐
                        ▼                               ▼
              embed the query                  websearch_to_tsquery
              (bge-small, 384-d)                        │
                        ▼                               ▼
              ORDER BY embedding <=> q          WHERE tsv @@ query
              (pgvector, HNSW)                  ORDER BY ts_rank_cd
              LIMIT 30                          LIMIT 30
                        │                               │
                 ranked list A                   ranked list B
                        └───────────────┬───────────────┘
                                        ▼
                            Reciprocal Rank Fusion (k=60)
                                        ▼
                          top 10, each carrying its evidence:
                          vector rank + similarity,
                          lexical rank + score, fused RRF score
```

### The code

| File | Job |
|---|---|
| `app/retrieval/embedding.py` | what text gets embedded; the `Embedder` interface; the fingerprint |
| `app/retrieval/embedders.py` | the real model (fastembed) and a deterministic stand-in |
| `app/retrieval/indexing.py` | writing vectors for ingested questions, idempotently |
| `app/retrieval/rrf.py` | fusion — pure arithmetic, no I/O |
| `app/retrieval/search.py` | `vector_search`, `lexical_search`, `hybrid_search` |
| `scripts/search_questions.py` | drive it by hand |

### K values

Two different numbers, and confusing them defeats the design:

| Setting | Default | Meaning |
|---|--:|---|
| `RETRIEVAL_VECTOR_K` | 30 | candidates the **vector** arm proposes |
| `RETRIEVAL_LEXICAL_K` | 30 | candidates the **lexical** arm proposes |
| `RETRIEVAL_FINAL_K` | 10 | results returned after fusion |
| `RETRIEVAL_RRF_K` | 60 | the damping constant |

Candidate K is deliberately larger than final K. A question ranked 22nd by
vectors and 2nd lexically deserves to be fused — and it can only be fused if
both retrievers were asked for more than the caller wants back. Setting
candidate K equal to final K would turn hybrid retrieval into two independent
top-10s stapled together.

### What Day 8 is *not*

* **No HTTP endpoint.** `GET /questions/search` is later. Nothing here is
  exposed over the API.
* **No reranking.** Day 8 stops at "ranked candidates". Day 9 adds a
  cross-encoder that consumes `hybrid_search`'s output unchanged and reorders
  it — see **[`docs/reranking.md`](reranking.md)**, which explains what a
  cross-encoder is and why it runs *after* this stage rather than instead of it.
* **No adaptive selection.** That is Phase 3.

### Embeddings are not an LLM call

They deliberately do **not** go through `app/llm/`. That layer exists to control
cost, routing, retries and failover for a metered remote API. This model runs
locally, costs nothing per call, and has no provider to fail over to. Sharing
the abstraction would buy nothing and make an offline component depend on an
online one.

---

## 11. How to run the embedding and indexing process

```bash
pip install -e "./backend[embeddings]"     # once: ~200 MB
cd backend
alembic upgrade head                        # adds the vector column + HNSW index

python scripts/ingest_question_bank.py      # validate → rows → embeddings
```

Embedding is **part of ingest, not a second command**. Two commands that must
both be run, in order, to leave the system consistent is a system that will
eventually be inconsistent — somebody edits a question, re-ingests, forgets to
re-embed, and search quietly answers for the previous wording. So the vectors
are written in the *same transaction* as the rows: a run either leaves both
agreeing, or leaves nothing.

```text
question-bank JSONL
        ↓  validate_bank()          — schema, taxonomy, duplicates
        ↓  upsert rows              — including search_document
        ↓  embed_questions()        — only what changed
        ↓  store vectors            — same transaction
        ↓  HNSW index               — maintained by Postgres
```

### Idempotency — it does not re-embed what has not changed

Each row stores `embedding_model` and `embedding_text_sha256` (a hash of the
exact text embedded, including a recipe version). A row whose pair still matches
is already correct and is skipped.

```
$ python scripts/ingest_question_bank.py
ingested: … 60 questions … embeddings: 60 embedded, 0 already current

$ python scripts/ingest_question_bank.py        # nothing changed
ingested: … 60 questions … embeddings: 0 embedded, 60 already current
```

Edit one question and re-run: exactly that one is re-embedded. Change the model
in settings: all 60 are, because vectors from two models are not comparable.

| Flag | Effect |
|---|---|
| *(default)* | embed anything missing or stale |
| `--no-embed` | write rows only; leave vectors NULL. Lexical search still works |
| `--reembed` | force every vector, ignoring fingerprints |
| `--dry-run` | validate only; touch no database |

`--reembed` exists for the one case a fingerprint cannot detect: the model file
changing underneath a name that stayed the same.

### Searching by hand

```bash
python scripts/search_questions.py "how do database indexes improve performance"
python scripts/search_questions.py "MVCC" --mode lexical
python scripts/search_questions.py "speed up slow lookups" --mode vector
```

`--mode` runs one retriever alone — the fastest way to *see* why hybrid is worth
having.

### Measured on this machine (60 questions)

| Step | Time |
|---|--:|
| model load, weights already on disk | 596 ms *(once per process)* |
| embed one query, warm | ~33 ms |
| embed 60 documents, one batch | 323 ms |
| vector SQL (`<=>` + ORDER BY + LIMIT) | 2.1 ms |
| lexical SQL (`tsv @@` + `ts_rank_cd`) | 1.0 ms |
| RRF fusion | 0.07 ms |
| **hybrid total** | **35.5 ms** |
| full ingest, nothing changed | ~2.3 s |

Medians over 20 runs. **These numbers describe 60 rows on one laptop and prove
nothing about production scale** — the SQL is fast because the table is tiny,
not because the index is doing work. What they do show is where the time goes:
embedding the query is ~90% of a hybrid search, and the database is noise. That
is the number to watch, and it is why a query-embedding cache is the obvious
first optimisation when one is actually needed.

---

## 12. How to run the tests

```bash
cd backend

# everything (needs `docker compose up` for the integration tests)
pytest

# just Day 8
pytest tests/unit/retrieval tests/integration/test_retrieval.py

# fusion only — pure functions, no database, instant
pytest tests/unit/retrieval/test_rrf.py

# the real model: opt-in, needs the [embeddings] extra
pytest -m embeddings
```

### The test seam, and an honesty rule

The default suite must be free, offline and fast — on a laptop, in CI, on
somebody else's fork. Downloading a 67 MB model is none of those. So the tests
drive the whole pipeline with **`HashingEmbedder`**: a real bag-of-words vector,
384-dimensional and L2-normalised like the real thing, built by hashing tokens
into slots. No model, no download, no randomness.

It is enough to test every *mechanism*: the vector column accepts it, `<=>`
orders by it, fusion combines it, ingest skips it when unchanged.

It is **not** semantic — it matches words, not meaning. So no test using it may
claim otherwise. Asserting that a hashed bag-of-words vector understands a
paraphrase would be asserting something false. That claim lives in
`tests/unit/retrieval/test_real_model.py`, against the real model, behind
`-m embeddings` — or it does not get made.

Those real-model tests are also deliberately **relative** ("this is closer than
that"), never absolute thresholds, for the reason given in §1.

| File | Tests | Covers |
|---|--:|---|
| `unit/retrieval/test_rrf.py` | 22 | fusion arithmetic, tie-breaking, determinism, `k` |
| `unit/retrieval/test_embedding.py` | 28 | the text recipe, fingerprints, the stand-in |
| `unit/retrieval/test_dimensions.py` | 5 | 384 agrees across code, model and migration |
| `integration/test_retrieval.py` | 43 | vector, lexical, hybrid, idempotency, the indexes |
| `unit/retrieval/test_real_model.py` | 10 | the real model's actual behaviour *(opt-in)* |
