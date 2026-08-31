# Retrieval evaluation — how well does it actually work?

*Phase 2, Day 10. Written for someone who has never studied ML. It follows
[`docs/retrieval.md`](retrieval.md) (stage 1) and
[`docs/reranking.md`](reranking.md) (stage 2).*

---

## 1. Why this exists

Days 8 and 9 answered **"does it run?"** — vectors are stored, queries return
results, the reranker reorders them. Every test so far checks *mechanism*: the
SQL is right, the fusion arithmetic is right, the fallback works.

None of that says whether the thing is any **good**.

That gap is where most retrieval systems quietly fail. It is entirely possible
to build a pipeline that runs perfectly, returns ten results in 40 ms, has
100% test coverage — and puts the right answer eighth. Nothing errors. Nothing
looks wrong. You only find out when someone uses it.

Worse, without measurement you cannot answer the question that actually decides
the architecture:

> The cross-encoder costs ~100 ms per candidate, about 40× the entire retrieval
> stage. **Is it worth it?**

You cannot reason your way to that answer. "It is a more sophisticated model"
is not evidence. Day 10 exists to replace the assumption with a number — and,
as it turned out, to overturn it.

---

## 2. Ground truth

To judge a retriever you need to already know the right answer. That known-right
answer is called **ground truth**, and it is just a list you wrote down by hand:

```
query:    "How can a relational database be made to answer lookups faster?"
relevant: db-index-001  (grade 2 — this is the question they meant)
          db-index-002  (grade 1 — related, reasonable to return)
          db-index-003  (grade 1 — related)
```

Run that query through the system, see where those ids land, and you can score
the ranking.

### It is not the question bank

This trips people up, so it is worth being blunt:

| | |
|---|---|
| **Question bank** — `data/question-bank/*.jsonl` | The 60 questions the system can *return*. The product. |
| **Evaluation set** — `evals/datasets/retrieval_queries.jsonl` | 40 *queries*, plus which questions are the right answers. The ruler. |

The bank is what is measured. The evaluation set is what measures it. They are
different files, different schemas, different purposes.

### Relevance grades

Labels are graded 0–2 rather than yes/no, because "relevant" is not binary:

```
2 = directly answers the query — the question they meant to find
1 = genuinely related, reasonable to return, but not the target
0 = not relevant  (never written down; absence means 0)
```

Recall and MRR need a yes/no answer, so they use **grade ≥ 1**. nDCG uses the
grades themselves — which is the entire reason for having them.

---

## 3. Recall@K — "did we find it at all?"

> Of all the questions that *should* have been returned, what fraction appeared
> in the top K?

```
recall@K = |{top K retrieved} ∩ {relevant}| / |{relevant}|
```

With one relevant question it is a simple yes/no:

```
retrieved = [B, A, C]      relevant = {A}

recall@3 = 1.0     A is 2nd, so it is inside the top 3   ✅
recall@1 = 0.0     only B is inside the top 1            ❌
```

With three relevant questions and two of them in the top K, recall is 0.67.

**What it ignores: position.** A relevant result at rank 1 and one at rank 10
score identically. Recall asks "is it in the list", never "is it near the top".
That is what MRR is for.

---

## 4. MRR — "how near the top was it?"

**Reciprocal rank** is one divided by the position of the *first* relevant
result:

```
rank 1  → 1/1  = 1.000        rank 4  → 1/4  = 0.250
rank 2  → 1/2  = 0.500        rank 10 → 1/10 = 0.100
rank 3  → 1/3  = 0.333        not found →     0.000
```

**MRR** ("Mean Reciprocal Rank") is that averaged over every query.

The shape matters. Moving the first hit from rank 2 → rank 1 gains **0.5**;
moving it from rank 9 → rank 8 gains **0.014**. MRR cares enormously about the
top of the list and almost not at all about the rest — which is exactly right
for a system where something looks at the first result and rarely scrolls.

**What it ignores: everything after the first hit.** A query with three relevant
questions scores identically whether the other two are at ranks 2 and 3 or
missing entirely. That is what nDCG is for.

---

## 5. nDCG — "how good is the whole ordering?"

Needed when several results are relevant *to different degrees*. Built in three
steps.

**Step 1 — gain.** How much is a result worth? Using the standard exponential
form `2^grade − 1`:

```
grade 0 → 0        grade 1 → 1        grade 2 → 3
```

So one perfect answer is worth three partial ones. That is a deliberate choice
of the formula, and it is why grading was worth doing.

**Step 2 — discount.** A result further down is worth less, divided by
`log2(rank + 1)`:

```
rank 1 → ÷1.00  (keeps 100%)      rank 3  → ÷2.00  (keeps 50%)
rank 2 → ÷1.58  (keeps 63%)       rank 10 → ÷3.46  (keeps 29%)
```

Logarithmic, not linear, because the gap between rank 1 and 2 matters far more
to a reader than the gap between rank 19 and 20.

Add them up and you have **DCG**:

```
grades = {A: 2, B: 1}      retrieved = [B, A, C]

rank 1  B   gain 1  ÷ log2(2) = 1.000
rank 2  A   gain 3  ÷ log2(3) = 1.893
rank 3  C   gain 0            = 0
                       DCG@3  = 2.893
```

**Step 3 — normalise.** DCG alone is not comparable between queries: one with
four relevant questions can out-score one with a single relevant question no
matter how well the second was served. So divide by the **ideal DCG** — the DCG
of the best ordering these labels allow:

```
ideal order = [A(2), B(1)]
iDCG@3 = 3÷log2(2) + 1÷log2(3) = 3.000 + 0.631 = 3.631

nDCG@3 = 2.893 / 3.631 = 0.797
```

**Interpretation.** 1.0 means the ordering is as good as the labels allow — not
"100% accurate". 0.0 means nothing relevant appeared in the top K. Anything
between is a fraction of the achievable ideal, comparable across queries.

---

## 6. Ablation study

An **ablation study** removes or replaces one component at a time and measures
what each removal costs. The word is borrowed from biology (removing tissue to
see what stops working); in engineering it just means *changing one thing and
measuring*.

Here, four configurations:

| | what it is |
|---|---|
| **Vector only** | stage 1 with the lexical arm removed |
| **Lexical only** | stage 1 with the vector arm removed |
| **Hybrid RRF** | both arms, fused — all of Day 8 |
| **Hybrid + reranker** | Day 8 plus Day 9's cross-encoder |

Comparing row 3 against rows 1 and 2 says what fusion is worth. Comparing row 4
against row 3 says what the reranker is worth. Neither question can be answered
by argument.

> **The rule this project follows:** the conclusion is whatever the numbers say.
> Not "the cross-encoder must be better, it is a bigger model." If reranking
> makes the metrics worse, that is the finding, and it gets written down.

---

## 7. The evaluation set

`evals/datasets/retrieval_queries.jsonl` — 40 queries, one JSON object per line:

```json
{
  "id": "eq-001",
  "query": "How can a relational database be made to answer lookups faster?",
  "relevant": {"db-index-001": 2, "db-index-002": 1, "db-index-003": 1},
  "kind": "semantic",
  "note": "Shares almost no vocabulary with db-index-001 ('B-tree index', 'read time'). The canonical case for vector search over lexical."
}
```

`note` is required. An unexplained label is one nobody can argue with, which is
the opposite of what ground truth should be.

**`kind`** records which retrieval situation the query was written to test, so
results can be broken down. An average over mixed cases hides exactly what an
ablation is for.

| kind | n | what it tests |
|---|--:|---|
| `semantic` | 20 | little or no word overlap with the target |
| `lexical` | 9 | exact terms, acronyms, concept keys |
| `hybrid` | 4 | both signals present |
| `rerank` | 3 | several plausible candidates, one clearly best — a *ranking* test |
| `hard` | 4 | ambiguous, cross-domain, or a known gap |

40 queries cover **55 of the 60** bank questions.

### How the labels were made

Every one of the 60 questions was read, and for each query the ids a person
asking it would want were chosen by hand. **The labels were written before any
retrieval was run against them**, so no label was picked to flatter a result.

They are *considered*, not authoritative — see §12.

---

## 8. How each mode is evaluated

```
40 queries
     │
     ├──► vector_search()           ──┐
     ├──► lexical_search()          ──┤
     ├──► hybrid_search()           ──┼──► top 10 ids ──► Recall@5, Recall@10
     └──► search_and_rerank()       ──┘                   MRR, nDCG@10
```

Each mode calls exactly the function the application calls. `runner.py` contains
no SQL and no model — it could not fake a result if it tried.

**Fairness rules**, which matter more than they look:

* Every mode is asked for the same **depth of 10**. A mode allowed to return 20
  while another returns 8 would win on recall for that reason alone.
* **Recall@5 is the first five of that same list**, not a second retrieval run
  with `k=5`. Re-running would let a mode reorder itself between measurements.
* The reranker's candidate pool (40) is separate from the depth (10): it scores
  40 candidates and returns the best 10.
* If reranking silently fell back to the hybrid order, the run **fails** rather
  than scoring it — otherwise the cross-encoder gets credit for Day 8's ranking.

---

## 9. Running it

```bash
# 1. the stack, and the bank ingested with embeddings
docker compose up -d
cd backend
alembic upgrade head
python scripts/ingest_question_bank.py

# 2. the evaluation (real models; ~3 minutes, mostly the cross-encoder)
python scripts/run_retrieval_eval.py --per-query

# 3. regenerate the committed report
python scripts/run_retrieval_eval.py --write-report

# faster: skip the slow stage
python scripts/run_retrieval_eval.py --no-rerank

# no model download at all — checks the runner, NOT quality
python scripts/run_retrieval_eval.py --stand-ins
```

### Reproducibility

Every input is pinned: the committed dataset, the committed bank,
`BAAI/bge-small-en-v1.5` and `BAAI/bge-reranker-base`. Both models are
deterministic on CPU. At 60 rows PostgreSQL chooses an exact sequential scan
over the approximate HNSW index, so even that source of variation is absent.

**Re-running produces identical rankings and identical metrics.** Only the
timings move. The configuration block printed above every table records the
dataset path, both model names, all four K values and whether real models or
stand-ins were used.

---

## 10. Results

Measured with real models over 40 queries, top 10 scored per mode. Full report:
[`evals/reports/retrieval_ablation.md`](../evals/reports/retrieval_ablation.md).

| Method | Recall@5 | Recall@10 | MRR | nDCG@10 | ms/query |
|---|---|---|---|---|---|
| Vector only | 0.850 | 0.883 | 0.908 | 0.877 | 66 |
| Lexical only | 0.208 | 0.208 | 0.300 | 0.269 | 1 |
| **Hybrid RRF** | **0.850** | **0.883** | **0.921** | **0.886** | 58 |
| Hybrid + reranker | 0.746 | 0.875 | 0.906 | 0.858 | 3448 |

The run was repeated end to end. **Every metric was identical to three decimal
places**; only the timings moved (the reranker column was 4379 ms/query on the
first run, 3448 on the second). That is the determinism claim in §9, measured
rather than asserted.

MRR broken down by what each query was written to test:

| kind | n | vector | lexical | hybrid | hybrid+rerank |
|---|--:|--:|--:|--:|--:|
| semantic | 20 | 0.917 | 0.000 | 0.917 | 0.895 |
| lexical | 9 | 0.889 | 0.778 | 0.944 | **1.000** |
| hybrid | 4 | 1.000 | 0.750 | 1.000 | 1.000 |
| rerank | 3 | 1.000 | 0.000 | 1.000 | 0.778 |
| hard | 4 | 0.750 | 0.500 | 0.750 | 0.750 |

---

## 11. Interpreting them

**Hybrid RRF is the best configuration measured.** Best MRR (0.921) and best
nDCG (0.886) of the four, at a third of the reranker's latency.

**Fusion helps, modestly.** Hybrid beats vector-only on MRR (0.921 vs 0.908) and
nDCG (0.886 vs 0.877) with identical recall. The gain is real but small, and its
source is visible in the by-kind table: fusion helps `lexical` queries
(0.889 → 0.944) and changes nothing for `semantic` ones. That is exactly what it
should do — lexical search is the arm that finds exact terms, and 20 of the 40
queries have no exact terms to find.

**Lexical alone is weak here (MRR 0.300), and that is expected, not a bug.**
`websearch_to_tsquery` joins terms with AND, so a conversational query like
*"How can a relational database be made to answer lookups faster?"* requires
every term to appear and matches nothing. It returned nothing at all for 28 of
40 queries. Lexical search is not a general-purpose retriever on this workload;
it is a precision instrument for exact terms, and fusion is what lets it
contribute without dragging everything else down.

**Reranking did not help.** MRR 0.921 → 0.906, nDCG 0.886 → 0.858, Recall@5
0.850 → 0.746 — worse on every metric, for **42× the latency**. Stated plainly:
on this evaluation set, the Day 9 cross-encoder makes the ranking worse.

But the by-kind table shows it is not uniformly bad:

* **`lexical` queries improved**, 0.944 → **1.000** — perfect. Where the query
  contains exact technical terms, the cross-encoder reads them in context and
  gets every one right.
* **`rerank` queries got worse**, 1.000 → 0.778 — the cases written specifically
  to reward a reranker. Hybrid already had them at rank 1, so there was nothing
  to gain and something to lose.
* **`semantic` queries got slightly worse**, 0.917 → 0.895.

**The mechanism, from the per-query data.** Hybrid RRF already puts the right
answer at **rank 1 for 34 of 40 queries**. When a ranking is already that good,
a reranker can only hold position or damage it. And the cross-encoder's scores
sit in a narrow negative band — for `eq-011` the top ten spanned −3.71 to −6.83
— so it is expressing weak preferences among candidates it finds roughly
equivalent, and reordering near-ties is a coin flip against a ranking that was
already right.

Concretely, for *"Does the order of columns in a multi-column index matter?"*
the cross-encoder scored a **linked-list question above the composite-index
question**, moving the correct answer from rank 1 to rank 3.

### A hypothesis that was tested and rejected

Day 9 chose to show the reranker the full `search_document` (question +
concepts + taxonomy + tags), justified by a single measured case. The `eq-011`
failure suggested that choice might be the problem, so it was measured properly
across all 40 queries:

| document shown to the reranker | Recall@5 | Recall@10 | MRR | nDCG |
|---|--:|--:|--:|--:|
| `search_document` (Day 9's choice) | 0.746 | 0.875 | **0.906** | **0.858** |
| question text only | 0.721 | 0.808 | 0.832 | 0.797 |

**The hypothesis was wrong.** Day 9's choice is clearly better in aggregate,
even though it is worse on the one case that suggested the idea. A single
example is not evidence — which is the whole argument for having an evaluation
set. Nothing was changed.

### What is *not* justified by this data

* **"The reranker is useless."** It was perfect on `lexical` queries. With a
  larger, harder bank where hybrid does not already achieve rank 1 on 85% of
  queries, there is far more room for it to help.
* **"Hybrid RRF is the right production default."** It is the best of four
  configurations on 40 queries against 60 documents. That is a finding, not a
  law.
* **Any conclusion from a difference of one or two queries.** With n = 40, one
  query is 2.5% of recall and up to 0.025 of MRR. The hybrid-vs-vector MRR gap
  (0.013) is *smaller than a single query*. It is suggestive, not significant —
  no confidence intervals were computed, and with this n they would be wide.

---

## 12. Limitations

Stated plainly, because an evaluation whose limits are hidden is worse than none.

1. **40 queries is small.** The plan called for 100 (§12.1). Every metric has
   wide error bars; differences below ~0.05 should be treated as noise.
2. **60 documents is very small.** Finding one of 60 is a far easier task than
   one of 100,000. Recall near 0.88 here says little about behaviour at scale,
   and it structurally *disadvantages the reranker*, whose value grows as
   stage 1 gets less certain.
3. **One annotator, no second opinion.** Labels were assigned by one person
   (the author) in one pass. No inter-annotator agreement was measured, so
   label noise is unknown. Some queries genuinely have arguable labels —
   `eq-024` is deliberately answerable from two domains.
4. **The queries were written by the same person who wrote the questions**,
   after reading them. Real users phrase things in ways nobody anticipates.
   This is the single largest threat to validity here.
5. **The question bank is drafted, not human-approved.** All 60 items are
   `review_status: "drafted"`. Evaluating retrieval over unreviewed content is
   fine — retrieval does not care whether an answer is correct — but the bank
   is not yet what Phase 2's exit gate requires.
6. **Binary-ish grading.** 36 of 40 queries have exactly one grade-2 answer, so
   nDCG behaves much like MRR here and its extra power is mostly unused.
7. **No statistical testing.** No confidence intervals, no significance tests.
8. **Timings are from one laptop**, not a benchmark.

### Why these results do not transfer to production

Every number above is conditional on *this* bank, *this* query set and *this*
annotator. Retrieval quality is a property of a **corpus/query pair**, not of an
algorithm. Change any of the following and the table can reorder:

* **The corpus grows.** At 60 documents almost anything works. Vector-only
  scoring 0.908 is partly a statement about how few things there are to confuse.
* **Real queries arrive.** Users type fragments, typos and jargon nobody wrote
  labels for.
* **The domain shifts.** Sprints 2 and 3 add OS, networks, API design, RAG and
  React. New confusable neighbours change every ranking.

The right way to use this table is as a **baseline to re-measure against**, not
as a verdict. Re-run it after every bank sprint; a metric that drops is a
regression worth investigating.

---

## 13. The Phase 2 architecture, end to end

```
  data/question-bank/*.jsonl          60 questions, git-versioned, validated
            │
            ▼  ingest (Day 7) + embed (Day 8), one transaction
  Postgres: questions
            ├── search_document   text + concepts + taxonomy + tags
            ├── tsv               GENERATED, GIN-indexed          ─┐
            └── embedding         vector(384), HNSW-indexed       ─┤
                                                                  │
  query ──┬──► bge-small-en-v1.5 ──► vector search  (cosine, <=>) ─┤ Day 8
          └──► websearch_to_tsquery ► lexical search (ts_rank_cd) ─┘
                          │
                          ▼  Reciprocal Rank Fusion, k=60
                    40 candidates
                          │
                          ▼  bge-reranker-base, one pass per candidate   Day 9
                    reordered top 8
                          │
                          ▼  Recall@K · MRR · nDCG@K · four-row ablation Day 10
                       measured
```

Each layer is independently testable and independently replaceable. `search.py`
does not import `rerank`; `rerank` does not run queries; `runner.py` holds no
SQL and no model. That separation is what made it possible to measure the
reranker's contribution in isolation — and to discover that, on this bank, it is
negative.

---

## 14. Running the tests

```bash
cd backend

# the metrics alone — pure functions, no database, instant
pytest tests/unit/evaluation/test_metrics.py

# metrics + the committed dataset's own validity
pytest tests/unit/evaluation

# the runner against real Postgres (deterministic stand-in models)
pytest tests/integration/test_retrieval_eval.py

# everything
pytest
```

The default suite **never downloads a model**. The integration tests drive the
runner with `HashingEmbedder` and `LexicalOverlapReranker`, which is enough to
prove every mode is wired to the right function and every summary is the mean of
its outcomes — and **not** enough to say anything about quality. No test in the
suite asserts a quality number; those live in the committed report, produced
with real models by an explicit command.

| File | Tests | Covers |
|---|--:|---|
| `unit/evaluation/test_metrics.py` | 46 | every metric, hand-worked, plus edge cases |
| `unit/evaluation/test_dataset.py` | 29 | the schema and the committed ground truth |
| `integration/test_retrieval_eval.py` | 19 | the runner, fairness, determinism, reporting |
