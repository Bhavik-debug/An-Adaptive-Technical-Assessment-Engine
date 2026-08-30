# The question bank

`data/question-bank/` is a **dataset artefact**, not a table someone types rows
into. It lives in git, is reviewed in pull requests, and is loaded into Postgres
by an ingest step that is a *projection* of the files rather than a second
source of truth. Plan §6 is the design; this file is how to work with it.

```
data/question-bank/
├── taxonomy.json        # the controlled vocabulary: domain -> topic -> subtopic
├── databases.jsonl      # one item per line, one domain per file
├── dsa.jsonl
└── system_design.jsonl
```

| Code | What it is |
|---|---|
| `backend/app/bank/taxonomy.py` | loads and validates `taxonomy.json` |
| `backend/app/bank/schema.py` | `BankItem` — what one item must look like |
| `backend/app/bank/loader.py` | reading the files + every check that spans two items |
| `backend/app/bank/ingest.py` | upsert into `topics` and `questions` |
| `backend/scripts/validate_question_bank.py` | the validator's command line |
| `backend/scripts/review_question_bank.py` | the manual review workflow |
| `backend/scripts/ingest_question_bank.py` | JSONL → Postgres |

---

## The schema

One JSON object per line. Field order in the file is fixed so that a diff of two
versions of an item is readable.

| Field | Type | Rule |
|---|---|---|
| `id` | string | `sys-cache-002` shape: lower-case hyphenated segments ending in three digits. Unique across the whole bank, and the primary key in Postgres — **never reuse or renumber one** |
| `topic` | string | a *topic*-level key from `taxonomy.json` |
| `subtopic` | string | a *subtopic*-level key whose parent is `topic` |
| `text` | string | the canonical question, 40–1200 chars |
| `difficulty_b` | float | IRT difficulty, `-3 ≤ b ≤ 3`, **higher is harder** — same scale as candidate ability θ |
| `discrimination_a` | float | IRT discrimination, 0.3–2.5. `1.0` until §5.11 calibration has real response data |
| `expected_concepts` | list | 3–6 entries of `{key, weight, hint}`; `weight` is an integer 1–3; keys are `lower_snake_case` and unique within the item |
| `common_misconceptions` | list | optional; what a plausible wrong answer looks like |
| `reference_answer` | string | non-empty, ≥200 chars. What a *strong* answer contains, detailed enough to grade against |
| `follow_up_seeds` | list | optional prompts for the follow-up probe (§8.5) |
| `anchor_terms` | list | 1–8 terms that **must appear verbatim in `text`**. §6.2 checks a re-rendered question still contains them |
| `time_estimate_s` | int | 60–900 |
| `tags` | list | 1–8 free-form tags for filtering |
| `source` | enum | `authored` \| `llm_drafted` \| `imported` |
| `review_status` | enum | `drafted` \| `reviewed` |
| `reviewed_by` | string \| null | required iff `reviewed` |
| `reviewed_at` | date \| null | required iff `reviewed` |
| `version` | int | bump when the text or concepts change materially |

Unknown fields are **rejected**, not ignored. A misspelt key in a hand-edited
line would otherwise silently drop `expected_concepts`, which is the field the
entire grading pipeline is built on.

### Why `difficulty_b` is not guesswork

`b` sits on the same scale as candidate ability θ (§9.3), so `b = 0` means "an
at-target candidate has about an even chance", `b = +1` is "above target",
`b = -1` is "below target". §6.4's rule is to seed by **comparison, not absolute
judgement**: keep a few reference items per topic pinned at −1, 0 and +1 and
rate every new item against them. §5.11 corrects the estimates from real
response data later; the job here is to be *consistent*, not precise.

### Why concept keys matter more than anything else

Keys are a **shared controlled vocabulary across questions**. Reuse is what lets
the report say "missed `cache_invalidation` across three different questions",
which is far more useful than three per-question scores. It also means a typo is
expensive: `hash_collision` and `hash_collisions` are two vocabulary entries,
every score derived from either is computed over half the evidence, and nothing
errors. The validator warns on any two keys one character apart for exactly this
reason.

A concept must be **observable in an answer** (`distinguishes write-through from
write-behind`, not `understands caching deeply`) and **independent** of the
others (if covering A guarantees covering B, they are one concept with weight 2).

---

## Validating

```bash
cd backend
python scripts/validate_question_bank.py            # report + exit code
python scripts/validate_question_bank.py --table    # the per-item review table
python scripts/validate_question_bank.py --strict   # warnings fail too
python scripts/validate_question_bank.py --require-reviewed   # the Phase 2 gate
```

The same `validate_bank()` runs in `tests/unit/bank/test_committed_dataset.py`
and in CI, so the command line and the pipeline cannot disagree about what
"valid" means.

**Errors** (these fail CI): invalid JSON; a schema violation; a duplicate id; an
unknown topic or subtopic, or a subtopic that belongs to a different topic; two
items from different domains in one file; a near-verbatim duplicate (word 4-gram
Jaccard ≥ 0.60).

**Warnings** (for a human): two questions that overlap heavily without being
duplicates; two concept keys one character apart.

---

## Reviewing — and the one rule that is not negotiable

> **Never mark an item `reviewed` unless a person actually read it.**

`review_status` is the only thing standing between "150 items" and "150 items
worth having". §6.4's cut-line is explicit: 110 items is acceptable, unreviewed
items never are. An item with a wrong concept key silently corrupts every score
derived from it, and it does so without erroring.

Passing `pytest` is *not* review. Schema validity says nothing about whether the
question is technically correct.

```bash
cd backend
python scripts/review_question_bank.py --pending                 # what is left
python scripts/review_question_bank.py --show dsa-graphs-002     # read one properly
python scripts/review_question_bank.py --approve dsa-graphs-002 --reviewer manas
```

`--approve` prints the item, asks for confirmation, then rewrites that one line
with `review_status: "reviewed"`, the reviewer's name and today's date, and
re-validates. The schema refuses `reviewed` without a named reviewer *and* a
date, and refuses `drafted` *with* either — so a half-finished review cannot
read as a finished one.

Budget ~90 seconds per item against this checklist (§6.4):

1. Is it technically correct?
2. Is it unambiguous, and answerable without hidden assumptions?
3. Does it actually test the listed concepts?
4. Are the concepts observable and independent, and are there ≥3 real ones?
5. Is `b` honest relative to the last ten items you rated?
6. Is the reference answer correct and sufficient to grade against?
7. Are `topic` and `subtopic` right?
8. Is it materially different from the other items in its subtopic?
9. Is it answerable in `time_estimate_s`?
10. Would a strong candidate plausibly miss the weight-3 concept? If not, weight
    it lower — a concept everyone covers carries no information.

If an item fails, edit the JSONL line and leave it `drafted`, or delete it. Do
not approve it with a note.

---

## Ingesting

```bash
cd backend
python scripts/ingest_question_bank.py --dry-run      # validate only
python scripts/ingest_question_bank.py                # everything
python scripts/ingest_question_bank.py --only-reviewed  # the production posture
```

Idempotent — an upsert keyed on the readable id, so running it twice is running
it once. It refuses to write anything if the dataset does not validate, because
a half-ingested bank is worse than no bank: the retrieval layer cannot tell the
difference.

Two deliberate behaviours:

- **Deletions are never automatic.** An item removed from the files is
  *reported*; `turns.question_id` references `questions.id`, and a silent
  cascade would erase interview history to tidy up a dataset edit.
- **`tsv` is never written by ingest.** It is a `GENERATED` column that Postgres
  recomputes on every write of `text`, so the search index cannot drift out of
  sync with the thing it indexes.

Any deployment that serves real candidates must run `--only-reviewed`.

---

## Adding items

1. Pick a subtopic and a difficulty band. Check what already exists there:
   `python scripts/validate_question_bank.py --table`.
2. Draft with Claude in the chat UI. This is *authoring*, not runtime — it costs
   the product nothing and must never go through the application's LLM
   infrastructure or spend NVIDIA quota.
3. Append to the `.jsonl` for that domain, with `source: "llm_drafted"`,
   `review_status: "drafted"` and both review fields `null`.
4. `python scripts/validate_question_bank.py` until clean.
5. Review each item yourself and approve it with `review_question_bank.py`.
6. Commit the JSONL. The diff is the review record.

If a subtopic does not exist, add it to `taxonomy.json` in the same pull
request — but sparingly. Keys are the primary key of `topics` and are referenced
by `questions` and `skill_states`, so a key is effectively permanent; and a
subtopic with one question is a subtopic whose θ estimate will never leave its
prior.
