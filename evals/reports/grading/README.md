# Grading-validity raw data

The artefacts the Day-19 report is computed from. Committed on purpose: a report
whose inputs are gone is a report nobody can check.

| File | What it is |
|---|---|
| `runs.jsonl` | One row per grader invocation. |
| `failures.jsonl` | One row per grading attempt that produced **no** score, with the error kind. |

## `runs.jsonl`

One JSON object per line, sorted by `(answer_id, prompt_version, variant,
run_index)` so a git diff stays readable.

That four-part key is what lets one table serve the whole harness:

| Slice | Key pattern |
|---|---|
| the main pass | `variant="base"`, `run_index=0` |
| test–retest | `variant="base"`, `run_index` 0–4 |
| bias probes | `variant` in `filler` / `confidence` / `jargon` / `concepts_shuffled`, paired against the `base` row for the same answer |
| prompt comparison | `prompt_version` varies |

Keeping them in one shape rather than four means the same metric code reads all
of them, and a probe cannot accidentally be compared against a baseline graded
under a different prompt.

### Every row carries the judgment it was scored from

`judgment` holds the **evidence-corrected** `GradeJudgment` — the concept
labels, the four rubric levels, the errors, the confidence. It is the largest
field in the file and it is there for three reasons:

1. It makes *"every number in the report is recomputed from these rows"*
   literally true rather than nearly true. `--report` makes no model calls.
2. The **position-bias probe** is a deterministic re-score of a stored judgment
   with the concept list shuffled — no second model call, which matters because
   re-asking the model would measure its own nondeterminism and could never
   return exactly zero.
3. The plan (§7.2) asks for the concept/rubric weighting to be **tuned against
   the labelled set**. With the judgment stored, that sweep is arithmetic over
   this file rather than another 120 paid calls.

It is the corrected judgment, not the model's raw output: the raw one would
re-introduce evidence spans that verification already downgraded, and re-scoring
it would not reproduce the row's own score.

### Reproducing anything in the report

```bash
cd backend

# recompute the whole report from this file. No model calls, no quota.
python scripts/run_grading_validity.py --report

# the position-bias probe, also offline - it re-scores stored judgments
python scripts/run_grading_validity.py --probe position
```

Both are safe to run repeatedly. Everything else in
`scripts/run_grading_validity.py` spends quota and is resumable: a row already
present for a key is never re-graded, so an interrupted pass costs only the call
it was in the middle of.

### Run one writer at a time

**`dump_runs` rewrites the whole file from an in-memory list.** That is what makes
the artefact sorted, diffable and crash-safe for a single process — the file is
never left half-written — and it means two processes writing it concurrently
will clobber each other: each loaded `existing` at *its* start, so whichever
writes last silently drops every row the other added in between.

This was learned the hard way during Day 19. Two overlapping invocations
destroyed a completed probe's rows twice before the cause was identified, and
the symptom was confusing rather than loud: row counts went *down* between
checks, and nothing failed.

So: one writer at a time. The resumability is what makes that cheap — re-running
a step after the previous one finishes re-grades nothing. Within a single
invocation, `--concurrency N` is safe: the dump is taken under an `asyncio.Lock`
and there is one process holding the list.

A file lock would be the real fix if this ever needs to run in parallel. It does
not today, and a lock that is never exercised is a lock nobody can trust, so the
constraint is written down here instead.

## `failures.jsonl`

Deduplicated by the same four-part key, newest entry winning — the pass is
resumable and re-run, so an answer the model cannot grade fails again on every
attempt, and three identical rows would make the violation rate look three
times worse than it is.

The dominant kind is `GradingContractError`, and it is **not** a schema failure:
the model returned structurally valid JSON that invented concept keys or failed
to classify keys it was given. `grader.check_judgment` catches it, the pipeline
does not retry it, and that answer therefore has no score — which is why the
report states the denominator on every line instead of assuming 120.
