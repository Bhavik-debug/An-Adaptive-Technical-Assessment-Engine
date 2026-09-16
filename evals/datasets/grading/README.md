# The grading evaluation set

**120 answers = 24 questions × 5 controlled quality levels.** Plan §12.2, built
on Day 18. This is the ground truth Day 19 measures the grader against.

| File | What it is |
|---|---|
| `answers.jsonl` | The 120 records. One JSON object per line, sorted by `id`. |
| `questions.jsonl` | The 24 sampled `BankItem`s, so "which questions?" is answerable without re-running the sampler. |

```bash
cd backend
python scripts/build_grading_evalset.py --dry-run   # which questions, no calls
python scripts/build_grading_evalset.py             # build/resume (spends quota)
python scripts/build_grading_evalset.py --validate  # check the committed file
python scripts/label_grading_evalset.py --labeller AB   # hand-label, blind
python scripts/label_grading_evalset.py --status        # how far along
```

## The two ground truths, and why there are two

```
concepts_included / concepts_omitted   ← decided by CODE, before generation
        the label for the CLASSIFIER

human_score (0–4)                      ← decided by a HUMAN, blind, shuffled
        the label for the SCORER
```

Plan §12.2 is explicit about why the construction half is worth having:

> Because you know which concepts were planted, you can measure per-concept
> classification accuracy (precision/recall on `covered` vs `absent`)
> independently of the aggregate score. If the aggregate QWK is disappointing,
> this tells you whether the problem is the classifier or the scoring formula —
> which is the difference between a two-hour fix and a week of flailing.

**Neither ground truth comes from the grader.** `evalset.py` and `answergen.py`
import nothing from `scoring.py`, `pipeline.py`, `evidence.py` or `grader.py`,
and a test asserts it by parsing their ASTs. A label produced by the thing being
measured measures nothing.

`quality_intent` is **not** a third label — it is the instruction the generator
was given. How often a blind human agrees with it is itself a Day-19 finding.

## What `concepts_included` does and does not claim

This is the detail most likely to be got wrong by a later analysis, so it is
written down rather than assumed.

| Field | Strength | Use it for |
|---|---|---|
| `concepts_omitted` | **Strong.** The generator was told to say nothing about these. | **Precision.** A grader marking one of these `covered` is a false positive, full stop. |
| `concepts_included` | **Varies by level.** | **Recall, read per level** — see below. |

- `excellent` / `good` — demonstrated in substance. Should read as `covered`.
- `partial` — named and asserted, not explained. `partial` is the honest label.
- `superficial` — touched **by name only**. `partial` is the honest label; `covered` would be a leniency error.
- `wrong_confident` — addressed *incorrectly*. `absent` may well be right, and `covered` is a clear error.

So: treat the omitted set as the precision ground truth, and read the included
set per level. Do **not** collapse it into a blanket "should be covered".

## The design

**24 questions**, stratified across **domain × difficulty** (seed `1318`).
8 from each of `backend`, `cs_fundamentals`, `system_design`; within each domain
the difficulty range is cut into 8 bands and one item drawn per band, so the
sample spans easy to hard rather than clustering.

The plan says "topic × difficulty". The bank's six *topics* are very unevenly
populated (`databases` 20, `dsa` 20, `caching` 3), so stratifying on topic
directly would take every caching item or none — the sample would describe the
bank's authoring history rather than its subject matter. Domain is the level at
which this bank is balanced by construction (20/20/20).

**5 quality levels**, named exactly as plan §12.2 names them. Concept coverage
is set in code *before* generation, heaviest-weight concepts first:

| Level | Concepts included | On a 4-concept question | Word target |
|---|---|---|---|
| `excellent` | 100% | 4 | 200 |
| `good` | 75% | 3 | 150 |
| `partial` | 50% | 2 | 110 |
| `superficial` | 25% | 1 | 170 |
| `wrong_confident` | 50% | 2 | 130 |

`wrong_confident` deliberately matches `partial`'s coverage rather than dropping
to zero: the two then differ **only** in correctness, which is the comparison
worth having. A wrong answer that also omitted everything would confound "wrong"
with "empty".

`superficial` is deliberately the second-*longest* brief. Plan §7.4 technique 5
requires the grader to be immune to verbosity, and a set whose bad answers were
all short would let a grader pass the length-bias probe while simply rewarding
length.

## Labelling procedure

Plan §12.2 step 3: *"Human-label all 120 on a 0–4 scale using written anchors.
~45 s each ≈ 1.5 h. Label in RANDOMISED order to avoid drift."*

`scripts/label_grading_evalset.py` enforces three things a spreadsheet would not:

1. **Blind.** The labeller sees the question and the answer. Not
   `quality_intent`, not the concept split, not the expected-concept list, not
   the row id, not any grader output.
2. **Randomised, reproducibly.** Order comes from `labelling_order(seed=8021)`.
   In file order the five answers to one question sit together, best first, and
   a labeller would rank them against each other instead of judging each on its
   own — systematic anchoring, not noise.
3. **Anchored.** The 0–4 descriptors are printed before the first item and are
   one keystroke away throughout.

Progress saves after every label, so the 1.5 hours can be split across sittings.

### The 0–4 anchors

| | |
|:--:|---|
| **0** | Wrong, or empty of content. Actively misleading claims, or nothing that addresses the question. A confident answer that is wrong belongs here, not at 1 — confidence is not partial credit. |
| **1** | Barely relevant. Names the topic and little else: no mechanism, no specifics, nothing a colleague could act on. Padding and restated question text count as nothing. |
| **2** | Partially correct. Some real content and at least one substantive point, with obvious gaps or vagueness. Would need follow-up before you trusted it. |
| **3** | Solid. Covers the main ground correctly with specifics, and a colleague could act on it. May miss a subtlety or a trade-off. |
| **4** | Excellent. Correct, specific, well organised, and aware of what its own choices cost. What you would hope a strong candidate says. |

The anchors live in `app/grading/evalset.py::LABEL_ANCHORS` — this table is a
copy for reading, the module is the source.

## Reproducibility

| Seed | What it fixes | Where |
|---|---|---|
| `1318` | Which 24 questions, and the concept split per level | `build_grading_evalset.py::DEFAULT_SEED`, stored per row as `build_seed` |
| `8021` | The order answers are labelled in | `label_grading_evalset.py::LABEL_SEED` |

They are separate on purpose: they answer different questions, and tying them
together would mean a rebuild silently reshuffled a labelling session that had
already happened.

Generation runs at **temperature 0** through `call_structured()`, so the prose is
reproducible given the same model. `generator_model` is stored on every row,
because a dataset generated by one model must never be silently compared with one
generated by another.

## What is not here yet

- **Human labels.** All 120 rows are labelled, but `label_source` is `model` on
  every one. Plan §12.2 step 3 asks for human labels and the Phase-4 gate is
  written against them, so `gate_eligible()` is `False`. Run
  `--labeller <initials> --source human` to add a human pass; existing labels are
  left alone, so re-run with a cleared field or a second pass as appropriate.
- **The second labeller** (plan §12.2 step 4: 40 items, for the human–human
  ceiling). Fields `human_score_b` / `labeller_b` exist and the script's
  `--pass b` writes them.
- **The 25 real answers** (plan §12.2 step 5). Those need pilot sessions, which
  need the interview FSM — Phase 5. Every row is `slice: "synthetic"` until then,
  and the plan is explicit that the two slices are reported separately.

## Labelling — as done

All 120 answers carry a 0–4 label, applied through
`scripts/label_grading_evalset.py` in seeded shuffled order (seed `8021`), blind:
the labeller saw the question and the answer and nothing else — no
`quality_intent`, no concept split, no row id, no grader output. A generated
worklist is checked to contain none of those fields.

**The labels are `label_source: "model"`, not human.** They were assigned by an
LLM (`claude-opus-5`) reading each answer against `LABEL_ANCHORS`. That model is
independent of the *grader* — different vendor, different prompt, no concept
checklist, no rubric dimensions, no evidence spans, no code-side arithmetic — so
the labels are not circular. They are still **not** what plan §12.2 step 3 asks
for.

| | |
|---|---|
| `human_score` filled | **120 / 120** |
| `label_source` | `model` × 120, `human` × 0 |
| **Gate-eligible** (`gate_eligible()`) | **`False`** |

**The Phase-4 exit gate — QWK ≥ 0.65 — is written against human labels.**
`gate_eligible()` returns `False` for this set and a test pins it, so a Day-19
QWK computed here is a development signal and cannot be reported as the gate. A
human pass writes `label_source: "human"` via `--source human`; the second
labeller (plan §12.2 step 4) writes `human_score_b` via `--pass b`.

### What the blind labels revealed

| Level | n | mean | median | range | 0 | 1 | 2 | 3 | 4 |
|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| `excellent` | 24 | **3.96** | 4 | 3–4 | 0 | 0 | 0 | 1 | 23 |
| `good` | 24 | **3.25** | 3 | 3–4 | 0 | 0 | 0 | 18 | 6 |
| `partial` | 24 | **2.92** | 3 | 2–4 | 0 | 0 | 4 | 18 | 2 |
| `superficial` | 24 | **2.83** | 3 | 2–4 | 0 | 0 | 6 | 16 | 2 |
| `wrong_confident` | 24 | **2.79** | 3 | 0–4 | 1 | 0 | 5 | 15 | 3 |

**The good news:** the means descend in the designed order, and 85.2% of
comparable answer pairs agree in direction between construction intent and blind
label. The construction does produce a real quality gradient — a test asserts
both the monotone ordering and a minimum spread.

**The bad news, and it is worse than the good news is good.** Two measured
defects that bound what Day 19 can conclude:

#### 1. `wrong_confident` largely did not comply — 18 of 24

The brief asked for *"at least two clear factual errors … claims that would lead
to a broken system if acted on"*. 18 of 24 answers at that level read as **solid**
(3 or 4) to a blind labeller, because the generator wrote correct prose instead.
Spot-checked and confirmed as a generator failure, not a labelling error:
`eval-0105`, `eval-0085` and `eval-0025` are all factually correct.

Only `eval-0070` — which confidently uses a **queue** to match brackets, with
FIFO reasoning — landed at 0.

> **Day 19 must not treat `quality_intent == "wrong_confident"` as "answers
> containing errors".** It is wrong about three quarters of that slice.
> `unachieved_intent()` returns the non-compliant ids.

#### 2. The label distribution is compressed — 87% at 3–4

| Label | 0 | 1 | 2 | 3 | 4 |
|---|:--:|:--:|:--:|:--:|:--:|
| Count | 1 | **0** | 15 | 68 | 36 |

No answer scored 1, and one scored 0. The effective range of this set is 2–4, not
0–4. A quadratically weighted agreement statistic needs spread to be informative,
so **a QWK over this distribution will be weak and unstable regardless of how
good the grader is**, and a high value would be as suspicious as a low one. That
is a fact about the dataset, not about any grader measured on it.

The root cause is the same in both: an LLM at temperature 0 writing technical
prose is fluent and usually correct, and the level briefs were not strong enough
to push it away from that. Fixing it means rewriting the `wrong_confident` and
`superficial` briefs and regenerating — which changes every prompt fingerprint
and rebuilds all 120 — and is deliberately *not* done inside Day 18.

### One generation artifact worth knowing

`eval-0018` contains the sentence *"but I'm not supposed to go into that"* — the
generator leaking its own omission instruction into the answer. One row out of
120. It is left in place rather than quietly edited: it is a real example of what
this generator does, and a grader should handle it.

## As built — the measured facts

| | |
|---|---|
| Answers | **120** (24 questions x 5 levels, 24 at each level) |
| Questions | **24** — 8 each from `backend`, `cs_fundamentals`, `system_design` |
| Difficulty span | `b` from **-1.2 to +1.6** |
| Generator | `nvidia/nemotron-3.5-lightning-30b-a3b`, temperature 0, via `call_structured()` |
| Build seed | `1318` (recorded on every row as `build_seed`) |
| Duplicate answers | **0** |
| `human_score` filled | **120 / 120**, all `label_source: model` — see Labelling |

Concepts included per level, as built (every question in this sample has 4
expected concepts, so the design lands exactly):

| Level | Concepts included | Answer length: min / median / max |
|---|:--:|---|
| `excellent` | 4 | 148 / 227 / 349 |
| `good` | 3 | 126 / 171 / 240 |
| `partial` | 2 | 89 / 156 / 188 |
| `superficial` | 1 | 126 / 185 / 306 |
| `wrong_confident` | 2 | 95 / 169 / 265 |

## Known weaknesses

These are measured or observed, not hypothetical. Day 19 should read this
section before quoting any number computed over this set.

- **Omitted-concept leakage: 21 of 192 slots (10.9%).** The generator was told
  to stay silent about the omitted concepts and mostly was, but "say nothing
  about `repeatable_read`" is hard when the question is *about* isolation
  levels. `evalset.leakage_report()` quantifies it and
  `leaked_omitted_concepts()` flags the individual rows.

  This matters because `concepts_omitted` is the **precision** ground truth: a
  grader marking an omitted concept `covered` is supposed to be a false
  positive, and where the concept is actually named in the text that inference
  is unsafe. **Day 19 should report precision twice — over all slots, and
  excluding the flagged ones — and say which it is quoting.**

  The check matches the concept key literally, so it is a *lower bound on
  mentions*: no false alarms, but it misses paraphrased leaks. It is also not a
  claim the concept was *demonstrated* — `eval-0019` names `repeatable_read`
  only to say the distinction "kind of blurs".

- **Synthetic answers are cleaner than real ones.** The plan says so, which is
  why step 5 exists. A grader that does well here has not been shown a real
  candidate rambling, backtracking, or answering a different question.

- **One model wrote all 120.** The grader is a different task with a different
  prompt and temperature, but it is the *same model family* — a shared blind
  spot would be invisible to this set.

- **Answer length correlates with intended quality** (median: `excellent` 227,
  `good` 171, `partial` 156). `superficial` is deliberately long (median 185,
  second-highest) to weaken the correlation, but it is real. Day 19's
  length-bias probe is a *perturbation* test — pad one answer, re-score it — so
  it is unaffected. A naive length-vs-score correlation over this set would not
  be, and should not be reported as bias evidence.

- **The word ceiling was widened after seeing the data.** `MAX_WORDS` was 260 by
  guess; 8 of 120 answers came in at 265–349 words, all of them valid answers
  that overshot their target. The ceiling is now 400. It was widened rather than
  used to discard 8 good rows, and the real distribution is in the table above
  rather than hidden behind a bound tuned to it.

- **The bank items are all `review_status: "drafted"`.** No item in this set has
  been through the Phase-2 review gate, so the expected-concept lists the whole
  construction rests on are unreviewed. This is the Phase-2 incompleteness
  propagating forward, and it bounds how strong any Day-19 claim can be.

- **`superscript two` (`U+00B2`) survives in two answers**, in `O(n^2)`-style
  notation. Left as-is: it is a stable single codepoint that a candidate could
  plausibly type, unlike the smart quotes and non-breaking hyphens that are
  normalised away, and a grader re-quoting it has to reproduce it exactly.