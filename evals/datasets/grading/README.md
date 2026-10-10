# The grading evaluation set

**120 answers = 24 questions × 5 controlled quality levels.** Plan §12.2, built
on Day 18. This is the ground truth Day 19 measures the grader against.

| File | What it is |
|---|---|
| `answers.jsonl` | The 120 records. One JSON object per line, sorted by `id`. |
| `questions.jsonl` | The 24 sampled `BankItem`s, so "which questions?" is answerable without re-running the sampler. |
| `reference_labels.jsonl` | **Day 19.** A third labelling pass, with a rationale and evidence lists per row. Kept in its own file rather than as more columns on `answers.jsonl` — see below. |

### Three label passes, and where each one lives

| Pass | Where | Labeller | Rubric | What it is good for |
|---|---|---|---|---|
| Day-18 pass A | `answers.jsonl` → `human_score` / `labeller`, `label_source: model` | `claude-opus-5[1m]` | `labels-v1` | Best-spread labels; **different model family from the grader** |
| Day-18 retired | `answers.jsonl` → `model_score` / `model_labeller` | `claude-opus-5` | `labels-v1` | The original pass, preserved |
| Day-19 blind | `reference_labels.jsonl` → `judgment.score` | `nvidia/nemotron-3.5-lightning-30b-a3b` | `labels-v2` | **Isolated from `quality_intent` by construction**; same model as the grader; severely compressed |

**All three are `label_source: "model"`. There are still no human labels, and
`gate_eligible()` is `False`.**

The Day-19 pass is a separate file for two reasons. `answers.jsonl` is the
*construction* record and has been stable since Day 18 — appending a third
scoring pass to it would make its diff unreadable. And a Day-19 row carries a
rationale, three evidence lists, a confidence and six provenance fields, which
does not fit a column.

```bash
cd backend
python scripts/build_grading_evalset.py --dry-run   # which questions, no calls
python scripts/build_grading_evalset.py             # build/resume (spends quota)
python scripts/build_grading_evalset.py --validate  # check the committed file
python scripts/label_grading_evalset.py --status        # how far along
python scripts/label_grading_evalset.py --labeller AB   # hand-label, blind, interactive

# a labelling pass done away from the terminal
python scripts/label_grading_evalset.py --worklist /tmp/items.txt   # blind, shuffled
python scripts/label_grading_evalset.py --from-file /tmp/scores.txt \
    --by-position --source human --labeller AB

# human_score is occupied by a model pass? move it aside first
python scripts/label_grading_evalset.py --retire-model-labels
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

### The row id leaks `quality_intent`, and the worklist used to print it

Worth writing down, because it was a real defect in the one artefact advertised
as blind.

Answer ids are allocated sequentially per question × level, so **`(N − 1) mod 5`
recovers the construction intent exactly**: `eval-0110` → 4 → `wrong_confident`,
`eval-0048` → 2 → `partial`. The interactive prompt never showed the id. The
`--worklist` file did — and the worklist is what a pass done away from the
terminal actually reads.

The worklist now carries **positions only**, and `--from-file --by-position`
applies scores back. No key file is needed, because the order is a pure function
of the committed seed, so there is nothing extra to leak either. The builder
moved out of the script into `evalset.blind_worklist()` so the property is
asserted by `TestTheBlindWorklist` — which rebuilds the worklist from position,
question and answer alone and compares — rather than trusted to an f-string in a
CLI.

A lexical check (*"the word `partial` must not appear"*) was tried first and
does not work: answer prose says "a partial update". The test is structural
instead, which catches any leaked field regardless of how its name spells.

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

## The Day-19 reference labels, and the rubric change that came with them

`reference_labels.jsonl` is produced by
`backend/scripts/build_reference_labels.py` and analysed by
`backend/scripts/report_reference_labels.py`, which writes
`evals/reports/reference_labels.md`. Read that report before quoting any number
computed against these labels — it leads with the two things wrong with them.

**What is better about this pass.** It is isolated from the construction intent
*by construction* rather than by discipline: the labelling prompt's
`required_inputs` are `("question", "answer")`, so there is nowhere to put
`quality_intent`, the concept split, or a previous label even by mistake. A test
renders every one of the 120 rows and rebuilds the prompt from the question and
the answer alone to prove nothing else got in. The Day-18 second pass could not
make that claim: its labeller had read this README.

**What is worse about it, and it is worse than the isolation is good.**

1. **The labeller is the same model that wrote the answers**, and the same model
   the grader runs on — the configured provider serves every capability tier
   from one model. A QWK between the grader and these labels measures two
   prompts to one model, not validity.
2. **Severe central tendency: 98 of 120 rows land on band 3** (81.7%). The
   Day-18 passes put 36–44 rows on band 4 and spread across three bands.
3. **It does not reproduce the designed quality gradient.** Means by intent are
   2.92 / 3.00 / 2.71 / 2.79 / 2.67 — `good` scores *above* `excellent`, and the
   spread is 0.33 against 1.00–1.17 for the Day-18 passes.

So on discriminating power this pass is the **weakest** of the three, and that is
reported rather than corrected: re-prompting the labeller until its histogram
looked better would be fitting the yardstick to the thing it measures.

### Rubric `labels-v2`

`LABEL_ANCHORS_V2` keeps all five of v1's rung descriptors **verbatim** —
a test asserts `LABEL_ANCHORS_V2 == LABEL_ANCHORS` — and adds one clause v1
left undefined, `ERROR_PRECEDENCE_V2`.

**The ambiguity it resolves.** v1's anchor 0 says *"a confident answer that is
wrong belongs here, not at 1"*; anchor 2 says *"partially correct … would need
follow-up"*. Neither covers the case that recurs in this dataset: an answer that
is detailed, specific and substantially correct, and **confidently wrong about
one of the three things the question asked**. Anchor 1 does not apply at all —
it describes *emptiness*, not partial wrongness. Three of 120 rows sat in that
gap under v1 and were scored by judgement rather than by the rubric.

v2's rule: no substantive error → score on the rungs; an error in the claim the
question principally asks for, or wrong throughout → 0, however fluent; an error
confined to one part of a multi-part question with the rest correct → **cap at
2**; omissions and vagueness with no false claims → no cap.

**Comparability.** Because the rung text is unchanged, a row scored under v1 and
a row scored under v2 are on the same scale wherever v1 was unambiguous; only
the ambiguous rows are affected. **Direction of the change:** capping a
detailed, half-correct answer at 2 moves labels *down*, away from the 3–4 mass
where 84% of this dataset already sits. It widens the distribution and makes the
grader's job harder. A rubric edit chosen to flatter a grader would have gone
the other way.

## Day 20: what this dataset was used to find

The dataset did its job, which is worth recording because the result is
unflattering to the grader rather than to the dataset.

- The **length-bias probe** built on these 120 answers found that 5 of 8 error
  claims the grader flagged were **fabrications** - quotations of sentences the
  candidate never wrote, including both `major` claims at 0.15 each. That is now
  fixed in code (`evidence.verify_evidence` verifies error claims exactly as it
  verifies evidence spans).
- The **injection suite** used 32 of these answers as attack carriers, with their
  committed Day-19 grades as controls. ASR = 25.0%.
- The **noise floor** comparison - contentless filler crossing the attack
  suite's own success threshold on 21% of answers against adversarial payloads'
  25% - is what identified one root cause behind three failing gates. That
  comparison is only possible because the length probe and the injection suite
  share a carrier set and a control set.

See `evals/reports/phase4_completion.md`.

## What is not here yet

- **Human labels. Still none.** All 120 rows are now labelled **three times**,
  and `label_source` is `model` on every one. Plan §12.2 step 3 asks for
  human labels and the Phase-4 gate is written against them, so
  `gate_eligible()` is `False`.

  To add a human pass: `--retire-model-labels` first (it moves the current pass
  into `model_score`/`model_labeller`, which `human_score` needs free), then
  `--labeller <initials> --source human`. `--from-file` **skips rows that
  already carry a score**, so without the retire step a human pass applies zero
  labels and exits 0; the script now says so loudly instead of reporting
  success.
- **The second labeller** (plan §12.2 step 4: 40 items, for the human–human
  ceiling). Fields `human_score_b` / `labeller_b` exist and the script's
  `--pass b` writes them.
- **The 25 real answers** (plan §12.2 step 5). Those need pilot sessions, which
  need the interview FSM — Phase 5. Every row is `slice: "synthetic"` until then,
  and the plan is explicit that the two slices are reported separately.

## Labelling — as done

**There are two passes over all 120 answers, and both are model passes.**

| Field | Pass | Labeller | Source |
|---|---|---|---|
| `human_score` / `labeller` / `label_source` | pass A, current | `claude-opus-5[1m]` | `model` |
| `model_score` / `model_labeller` | the retired first pass | `claude-opus-5` | model by construction |
| `human_score_b` / `labeller_b` | plan §12.2 step 4, not collected | — | — |

Both were applied through `scripts/label_grading_evalset.py` in seeded shuffled
order (seed `8021`), against `LABEL_ANCHORS`, from a worklist carrying the
question and the answer and nothing else.

**Neither pass is human, and `gate_eligible()` is `False`.**

| | |
|---|---|
| `human_score` filled | **120 / 120** |
| `label_source` | `model` × 120, `human` × 0 |
| `model_score` filled (retired pass) | **120 / 120** |
| **Gate-eligible** (`gate_eligible()`) | **`False`** |

**The Phase-4 exit gate — QWK ≥ 0.65 — is written against human labels.**
`gate_eligible()` returns `False` for this set and a test pins it, so a Day-19
QWK computed here is a development signal and cannot be reported as the gate.
A test also asserts that **two model passes do not add up to a human one** —
stacking provisional labels must never satisfy the gate.

### Why the first pass was kept rather than overwritten

`human_score` had to be freed for a second pass, and the choice was to destroy
the first pass or move it. It was moved, into `model_score`/`model_labeller`:
holding both on one row makes *"how closely do two independent-ish passes
agree?"* a question this dataset can answer, and that is unanswerable once the
column is gone. `label_agreement()` computes it.

### The two passes agree closely — and that is reproducibility, not validation

| | |
|---|---|
| Rows compared | 120 |
| Exact agreement | **90 / 120 (75%)** |
| Within one point | **120 / 120 (100%)** |
| Max disagreement | **1 point** |
| Mean signed error (retired − current) | **−0.03** |
| Mean absolute error | **0.25** |

Excluding the six rows this README had already described (see *Known
weaknesses*): 84 / 114 exact (73.7%), MAE 0.26, still never more than one point
apart. So the contamination moves the headline agreement by about a point and a
half — it is not what produces the result, but the clean figure is the one to
quote.

Thirty rows differ, every one of them by exactly one point: 17 where the current
pass was the more generous, 13 where the retired pass was.

**This number does not validate either pass.** `claude-opus-5` and
`claude-opus-5[1m]` are the same model family, so this measures whether the
anchors are applied *reproducibly*, not whether they are applied *correctly*.
Both passes can be wrong in the same direction for the same reason, and nothing
in a 0.25 MAE would show it. `label_agreement()` returns a `sources` key for
exactly this reason, and the script prints which comparison it is:

```
model vs model (same family - reproducibility, NOT validation)
```

The genuinely useful version of this statistic is the **human vs model** one,
which answers whether model labelling could stand in for the 1.5 hours of human
time on a future set. **That comparison has not been made**, because there is no
human pass to make it against.

### What the blind labels revealed

Pass A (current, `claude-opus-5[1m]`):

| Level | n | mean | median | range | 0 | 1 | 2 | 3 | 4 |
|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| `excellent` | 24 | **3.79** | 4 | 2–4 | 0 | 0 | 1 | 3 | 20 |
| `good` | 24 | **3.50** | 4 | 2–4 | 0 | 0 | 1 | 10 | 13 |
| `partial` | 24 | **2.96** | 3 | 2–4 | 0 | 0 | 5 | 15 | 4 |
| `superficial` | 24 | **2.88** | 3 | 2–4 | 0 | 0 | 7 | 13 | 4 |
| `wrong_confident` | 24 | **2.79** | 3 | 0–4 | 1 | 1 | 3 | 16 | 3 |

The retired first pass (`claude-opus-5`), for contrast:

| Level | n | mean | median | range | 0 | 1 | 2 | 3 | 4 |
|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| `excellent` | 24 | 3.96 | 4 | 3–4 | 0 | 0 | 0 | 1 | 23 |
| `good` | 24 | 3.25 | 3 | 3–4 | 0 | 0 | 0 | 18 | 6 |
| `partial` | 24 | 2.92 | 3 | 2–4 | 0 | 0 | 4 | 18 | 2 |
| `superficial` | 24 | 2.83 | 3 | 2–4 | 0 | 0 | 6 | 16 | 2 |
| `wrong_confident` | 24 | 2.79 | 3 | 0–4 | 1 | 0 | 5 | 15 | 3 |

**The good news:** the means descend in the designed order under both passes
(pass A: 3.79 / 3.50 / 2.96 / 2.88 / 2.79, spread 1.00), and 82.6% of comparable
answer pairs agree in direction between construction intent and blind label
(123 concordant, 26 discordant, 91 tied of 240 within-question pairs). The
construction does produce a real quality gradient — a test asserts both the
monotone ordering and a minimum spread.

**The `excellent`/`good` boundary is thinner than it looks.** Pass A separates
those two levels by 0.29 of a point, against the retired pass's 0.71, because
pass A found real defects in four `excellent`-intent rows and marked them down:
`eval-0046` (a Dijkstra answer whose worked example does not demonstrate the
failure it narrates) scored **2**, and `eval-0011`, `eval-0016` and `eval-0116`
scored 3 — one for an internal contradiction about whether images belong in
object storage or the database, one for a garbled transaction interleaving, one
for a reason stated in the wrong causal direction. Whether those markdowns are
right is a judgement call; that they exist means **`quality_intent ==
"excellent"` is not a safe proxy for "correct answer"** any more than
`wrong_confident` is for "contains errors".

**The bad news, and it is worse than the good news is good.** Two measured
defects that bound what Day 19 can conclude:

#### 1. `wrong_confident` largely did not comply — 19 of 24

The brief asked for *"at least two clear factual errors … claims that would lead
to a broken system if acted on"*. 19 of 24 answers at that level read as **solid**
(3 or 4) to a blind labeller — 18 under the retired pass — because the generator
wrote correct prose instead. Spot-checked and confirmed as a generator failure,
not a labelling error: `eval-0105`, `eval-0085` and `eval-0025` are all
factually correct.

Only **five rows at that level complied**, and both passes agree on all five:

| Row | Pass A | Retired | What it actually gets wrong |
|---|:--:|:--:|---|
| `eval-0070` | **0** | 0 | Matches brackets with a **queue**, complete with FIFO justification |
| `eval-0045` | **1** | 2 | Asserts `(lo + hi) / 2` "works fine because the sum of two indices rarely overflows", citing the "elegance of the single expression" — the exact misconception the question probes |
| `eval-0020` | 2 | 2 | — |
| `eval-0090` | 2 | 2 | — |
| `eval-0120` | 2 | 2 | — |

> **Day 19 must not treat `quality_intent == "wrong_confident"` as "answers
> containing errors".** It is wrong about four fifths of that slice.
> `unachieved_intent()` returns the non-compliant ids.

#### 2. The label distribution is compressed — 84% at 3–4

| Label | 0 | 1 | 2 | 3 | 4 | 3–4 share |
|---|:--:|:--:|:--:|:--:|:--:|:--:|
| Pass A (current) | 1 | 1 | 17 | 57 | 44 | **84.2%** |
| Retired pass | 1 | 0 | 15 | 68 | 36 | 86.7% |

Two rows of 120 sit at 0 or 1. The effective range of this set is **2–4, not
0–4**, under either pass. A quadratically weighted agreement statistic needs
spread to be informative, so **a QWK over this distribution will be weak and
unstable regardless of how good the grader is**, and a high value would be as
suspicious as a low one. That is a fact about the dataset, not about any grader
measured on it.

A second model pass did **not** fix this, which is the useful finding: the
compression is a property of the *answers*, not of whoever read them. Across
both passes — 240 labels over 120 rows — exactly **two rows** were ever scored
below 2 by either pass: `eval-0070` (0 in both) and `eval-0045` (1 in pass A, 2
in the retired pass).

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
| `model_score` filled | **120 / 120** — the retired first pass |
| Human labels | **0** — `gate_eligible()` is `False` |

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

- **Both label passes are models of the same family, and the second was not
  fully blind.** `claude-opus-5` and `claude-opus-5[1m]` are not two opinions.
  Worse, the pass-A labeller had **read this README before labelling**, which
  disclosed the retired pass's aggregate distribution (`0:1, 1:0, 2:15, 3:68,
  4:36`), its per-level means, and the specific properties of six individual
  rows. Pass A was applied from the blind worklist and per-row retired scores
  were never consulted, but **the anchoring is real and the 75% exact agreement
  is inflated by an unknown amount because of it.** A genuinely blind third pass
  would be worth more than this one.

  The six disclosed rows, which should be treated as **contaminated**:

  | Row | What the README disclosed | Pass A | Retired |
  |---|---|:--:|:--:|
  | `eval-0070` | "confidently uses a queue to match brackets", "landed at 0" | 0 | 0 |
  | `eval-0025` | "factually correct" | 3 | 3 |
  | `eval-0085` | "factually correct" | 3 | 3 |
  | `eval-0105` | "factually correct" | 3 | 3 |
  | `eval-0018` | contains "but I'm not supposed to go into that" | 2 | 2 |
  | `eval-0019` | names `repeatable_read` only to say it "kind of blurs" | 2 | 2 |

  Pass A agrees with the retired pass on all six. `eval-0070` is the sharpest
  case: its documented score of 0 was known in advance. The queue/FIFO error is
  unmistakable on reading the answer, so the label is very likely right anyway —
  but "very likely right anyway" is not the same as independently derived, and
  these six rows cannot be used as evidence of inter-pass agreement.

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