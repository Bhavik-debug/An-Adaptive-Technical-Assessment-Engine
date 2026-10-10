# Grading — how the grader is measured, and what the measurements are worth

*Phase 4, Days 16–20 and the remediation diagnostic. Written for someone who has
never studied measurement. It is the companion to
[`docs/evaluation.md`](evaluation.md), which does the same job for retrieval.*

---

## 1. Why a grader needs its own evaluation

The grader takes a question, a candidate's answer, and a list of concepts the
answer ought to demonstrate, and returns a number between 0 and 1. The tests
prove the number is *computed* correctly — the arithmetic is pure, the schema is
enforced, the same judgment always produces the same score.

None of that says the number is **right**.

That is a different kind of question, and it needs a different kind of answer.
"The tests pass" tells you the machine works. It does not tell you whether a
good answer scores higher than a bad one, whether the same answer scores the same
twice, or whether a candidate can raise their score by padding.

So Phase 4 built a measurement apparatus around the grader: a 120-answer
evaluation set, reference labels, a correlation statistic, and four probes that
try to move the score without changing what the answer actually says. The plan
sets numeric gates on all of them.

**The honest summary, as of the diagnostic pass: the apparatus works well enough
to have found both its subject and itself failing.** Two gates pass, one cannot
be measured at all, and three fail — two of them for reasons that turned out to
be defects in the measurement rather than in the grader.

---

## 2. The three layers, and why each can be wrong independently

```
   the answer         ->  the model's judgment  ->  verification  ->  the score
   (dataset)              (concepts + rubric)       (span checks)     (arithmetic)
```

Each arrow can introduce an error, and the whole point of persisting intermediate
results is to be able to say *which arrow*.

| Layer | Deterministic? | Can it be wrong? |
|---|---|---|
| The dataset | yes | **Yes — see §5.** Its "ground truth" is largely false. |
| The model's judgment | **no** | Yes, and it varies run to run on identical input. |
| Verification | yes | **Yes — see §4.** It removes credit that was earned. |
| The arithmetic | yes | No. It reproduces bit-exactly. |

The scoring arithmetic is the only layer that has never been wrong. Every
Phase-4 failure lives in one of the other three.

---

## 3. What "measuring bias" means, and the trap in it

A bias probe asks a simple question: **if I change the answer in a way that adds
no technical content, does the score move?** It should not. If it does, the
grader is rewarding something other than correctness, and a candidate who learns
what will get a score they did not earn.

Four probes exist:

| Probe | The change | Should the score move? |
|---|---|---|
| Length | 207 words of deliberately contentless filler | no |
| Confidence | a confident-sounding prefix | no |
| Jargon | technical words swapped for buzzwords | no |
| Position | the expected-concept list shuffled | **no, exactly zero** |

The position probe is different from the other three, and the difference is
instructive. It does **not** re-call the model: it re-scores the stored judgment
with the concept list shuffled eight ways. That is deliberate, and
`position_invariance` says why — scoring indexes by concept key and sums over a
mapping, so it is order-free in IEEE-754 as well as in exact arithmetic, whereas
re-asking the model would measure the model's nondeterminism and could never
return exactly zero.

So it measures 0.000000 on all 120 rows and it always will. **It is a real gate
and it passes — but understand what it certifies:** that the *architecture* puts
scoring in code rather than in the model. It says nothing about whether the model
would judge a shuffled list the same way, and after §3 you should assume it would
not.

### The trap

The other three probes were specified as: *for each answer, grade it once
normally, grade it once with the change, take the difference, and report the
largest difference you see.*

Each of those words is reasonable. Together they are a trap, because the middle
layer is **not deterministic**. Grade the same answer twice, with no change at
all, and the score moves:

```
eval-0031, same text, same prompt, five runs:
  [0.7825, 0.8044, 0.9781, 0.8481, 0.8044]     range = 0.1956
```

The length gate asks for a maximum change below **0.05**. The grader moves by
**0.1956** when nothing changes. So "the largest difference over 24 answers" is
mostly a measurement of how noisy the grader is — an *order statistic* of the
noise, not a measurement of bias.

The numbers make it unambiguous:

| Condition | text change | inflates the score by more than +0.05 |
|---|---|--:|
| **repeat run, identical input** | **none** | **21.4%** |
| contentless filler | semantically none | 20.8% |
| confidence prefix | semantically none | 37.5% |
| jargon substitution | semantically none | 15.0% |
| **crafted prompt-injection payload** | adversarial | **21.9%** |

A crafted attack is **no more likely** to raise the score than running the same
grader twice on the same text. And when the two rows that set the headline
maxima were re-sampled four times per condition in a single sitting, the
length-bias maximum fell from −0.2385 to a mean of −0.0375, and the
confidence-bias maximum **changed sign** from +0.3044 to a mean of −0.1138.

### The lesson, stated generally

> **A paired-difference metric is only meaningful relative to the variance of
> the thing being paired.** Before reporting any such metric, run a no-change
> arm and publish it as the noise floor. If your effect is not larger than your
> floor, you have not measured an effect.

This does **not** mean there is no bias. Mean |Δ| on the filler probe is 0.0723
against a repeat-run floor of 0.0372 — about twice the floor, consistent with a
real but modest length effect hiding underneath. **Neither bias gate is reported
as passing.** What the diagnostic established is that their reported magnitudes
are not trustworthy, and the confidence probe's reported *direction* is not
either.

The same reasoning applies to the security metric. Attack success rate is a
paired difference too, so a 21.9% ASR against a 21.4% noise floor says almost
nothing about attacks. One exception survives: two `delimiter_escape` cases
reproduce across both prompt arms with identical deltas, and they have a real
mechanism — the grader is shown the literal marker `[removed-delimiter]` where a
smuggled fence sentinel used to be, and the active prompt never explains what it
means.

---

## 4. Why the grader is so noisy: a verifier that is right about the rule and wrong about the case

The grader must quote the candidate's own words as evidence for every concept it
credits. `verify_evidence` checks that the quote is actually in the answer and
removes the credit if it is not. The idea is sound and it is the strongest
defence in the pipeline — it is why no prompt-injection payload has ever had a
concept credited to it.

**But it removes credit from 15.4% of the concepts the grader claims** — 70 of
454 on the base pass, affecting 55 of 120 answers. It records each one as
`FABRICATED`.

Seven were captured live and inspected. **None is a fabrication.**

| What the model offered as its quote | What it actually is |
|---|---|
| `push openers, pop and compare on each closer` | a précis, in the model's own words |
| `one pass over the characters` | a light paraphrase |
| `you read the string from left to right ... when you hit a closing bracket you pop the top ...` | **two genuinely verbatim excerpts, joined with an ellipsis** |

The third case is the clearest defect. The model quoted two real,
non-contiguous passages and joined them with `...`. The first 104 characters
match the answer exactly. The substring test fails **only on the join** — and the
candidate loses the concept.

Splitting a candidate span on `...` and requiring every segment to verify would
fix that case **with no loss of strictness whatsoever**: each segment still has
to be an exact quotation, so an invented segment still fails. (Not yet
implemented — the diagnostic pass was diagnosis only.)

### Why one quote is worth so much

```
score = 0.65 x concept_score + 0.35 x rubric_score - penalty
```

Concepts carry weights of 2–3 out of a question total of 9. Dropping one heavy
concept from `covered` to `absent` removes `3/9 × 0.65 = 0.2167`.

> **One unverifiable quote is worth 4.3 times the entire length-bias tolerance.**

And because the model paraphrases non-deterministically, that 0.2167 step lands
or does not land at random. That is the noise in §3, and it is why three separate
metrics looked broken.

---

## 5. The dataset's ground truth is mostly false

This defect is completely independent of the previous two, and it bounds the
primary gate.

The evaluation set was built deliberately: for each question, code chose which
concepts an answer should demonstrate (`concepts_included`) and which it should
stay silent about (`concepts_omitted`), then asked a model to write an answer to
that specification at one of five quality levels. Because the split was decided
*by code, before generation*, it is the only ground truth in the dataset that no
model produced.

It does not hold up:

```
included concepts graded covered : 245/288 = 85.1%
omitted  concepts graded absent  :  58/192 = 30.2%
```

134 of 192 supposedly-omitted concepts were credited. The decisive check is
whether those credits rest on real text — and **all 134 do**, on spans that
verify verbatim against the answer. The grader is not imagining coverage. **The
generator ignored the instruction**, and the answers genuinely explain the
concepts the dataset records as absent:

```
eval-0010, quality_intent = wrong_confident, deadlock_detection recorded as OMITTED:
  "The database engine builds a wait-for graph to spot the cycle, and then it
   picks one transaction to roll back as the victim so the other can finish."
```

The leakage detector that was supposed to catch this looks for the concept's own
key terms, so it flags **21 of 134 — 16%**. A correct paragraph that happens not
to reuse the keywords is invisible to it.

### The consequence: five quality levels that are not five quality levels

```
intent               n  ref mean  grader mean
excellent           24      2.92        0.808
good                24      3.00        0.811
partial             24      2.71        0.705
superficial         24      2.79        0.689
wrong_confident     24      2.67        0.682
```

`good` outscores `excellent` on both the reference labels and the grader. The
total spread across five deliberately distinct levels is **0.129** — smaller
than the 0.1956 range the same grader produces on identical input.

### The consequence for the headline statistic

Quadratic weighted kappa measures agreement *beyond chance*, and it computes
"chance" from how the two raters distribute their labels. The reference labels do
not distribute:

```
distribution: {0: 1, 1: 2, 2: 17, 3: 98, 4: 2}     81.7% in a single band
```

With a marginal that degenerate the expected-disagreement denominator is tiny and
the statistic is weakly identified. The committed `0.4252` is dominated by how
the grader spreads 98 band-3 answers across bands 2, 3 and 4.

> **This bounds the QWK gate independently of the human-label problem.**
> Collecting 120 human labels against a dataset whose intended quality levels do
> not separate would produce a number nearly as hard to interpret as this one.

---

## 6. Circularity: one model is playing every role

`NvidiaProvider.model_for()` returns `nemotron-3.5-lightning-30b-a3b` for every
tier. So the model that **wrote** the answers, the model that **grades** them, and
the model that produced the **reference labels** are all the same model.

A grader agreeing with itself is not evidence that it agrees with a person. The
measured ordering is exactly what circularity predicts: QWK is highest against
the same-model reference (0.4252) and lowest against the reference produced by a
different model family (0.2970).

Every number in this document is bounded by this, and the only fix is a second
model family.

---

## 7. No label in this dataset is human-generated

Stated plainly because it is the single most important caveat in Phase 4, and
because it would be easy to lose in the arithmetic.

- `answers.jsonl` carries 120 labels. All 120 have `label_source = "model"`.
  **Zero have `label_source = "human"`.**
- `reference_labels.jsonl` carries 120 provisional reference labels. The
  `ReferenceLabel` type makes `label_source` a `Literal["model"]`, so it is not
  possible to record one as human.
- `gate_eligible()` requires 120 rows with `label_source == "human"` and returns
  **`False`**.
- The committed QWK baseline records `reference_label_source: "model"` and
  `is_acceptance_evidence` is permanently `False` for it.

The plan's primary Phase-4 gate is *QWK ≥ 0.65 against a human reference on 120
answers*. It is **not failed — it is unmeasured**, and nothing in the repository
is permitted to report it as anything else.

---

## 8. Gate status

| Criterion | Required | Measured | Status |
|---|---|--:|---|
| QWK vs **human** reference | ≥ 0.65 on 120 | no human labels exist | **Unverified — ineligible** |
| Length bias | max \|Δ\| < 0.05 | 0.2385 as stored; −0.0375 mean on replication | **FAIL — magnitude not established** |
| Confidence bias | max \|Δ\| < 0.03 | 0.3044 as stored; −0.1138 mean, opposite sign | **FAIL — direction not established** |
| Position/order bias | exactly 0 | **0.000000** on 120 rows × 8 shuffles, code-side | **PASS** |
| Injection ASR | 0% | **21.9%** (7/32), at a 21.4% noise floor | **FAIL** |
| CI regression gate | fail on > 0.05 | demonstrated both ways, 32 tests | **PASS** |

Secondary: jargon probe mean Δ −0.0562 (**PASS**); test–retest σ 0.0386 but over
**6 items, not the planned 40** — insufficient evidence, **not** a pass;
first-attempt schema-violation rate ~2.5% against a <0.5% target, 0% after retry.

---

## 9. Where the numbers live, and how to re-derive them

| Artefact | What it holds |
|---|---|
| `evals/datasets/grading/answers.jsonl` | the 120 answers, their construction split, and the model label pass |
| `evals/datasets/grading/reference_labels.jsonl` | the provisional model reference labels |
| `evals/reports/grading/runs.jsonl` | every grading run, with its **evidence-corrected** judgment |
| `evals/reports/grading/injection.jsonl` | the 64 injection rows (32 cases × 2 prompt arms) |
| `evals/baselines/grading_qwk.json` | the committed baseline the CI gate compares against |
| `evals/reports/phase4_diagnostic.md` | the diagnostic pass, with every command |

```bash
cd backend
python scripts/run_grading_validity.py      # the validity harness
python scripts/run_injection_suite.py       # the injection suite
python scripts/recheck_bias_probes.py       # re-derive the probes offline, no provider
python scripts/diagnose_phase4.py --offline # the diagnostic, no provider
```

### What is *not* recorded, and what that cost

Four fields are missing, and each one blocked part of the diagnosis:

1. **The raw judgment.** `runs.jsonl` stores the judgment *after* verification
   corrected it, so the model's original claim is gone.
2. **The offending span.** When verification downgrades a concept it sets
   `evidence = None`. §4 therefore required re-grading a row repeatedly until a
   span happened to fail, at provider cost, to see even one.
3. **The confidence gate's re-grade samples.** Only `gate_fired` is stored, not
   the three sample scores behind a gated median.
4. **The injection suite's judgments and failures.** Neither is persisted, so
   per-stage attribution for an attack is impossible after the fact.

> **A pipeline that cannot explain its own numbers after the fact will be
> diagnosed by sampling, at provider cost, every time.** Persisting the raw
> judgment alongside the corrected one is cheap; re-deriving it from the provider
> is not.
