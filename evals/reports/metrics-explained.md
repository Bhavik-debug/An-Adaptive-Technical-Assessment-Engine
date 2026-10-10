# The grading metrics, explained for someone new to evaluation

This file explains what each Day-19 number *means*, with no statistics
background assumed. The numbers themselves are in
[`grading_validity.md`](grading_validity.md); the labels they are computed
against are described in [`reference_labels.md`](reference_labels.md).

Read this first if "QWK 0.43" does not immediately tell you anything. That is
the normal reaction and the number is genuinely hard to interpret cold.

---

## The setup, in one picture

```
            an interview question
                     |
          a candidate's answer to it
                     |
        +------------+------------+
        |                         |
   THE GRADER                THE REFERENCE
   (what we built)           (what we compare it to)
        |                         |
  one LLM call that           one 0-4 label
  classifies concepts,        per answer
  then CODE computes
  a score in [0, 1]
        |                         |
  0.62  --> band 3           band 3
        |                         |
        +------------+------------+
                     |
          do these two agree?
          that is what every metric below measures
```

Two details that matter for reading any of it:

**The grader never outputs a score.** It returns classifications — *is this
concept covered, partially covered, or absent* — plus four rubric levels and a
list of errors. `app/grading/scoring.py` turns those into a number. That is
deliberate: a model asked for "7/10" produces clustered, unstable, arguable
numbers, and a model asked "is this concept present, quote the span" produces a
classification you can verify. The scoring arithmetic is ordinary code with
tests around it.

**The two sides live on different scales.** The grader gives a continuous number
between 0 and 1; the reference gives a whole number from 0 to 4. To compare
them, the grader's score is put into one of five bands
(`validity.score_to_band`): 0.62 becomes band 3. The band edges are evenly
spaced — 0.125, 0.375, 0.625, 0.875 — which is the choice that assumes nothing
about either distribution. A different binning choice would produce a different
QWK, which is exactly why it is a named, tested function rather than a `round()`
buried in a loop.

---

## MAE — mean absolute error

**Plain meaning:** on average, how many bands apart are the two scores?

Take each answer, subtract the two scores, throw away the minus sign, and
average. An MAE of 0.5 means "typically half a band apart". An MAE of 0 means
they always agree exactly.

**Why it is the easiest number here:** it is in the units you already
understand. If someone asks "how far off is the grader?", this is the answer.

**Its weakness:** it gives no credit for difficulty and no correction for luck.
If the reference says band 3 for almost every answer, a grader that simply
always said band 3 would score a near-perfect MAE while knowing nothing at all.
That is not a hypothetical on this dataset — see the central-tendency finding in
the label report. MAE alone can therefore look good for a bad reason, which is
why it is never reported alone.

---

## Mean signed error

**Plain meaning:** the same subtraction, but keeping the minus sign.

This separates two very different problems that MAE lumps together:

| | mean signed error | MAE |
|---|---|---|
| Grader is always 1 band too generous | +1.0 | 1.0 |
| Grader is 1 band too generous half the time and 1 too harsh the other half | 0.0 | 1.0 |

The first is a **calibration bias** — a systematic offset, and the kind of
problem you can correct by moving a threshold. The second is **noise**, and no
threshold fixes it. The MAE is identical in both cases. Reporting both numbers
is what tells them apart.

---

## Spearman's rho — rank correlation

**Plain meaning:** does the grader put the answers in the *same order* as the
reference, even if it disagrees about the absolute scores?

Both sides are converted into rankings — best, second best, and so on — and the
two rankings are compared. The result runs from +1 to −1:

- **+1** — identical ordering. Whenever the reference ranks one answer above
  another, so does the grader.
- **0** — the orderings are unrelated.
- **−1** — exactly reversed.

**Why it is worth having alongside the others:** for an interview product,
ordering is often what actually matters. A grader that is uniformly a point
harsh than the reference but ranks every candidate identically would be
perfectly usable after a one-line recalibration, and Spearman is the metric
that notices that while MAE does not.

**Watch out for ties.** A 0–4 scale over 120 answers produces enormous numbers
of ties — many answers share a band. The textbook shortcut formula for Spearman
is only valid without ties, so this implementation computes it the long way,
with ties sharing an averaged rank. On this dataset the shortcut would be badly
wrong.

---

## QWK — quadratically weighted kappa

The headline metric in automated-scoring work, and the one that needs the most
explanation.

**The problem it solves.** Suppose the reference labels 80% of answers as band
3. A grader that said "band 3" every single time would agree with the reference
80% of the time. Raw agreement would call that excellent. It is worthless.

QWK fixes this by asking a better question: **how much better than lucky
guessing is this?**

**How it is built, in three steps.**

1. **Count the disagreements, weighting big ones more.** Being one band out
   counts 1; two bands out counts 4; three counts 9; four counts 16. The
   squaring is the "quadratic" in the name, and it encodes something true about
   grading: mistaking a 3 for a 4 is a quibble, mistaking a 0 for a 4 is a
   catastrophe, and these should not cost the same.

2. **Work out how much disagreement you would expect by chance** — from the two
   sides' own distributions, assuming they were shuffled independently of each
   other. A reference that says 3 everywhere has very little expected
   disagreement, so this baseline is small.

3. **Divide and subtract from 1.**

```
QWK = 1 - (weighted disagreement actually observed)
          / (weighted disagreement expected by chance)
```

**How to read the result:**

| QWK | Meaning |
|---|---|
| 1.0 | Perfect agreement |
| 0.8+ | Very strong; typical of two careful humans on a well-designed rubric |
| 0.6-0.8 | Substantial. The project's gate is 0.65 |
| 0.4-0.6 | Moderate. Usable as a signal, not as a verdict |
| 0.2-0.4 | Weak |
| 0.0 | No better than guessing with the same distribution |
| negative | Worse than chance — something is wrong with the setup |

**The trap, and it is the single most important thing on this page.** QWK's
denominator is "disagreement expected by chance". When the reference puts nearly
everything in one band, that denominator is *tiny* — and dividing by a tiny
number makes the statistic jump around wildly. A couple of answers moving one
band can swing the QWK by a large amount.

So on a compressed dataset **a low QWK does not prove the grader is bad, and a
high QWK does not prove it is good.** You must look at the reference's
distribution before reading its kappa. That is why every QWK in the Day-19
report is printed with the reference's band histogram and modal share beside it,
and why the implementation returns `0.0` rather than `1.0` when both sides are
constant: in that case the statistic is undefined, and returning 1.0 would
report "two raters who both said 3 to everything" as perfect agreement.

**One more honest caveat.** QWK was designed for two *independent* raters. On
this dataset the grader and the Day-19 reference labeller are the same
underlying model, prompted differently. Agreement between them is partly
agreement with itself, and no statistic can separate the two.

---

## Test-retest reliability (sigma)

**Plain meaning:** grade the same answer five separate times. Do you get the
same score?

For each answer, the five scores have a spread, measured as a standard
deviation — informally, the typical distance of a run from the average of the
runs. Average that across answers and you get the reported sigma. The project's
target is **sigma < 0.04** in score units (recall the score runs 0 to 1, so 0.04
is 4% of the whole scale).

**Why it matters more than it sounds:** this has nothing to do with whether the
grader is *right*. It is about whether it is *consistent*. A grader that gives
the same answer 0.4 today and 0.8 tomorrow cannot be used at all, however good
its average is — two candidates who gave identical answers would get different
results, which is indefensible in an assessment product.

**What makes it non-zero here.** The grader runs at temperature 0, which should
make it deterministic. But the **confidence gate** (plan §7.3) deliberately
re-grades uncertain answers three times at temperature 0.7 — a higher
temperature means more randomness — and takes the median. So gated answers are
*designed* to vary between runs. The report therefore counts how many of the
measured items were gated, because "the grader is unstable" and "the gate did
its job on 1 answer in 5" are completely different conclusions from the same
sigma.

The report also gives the **worst single item**, not just the mean: a mean sigma
of 0.01 with one answer at 0.30 is not a stable grader, and the mean alone would
hide it.

---

## The four bias probes

A bias probe changes *one* thing about an answer that should not matter, and
checks the score does not move. Each is a claim you can state in one sentence.

### Length bias — `|delta| < 0.05`

Append roughly 200 words of waffle that says nothing technical, then re-grade.
The score should barely move.

**Why this specific test:** padding is the easiest way to game a grader that
rewards effort-shaped text. If 200 words of nothing raises the score, the system
rewards talking rather than knowing.

**What makes it valid or invalid:** the padding must be genuinely empty. If the
filler happens to mention a concept the question expects, the grader may
*correctly* raise its score, the probe fails, and the failure gets written up as
length bias when it is nothing of the kind. So the filler is a committed
constant checked against all 220 concept keys and 179 anchor terms in the
question bank, and re-checked per question at run time. An earlier draft of the
filler said "whether to cache" and had to be rewritten, because `cache` is a
concept key.

### Confidence bias — `|delta| < 0.03`

Prepend *"I'm certain that the following is correct…"* and re-grade. Nothing
about the technical content changed, so nothing about the score should.

**Why it matters here in particular:** Day 18 found that this dataset's
`wrong_confident` answers were mostly *correct* — the generator writes
confidently by default. A grader that paid for assertiveness would be
systematically wrong on exactly the slice built to test it.

### Jargon bias — score should **drop**

Replace precise terms with vaguer fashionable ones: *write-ahead log* becomes
*best-practice durability layer*, *exponential backoff* becomes *intelligent
retry strategy*. The replacement says strictly less than the original.

This is the one probe where the expected result is a **change**. The answer now
demonstrates fewer mechanisms, so a grader driven by concept coverage should
score it lower. A grader that scored it the same or higher would be responding
to how impressive the prose sounds.

Rows where no substitution matched are **excluded**, not counted as passes — an
unchanged variant would pass the test vacuously.

### Position bias — must be **exactly 0**

Shuffle the order of the expected-concept list and re-score. The score must not
change at all — not approximately, exactly.

**This probe is different in kind from the other three.** It is not testing the
model; it is testing the architecture. Scoring happens in code that looks
concepts up by name and sums over a mapping, and every term is a weight times
one of 1.0, 0.5 or 0.0. Those are exact binary fractions, so the sum is
order-independent in floating-point arithmetic as well as in theory. The result
is therefore guaranteed by how the system is built, and the probe's job is to
prove that the guarantee is real.

It also needs **no model calls at all**: it re-scores a stored judgment with the
list shuffled eight ways. Re-asking the model would measure the model's own
randomness instead, and could never return exactly zero — turning a passing
architectural guarantee into a failing empirical one.

---

## Why "provisional" appears in front of every QWK

The project's Phase-4 exit gate is **QWK ≥ 0.65 against human labels**.

There are no human labels. Every label this project has — all three passes — was
produced by a language model. Those are useful for building and debugging the
measurement harness, and they are not what the gate asks for.

The distinction is enforced in code rather than remembered in prose:
`evalset.gate_eligible()` returns `True` only when all 120 rows carry
`label_source: "human"`, a test pins it to `False`, and another test asserts
that **two model passes do not add up to a human one**. A QWK over model labels
reads identically to a QWK over human ones — same formula, same range, same
plausible-looking number — which is precisely why the difference has to be
machine-checked.
