# Convergence — how fast, and how reliably, does θ̂ reach the truth?

Day 13 asked **where the error ends**: one MAE per policy, and eight sampled
points of a curve printed as a text table. Day 14 asks **how it got there**.

That is not the same question, and in this experiment it does not have the same
answer. Day 13's headline was that random selection edges adaptive on the
all-subtopic MAE, 0.422 to 0.433. Day 14's curves show *why*, and the shape is
not the one that number suggests: **adaptive is ahead for the first eight
questions and behind from the ninth**. Nothing in the Day 13 numbers was wrong.
The endpoint was simply the last frame of a film nobody had watched.

Everything below is a statement about a **synthetic environment**. See
[`docs/simulation.md`](simulation.md), §2 "What this cannot tell us", which
applies to every number on this page without exception.

---

## 1. What Day 14 measures

Seven quantities, all read off the same runs Day 13 published:

| # | question | how it is answered |
|---|---|---|
| 1 | How wrong is the estimate after *n* questions? | mean and median MAE per question count, with the interquartile range |
| 2 | Is the remaining error spread or concentrated? | RMSE per question count, read beside MAE |
| 3 | What does one candidate's estimate actually do? | θ̂ against the hidden θ, per subtopic, per policy |
| 4 | How many candidates are close enough yet? | share reaching MAE ≤ 0.5, cumulative and instantaneous |
| 5 | Why do sessions end, and *when*? | stopping reason crossed with question number |
| 6 | How much of the blueprint has been touched? | subtopics measured per question count |
| 7 | Does the estimate settle or oscillate? | mean \|Δθ\| per question, reversal rate, MAE-increase rate |

**No new experiment was run.** Day 14 re-runs Day 13's `run_experiment` with the
same seed, the same environment, the same policies and the same production
weights. Day 13's committed report regenerates byte-identically after every
change in this phase — asserted by regeneration, not by assumption.

**No number here is fed back into the system.** No weight, threshold or
stopping constant was changed. `docs/simulation.md` §9 explains why the ablation's
tempting result was left alone; the same reasoning applies here.

---

## 2. How convergence is defined

**Both criteria are Day 13's, unchanged. Day 14 introduced no new threshold.**
That is a deliberate constraint: a chart drawn to a different definition than
the table beside it is a trap, and inventing a threshold that a policy happens
to clear is the easiest way to manufacture a result.

**MAE ≤ 0.5.** `ExperimentConfig.convergence_mae`. One sixth of the [−3, +3]
scale, about the width of one difficulty band. *An arbitrary design choice*, and
it is reported next to the full error curve precisely so a reader who dislikes
it can ignore the threshold and read the curve.

**worst-topic RD ≤ 0.35.** `ExperimentConfig.convergence_se`, plan §8.6's own
criterion, applied at **topic** level through Day 11's `roll_up` — the same
precision-weighted aggregation Day 12's stopping rule uses. A topic nobody asked
about counts as infinite, so "we have no idea" can never satisfy a precision
threshold.

**Reached, and still reached.** "Convergence at question *n*" reuses
`metrics.first_index_at_or_below`, evaluated at every *n* rather than once. That
makes it **cumulative**: a session counts as converged from the first moment it
touched the threshold, which is exactly what `SessionResult.items_to_mae` means,
so the right-hand end of the chart equals Day 13's `reached` column.

But MAE is **not monotone** — an estimate can cross 0.5 and cross back — so
`at_or_below_curve` reports the *instantaneous* share as well, and the gap
between the two at the final question is a direct measurement of how much
convergence gets undone. Reporting only the cumulative curve would hide that;
reporting only the instantaneous one would disagree with Day 13. Both are
printed.

**Censoring is preserved.** A session that never meets a criterion is counted,
not dropped. Averaging over only the sessions that converged is the classic way
to make a policy that rarely converges look fast.

---

## 3. How trajectories are generated

### The record

Day 13's session loop now appends one `TrajectoryStep` per turn, plus one for
the cold start:

```
TrajectoryStep
    items_asked          0 for the cold start
    item_id / subtopic / topic / score / delta_theta    None at the cold start
    theta[subtopic]      what the engine believed, for EVERY blueprint subtopic
    rd[subtopic]
    subtopics_measured
    mae, rmse, worst_topic_rd
```

Three properties make this an observation rather than an intervention:

1. **It is written after the update, from state the policy already had.** No
   new arithmetic enters the loop.
2. **It draws no randomness.** Neither the response stream nor the ε-greedy
   stream is touched, so the sequence of questions is unchanged.
3. **No policy can read it.** A strategy's arguments are still
   `(state, bank, rng)`, and `SelectionState` still holds no reference to a
   candidate or to a trajectory.

The check that all three hold is not an argument, it is a regeneration: Day 13's
`evals/reports/selection_simulation.md` is **byte-identical** before and after.

`mae_trajectory` and `rd_trajectory`, which Day 13 stored as separate lists, are
now derived views onto this one record — so there is exactly one place a
per-turn number can come from, and the chart and the table cannot drift.

### Ground truth, and where it may be used

Ground truth enters a `TrajectoryStep` only as an *error* — `mae`, `rmse` — and
only because the step is written by the simulator, after the policy has already
chosen. The rule from Day 13 is unchanged and unweakened:

> `SyntheticCandidate.true_theta` is read by `response.graded_score`, whose only
> output is a score in [0, 1]. Everything else that touches it is evaluation.

Day 14 adds a second reader — `runner._record_step` — and it is evaluation. Its
output goes into a `SessionResult`, which is constructed after the session has
ended.

### Averaging across candidates

Sessions have different lengths (Day 13: adaptive 18.9 questions, random 19.0,
fixed 16.6). Every curve **forward-fills** a short session at its last value
rather than truncating it, which is Day 13's own rule and matters more here than
it did there: truncation would drop the early-stopping sessions from the tail of
every average, biasing every curve towards whichever policy runs longest. A
stopped session's estimate does not move, so repeating it is the honest reading.

**The spread is shown, not just the mean.** Day 13 measured the standard
deviation across candidates at 0.21–0.25 while the gap between policies is
0.07 — three times smaller. A chart of means alone over a difference that size
would be misleading by omission, so Figure 2 exists.

---

## 4. What each figure means

All ten live in [`evals/reports/convergence/`](../evals/reports/convergence),
embedded in [`evals/reports/convergence.md`](../evals/reports/convergence.md)
with the numbers they were drawn from. Regenerate; do not edit.

| figure | what it shows | how to read it |
|---|---|---|
| `mae-vs-questions` | mean MAE after *n* questions, three policies | The headline. Over **every** blueprint subtopic, unmeasured ones at the prior — the number a report would print. |
| `mae-spread` | median MAE with the interquartile range, one panel per policy | Small multiples, shared scale. The band is where the middle half of the 200 candidates sit — a population spread, **not** a confidence interval, and far wider than the gaps between policies. |
| `rmse-vs-questions` | mean RMSE after *n* questions | Read beside MAE. RMSE pulling away from MAE means the error is concentrating in a few subtopics rather than spreading thin. |
| `convergence-reach` | share of candidates that have reached MAE ≤ 0.5 | Cumulative. The table beneath gives the instantaneous reading and the gap. |
| `coverage-vs-questions` | mean subtopics with at least one question | The other half of the MAE story: an untouched subtopic contributes its whole prior error. |
| `stability` | mean \|Δθ\| on the answer at each position | A falling curve means the estimate is settling. The dashed line is Day 12's own `SMALL_DELTA_THETA` = 0.05 — drawn, not tuned. |
| `stopping-by-question` | stopping reason crossed with session length | Stacked columns, one panel per policy. Row totals equal Day 13's stopping table. |
| `trajectory-c175` / `-c059` / `-c089` | θ̂ against hidden θ, one panel per subtopic | Three candidates at the 10th, 50th and 90th percentile of the **adaptive** policy's final-MAE ordering. The dashed muted line is ground truth and is never a series colour. A flat stretch is a run of turns spent elsewhere; a panel where every line sits at 0 is a subtopic nobody asked about. |

**Two figures were considered and deliberately not drawn**, because their
absence otherwise looks like an oversight:

- *A curve of worst-topic RD.* It is `inf` for most of a short session and never
  crosses any threshold within 20 questions, so the chart would be an
  unplottable infinity followed by a flat line. Reported as a number instead.
- *A scatter of coverage against final MAE.* Subtopics-measured is an integer
  from 1 to 6, so 600 points in six vertical stripes is a strip plot pretending
  to be a scatter. The binned table says the same thing legibly.

### Why the charts are hand-written SVG

`app/simulation/charts.py` emits SVG as text. No plotting library, for three
reasons: the brief requires two runs to produce the same plots, and a PNG stamps
metadata and renders through whatever FreeType happens to be installed; CI
installs `./backend[dev]` and nothing else, so matplotlib would add ~50 MB of
dependency to draw axes and polylines; and the charts are lines, a shaded band
and stacked columns, which is an amount of geometry that can be asserted on
directly. The output is byte-identical across runs and legible in `git diff`.

Colour follows the project's data-visualisation guidance: a fixed slot order so
a policy keeps its hue in every figure, the three colour-vision-validated slots
for the three policies, a muted dashed reference channel that can never be
mistaken for a series, text in ink rather than in the data colour, a legend
whenever there are two or more series, and end labels dropped rather than nudged
when they would collide. Both light and dark palettes are declared; every figure
also ships its numbers as a table, which is the reading that does not depend on
colour at all.

---

## 5. Results

200 candidates, 20-question budget, seed 20260101 — the same run Day 13
published.

### MAE against questions asked

| questions | adaptive | random | fixed | adaptive − random |
|---:|---|---|---|---|
| 0 | 0.757 | 0.757 | 0.757 | +0.000 |
| 3 | **0.684** | 0.692 | 0.730 | −0.008 |
| 6 | **0.625** | 0.628 | 0.689 | −0.003 |
| 8 | **0.589** | 0.591 | 0.651 | −0.002 |
| 9 | 0.574 | **0.572** | 0.643 | +0.002 |
| 12 | 0.535 | **0.521** | 0.585 | +0.014 |
| 17 | 0.475 | **0.453** | 0.516 | +0.022 |
| 20 | 0.433 | **0.422** | 0.495 | +0.011 |

**The crossover is at question 9.** Adaptive is ahead on all-subtopic MAE for the
first eight questions of the interview and behind from the ninth onwards. The
gap in its favour peaks at question 3 (−0.008) and against it at question 17
(+0.022), then narrows slightly as both approach the budget.

### Coverage against questions asked

| questions | adaptive | random | fixed |
|---:|---|---|---|
| 3 | 2.43 | 2.48 | 3.00 |
| 6 | 3.33 | 3.97 | 4.70 |
| 9 | 3.80 | 4.84 | 4.70 |
| 12 | 4.27 | 5.38 | 4.70 |
| 15 | 4.58 | 5.65 | 5.50 |
| 20 | 4.71 | 5.84 | 5.50 |

Adaptive and random are level for three questions and then separate steadily; by
question 8 — the last one adaptive wins — random has already touched one whole
subtopic more (4.63 against 3.63), and the gap keeps widening to 1.13 subtopics
at the budget.

The fixed policy's curve is a **staircase**: 4.70 from question 5 through 13,
then a step to 5.50 at question 14. That is a structural consequence of its
proportional interleaving against quotas of {8, 7, 5} — the fifth and sixth
subtopics simply are not on the front of the paper.

### Convergence

| policy | items to MAE ≤ 0.5 | reached | half of the population by | ever reached | still ≤ 0.5 at the end | undone |
|---|---|---|---|---|---|---|
| adaptive | 6.4 | 149/200 | question 9 | 74.5% | 72.0% | −2.5% |
| random | 6.1 | 152/200 | question 8 | 76.0% | 75.0% | −1.0% |
| fixed | 6.1 | 126/200 | question 13 | 63.0% | 62.0% | −1.0% |

`items to SE ≤ 0.35`: **0 of 200 for every policy**, exactly as Day 13 reported.
No RD-based criterion fires within a 20-question budget, so no figure was drawn
for it.

### Spread across candidates, at the budget

| policy | p25 | median | p75 | max |
|---|---|---|---|---|
| adaptive | 0.275 | 0.382 | 0.520 | 1.397 |
| random | 0.277 | 0.370 | 0.498 | 1.411 |
| fixed | 0.328 | 0.436 | 0.614 | 2.159 |

The interquartile range is roughly 0.24 wide for each policy. The mean gap
between adaptive and random is 0.011. **The difference between the two policies
is about one twentieth of the difference between two candidates**, which is the
single most important thing to hold in mind when reading any of the curves
above.

### Stability

| policy | mean \|Δθ\| | last 5 answers | reversal rate | turns where MAE rose |
|---|---|---|---|---|
| adaptive | 0.155 | 0.138 | 36.3% | 26.8% |
| random | 0.153 | 0.124 | 34.2% | 25.7% |
| fixed | 0.131 | 0.082 | 32.5% | 24.5% |

Per-question movement, mean \|Δθ\|:

| question | 1 | 5 | 10 | 20 |
|---|---|---|---|---|
| adaptive | 0.187 | 0.182 | 0.131 | 0.117 |
| random | 0.172 | 0.174 | 0.161 | 0.097 |
| fixed | 0.062 | 0.121 | 0.139 | 0.052 |

### Stopping

| policy | item budget | mean questions | no new information | mean questions |
|---|---|---|---|---|
| adaptive | 172 | 20.0 | 28 | 11.9 |
| random | 174 | 20.0 | 26 | 12.3 |
| fixed | 139 | 20.0 | 61 | 9.0 |

Crossing the reason with the question number is what makes this table say
something Day 13's three totals could not — see §9.

---

## 6. Adaptive vs random vs fixed

**Read the two comparisons differently. This has not changed since Day 13 and
Day 14 does not weaken it.** `random` sees the *same* eligible pool as
`adaptive`, including the θ-dependent `|b − θ| ≤ 1.5` window, so that contrast
isolates the ranking objective. `fixed` cannot apply that window — it reads the
running estimate, and a policy consulting it would not be fixed — so the
adaptive-vs-fixed contrast bundles the window with the objective and ε-greedy,
and **cannot attribute a gap to any single mechanism**.

### Adaptive against random — a lead that is spent, not absent

Day 13's conclusion stands: on the all-subtopic MAE at the budget, random wins,
0.422 to 0.433. Day 14 adds the mechanism.

**Adaptive is not worse per question.** It is ahead for eight questions. What it
does is trade breadth for depth, and in this environment the trade stops paying
at question nine. Two facts make the trade legible:

1. **An untouched subtopic contributes its whole prior error to the MAE**, and
   the prior is θ = 0 against a population whose ability is drawn around 0 with
   sd ≈ 1.0. So the marginal value of a *first* question in a new subtopic is
   large and the marginal value of a *fourth* question in an old one is small —
   arithmetic that has nothing to do with either policy.
2. **Adaptive's concentration is structural, not accidental.** Day 13 located
   it: Day 12's coverage term is topic-level while θ is subtopic-level, so
   nothing in the objective spreads questions *within* a topic; and resume
   affinity is the only always-on, non-satiating, subtopic-level term.

Both were already in Day 13. What Day 14 shows is that the resulting deficit
takes about nine questions to overtake the depth advantage, which is a fact
about the *budget* as much as about the policy — and the extended 60-item run
(Day 13, §9) is consistent with that reading, since the gap widens there rather
than closing.

### Adaptive against fixed — the clearest result on the page

Adaptive beats fixed at every question count from the first to the twentieth,
on MAE (0.433 vs 0.495), RMSE (0.538 vs 0.575), reach rate (74.5% vs 63.0%) and
time-to-half-the-population (question 9 vs question 13). It is behind fixed on
neither error curve at any point. **This bundles at least three mechanisms and
attributes the difference to none of them.**

### Where adaptive is unambiguously better

Conditional on having measured the same number of subtopics, adaptive is the
most accurate policy — see §7. Day 13 already reported the unconditional version
of this (MAE(seen) 0.333 against 0.415 and 0.494; difficulty gap 0.136 against
0.756 and 1.305).

**No hypothesis test was run and no significance is claimed** — here, in the
report, or in the figures. The differences between adaptive and random are
small relative to the between-candidate spread, and this experiment was not
designed to resolve them.

---

## 7. Coverage and convergence

The clearest new result on this page:

| subtopics measured | adaptive: sessions | MAE | random: sessions | MAE | fixed: sessions | MAE |
|---:|---:|---|---:|---|---:|---|
| 2 | 2 | 0.494 | 2 | 0.545 | — | — |
| 3 | 4 | 0.395 | — | — | 29 | 0.618 |
| 4 | 90 | 0.501 | 4 | 0.381 | 3 | 0.565 |
| 5 | 58 | 0.402 | 15 | 0.377 | 6 | 0.553 |
| **6** | **46** | **0.339** | **179** | **0.425** | **162** | **0.470** |

**Among the sessions that reached all six subtopics, adaptive has the lowest
error by a wide margin** — 0.339 against random's 0.425 and fixed's 0.470. That
is a larger gap than the 0.011 that separates the policies overall, and it runs
the *other way*.

The reading this supports: adaptive's measurement is better, its breadth is
worse, and the headline MAE is the sum of the two. Where the breadth handicap is
removed, the measurement advantage is what remains.

**The reading this does not support, and the caveat is not optional.** These
bins are **not a controlled comparison**. A session lands in a bin because of
what its policy did to *that candidate*, so a bin also selects on candidate
difficulty, session length and luck — and the adaptive six-subtopic bin holds 46
sessions against random's 179, so they are not the same slice of the population.
The table describes an association. It does not establish that adding a subtopic
to a given adaptive session would lower its MAE, and it is not evidence for any
weight change.

**What it does establish** is that the coverage/accuracy trade-off Day 13
identified is real, has a measurable size, and has a crossover point in this
environment at about nine questions.

---

## 8. Stability

**Every policy settles; none oscillates.** Mean \|Δθ\| falls from 0.187 to 0.117
(adaptive) and 0.172 to 0.097 (random) over twenty questions, and the late-window
means (0.138, 0.124, 0.082) are all well above Day 12's 0.05 "no new
information" threshold — which is consistent with that condition firing in only
26–28 of 200 adaptive and random sessions.

**The reversal rate is the informative number, and it says the estimates have
not arrived.** It counts consecutive updates *to the same subtopic* whose signs
disagree. An estimate sitting on the truth and being pushed around by grading
noise produces ≈ 50%; a run of 32–36% means every policy's θ̂ is still travelling
in one direction when the interview ends. In this environment that is the
expected shape — the estimate starts at 0 and the population has sd ≈ 1.0, so
most sessions spend all twenty questions walking towards their candidate — but
it is worth stating plainly: **at a 20-question budget nothing here has
converged in the strict sense.** MAE ≤ 0.5 is a usefulness threshold, not a
convergence proof, and 0 of 600 sessions met the precision criterion.

MAE rose on roughly a quarter of all turns for every policy (24.5–26.8%). Some
of that is unavoidable and not a defect: an answer that is informative about one
subtopic is silent about five, and the all-subtopic mean can rise when a
well-measured subtopic's estimate steps past the truth.

**The fixed policy's movement curve is the odd one.** It starts at 0.062 —
a third of adaptive's — rises to a peak of 0.139 around question 10, and falls
to 0.052. The easy-first paper opens with questions the model already predicts
correctly, so the first updates carry almost no news; the paper only becomes
informative in the middle. That is the same mechanism that makes 29 of its
sessions stop at question 3.

---

## 9. Stopping behaviour

Day 13 reported three totals per policy. Crossing them with the question number
turns one of those totals into a finding:

**29 of the fixed policy's 61 `no_new_information` stops fire at question 3** —
the earliest the rule can possibly fire, since it needs three consecutive
answers moving θ by less than 0.05. Nearly 15% of all fixed sessions end after
three questions.

This is not a bug in the stopping rule; the rule is behaving exactly as designed.
It is a property of an easy-first paper: the first three questions are ones the
cold-start model predicts confidently, the updates are tiny, and Day 12
concludes — correctly, on the evidence available to it — that nothing is being
learned. A fixed test can therefore terminate itself before it has measured
anything, and **the aggregate "61 sessions stopped for no new information" hides
completely whether that was a test that ran its course or a test that gave up.**

For adaptive and random the same reason fires late and thinly spread — means of
11.9 and 12.3 questions, no single question number carrying more than four
sessions. That is the shape of a test running out of things to learn, which is
what the condition is for.

The item budget ends 86% of adaptive sessions and 87% of random ones, always at
exactly question 20. Day 13's calibration observation is unchanged and worth
restating: **at the plan's budget the item budget ends nearly every interview
and the precision condition is inert.** Day 12's RD < 0.40 threshold was not
changed, and moving a stopping constant so that a synthetic run reaches it would
be fitting the system to the experiment.

---

## 10. Limitations

Every limitation in [`docs/simulation.md`](simulation.md) §2 applies unchanged.
These are the ones Day 14 adds or sharpens:

1. **One environment, one seed.** Every number on this page comes from seed
   20260101. The curves are not averaged over environments, and no test asserts
   they are stable under a different seed — because they are not expected to be,
   at the size of the differences involved.
2. **No statistical test.** None. Not on the crossover, not on the coverage
   bins, not on the stability rates. Differences of 0.011 MAE against a
   between-candidate spread of 0.24 would need a paired test this experiment
   was not designed around, and claiming significance without it would be worse
   than saying nothing.
3. **The coverage bins are observational.** §7 states this at length. They
   describe an association within a policy's own sessions and establish no
   cause.
4. **The crossover at question 9 is a property of this environment.** Six
   subtopics, a 20-question budget, quotas of {8, 7, 5}, and a population drawn
   with subtopic sd 0.6 around an overall sd 0.8. Change the taxonomy size or
   the budget and the crossover moves; the extended 60-item run puts it
   somewhere different again.
5. **Forward-filling is a modelling choice.** A stopped session is held at its
   last estimate. That is the honest reading of "what would a report say at
   question 20", but it means the tail of every curve mixes finished sessions
   with frozen ones, and the fixed policy — which stops earliest and most often —
   is the most affected.
6. **Adaptive-vs-fixed remains uncontrolled.** The difficulty window, the
   objective and ε-greedy vary together. Every result in §6's second half
   inherits that.
7. **The trajectory figures are three candidates out of two hundred.** They are
   chosen deterministically at the 10th, 50th and 90th percentile of the
   adaptive policy's ordering, which is stated in each caption. They illustrate;
   they do not evidence.
8. **Synthetic candidates answer the model the engine assumes.** This is the
   big one and it is not a Day 14 limitation, it is the experiment's. A real
   candidate is not a 2PL curve with Beta noise, and nothing here transfers to
   one without validation that has not been done.

---

## 11. What remains uncertain

- **Whether the crossover survives a real bank, a real taxonomy or real
  candidates.** Unknown, and this experiment cannot say.
- **Whether adaptive's conditional advantage at equal coverage is causal.** §7's
  table is consistent with it and does not establish it. A controlled version
  would need a policy that matches random's coverage while keeping adaptive's
  ranking — which is a design change, not an analysis.
- **Whether 0.40 is the wrong precision threshold or 20 questions is the wrong
  budget.** Day 13 could not separate these two readings and Day 14 cannot
  either. Both remain open, and neither constant was moved.
- **What the resume term is worth.** The synthetic resume carries no ability
  signal by construction, so this experiment measures the term's cost in
  measurement accuracy and is structurally incapable of measuring its benefit in
  perceived relevance. Resume affinity stays at 0.15.
- **Whether the reversal rate would reach 50% at a larger budget.** The 60-item
  run would answer this and was not analysed here.

---

## 12. Reproducibility

```bash
cd backend
python scripts/run_convergence_analysis.py                  # tables on stdout
python scripts/run_convergence_analysis.py --write-report   # + figures into evals/reports/
python scripts/run_convergence_analysis.py --extended       # the 60-item budget instead
python scripts/run_convergence_analysis.py --seed 7         # a different world
```

Zero API tokens, no database, no network; about fifteen seconds for the plan's
200 candidates. Everything derives from `--seed` through BLAKE2b, and two runs
with the same seed produce **byte-identical tables and byte-identical SVG
files** — which is asserted by test and verified by regeneration, and is the
reason the charts are emitted as text rather than rasterised.

Experiment configuration stays in `app/simulation/config.py`, unchanged. Day 14
added no experiment parameter and no threshold.

---

## 13. Deferred work

- ~~**Day 15 — the blueprint builder.**~~ **Done** — `app.blueprint`,
  [`docs/blueprint.md`](blueprint.md). It turns role + level + duration + JD
  weights into quotas and budgets, and `config.split_budget_by_jd` now delegates
  its division there. Deriving the item budget from the *committed* bank's pacing
  produced a number worth reading beside §5: a real 45-minute assessment fits
  **8 questions**, and the adaptive/random crossover measured above is at
  question 9.
- **A convergence analysis of the extended 60-item run.** The script takes
  `--extended` and produces the whole figure set for it; that output is not
  committed and its results are not discussed here. It is where the RD criterion
  first fires and where the reversal rate could plausibly approach 50%.
- **Paired statistical testing across seeds.** The right way to settle whether
  the crossover is real. It needs multiple environments and a stated
  hypothesis, which is a different experiment from this one.
- **A coverage-matched policy comparison.** The controlled version of §7.
- **Difficulty calibration from the exploration data** (plan §5.11) — still what
  ε-greedy's off-policy data is *for*, and still not done.
- **Any weight change.** Not deferred so much as declined: fitting weights to
  one synthetic experiment is fitting the experiment.
