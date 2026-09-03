# Simulation — does the adaptive selector actually work?

*Phase 3, Day 13. Written for someone who has never run an experiment on a
machine-learning system. Nothing here assumes you know what a Beta distribution,
a control variate, or an ablation study is.*

*Day 11 built the ability model ([`docs/adaptive-selection.md`](adaptive-selection.md)
covers it), Day 12 built the selector. This is the day the project stops
asserting that they work and starts measuring it.*

---

## The problem with Day 12

Day 12 finished with a working adaptive question-selector: hard constraints,
a six-term weighted score, ε-greedy exploration, a stopping rule. Every part of
it was tested — the arithmetic is right, the constraints hold, the SQL agrees
with the Python.

None of that is evidence that the **policy** is any good.

A test can tell you `score_item` computes the formula you wrote down. It cannot
tell you that asking questions in that order measures a candidate better than
asking them at random. Those are different claims, and only one of them had been
checked. Day 12's own report said so, and deferred both open questions here:

> *"These six weights are design choices, not measured optima — the ablation that
> could justify them is Day 13 and has not been run."*

So Day 13 runs the experiment. It costs **zero API tokens**, needs no database
and no network, and takes about three minutes.

---

## 1. Why simulate at all?

To know whether an ability estimate is *correct* you have to know the truth. With
real candidates you never do — that is the entire reason the engine exists.

A simulation inverts the problem. **We invent 200 people, decide exactly how good
each of them is, hide that from the engine, and see whether it can work it out.**

```
        we choose it                      the engine never sees it
             │                                       │
             ▼                                       ▼
   ground-truth θ  ──────►  synthetic answer  ──────►  estimated θ̂
   arrays  = +0.8            "scored 0.71"             arrays  = +0.6
   graphs  = -0.4                                      graphs  = -0.3
   caching = +1.2                                      caching = +0.9
             │                                       │
             └──────────────►  |θ̂ − θ| ◄─────────────┘
                              the error we can finally measure
```

The engine gets the same thing it would get from a real interview — a graded
score between 0 and 1 — and nothing else. We get to grade its homework, because
we wrote the answer key.

---

## 2. What this **cannot** tell us

This section is deliberately before the results, because it is the easiest claim
in the project to overstate.

The simulated candidates answer according to the *same 2PL curve the engine
assumes*. That makes the experiment a check that the implementation is coherent
and that one policy beats another **under the model's own assumptions**. It is
circular with respect to the model itself, and no amount of extra candidates
fixes that.

| a simulation **can** show | a simulation **cannot** show |
|---|---|
| the implementation recovers abilities it was given | that real people behave like a 2PL curve |
| adaptive beats random under these assumptions | that it beats random on real candidates |
| which scoring components change behaviour here | which components matter in a real interview |
| the stopping rule fires when the arithmetic says it should | that stopping there is the right call for a person |
| the code is reproducible and constraint-obeying | anything about question quality, fairness, or hiring validity |

Every number below is a statement about a synthetic world. Treat "adaptive
reduces error by X" as "adaptive reduces error by X **in this simulation**", and
the rest of the sentence is honest.

---

## 3. The synthetic world

Everything is generated from one integer seed, so the whole experiment is
reproducible: `python scripts/run_selection_simulation.py --seed 20260101`.

### Why not the real question bank?

The committed 60-item bank is a good dataset and a poor laboratory. Measured
directly:

| the real bank | why it cannot support this experiment |
|---|---|
| 60 items | a 20-item session eats a third of it, so the policies end up choosing between the same few leftovers |
| 35 subtopics, 1–3 items each | a subtopic can be asked at most three times, so θ there can never converge |
| difficulty only −1.2 to +1.6 | a candidate at θ = −2.5 has **nothing** inside the `\|b − θ\| ≤ 1.5` window |
| every discrimination = 1.0 | `a` has no variance, so the one place the response model uses it is inert |
| 150–420 s per item | twenty items is 110 minutes; any realistic time budget stops every session at item five |

So Day 13 builds a controlled bank instead — and pays for it with the caveat in
§2, which is the right trade for an experiment whose job is to isolate one
variable.

### The bank — 192 questions

6 subtopics × 32 items. Each item gets:

- **difficulty `b`** spread evenly across −2.5 … +2.5 *within its subtopic*, plus
  a small jitter. So every subtopic has something askable whatever the candidate's
  ability — the difficulty window can never starve.
- **discrimination `a`** drawn from 0.7 … 1.6. This *is* used by the response
  model and by the RD update, and is *not* used by Day 12's selection score. The
  world therefore distinguishes a sharp item from a flat one while the policy
  cannot — which is exactly the asymmetry the real system has.
- **time estimate** from {60, 90, 120, 150, 180} seconds.
- **an embedding**, 16 numbers, so the redundancy term has something real to
  measure. Items in the same subtopic sit at cosine 0.87 on average and never
  below 0.63; items in different subtopics average 0.00 and never exceed 0.49.
  Every same-subtopic pair is closer than every cross-subtopic pair — asserted by
  test, because a geometry without that property would make the redundancy
  penalty react to noise.

### The taxonomy — 3 topics, 6 subtopics

```
algorithms → arrays, graphs          JD weight 0.9, quota 8 items
databases  → indexing, transactions  JD weight 0.7, quota 7 items
systems    → caching, queues         JD weight 0.5, quota 5 items
```

Six subtopics is a compromise, and the losing side is worth naming. Every answer
lands on exactly one subtopic, so with a 20-item budget:

- 16 subtopics would give 1.25 observations each and nothing would converge;
- 2 subtopics would give 10 each and the coverage half of the objective would have
  nothing to do.

Six puts ~3.3 observations on each — enough for θ to travel a long way from its
prior, not enough for RD to reach Day 12's 0.40 threshold. That second fact is a
result, not a design flaw, and §9 reports it.

### The candidates — 200 people

Ability is drawn in two levels:

```
overall_i     ~ Normal(0, 0.8)                  how good this person is overall
θ_i,subtopic  ~ Normal(overall_i, 0.6)          how good they are at each thing
```

clamped into −2.5 … +2.5 so the bank can actually probe them.

The two-level draw matters. Drawing each subtopic independently would mean a
candidate strong at arrays is no more likely to be strong at graphs — not how
people are, and it would destroy the correlation that makes an early answer
informative about the next subtopic. Drawing one number per candidate would
destroy the per-subtopic variation the engine exists to find.

Each candidate also gets a **synthetic resume**: two subtopics, each with an
affinity in 0.5 … 1.0.

> **The resume is drawn from a stream that never sees ground truth.** This is
> deliberate. A resume that correlated with true ability would hand the adaptive
> policy an information channel the baselines do not use, and "CAT wins" would
> stop being a statement about selection. The cost is that the ablation can only
> measure what the resume term **costs** in measurement accuracy, never what it
> delivers in perceived relevance — which this simulation cannot measure at all.

---

## 4. The response model

Plan §8.6: *"draw a score from a Beta distribution centred on p(θ_i, b_j)"*.

**Step 1 — the expected score comes from Day 11, unchanged.**

```
p = probability_correct(true_θ[subtopic], item.b, item.a)
```

That is the production 2PL function, called with the *hidden* θ instead of the
estimated one. This is the only place in the whole package that reads ground
truth.

**Step 2 — the observed score is a Beta draw centred on it.**

```
α = p · k                 k = 10, the "concentration"
β = (1 − p) · k
score = Beta(α, β)        mean exactly p, variance p(1−p)/(k + 1)
```

**Why Beta and not a coin flip.** Day 11's `update_ability` accepts a *soft*
score in [0, 1], because a graded answer is a rubric total, not a right/wrong
bit. A Bernoulli draw would throw that away and would test a model the system
does not use. Beta is the natural distribution on [0, 1], and parameterising it
by mean and concentration lets `p` stay exactly the expected score while `k`
controls how noisy the grader is.

`k = 10` gives a standard deviation of 0.151 on a coin-flip item — roughly the
spread two competent human graders show on the same answer. Larger `k` is a more
reliable grader.

`p` is clamped into [0.01, 0.99] before it becomes Beta parameters, because at
`p = 0` exactly `α = 0` and the distribution is undefined — precisely at the ends
of the difficulty range where the bank is widest.

---

## 5. The three policies

| | no repeats | topic cap | time fits | `\|b−θ\| ≤ 1.5` | how it ranks |
|---|:--:|:--:|:--:|:--:|---|
| **adaptive** | ✅ | ✅ | ✅ | ✅ | Day 12's six-term score + ε-greedy |
| **random** | ✅ | ✅ | ✅ | ✅ | uniform |
| **fixed** | ✅ | ✅ | ✅ | ❌ | a predetermined sequence |

**Adaptive** calls `app.selection.choose_next` directly. Not a re-implementation,
not a simplification — the shipped function, with the shipped weights, so there
is no second selector that could drift from the first.

**Random** calls Day 12's own `filter_eligible` and then draws uniformly. It sees
*exactly* the same eligible pool the adaptive policy scores, so the difference
between the two is precisely the value of the objective plus ε-greedy, and
nothing else.

**Fixed** walks a predetermined exam paper and never consults θ. The one
asymmetry in the table is deliberate: the difficulty window is a *function of the
running estimate*, so a "fixed" policy that applied it would be adapting. It
still honours the three constraints that do not depend on θ, because those are
properties of the session rather than of the policy.

So the two comparisons answer **different questions, and they are not equally
clean**. This distinction governs how every number in §9 may be read.

**adaptive vs random — a controlled comparison of the ranking objective.**
Both policies see *exactly* the same eligible pool, because `RandomStrategy`
calls Day 12's own `filter_eligible` with the same difficulty window. Every hard
constraint, including the θ-dependent one, is held identical. The **only**
difference is whether the six-term score and ε-greedy choose within that pool, or
a uniform draw does. Any gap between them is attributable to the ranking
objective and to nothing else, which makes this the comparison that actually
tests what Day 12 built.

**adaptive vs fixed — a comparison of two different systems, not a controlled
one.** The fixed policy *cannot* apply the `|b − θ| ≤ 1.5` filter, because that
filter is a function of the running ability estimate and a policy consulting it
would no longer be fixed. So this contrast bundles together at least three
changes at once: the difficulty window, the ranking objective, and ε-greedy. It
answers "is an adaptive test better than a non-adaptive one?" — a worthwhile
question, and the one plan §8.6 asks — but it **cannot** attribute a difference
to any single mechanism, and no result below does so.

Reading them the wrong way round is the most likely misinterpretation of this
whole experiment: a large adaptive-vs-fixed gap is *not* evidence that the
weighted objective is doing the work, because most of that gap could come from
the difficulty window alone.


### How the fixed sequence is built

The plan says *"fixed sequence (easy → hard)"*. Read literally over a 192-item
bank that means **the twenty easiest questions in existence**, all clustered near
b = −2.5 — which no real fixed test looks like, and which would be a deliberately
feeble baseline. A rigged comparison proves nothing.

So the paper is built the way a paper exam actually is:

1. **Within a topic**, take that topic's quota of items at evenly spaced
   percentiles of its difficulty-sorted list — the easy, medium and hard
   questions — and present them in increasing difficulty.
2. **Across topics**, interleave proportionally: at each position the topic
   furthest behind its quota share goes next.

The result satisfies the blueprint by construction, so the topic cap never binds
for this policy and cannot disadvantage it. The remaining items follow as
fallback for a turn where the front of the paper does not fit the clock.

---

## 6. Fairness controls

The comparison is only worth reading if the three policies really are sitting the
same exam. Five mechanisms enforce that.

**1. One environment object.** The bank and the 200 candidates are built once and
handed to all three policies — the mechanical form of "same questions, same
people, same difficulties, same discriminations, same topic distribution".

**2. Common random numbers.** This is the important one. The generator for a
graded answer is keyed by **(candidate, item)** and nothing else:

```
seed(experiment_seed, "response", candidate_id, item_id)
```

So candidate `c007` answering `arrays-19` produces **the same score** under CAT,
under random selection and under the fixed sequence, whether it was their first
question or their twentieth. Any difference in the results is caused by *which
questions each policy chose*, never by one of them getting luckier grading. This
is the standard variance-reduction technique for comparing policies on a shared
environment, and it means the comparison stabilises with far fewer candidates
than independent draws would need.

**3. A shared policy stream.** ε-greedy's exploration draws come from
`seed(experiment_seed, "policy", candidate_id)` — not keyed by the strategy, so
no policy and no ablation variant gets a luckier sequence of coin flips.

**4. Identical budgets and stopping rules.** Same item budget, same time budget,
same cold start (θ = 0, RD = 1.30 everywhere), same Day 12 `should_stop` — which
is *not* modified for the simulation, even where that would flatter the results.

**5. Ground truth is structurally unreachable.** A strategy's only arguments are
`(state, bank, rng)`. `SelectionState` holds no reference to a candidate, so
there is no field through which the answer key could leak. Two tests pin this: a
signature check, and a behavioural one — two candidates who differ *only* in
hidden ability must receive the same first question, since before any answer
exists there is no legitimate signal distinguishing them.

---

## 7. What is measured

**Estimation error.** `|θ̂ − θ_true|` per subtopic, reported two ways:

- **MAE** — over *every* blueprint subtopic, with unmeasured ones left at the
  cold-start prior θ = 0. This is the number a report would actually print, and
  it is the honest headline: a policy that never asked about graphs still has an
  opinion about graphs.
- **MAE(seen)** — over only the subtopics a policy asked about. The optimistic
  reading, and it must be read next to **subtopics covered**, because measuring
  two subtopics beautifully while ignoring four is not a better assessment.

RMSE is reported alongside MAE. Squaring before averaging makes one badly-missed
subtopic count for more than several slightly-missed ones, so RMSE much larger
than MAE means the average is hiding a subtopic nobody pinned down.

**Convergence.** Two criteria, both explicit:

- **items to MAE ≤ 0.5** — "close enough to the truth to place someone in the
  right band". 0.5 is one sixth of the θ scale. *A design choice, and an
  arbitrary one*; it is reported next to the full error curve so a reader who
  dislikes the threshold can ignore it.
- **items to SE ≤ 0.35** — plan §8.6's own criterion, applied at **topic** level
  through Day 11's `roll_up`: the same precision-weighted aggregation Day 12's
  stopping rule uses, one notch tighter than its 0.40. A topic nobody asked about
  counts as infinite, so "we have no idea" can never satisfy a precision
  threshold.

A session that never meets a criterion is **censored** — counted separately, not
dropped. Averaging over only the sessions that converged is the classic way to
make a policy that rarely converges look fast.

**Questions used, stopping reasons, coverage compliance** (the share of sessions
meeting every topic quota) and **difficulty appropriateness** (`|b − θ|` at the
moment each item was chosen) round it out. The last two are plan §8.6's own
extra requests.

---

## 8. The weight ablation

Day 12 shipped six weights and admitted it had no evidence for them. An
**ablation study** is the standard way to get some: take the finished system,
switch off one component at a time, and measure what each removal costs.

Seven configurations — the full objective, then one with each weight set to zero.

**No renormalisation, and none is needed.** Within a single selection every item
is scored with the same weights, so multiplying all six by a constant multiplies
every score and leaves the argmax and the top-5 exactly where they were. Rescaling
the survivors would therefore change *nothing* about the decisions while making
the study harder to describe. "Remove a component" means precisely "its weight is
0". (A test pins the invariance: doubling all six weights never changes a pick.)

The weights are varied through an `ObjectiveWeights` value passed as an argument,
never by mutating a module constant. A simulation that monkeypatched
`COVERAGE_WEIGHT` would leak across runs and silently change the behaviour of
every other caller in the process — exactly the class of bug an experiment must
not introduce into the system it is measuring. **The production defaults are
unchanged, and a test asserts it after the ablation has run.**

---

## 9. Results

The numbers live in
[`evals/reports/selection_simulation.md`](../evals/reports/selection_simulation.md)
and are regenerated by the script rather than edited. The headline, at 200
candidates and a 20-item budget:

| policy | MAE | RMSE | MAE(seen) | subtopics covered | items | abs(b−θ) | coverage |
|---|---|---|---|---|---|---|---|
| adaptive | 0.433 | 0.538 | **0.333** | 4.71 / 6 | 18.9 | **0.136** | 86.0% |
| random | **0.422** | **0.504** | 0.415 | 5.84 / 6 | 19.0 | 0.756 | 87.0% |
| fixed | 0.495 | 0.575 | 0.494 | 5.50 / 6 | 16.6 | 1.305 | 69.5% |

Spread across the 200 candidates (MAE): adaptive median 0.382, sd 0.229; random
median 0.370, sd 0.209; fixed median 0.436, sd 0.254. The differences are small
relative to that spread — worth remembering before any of them is called
decisive.

### Observed

1. **Adaptive measures the subtopics it investigates far more precisely.**
   MAE(seen) 0.333 against random's 0.415 and fixed's 0.494 — 20% and 33% lower.
   The gap `abs(b − θ)` at selection time is 0.136 against 0.756 and 1.305, so
   the information term is doing exactly what plan §5.10 says it should.
2. **Adaptive covers fewer subtopics than either baseline**: 4.71 of 6, against
   random's 5.84 and fixed's 5.50.
3. **On the all-subtopic MAE the two are close, and random edges it**: 0.422
   against 0.433. The precision gain is slightly more than cancelled by the
   coverage loss.
4. **Against fixed, adaptive wins on accuracy and difficulty matching and loses
   on breadth.** Not on every metric — that claim would be false:

   | adaptive vs fixed | adaptive | fixed | |
   |---|---|---|---|
   | MAE | **0.433** | 0.495 | adaptive better |
   | RMSE | **0.538** | 0.575 | adaptive better |
   | MAE(seen) | **0.333** | 0.494 | adaptive better |
   | abs(b − θ) | **0.136** | 1.305 | adaptive better |
   | coverage compliance | **86.0%** | 69.5% | adaptive better |
   | subtopics measured | 4.71 | **5.50** | **fixed better** |
   | items asked | 18.9 | 16.6 | fixed used less of the budget |

   The last two rows are the honest qualifications. Fixed reaches more subtopics
   than adaptive does, and it spends fewer of its twenty items — though the
   second is not a win on its own, because it stops early with a *worse*
   estimate (§Failure cases), and fewer items only counts as efficiency at equal
   accuracy.
5. **Nothing converges on the RD criterion.** Zero of 200 sessions reach
   SE ≤ 0.35 under any policy; the best worst-topic RD reached at 20 items is
   0.546, and the mean is 0.674.
6. **Convergence to MAE ≤ 0.5** takes 6.4 items for adaptive, 6.1 for random and
   6.1 for fixed — but is *reached* by 149, 152 and 126 of 200 sessions
   respectively. Adaptive is not faster on this criterion here.

### Interpretation

**Read §5's distinction first.** Adaptive-vs-random is the controlled comparison:
same eligible pool, same difficulty window, only the ranking objective differs.
Adaptive-vs-fixed bundles the difficulty window together with the objective and
ε-greedy, so it can say *whether* an adaptive test beats a non-adaptive one but
cannot attribute the gap to any one mechanism — and most of that gap plausibly
comes from the difficulty window rather than from the six-term score.

**The precise conclusion this experiment supports:**

> Adaptive selection substantially improves **precision on the subtopics it
> investigates** and **difficulty matching** — MAE(seen) 0.333 vs 0.415, and a
> difficulty gap of 0.136 vs 0.756, both against random on an identical
> eligibility pool. In this synthetic environment it currently sacrifices enough
> **coverage** (4.71 of 6 subtopics vs 5.84) that random **slightly wins on
> overall MAE**, 0.422 to 0.433. Against the fixed sequence, adaptive is more
> accurate and far better matched to ability, but reaches fewer subtopics.

Every clause of that is measured. None of it generalises past this environment
(§2).

Observation 3 is a **genuine finding against the current objective**, and the
ablation locates it precisely. Two mechanisms combine:

- **Day 12's coverage term is defined at topic level, but θ is estimated at
  subtopic level.** Nothing in the objective pushes the policy to spread across
  the *subtopics* inside a topic; subtopic coverage happens only as a side effect
  of the redundancy penalty and of the hard per-topic cap.
- **The resume term is the only always-on, subtopic-level bias in the objective,
  and it never satiates.** Information shrinks as the estimate converges toward
  `b`; coverage deficit shrinks as a quota fills. Resume affinity is constant for
  the whole session, so its influence accumulates.

Measured directly — the share of items landing on the two resume-mentioned
subtopics, where uniform selection would give 33.3%:

| objective | share on resume subtopics |
|---|---|
| full | **59.9%** |
| without the resume term | 33.0% |
| without the redundancy term | 62.2% |
| random selection | 33.4% |

### The ablation

| objective | MAE | ΔMAE | MAE(seen) | subtopics | items | abs(b−θ) | coverage |
|---|---|---|---|---|---|---|---|
| full | 0.4329 | — | 0.3325 | 4.71 | 18.86 | 0.136 | 86.0% |
| no information | 0.4719 | **+0.0390** | 0.3639 | 4.39 | 18.57 | 0.802 | 85.0% |
| no JD | 0.4408 | +0.0079 | 0.3483 | 4.76 | 18.45 | 0.134 | 82.5% |
| no resume | 0.3750 | **−0.0579** | 0.3741 | 5.99 | 19.55 | 0.107 | 94.5% |
| no coverage | 0.4433 | +0.0104 | 0.3436 | 4.61 | 18.38 | 0.138 | 83.5% |
| no redundancy | 0.4495 | +0.0166 | 0.3083 | 4.11 | 19.00 | 0.130 | 89.5% |
| no time | 0.4419 | +0.0090 | 0.3414 | 4.69 | 18.82 | 0.131 | 86.5% |

A **positive** ΔMAE means removing the component made the estimate worse — the
component was earning its place.

- **Information (+0.039) is the clear winner.** Removing it is the largest
  degradation of any component and sends the difficulty gap from 0.136 to 0.802.
  This is the one weight the experiment positively supports.
- **Redundancy (+0.017) is the surprise.** It turns out to be doing *coverage*
  work: semantic similarity is highest within a subtopic, so penalising it pushes
  the policy away from asking the same subtopic again. Removing it drops
  subtopics covered from 4.71 to 4.11 — while *improving* MAE(seen) to 0.308,
  which is the precision/coverage trade-off in miniature.
- **Coverage (+0.010), time (+0.009) and JD (+0.008) are small and positive.**
  Each is worth something here; none is decisive, and all three are well inside
  the noise a different seed could produce.
- **Resume (−0.058) is the only component that costs accuracy**, and by the
  largest margin of any row. Removing it raises subtopics covered from 4.71 to
  5.99 and coverage compliance from 86.0% to 94.5%, and would put adaptive
  (0.375) ahead of random (0.422).

**This is a diagnostic finding, not a recommendation.** It does *not* say
"delete the resume term", and **resume affinity has been left at 0.15,
untouched**. The synthetic resume carries **no ability signal by construction** (§3),
so in this environment it can only ever cost accuracy — the experiment measures
its price and is structurally incapable of measuring its benefit, which is
perceived relevance to a candidate. What the experiment *does* establish is a
mechanism: **a constant, non-satiating, subtopic-level term inside a topic-level
coverage scheme causes subtopic starvation.** That mechanism holds whether or not
the resume is informative, and it is a real property of the current objective.

Nothing in Day 12 was changed in response — not the resume weight, not any of
the other five. Retuning on a single synthetic experiment is fitting the
experiment rather than the system. What is recorded is the *diagnosis*: where the
coverage loss comes from, and which structural property of the objective produces
it. Acting on it belongs to a later day with evidence this experiment cannot
supply.

### Failure cases

- **Subtopic starvation — the central negative finding.** Adaptive reaches 4.71
  of 6 subtopics, fewer than *either* baseline (random 5.84, fixed 5.50). Because
  an unmeasured subtopic keeps the cold-start prior, that shortfall is what
  converts a real precision advantage into a slight loss on overall MAE. The
  mechanism is diagnosed above; nothing was changed in response.
- **A whole topic goes untouched** in 4 of 200 adaptive sessions, against 2 for
  random and 0 for fixed. At a 60-item budget it gets *worse* — 23 of 200 —
  because larger quotas mean the hard cap binds later, so the resume and JD pull
  can concentrate for longer before anything forces diversification. The soft
  coverage term is too weak to produce coverage on its own.
- **The fixed policy collapses at longer budgets.** At 60 items it stops after
  16.2 items on average, with **all 200** sessions ending on `no_new_information`
  and coverage compliance at 0%: its easy-first paper produces a run of
  confidently-predicted answers, the estimate stops moving, and the stopping rule
  correctly concludes nothing more is being learned.
- **Adaptive stops slightly earlier than random** (18.9 vs 19.0 items at budget
  20, 38.3 vs 40.3 at 60) for a milder version of the same reason: converging to
  small updates trips `no_new_information`, and an error metric measured at the
  end then rewards the policy that kept asking.
- **Neither adaptive nor random converges faster on MAE ≤ 0.5** (6.4 vs 6.1
  items). The precision advantage shows up in *where* the error ends up, not in
  how fast it crosses a loose threshold.

### Calibration finding: the precision stopping rule cannot fire

Day 12 stops when every required topic reaches RD < 0.40. In this environment
that **never** happens within the plan's 20 items, and happens twice in 200
sessions at 60 items.

The arithmetic explains it. RD updates as `1/sqrt(1/RD² + a²·p(1−p))`, so one
informative answer adds about 0.33 of precision. A topic with two subtopics starts
at 2 × 1/1.30² = 1.18 of precision and needs 1/0.40² = 6.25 — roughly **15
informative answers per topic**, about 45 items for a three-topic blueprint and
more than 20 for even one topic.

**This is an observation about the budget, not a defect to fix today.** Two
readings are consistent with it and this experiment cannot separate them: either
0.40 is tighter than a 20-item interview can earn, or a 20-item interview over
six subtopics is thinner than the threshold assumes. Deciding between them needs
evidence this simulation does not contain — real sessions, a real bank, and a
real answer to "how precise does a topic estimate have to be before a report may
quote it?".

So Day 12's threshold was **not** changed. Moving a stopping constant so that a
synthetic run reaches it is fitting the system to the experiment, which is the
exact failure this day exists to avoid. The practical consequence is recorded
instead: at the plan's budget the item budget is what ends an interview, the
precision condition is inert, and any future claim that the engine "stops when it
is confident enough" has to reckon with that.

## 10. Reproducibility

```bash
cd backend
python scripts/run_selection_simulation.py                # the plan's experiment
python scripts/run_selection_simulation.py --ablation     # plus the weight ablation
python scripts/run_selection_simulation.py --write-report # everything, into evals/reports/
python scripts/run_selection_simulation.py --seed 7       # a different world
```

Every stochastic component derives its seed from the one integer, through BLAKE2b
rather than Python's `hash()` — which is salted per process and would make the
"same" experiment differ between two runs of the same command. Two runs with the
same seed produce byte-identical tables; a test asserts it on the whole
experiment, not just on a component.

---

## 11. What Day 13 does not do

- **No chart.** The convergence plot with error bands is Day 14. This computes the
  curve and prints eight numbers of it; it does not plot.
- **No blueprint builder.** `split_budget_by_jd` divides an already-given budget
  by already-given JD weights. Turning role + level + duration into those inputs
  is Day 15.
- **No weight changes.** See §9.
- **No difficulty calibration.** Fitting `b` from observed responses is plan §5.11
  — and is what ε-greedy's exploration data is *for*.
- **No grading, follow-ups, session persistence, HTTP endpoint, or deployment.**
