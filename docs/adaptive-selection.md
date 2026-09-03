# Adaptive item selection — which question to ask next

*Phase 3, Day 12. Written for someone who has never studied statistics or
machine learning. Nothing here assumes you know what Fisher information, item
response theory, or an exploration/exploitation trade-off is.*

*Day 11 built the **estimate** — how good is this candidate, and how sure are we?
That is `backend/app/ability.py`. This document is about the **decision** built on
top of it: given that estimate and a bank of questions, which one do we ask?*

---

## The problem

An interview has a budget — maybe twelve questions and thirty minutes. A fixed
list wastes most of it. Ask a senior engineer twelve easy questions and you learn
almost nothing you did not know after the second one; ask a junior twelve hard
ones and you learn nothing at all, and they leave feeling humiliated.

An adaptive test picks each question *after* seeing the previous answers. That is
the difference between:

| | fixed test | adaptive test |
|---|---|---|
| question 5 depends on answers 1–4 | ❌ | ✅ |
| items needed for a given confidence | ~20 | ~10 |
| feels like | a form | a conversation |

Day 12 is the code that makes that choice. It has three stages and a fourth
question it answers separately:

```
      Day 11: ability state          θ and RD, per subtopic
                    │
                    ▼
      ┌─────────────────────────────┐
      │ 1. hard constraints (SQL)   │   what is even allowed?
      └─────────────────────────────┘
                    │  the eligible pool
                    ▼
      ┌─────────────────────────────┐
      │ 2. weighted scoring         │   how good is each allowed item?
      └─────────────────────────────┘
                    │  ranked candidates
                    ▼
      ┌─────────────────────────────┐
      │ 3. ε-greedy choice          │   usually the best, sometimes not
      └─────────────────────────────┘
                    │
                    ▼
              the next question

      and, separately:  should there be a next question at all?
                        → the stopping rule
```

Everything lives in `backend/app/selection/`:

| file | stage | what it holds |
|---|---|---|
| `information.py` | — | Fisher information and its normalised form |
| `state.py` | — | what the policy reads: θ/RD, quotas, JD, resume, history, clock |
| `constraints.py` | 1 | the four hard filters, as SQL and as a Python predicate |
| `objective.py` | 2 | the six-term weighted score |
| `policy.py` | 3 | ranking and the ε-greedy draw |
| `stopping.py` | — | the four stopping conditions |

---

## 1. What makes a question informative

### The prediction we already have

Day 11 gives us two numbers per subtopic:

- **θ (theta)** — the candidate's ability, roughly −3 to +3, where 0 is "at the
  target level for this role".
- **RD** — how unsure we are about θ. Roughly 0.3 (measured several times) to
  1.3 (never measured).

Every question in the bank has a **difficulty `b`** on the *same scale*. That
shared scale is the whole trick, because it means you can subtract them. The
model's prediction that the candidate answers a question well is

```
p = 1 / (1 + exp(-a·(θ − b)))          the 2PL model, app/ability.py
```

which is just a smooth S-curve: far above the difficulty → p near 1, far below →
p near 0, exactly at it → p = 0.5.

### Fisher information

**Fisher information measures how much an observation will teach you about a
quantity you are trying to estimate.** For this model it is:

```
I(θ, b) = a² · p · (1 − p)
```

and the interesting factor is `p · (1 − p)`:

| situation | p | `p·(1−p)` | what you learn |
|---|---:|---:|---|
| way too easy | 0.95 | 0.0475 | almost nothing |
| perfectly matched | **0.50** | **0.2500** | **the most possible** |
| way too hard | 0.05 | 0.0475 | almost nothing |

`p·(1−p)` is largest at p = 0.5, and p = 0.5 happens exactly when **b = θ**.

**Why, in plain language.** If you are already confident someone will get a
question right, watching them get it right tells you nothing you did not already
believe. The informative question is the one where you genuinely cannot predict
the outcome. *Uncertainty in the prediction is information in the result.*

**The analogy.** Binary search. You do not check the first element of the array
or the last one — you check the middle, because that is the probe that halves
what you do not know. Adaptive testing is binary search over ability, and Fisher
information is its continuous, probabilistic version.

### Two information numbers, kept apart on purpose

This is the one place a reader can get confused, so the code keeps them in two
separate functions with two separate names.

| | formula | range | used for |
|---|---|---|---|
| **Fisher information** | `a² · p · (1−p)` | `[0, a²/4]` | the honest quantity; it is the term Day 11's RD update adds to precision |
| **normalised selection information** | `p · (1−p) / 0.25` | `[0, 1]` | the first term of the Day 12 scoring objective |

Two things differ.

1. **The `a².`** Fisher information includes the item's discrimination; the
   selection objective does not. That is deliberate and comes from the plan,
   whose `score_item` writes `p = sigmoid(state.theta[q.subtopic] −
   q.difficulty_b)` with no `a` in it. If `a²` were in the selection score, a
   sharp item would get up to four times the information term of a flat one, and
   that would silently re-weight the whole six-term sum against the other five
   terms. `a` still does its real jobs elsewhere: it shapes the prediction inside
   `probability_correct`, and it drives the RD update.
2. **The `/ 0.25`.** Dividing by the maximum turns `p·(1−p) ∈ [0, 0.25]` into
   `[0, 1]`, so the term is comparable with the other five and the weight 0.40
   means what it looks like it means. It is a pure rescaling — it changes nothing
   about *which* item is most informative.

`fisher_information()` and `normalised_information()` are separate functions and
neither silently converts into the other.

---

## 2. Why information is not the whole story

Suppose the engine simply asked the most informative question every time. Watch
what happens:

- Information is highest where uncertainty is highest, so it finds the single
  subtopic it knows least about — and asks about it twelve times in a row.
- The job description is never consulted, so a candidate applying for a
  backend role gets grilled on whichever corner of the taxonomy happened to have
  the widest error bar.
- Two near-identical questions both look maximally informative, so it asks both,
  back to back.
- A six-minute question is chosen with two minutes left on the clock.

Every one of those is a real failure and none of them is a bug in the maths.
Pure information maximisation optimises a measurement objective, and an interview
has more than one objective. **Hence the weighted score: information is 40% of
it, not 100%.**

### The objective

```python
def score_item(q, state):
    p    = sigmoid(state.theta[q.subtopic] - q.difficulty_b)
    info = p * (1 - p)                              # ∈ [0, 0.25]

    return (
        0.40 * (info / 0.25)                        # information gain
      + 0.25 * state.jd_weight[q.topic]             # role alignment
      + 0.15 * resume_affinity(q, state.resume)     # personalisation
      + 0.15 * coverage_deficit(q.topic, state)     # quota shortfall
      - 0.10 * redundancy(q, state.asked)           # similarity to asked
      - 0.05 * (q.time_estimate_s / state.time_left)
    )
```

That is `app/selection/objective.py::score_item`, term for term.

| term | weight | why it is there | why it is not bigger |
|---|---:|---|---|
| information gain | **+0.40** | measuring the candidate is the primary objective | at 1.00 it asks twelve questions about one subtopic and ignores the job |
| JD weight | **+0.25** | the candidate asked to be assessed for a *specific role*; an assessment that drifts off it produces a report about someone else | it is alignment, not measurement — it cannot be allowed to pick easy on-topic questions over informative ones |
| resume affinity | **+0.15** | personalisation: ask about what they say they have done | it is a *product* feature, not a measurement feature. Let it dominate and the engine measures whatever flatters the resume |
| coverage deficit | **+0.15** | pushes toward topics still behind their quota, so the blueprint is met before the budget runs out | it is a tie-breaker among informative items, not a reason to ask a useless one |
| redundancy | **−0.10** | two near-identical questions produce *correlated* evidence, which looks like two independent measurements and inflates apparent confidence | a mild nudge; genuine near-duplicates are a bank-quality problem, not a scoring one |
| time cost | **−0.05** | a six-minute item late in a thirty-minute session is a bad trade however informative | smallest, because the hard time filter already removed everything that does not fit — this only orders what remains |

### The honest caveat

> **The six selection weights are design choices, not mathematically optimised
> parameters.**

Nothing in this repository has yet demonstrated that
`0.40 / 0.25 / 0.15 / 0.15 / −0.10 / −0.05` beats any other set of numbers. They
encode a *priority ordering* that can be argued with, and they are written as
named constants so that arguing with them is a one-line diff rather than a hunt
through an expression.

The experiment that would turn them into an empirical claim — run the simulation
with information-only selection, with the full objective, and with random
selection, and report all three — is plan §8.6 and is **deferred to Day 13**.
Until it has run, no document or docstring in this project calls these weights
optimal. A designer who knows which of their constants are principled and which
are chosen is more trustworthy than one who claims all of them are principled.

### One consequence worth knowing

The four positive weights sum to **0.95**, not 1.0 — the plan does not make them
a probability distribution. So a score lives in `[−0.15, 0.95]`. It is a
**ranking key**, never a percentage, and nothing may present it as one.

### The three derived terms

**`resume_affinity(q, resume)` → `[0, 1]`.** Returns the first of: an explicit
score for the question's *subtopic* key; an explicit score for its *topic* key;
the cosine similarity between a resume vector and the question's vector, with
negatives clamped to 0; otherwise 0.0. Specific beats general, which is why the
subtopic is consulted first.

*This is a minimal interface, and it is meant to be.* Resume parsing, skill
extraction and resume-seeded θ priors are Phase 5. `ResumeProfile` is the shape
Day 12 needs from that future work and nothing more — no LLM call, no network,
no file reading, no new service.

**`coverage_deficit(topic, state)` → `[0, 1]`.**

```
deficit = max(0, target - already_asked) / target
```

1.0 for a topic that has had none of its required items, 0.0 once the quota is
met, proportionally in between — so a topic needing three more of four outranks
one needing one more of four. A topic with no target, a target of zero, or an
over-served quota all score 0.0; the floor makes this a nudge toward the
underserved rather than a punishment of the served.

The `target` numbers come from the session blueprint. **Day 12 consumes that
mapping; it does not build one.** The blueprint builder is Day 15.

**`redundancy(q, asked)` → `[0, 1]`.** The cosine similarity to the *most*
similar already-asked question — maximum, not mean. A candidate that is a
near-duplicate of item 3 is redundant whether or not it resembles items 1, 2 and
4, and averaging over ten asked questions would dilute that single collision into
nothing, which is exactly the case the term exists to catch.

It reuses the vectors Phase 2 already stores (`BAAI/bge-small-en-v1.5`, written
by `app/retrieval/indexing.py`) rather than building a second notion of
"similar". Nothing here embeds anything — no model is loaded and no network is
touched. A question with no stored vector contributes no similarity rather than
a fabricated one, and nothing asked yet means zero.

---

## 3. The hard constraints — and why they run first

Four rules are applied **before** any item is scored:

```sql
WHERE id NOT IN (asked_ids)                    -- never repeat
  AND topic_key   IN (topics_with_quota_left)  -- respect the blueprint
  AND ABS(difficulty_b - :theta_for_topic) <= 1.5
  AND time_estimate_s <= :time_remaining
```

**Why filters and not penalties.** A penalty is a number added to a score, and a
large enough bonus elsewhere will always outweigh it. "Never repeat a question"
implemented as `−0.5 × already_asked` is not *never*, it is *usually* — and the
one session where a high JD weight and a big coverage deficit outvote it is the
session a real candidate sees the same question twice. These four are **policy**,
not preference: an item that fails any of them must be *impossible* to select,
whatever the other terms say. The only way to get that guarantee is to remove it
from the pool.

Three more reasons the plan puts them in SQL:

- **The database is where the questions are.** Filtering there ships back the
  survivors instead of the bank. Scoring then touches tens of rows and never
  grows into a full-table load as the bank grows.
- **An index already fits.** `ix_questions_subtopic_key_difficulty_b` is exactly
  the `(subtopic, difficulty)` shape the difficulty window asks for.
- **It keeps scoring total.** `score_item` divides by `state.time_left` and reads
  `state.theta[...]`; it never has to ask "is this item even allowed?", because
  by then the question cannot arise.

### The four rules

**No repeats.** A question already asked in this session can never be selected
again. The rule is about *ids*: two different questions with identical wording
are not a repeat, they are a redundancy problem, and conflating the two would
turn a soft penalty into a hard filter.

**Per-topic quota.** Only topics with `asked < target` are eligible. At the
quota, or over it, the topic disappears from the pool entirely. This is what
stops one topic eating the interview; the coverage-deficit term is the gentle
version of the same idea, and the cap is the version that cannot be outvoted.

**The difficulty window, `|b − θ| ≤ 1.5`.** Inclusive: a gap of exactly 1.5 is
allowed, 1.51 is not. θ is the estimate for the item's **subtopic** — the level
the state is canonically stored at, and the same number `score_item` uses, so the
filter and the score are always talking about the same thing. A subtopic that has
never been measured falls back to the cold-start prior (θ = 0, RD = 1.30) rather
than being dropped.

This filter does double duty: items this far from ability carry almost no Fisher
information *and* they bore or demoralise the candidate. Measurement and user
experience agree here, which is rare and worth saying out loud.

**Time remaining.** `time_estimate_s <= time_remaining`, inclusive — a question
that uses the last 120 seconds is still askable. With no time left nothing with a
positive estimate fits, and the query is short-circuited without a round trip.

### SQL safety

Every value a caller supplies is a **bound parameter**. Nothing is formatted into
SQL text — not the asked ids, not the topic keys, not θ, not the time remaining,
not the limit. The per-subtopic θ becomes a `CASE questions.subtopic_key WHEN :k1
THEN :t1 … ELSE :prior END`, which needs no server-side object and keeps the
whole thing one statement. The query selects only the seven columns selection
reads — never the prose, the reference answer or the concept checklist.

A unit test compiles the statement against the PostgreSQL dialect and asserts
that a subtopic key of `caching'; DROP TABLE questions; --` appears in the
*parameters* and not in the SQL.

### The rule is written twice, on purpose

`constraints.py` carries the same four rules as a Python predicate
(`ineligibility_reason`) and as SQL. The Python one is the specification — it is
exercised exhaustively offline and it reports *which* constraint an item failed,
which is what you want at 2am when the pool came back empty. The SQL one is the
fast path over the real table.

Two implementations of one rule drift unless something checks them, so
`tests/integration/test_selection_sql.py` runs both over the real 60-question
bank in a real PostgreSQL and asserts they select the same rows — including
mid-interview, with asked ids, a partly-served quota, moved θs and a shrunken
clock. `select_next_item` also re-applies the Python predicate to whatever SQL
returned: one comparison per surviving row, it can only ever *remove* items, and
it makes the guarantee hold even if the SQL is one day edited wrongly.

---

## 4. ε-greedy — why the engine is deliberately a little random

**90% of the time take the highest-scoring item. 10% of the time sample
uniformly from the top 5.** That is `ε = 0.10`.

A pure argmax policy has three problems, and ε-greedy is the standard, boring,
name-it-and-move-on answer to all three:

1. **It is memorisable.** Two candidates with similar backgrounds would get the
   same questions in the same order, and the second one has a friend who took it
   yesterday.
2. **It never learns about the rest of the bank.** Argmax only ever asks items
   near θ, so the difficulty of everything else stays exactly as the author
   guessed it. Difficulty recalibration — fitting `b` to real observations, plan
   §5.11 — needs data gathered *off* the greedy policy to have anything to fit.
   This is the strongest of the three arguments.
3. **It cannot recover from its own mistakes.** If a mis-authored `b` makes one
   item look best under a slightly wrong θ, argmax keeps choosing it and never
   gathers the evidence that would fix it.

**Why the top 5 and not the whole bank.** The exploration set is the top of a
list that has *already* passed every hard constraint and been scored. The worst
thing exploration can do is ask the fifth-best allowed question — which bounds
the cost of exploring to something a candidate would not notice, and is what
makes 10% affordable. Sampling from the whole bank would not be exploration, it
would be a bug.

**Reproducibility.** The generator is a parameter (`rng: random.Random`),
defaulting to a module-level instance — the same injectable-source convention
`app/llm/router.py` uses for retry jitter. Passing `random.Random(7)` makes an
entire session reproducible, which is what lets the tests pin the exploitation
path, the exploration path and the top-5 restriction with no flaky assertions and
no monkeypatching of a global. Exactly one `random()` draw is made per selection,
before the branch, so a seeded sequence is predictable whichever way each
decision goes.

Ranking breaks ties by item id. That is not cosmetic: scores are floats built
from six terms, exact ties are common (two items in the same subtopic with the
same `b` and the same time estimate score identically), and without a total order
the outcome would depend on row order from the database.

---

## 5. The stopping rule

```
STOP when:  all required topics have RD < 0.40        # sufficient precision
        or  items_asked >= item_budget                # out of questions
        or  time_elapsed >= time_budget               # out of time
        or  3 consecutive items with |Δθ| < 0.05      # nothing new arriving
```

**Any** of them, not all — three are limits and one is an achievement, and an
interview ends at whichever comes first. `should_stop()` reports *every* reason
that fired, because "we ran out of time **and** we had already reached precision"
is a good interview while "we ran out of time and had not" is a truncated one,
and a report that can tell them apart is worth the extra tuple.

**1. Sufficient precision — `RD < 0.40` for every required topic.** This is the
point of an adaptive test. Once every topic the blueprint required is measured to
within about ±0.78 at 95%, more questions buy precision nobody needs. Topic RD is
not stored: it is derived from the subtopic states by Day 11's `roll_up()`, the
same precision-weighted aggregation a report uses, so the number that ends the
interview and the number that appears in the report cannot disagree. Note that
aggregation *shrinks* RD — three measured subtopics pin a topic better than any
one of them — so this can be satisfied without every individual subtopic reaching
0.40. Two guards: a required topic with no measured subtopic is **not** satisfied
(you cannot claim precision about something never asked), and an *empty* required
set returns `False` rather than a vacuous `True`, because a stopping rule that
fires before the first question is a bug, not an achievement.

**2. The item budget — `items_asked >= item_budget`.** Inclusive: an interview
that has asked its 12th of 12 items is finished, not one item short.

**3. The time budget — `time_elapsed >= time_budget`.** Likewise inclusive.

**4. Three consecutive items with `|Δθ| < 0.05`.** The interesting one: the
"we have learned everything this bank can tell us about this candidate" detector,
and what lets a strong candidate finish early instead of grinding through a fixed
twenty questions. It catches the case the RD condition misses — the estimate has
settled even though RD has not yet crossed its threshold. *Three* in a row rather
than one, because a single small step happens routinely when a prediction happens
to be right; a run of three is a pattern. The count is the **trailing** run, so
one substantial step anywhere resets it for free: `[0.01, 0.01, 0.40, 0.01]`
counts 1, not 3. The comparison is strict — a step of exactly 0.05 is not small.

All five thresholds are keyword arguments, so a caller or a test can vary one
without any of them being quietly redefined.

---

## 6. How Day 12 uses Day 11

```
        Day 11  ────────────────────────────────►  Day 12
  probability_correct(θ, b, a)   ─────────────►  the p inside every information term
  AbilityState.theta             ─────────────►  the difficulty window, and the p above
  AbilityState.rd                ─────────────►  (via roll_up) the precision stopping condition
  roll_up(subtopics, parent_of)  ─────────────►  topic-level RD, computed not stored
  RD_MAX                         ─────────────►  the cold-start prior for an unmeasured subtopic
```

Nothing in Day 11 was modified. In particular the selection layer does **not**
re-implement the sigmoid, does not keep its own θ, and does not store a topic-level
RD — it calls Day 11's functions, and where the answer is derivable it derives it.
`SelectionState` holds Day 11's `AbilityState` objects directly, keyed by
subtopic, and falls back to a documented prior (θ = 0, RD = 1.30) for a subtopic
nobody has been asked about yet.

One pleasant consequence of the same design, worth noting even though the
mechanism that enables it is deferred: a subtopic untested for a long time would
have an inflated RD, a high RD means a wide, uncertain estimate, and the
information term therefore favours re-testing it. Spaced repetition would fall
out of the mathematics with no scheduler and no separate feature. (Cross-session
RD inflation needs a clock and a `last_tested_at`; it is deferred with session
persistence.)

---

## 7. What is deliberately not here

Every item below is future-day work. None of it is implemented, and no interface
in `app/selection/` hides a partial version of it.

| deferred | plan section | day |
|---|---|---|
| the simulation harness — 200 synthetic candidates, the Beta response model, CAT vs random vs fixed-sequence | §8.6 | 13 |
| the convergence chart, items-to-reach-SE-0.35, coverage compliance, the `\|b−θ\|` histogram | §8.6 | 14 |
| the **weight ablation** that would justify the six weights empirically | §8.6 | 13 |
| the **blueprint builder** — topic quotas, item budget, time budget. Day 12 *consumes* a quota mapping; it does not build one | §3 | 15 |
| difficulty calibration — fitting `b` from observations. This is what ε-greedy's exploration data is *for* | §5.11 | later |
| the follow-up policy | §8.5 | later |
| grading, the event log, session persistence, the FSM | §7, §11 | later |
| resume parsing and resume-seeded θ priors — `ResumeProfile` is the interface, not the producer | §9.4 | Phase 5 |
| JD parsing — `jd_weights` is likewise consumed, not built | | Phase 5 |
| the relaxation ladder for an exhausted pool (widen the window to 2.0, then relax the topic constraint, then end the topic early and log why) | §8.7 | later |
| cross-session RD inflation | §9.5 | later |
| any HTTP endpoint. Nothing in `app/selection/` is exposed over the API | | later |
| deployment — days 1–29 are local only | | 30 |

An empty candidate pool is reported as an empty pool (`select_next_item` returns
`None`), never raised and never worked around. Deciding what to do about it — the
relaxation ladder above — belongs to the session orchestrator, which does not
exist yet.

---

## 8. Testing

231 tests, all of them fast:

| file | tests | what it pins |
|---|---:|---|
| `tests/unit/selection/test_information.py` | 41 | `I ≥ 0`, finite, maximised at p = 0.5, scales as `a²`, normalised form in `[0,1]` |
| `tests/unit/selection/test_objective.py` | 56 | every term alone, the exact weighted formula by hand, and the six "this can only help / only hurt" properties |
| `tests/unit/selection/test_constraints.py` | 40 | the four rules at, just inside and just outside each boundary; the compiled SQL; parameter binding |
| `tests/unit/selection/test_policy.py` | 28 | ranking, exploitation, exploration, the top-5 restriction, seeded reproducibility |
| `tests/unit/selection/test_stopping.py` | 42 | all four conditions alone and together, the reset, the strict thresholds |
| `tests/unit/selection/test_pipeline.py` | 11 | Day 11 + Day 12 over a six-turn simulated interview |
| `tests/integration/test_selection_sql.py` | 13 | SQL and Python select the same rows, over the real bank in a real PostgreSQL |

**On property tests.** Hypothesis is not a dependency of this project and none was
added. The property classes sweep deterministic `itertools.product` grids, in the
same style as `tests/unit/test_ability.py` — less coverage than a randomised
search, but they fail identically on every machine and in every CI run, which for
a numerical core is the better trade.

**On flakiness.** Not one test in this package depends on an unseeded random
draw. Every ε-greedy test either scripts the `random()` value explicitly or seeds
a real `random.Random`, including the one that counts how often exploration
happens over 1000 draws.
