# The blueprint — what assessment do we need?

*Phase 3, Day 15. The last piece of the adaptive engine, and the smallest.*

Days 11–14 built something that could measure a candidate, choose the next
question, decide when to stop, and prove — in a synthetic world — how well it
did all three. Every one of them took its **requirements** from somewhere else:
a hand-built test fixture, an `ExperimentConfig`, a literal in a docstring.

Day 12 said so itself, and had said so since it was written:

> `coverage_targets` — `topic_key -> how many items this topic must get`.
> Consumed, never built: the blueprint builder that produces it is plan §3,
> Day 15, and is **DEFERRED**.

This is that producer.

---

## 1. What the blueprint is

One immutable value that answers **"what assessment do we need?"** — and nothing
else.

```
Assessment inputs          role, level, duration, mode, JD weights
        ↓
   AssessmentSpec          the request; may be under-specified
        ↓
  build_blueprint          divide the budget; check it is possible
        ↓
    Blueprint              the contract; fully determined, validated, frozen
        ↓
  Adaptive selector        which question next?          (Day 12, unchanged)
        ↓
  Ability estimator        what do we now believe?       (Day 11, unchanged)
        ↓
  Stopping rule            should the assessment end?    (Day 12, unchanged)
```

| layer | question it answers |
|---|---|
| **Blueprint** (Day 15) | what assessment do we need? |
| **Selector** (Day 12) | which question should we ask next? |
| **Estimator** (Day 11) | what do we currently believe about the candidate? |
| **Stopping** (Day 12) | should the assessment end? |

**A blueprint never selects a question.** It never looks at an item, never scores
one, and never touches an ability estimate. That boundary is the reason the
package is four small modules rather than a second selection engine.

---

## 2. Why it exists

Three concrete reasons, in the order they bite.

**Somebody has to produce the six values Day 12 requires.** `SelectionState`
needs `coverage_targets`, `jd_weights`, `time_left_s` and `prior`; `should_stop`
needs `required_topics`, `parent_of`, `item_budget` and `time_budget_s`. Until
now every one of those came from a test fixture. A production session cannot.

**The requirements have to be validated together.** Each of those values is
individually plausible and the *combination* can still be impossible: quotas that
sum past the item budget, a 45-minute clock holding a 30-question budget, six
required topics and five questions. Nothing could catch those before, because
nothing held them in one place.

**A session's requirements must survive the session.** `interview_sessions.blueprint`
is a JSONB column, and `app/models/interview.py` has described it as *"computed
once at session creation by the blueprint builder"* since Day 2. An interview
resumed next week, replayed from its event log, or re-graded months later must be
judged against the blueprint it actually ran under — not against whatever the
calibration tables say today. So the blueprint serialises, and `from_dict`
re-validates on the way in.

---

## 3. The schema

### `AssessmentSpec` — the request

| field | type | default | meaning |
|---|---|---|---|
| `role` | `str` | — | the job this assessment is for. Recorded, never parsed. |
| `topics` | `tuple[TopicRequest, ...]` | — | what must be covered. At least one, at least one required. |
| `level` | `str` | `"mid"` | `junior` / `mid` / `senior` / `staff`. Sets the cold-start θ. |
| `mode` | `str` | `"standard"` | `screening` / `standard` / `deep_dive`. Sets the pace. |
| `duration_minutes` | `int` | `45` | 5–360. **Minutes**, and the ceiling exists to catch seconds. |
| `item_budget` | `int \| None` | `None` | `None` derives it from duration and mode. |
| `notes` | `str` | `""` | for whoever reads the stored blueprint. Never read by code. |

### `TopicRequest` — one area, and how much it matters

| field | type | default | meaning |
|---|---|---|---|
| `topic_key` | `str` | — | a taxonomy **topic** key. Quotas are stated at topic level. |
| `jd_weight` | `float` | `1.0` | role alignment in `[0, 1]`. |
| `subtopic_keys` | `tuple[str, ...]` | `()` | the sub-areas that count. Filled from the taxonomy when empty. |
| `required` | `bool` | `True` | required topics block the precision stopping rule. |
| `min_items` | `int` | `0` | minimum coverage floor. |
| `max_items` | `int \| None` | `None` | cap — plan §8.7's guard against topic starvation. |

### `Blueprint` — the contract

| field | type | meaning |
|---|---|---|
| `role`, `level`, `mode` | `str` | carried through from the spec. |
| `item_budget` | `int` | how many questions. Day 12's `item_budget`. |
| `time_budget_s` | `float` | the clock. Day 12's `time_budget_s` and the initial `time_left_s`. |
| `topics` | `tuple[TopicPlan, ...]` | one per topic, with its **resolved** quota. |
| `target_theta` | `float` | the ability aimed at, from `level`. Feeds `prior_ability()`. |
| `difficulty_window` | `float` | Day 12's `abs(b − θ) ≤ 1.5` hard filter, carried so the argument has one source. |
| `rd_target` | `float` | Day 12's `RD < 0.40` precision target. |
| `notes` | `str` | passed through. |
| `version` | `int` | the serialised shape. Refused if it is not this build's. |

`TopicPlan` is a resolved `TopicRequest`: `topic_key`, `quota` (a number now),
`jd_weight`, `required`, `subtopic_keys`.

### The two derived tables

**`jd_weight` does two jobs**, and it is worth separating them: it is the
proportion the item budget is divided by, **and** it is passed through unchanged
into `SelectionState.jd_weights`, where it is one of Day 12's six scored terms.
It is deliberately not renormalised on the way through — that would change the
score.

**`parent_of`** (`subtopic → topic`) is derived from the plans, never stored.
Day 11 keeps θ per *subtopic* and Day 12 quotas per *topic*, so something has to
join them; `roll_up` and `precision_reached` both require exactly this mapping.
Deriving it follows the rule `SelectionState` states for `asked_ids`: two sources
of truth drift.

---

## 4. The two calibrated numbers

Everything derived from `level` and `mode` is a table, so *"why 8 questions?"*
has an answer with a number in it.

### Pacing — `MODE_SECONDS_PER_ITEM`

Measured over all **60 committed bank items**, `time_estimate_s` has quartiles
**p25 = 292.5, p50 = 330, p75 = 390** (mean 324, range 150–420). Each mode takes
one of them, rounded to the 30-second granularity every authored estimate already
uses:

| mode | s/item | quartile | what it means |
|---|---|---|---|
| `screening` | 300 | p25 | the bank's quicker items — breadth first |
| `standard` | 330 | p50 | the median item |
| `deep_dive` | 390 | p75 | the slower items — fewer, harder |

```
item_budget = floor(duration_minutes × 60 ÷ seconds_per_item)
```

| duration | screening | standard | deep_dive |
|---|---|---|---|
| 30 min | 6 | 5 | 4 |
| 45 min | 9 | **8** | 6 |
| 60 min | 12 | 10 | 9 |
| 90 min | 18 | 16 | 13 |

Floor division, not rounding: budgeting for a question the clock cannot finish
promises the candidate something Day 12's time constraint will take away — after
it has already been counted in a quota.

### Seniority — `LEVEL_TARGET_THETA`

| level | target θ |
|---|---|
| `junior` | −0.5 |
| `mid` | **0.0** |
| `senior` | +0.5 |
| `staff` | +1.0 |

Used for exactly one thing: the cold-start prior.

```python
Blueprint.prior_ability() == AbilityState(theta=target_theta, rd=RD_MAX)
```

**`mid` is exactly 0.0, so a default blueprint reproduces Day 12's `PRIOR_ABILITY`
byte for byte** — a test pins it. Nothing about the selector's behaviour moves
unless a caller explicitly picks another level. That is why this is a level and
not a flag: the choice is visible in the request.

**The uncertainty is never reduced by the level.** RD stays at `RD_MAX`. A level
is a claim about where to start looking, not evidence about this person, and
claiming precision we do not have is what `RD_MAX` exists to prevent.

**These four numbers are reasoned, not measured.** They are half a difficulty
band apart, which is the smallest step that moves the window meaningfully without
asserting more about "senior" than anyone can defend. Validating them in
simulation is deferred — see §8.

---

## 5. Validation rules

All raise `BlueprintError` (a `ValueError`, matching `TaxonomyError`), and every
message names the offending field *and* its value. Validation happens in
`__post_init__`, so a frozen, invalid blueprint cannot exist — there is no window
in which one could be written to a database.

### On the request (`AssessmentSpec` / `TopicRequest`)

| # | rule | why |
|---|---|---|
| 1 | at least one topic | an empty blueprint makes every question ineligible and ends the session at turn zero |
| 2 | at least one **required** topic | with none, the precision stopping rule can never be satisfied and the assessment has no goal |
| 3 | `role` not blank | it is what the report is about |
| 4 | no duplicate `topic_key` | merge the entries rather than relying on which wins |
| 5 | no subtopic under two topics | a subtopic has exactly one parent; two make `parent_of` ambiguous |
| 6 | no subtopic listed twice under one topic | a duplicated requirement is a typo |
| 7 | no key used as both topic and subtopic | `topics.key` is one primary key across all three levels |
| 8 | `jd_weight ∈ [0, 1]`, finite | it is a scored term, not a free scalar |
| 9 | `min_items ≥ 0` | a negative floor is meaningless |
| 10 | `max_items > 0` when set | capping at zero is a confusing way to say "leave it out" — use `required=False` |
| 11 | `max_items ≥ min_items` | otherwise the two contradict |
| 12 | `level` and `mode` known | a typo must not silently select a default |
| 13 | `5 ≤ duration_minutes ≤ 360`, integer | the ceiling catches a duration passed in seconds; the message says so |
| 14 | `3 ≤ item_budget ≤ 200` when stated | the stopping rule needs three answers before it can fire at all |

### On the derivation (`build_blueprint`)

| # | rule | why |
|---|---|---|
| 15 | the duration must fit `MIN_ITEM_BUDGET` questions at the mode's pace | a 15-minute deep dive is one question |
| 16 | per-topic minimums must not exceed the item budget | otherwise a promised floor cannot be met |
| 17 | per-topic maximums must be able to absorb the item budget | otherwise `item_budget` does not mean what it says |
| 18 | every key must exist in the taxonomy, at the right level, under the right parent | *when a taxonomy is supplied* — a typo should fail here, not as an empty pool at question one |

### On the contract (`Blueprint`)

| # | rule | why |
|---|---|---|
| 19 | quotas must not exceed the item budget | a quota the session can never reach is a coverage requirement that silently fails |
| 20 | quotas must not all be zero | nothing would be asked |
| 21 | a **required** topic must have a non-zero quota | a required topic that can never be asked is the same silent failure |
| 22 | `item_budget × seconds_per_item ≤ time_budget_s` | the session would run out of clock before it ran out of questions |
| 23 | `time_budget_s`, `target_theta`, `difficulty_window`, `rd_target` finite and in range | a NaN here would poison every downstream comparison |
| 24 | `version` must be this build's | a stored blueprint from an older shape is a migration to do deliberately, not a `KeyError` in production |

The taxonomy check (18) is **optional** by design: the blueprint layer is usable
without loading the bank, and Day 13's simulation builds blueprints over a
synthetic taxonomy that is not in `data/question-bank/`. Requiring the file would
couple an experiment to a dataset it does not use.

---

## 6. How it plugs into the selector

Two methods, and neither is a new abstraction — every argument they produce was
already declared by Day 11 or Day 12.

```python
state = blueprint.selection_state(
    ability=ability,          # Day 11 states, keyed by subtopic
    asked=asked,              # what has been used
    time_left_s=time_left,    # defaults to the whole budget
    resume=resume,            # Day 12's ResumeProfile, untouched
)
selection = choose_next(state, bank, rng=rng,
                        difficulty_window=blueprint.difficulty_window)

decision = should_stop(**blueprint.stopping_arguments(
    ability=ability,
    items_asked=len(asked),
    time_elapsed_s=elapsed,
    recent_theta_deltas=deltas,
))
```

`stopping_arguments` returns a mapping to be splatted rather than calling
`should_stop` itself. That is deliberate: the blueprint supplies the
*requirements*, Day 12 makes the *decision*, and this package must never become a
second place where a stopping rule lives.

**`app.selection` does not import `app.blueprint`.** The arrow points one way, so
Day 12 stays usable and testable without Day 15 — and a test asserts the absence
of the cycle.

### What Day 12 got, without changing

| Day 12 argument | now produced by |
|---|---|
| `SelectionState.coverage_targets` | `blueprint.coverage_targets` |
| `SelectionState.jd_weights` | `blueprint.jd_weights` |
| `SelectionState.time_left_s` | `blueprint.time_budget_s` (at turn zero) |
| `SelectionState.prior` | `blueprint.prior_ability()` |
| `should_stop(required_topics=…)` | `blueprint.required_topics` |
| `should_stop(parent_of=…)` | `blueprint.parent_of` |
| `should_stop(item_budget=…)` | `blueprint.item_budget` |
| `should_stop(time_budget_s=…)` | `blueprint.time_budget_s` |
| `should_stop(rd_target=…)` | `blueprint.rd_target` |
| `choose_next(difficulty_window=…)` | `blueprint.difficulty_window` |

---

## 7. An example

```python
spec = AssessmentSpec(
    role="Backend Engineer",
    level="senior",
    mode="standard",
    duration_minutes=45,
    topics=(
        TopicRequest("dsa",       0.9, ("arrays", "graphs", "trees")),
        TopicRequest("databases", 0.7, ("indexing", "transactions")),
        TopicRequest("caching",   0.4, ("cache_invalidation",),
                     required=False, max_items=2),
    ),
)
blueprint = build_blueprint(spec, taxonomy=load_taxonomy())
```

```
role                 Backend Engineer
level                senior
mode                 standard
item budget          8                       45 min ÷ 330 s, floored
time budget          2700 s
topic quotas         dsa=4, databases=3, caching=1
required topics      databases, dsa          caching is optional
subtopics            6
target theta / RD    0.5 / 1.3               senior cold start
difficulty window    1.5                     Day 12's
rd target            0.4                     Day 12's
```

Stored as:

```json
{
  "version": 1,
  "role": "Backend Engineer",
  "level": "senior",
  "mode": "standard",
  "item_budget": 8,
  "time_budget_s": 2700.0,
  "target_theta": 0.5,
  "difficulty_window": 1.5,
  "rd_target": 0.4,
  "notes": "",
  "topics": [
    {"topic_key": "dsa", "quota": 4, "jd_weight": 0.9, "required": true,
     "subtopic_keys": ["arrays", "graphs", "trees"]},
    {"topic_key": "databases", "quota": 3, "jd_weight": 0.7, "required": true,
     "subtopic_keys": ["indexing", "transactions"]},
    {"topic_key": "caching", "quota": 1, "jd_weight": 0.4, "required": false,
     "subtopic_keys": ["cache_invalidation"]}
  ]
}
```

### How the quotas were reached

Budget 8. The two required topics take a floor of one each, leaving 6 to divide
by `0.9 : 0.7 : 0.4` (total 2.0):

```
dsa        6 × 0.45 = 2.70  →  2, remainder 0.70
databases  6 × 0.35 = 2.10  →  2, remainder 0.10
caching    6 × 0.20 = 1.20  →  1, remainder 0.20
```

Five allocated, one spare, and it goes to the largest remainder — `dsa`. Final:
`dsa = 1 + 3 = 4`, `databases = 1 + 2 = 3`, `caching = 0 + 1 = 1`. Sum = 8.

**Largest remainder, not repeated flooring**, for the reason it is used to
allocate parliamentary seats: flooring silently starves the smallest share and
rounding overspends the budget. Ties break by topic name, so two runs agree — a
property Day 13's reproducibility guarantee runs through.

---

## 8. The pacing finding

Deriving the item budget from the committed bank produced a number worth stating
plainly:

> **A 45-minute standard assessment fits 8 questions, not 20.**

Day 13's simulation ran a 20-item budget over a 2700-second clock — 135 s per
item. The committed bank's median item is **330 s**. The simulation was not wrong
to do that: it says so in `app/simulation/config.py`, which lists *"150–420 s per
item: twenty items is 110 minutes"* as one of five reasons the real bank cannot
serve as an experimental environment, and builds a synthetic bank averaging 120 s
instead.

But it means the two numbers are not comparable, and the honest version is:

- Day 13/14's conclusions are about a synthetic world with a 20-item budget;
- a real 45-minute interview on the real bank has **8 questions** to work with;
- Day 14 measured the adaptive/random crossover at question **9**.

**Adaptive was ahead for the whole length of a real 45-minute assessment.** That
is a suggestive alignment and nothing more — it compares a synthetic curve to a
real bank's pacing, two things measured in different worlds. It is recorded here
because it is the obvious next experiment, not because it is a result. Running the
simulation at realistic budgets is deferred (§9).

No constant was changed on the strength of it.

---

## 9. What is intentionally deferred

- **JD parsing and resume parsing.** A spec takes JD weights as numbers; deriving
  them from a job description is Phase 5. `ResumeProfile` remains Day 12's
  minimal interface, and resume-seeded per-subtopic priors (plan §9.4) are not
  built.
- **Persistence.** `to_dict` / `from_dict` produce and consume the JSONB shape.
  Writing the row, and the session FSM around it, is Phase 5.
- **Validating a blueprint against the *bank*.** `check_against_taxonomy` verifies
  the keys exist in the controlled vocabulary. Whether enough questions exist at
  the right difficulty to satisfy a quota is a database question, and what to do
  when they do not is plan §8.7's relaxation ladder — still deferred.
- **Simulating the level-seeded prior.** `LEVEL_TARGET_THETA` is reasoned, not
  measured. Day 13's harness could test whether starting a senior candidate at
  θ = +0.5 helps or hurts; that experiment has not been run.
- **Re-running the simulation at realistic budgets.** §8's finding is the reason
  to, and Day 13's environment is deliberately not the real bank, so this is a
  design question rather than a parameter change.
- **A blueprint API endpoint.** Nothing here is served over HTTP.

---

## 10. Where the code is

```
backend/app/blueprint/
    spec.py         AssessmentSpec, TopicRequest, the calibration tables, BlueprintError
    quotas.py       allocate_quotas — largest remainder with floors and caps
    blueprint.py    Blueprint, TopicPlan, validation, serialisation, the two adapters
    builder.py      build_blueprint — the one place a spec becomes a contract
```

```
backend/tests/unit/blueprint/
    test_quotas.py       the arithmetic, hand-worked; Day 13 compatibility
    test_builder.py      derivation, and every rejection
    test_blueprint.py    accessors, serialisation, construction-time validation
    test_integration.py  Blueprint → Selector → Ability → Stopping, end to end
```

`app.simulation.config.split_budget_by_jd` now delegates to
`app.blueprint.allocate_quotas`, so there is a single largest-remainder
implementation. A test pins that the unconstrained path returns exactly what
Day 13's own implementation returned, over 50 budget/weight combinations, and
Day 13's committed report regenerates byte-identically.
