"""Stage 3 - ranking the scored survivors and picking one, epsilon-greedily.

Plan section 8.3, step 3.  The three stages are kept as three functions on
purpose::

    eligible_items(...)          stage 1  hard constraints, in SQL
            |
    score_item(...)              stage 2  the weighted objective
            |
    rank_items(...)              stage 3a deterministic ordering
            |
    epsilon_greedy_select(...)   stage 3b exploit, or explore

Collapsing them into one ``pick_a_question()`` would be shorter and would cost
the four things that matter most here: you could not test the constraint layer
without the scoring layer, you could not explain a choice after the fact, you
could not push the filter into SQL, and you could not reproduce a session.

Why explore at all
------------------

A pure argmax policy is a policy with three problems, and epsilon-greedy is the
standard, boring, name-it-and-move-on answer to all three:

1. **It is memorisable.**  Two candidates with the same background would get the
   same questions in the same order, and the second one has a friend who took it
   yesterday.
2. **It never learns about the rest of the bank.**  Argmax only ever asks items
   near theta, so the difficulty estimates of everything else stay exactly as
   the author guessed them.  Difficulty recalibration (plan section 5.11) needs
   observations *off* the greedy policy to have anything to fit.  That is the
   off-policy-data argument, and it is the strongest of the three.
3. **It cannot recover from its own mistakes.**  If a mis-authored ``b`` makes
   one item look best under a slightly wrong theta, argmax will keep choosing
   it and never gather the evidence that would fix it.

``epsilon = 0.10``: 90% of the time take the top-scoring item; 10% of the time
sample **uniformly from the top 5**.  Not from the whole bank - the exploration
set is the top of a list that has already passed every hard constraint and been
scored, so the worst thing exploration can do is ask the fifth-best allowed
question.  That bounds the cost of exploring to something a candidate would not
notice, which is what makes 10% affordable.

Randomness that a test can pin down
-----------------------------------

The generator is a parameter, defaulting to a module-level
:class:`random.Random`.  Passing ``random.Random(7)`` makes the whole pipeline
reproducible, which is what lets the tests assert the exploitation path, the
exploration path and the top-5 restriction without a single flaky assertion and
without monkeypatching a global.  It is the same injectable-source convention
``app.llm.router`` already uses for its retry jitter.

Exactly one ``random()`` draw is made per selection, before the branch, so a
seeded sequence is predictable regardless of which way the branch goes.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession

from app.selection.constraints import (
    DEFAULT_CANDIDATE_LIMIT,
    DIFFICULTY_WINDOW,
    eligible_items,
    filter_eligible,
)
from app.selection.objective import (
    DEFAULT_WEIGHTS,
    ObjectiveWeights,
    ScoreBreakdown,
    score_items,
)
from app.selection.state import CandidateItem, SelectionState

#: Plan section 8.3.  Explore one selection in ten.
EPSILON = 0.10

#: How many of the ranked survivors exploration may choose between.
EXPLORATION_POOL_SIZE = 5

#: Not a security decision - it decides which of five acceptable questions to
#: ask - so the standard generator is the right tool, exactly as in
#: ``app.llm.router``'s retry jitter.
_DEFAULT_RNG = random.Random()  # noqa: S311


@dataclass(frozen=True, slots=True)
class Selection:
    """The chosen item and the evidence for why it was chosen.

    Carries what plan section 8.7 requires in the event log - pool size, the
    top five with their scores, the chosen id, and (inside ``chosen``) theta's
    consequences and the information term.  Writing that log is the session
    orchestrator's job and does not exist yet; producing the record it will
    write is this layer's.
    """

    chosen: ScoreBreakdown
    #: True when the exploration branch was taken.  It reports the *branch*, not
    #: whether the outcome differed - with a single candidate in the pool, both
    #: branches choose the same item.
    explored: bool
    #: How many items survived the hard constraints and were scored.
    pool_size: int
    #: The ranked exploration pool: the top ``EXPLORATION_POOL_SIZE``, or fewer.
    top: tuple[ScoreBreakdown, ...]

    @property
    def item(self) -> CandidateItem:
        return self.chosen.item

    @property
    def item_id(self) -> str:
        return self.chosen.item.id


def rank_items(scored: Sequence[ScoreBreakdown]) -> list[ScoreBreakdown]:
    """Highest score first, ties broken by item id.

    The tie-break is not cosmetic.  Scores are floats built from six terms, and
    exact ties are common in practice - two items in the same subtopic with the
    same ``b`` and the same time estimate score identically to the last bit.
    Without a total order, ``sorted`` would fall back on input order, which
    comes from the database, which is only stable because stage 1 asks for
    ``ORDER BY id``.  Making the rule explicit here means a seeded run is
    reproducible even if that ever changes.
    """
    return sorted(scored, key=lambda entry: (-entry.total, entry.item.id))


def epsilon_greedy_select(
    ranked: Sequence[ScoreBreakdown],
    *,
    rng: random.Random | None = None,
    epsilon: float = EPSILON,
    pool_size: int = EXPLORATION_POOL_SIZE,
) -> Selection | None:
    """Take the argmax, or - with probability ``epsilon`` - sample the top ``pool_size``.

    ``ranked`` must already be ordered by :func:`rank_items`; this function does
    not re-sort, so that "the top 5" means the same thing to it as to the caller
    that logged them.

    Returns ``None`` for an empty list.  That is the pool-exhaustion case of
    plan section 8.7 - no item satisfied the hard constraints - and it is
    reported rather than raised, because "no question fits" is a decision the
    session has to make (relax the window, end the topic), not a crash.
    """
    if not 0.0 <= epsilon <= 1.0:
        raise ValueError(f"epsilon must be in [0, 1], got {epsilon!r}")
    if pool_size < 1:
        raise ValueError(f"pool_size must be at least 1, got {pool_size!r}")
    if not ranked:
        return None

    generator = _DEFAULT_RNG if rng is None else rng
    pool = tuple(ranked[:pool_size])

    # One draw, always, before the branch: a seeded generator then produces the
    # same decision sequence whichever way each decision goes.
    explore = generator.random() < epsilon
    chosen = generator.choice(pool) if explore else pool[0]

    return Selection(chosen=chosen, explored=explore, pool_size=len(ranked), top=pool)


def choose_next(
    state: SelectionState,
    items: Sequence[CandidateItem],
    *,
    rng: random.Random | None = None,
    epsilon: float = EPSILON,
    pool_size: int = EXPLORATION_POOL_SIZE,
    difficulty_window: float = DIFFICULTY_WINDOW,
    weights: ObjectiveWeights = DEFAULT_WEIGHTS,
) -> Selection | None:
    """The whole policy over an in-memory candidate list: filter, score, rank, pick.

    Pure - no database, no clock - so a simulation or a test drives the complete
    decision path in microseconds.  The stages are called, not inlined; this
    function is composition and nothing else, and it is the only place that
    knows the order they go in.

    ``weights`` defaults to the production objective; it is threaded through for
    the Day 13 ablation and changes nothing when omitted.
    """
    eligible = filter_eligible(items, state, difficulty_window=difficulty_window)
    ranked = rank_items(score_items(eligible, state, weights=weights))
    return epsilon_greedy_select(ranked, rng=rng, epsilon=epsilon, pool_size=pool_size)


async def select_next_item(
    session: AsyncSession,
    state: SelectionState,
    *,
    rng: random.Random | None = None,
    epsilon: float = EPSILON,
    pool_size: int = EXPLORATION_POOL_SIZE,
    difficulty_window: float = DIFFICULTY_WINDOW,
    limit: int = DEFAULT_CANDIDATE_LIMIT,
    weights: ObjectiveWeights = DEFAULT_WEIGHTS,
) -> Selection | None:
    """The same policy, with stage 1 executed in the database.

    The hard constraints are applied **in SQL**, so what comes back over the
    wire is already the eligible pool and scoring never sees the rest of the
    bank.  :func:`choose_next` then re-applies the same predicate in Python.
    That second pass is not redundancy for its own sake: it costs one comparison
    per surviving row, it makes the guarantee "an ineligible item cannot be
    selected" hold even if the SQL is one day edited wrongly, and it can only
    ever remove items - so it fails safe.  ``tests/integration`` asserts that on
    a real bank it removes nothing, which is how the two expressions of the rule
    are kept honest.
    """
    candidates = await eligible_items(
        session, state, difficulty_window=difficulty_window, limit=limit
    )
    return choose_next(
        state,
        candidates,
        rng=rng,
        epsilon=epsilon,
        pool_size=pool_size,
        difficulty_window=difficulty_window,
        weights=weights,
    )


__all__ = [
    "EPSILON",
    "EXPLORATION_POOL_SIZE",
    "Selection",
    "choose_next",
    "epsilon_greedy_select",
    "rank_items",
    "select_next_item",
]
