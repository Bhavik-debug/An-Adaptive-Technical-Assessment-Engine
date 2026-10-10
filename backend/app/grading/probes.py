"""The four bias probes - Day 19, plan section 12.1.

| Probe | Method | Assertion |
|---|---|---|
| Length bias | Append 200 words of on-topic but contentless filler | `|dscore| < 0.05` |
| Confidence bias | Prepend *"I'm certain that..."* | `|dscore| < 0.03` |
| Jargon bias | Replace correct terms with buzzwords, no added substance | Score **drops** |
| Position bias | Shuffle the order of `expected_concepts` | Score changes by **exactly 0** |

**This module holds only the perturbations, and they are pure.** Building the
variant text is separated from grading it for one reason: *the probe is only as
good as the filler*. A length probe whose 200 words accidentally mention cache
invalidation is not measuring length bias, it is measuring whether the grader
noticed a new concept - and it would fail, correctly, while looking like a bias
finding. So the filler is a committed constant with tests asserting it contains
none of the bank's concept keys, and :func:`filler_problems` re-checks that
against the specific question being probed at run time.

The position probe needs no text transform at all and no LLM call: scoring is
code-side and indexes concepts by key, so shuffling the list must change the
score by exactly zero. Plan section 12.1 calls it "a test that proves your
architecture, not just your prompt", and it is the one probe that can be
asserted rather than measured.
"""

from __future__ import annotations

import random
import re
from collections.abc import Sequence
from typing import Protocol, TypeVar

from app.bank.schema import ExpectedConcept
from app.grading.schema import GradeJudgment

#: ``shuffled_concepts`` is order-only, so it is generic in the element type:
#: it permutes expected concepts in the probe and plain strings in the tests.
_T = TypeVar("_T")

#: Plan section 12.1: "200 words of on-topic but contentless filler".
#:
#: **What makes this filler and not content.** Every sentence is about the
#: *activity* of answering - process, caveats, how one might think about a
#: problem - and none of it names a mechanism, a technology, a trade-off, a
#: complexity class or a failure mode. It is the padding a nervous candidate
#: produces under time pressure, which is exactly the thing a grader must not
#: reward.
#:
#: It deliberately contains no technical nouns at all. An earlier draft said
#: "whether to cache" and had to be rewritten: ``cache`` is a concept key in
#: this bank, so that draft would have made the length probe a concept probe.
#: :func:`filler_problems` is the check that stops that happening again.
FILLER_200_WORDS = """
It is worth stepping back for a moment before going further, because questions
like this one reward a little care in how the answer is framed rather than a
rush to the first thing that comes to mind. In my experience the useful habit is
to say plainly what is being asked, then to work through it in the order that
makes the reasoning easiest to follow, and to flag anywhere the answer would
change if the circumstances were different. I have found that people who do this
tend to communicate more clearly under pressure, and that reviewers find the
result easier to check afterwards, which matters more than it might seem. There
is always a temptation to say more than is needed, and I am conscious of that
here, so I will try to keep the remainder focused on what was actually asked.
Different teams I have worked with have had different conventions about how much
detail to give in an answer like this, and I do not think there is one correct
amount; it depends a great deal on who is going to read it. I mention this
mainly to be transparent about my approach rather than because it changes the
substance of what follows.
""".strip()

#: The filler's exact length, asserted by a test. Plan section 12.1 specifies
#: "200 words", and a probe whose padding quietly drifted to 300 would be a
#: stricter test than the gate it reports against - so the count is pinned
#: rather than eyeballed.
FILLER_TARGET_WORDS = 200
FILLER_WORD_TOLERANCE = 15

#: Plan section 12.1: 'Prepend "I'm certain that..."'.
#:
#: Confidence with no added substance. The grader must not pay for assertiveness
#: - and this project's own dataset is the argument for the probe: Day 18 found
#: that the ``wrong_confident`` level mostly produced *correct* prose, so the one
#: thing already known about this generator is that it writes confidently.
CONFIDENCE_PREFIX = (
    "I'm certain that the following is correct, and I have no doubt about any " "part of it. "
)

#: Plan section 12.1: "Replace correct terms with buzzwords, no added substance".
#:
#: Each substitution replaces a precise technical term with a vaguer, more
#: fashionable phrase that says strictly less. The assertion is that the score
#: **drops**: the answer now names fewer mechanisms, so a grader driven by
#: concept coverage should notice, and one driven by how impressive the prose
#: sounds should not.
#:
#: Ordered longest-first at substitution time so that a shorter key cannot
#: consume part of a longer phrase.
JARGON_SUBSTITUTIONS: dict[str, str] = {
    "write-ahead log": "best-practice durability layer",
    "exponential backoff": "intelligent retry strategy",
    "request coalescing": "smart traffic shaping",
    "materialized view": "optimised data layer",
    "select for update": "enterprise-grade locking",
    "optimistic concurrency control": "modern concurrency approach",
    "pessimistic locking": "traditional locking approach",
    "lost update": "data integrity concern",
    "non-repeatable read": "consistency anomaly",
    "cache stampede": "traffic spike scenario",
    "dead-letter queue": "failure-handling mechanism",
    "object storage": "cloud-native storage solution",
    "binary search": "efficient lookup approach",
    "dynamic programming": "advanced algorithmic technique",
    "bellman-ford": "a more robust algorithm",
    "idempotent": "resilient",
    "atomicity": "transactional integrity",
    "durability": "persistence guarantees",
    "isolation": "concurrency safety",
    "fsync": "low-level disk operation",
    "stack": "appropriate data structure",
    "queue": "appropriate data structure",
    "b-tree": "scalable index structure",
    "o(n)": "efficient complexity",
    "o(n^2)": "less efficient complexity",
    "quadratic": "suboptimal",
    "amortised": "averaged out",
    "amortized": "averaged out",
}


def pad_with_filler(answer: str, *, filler: str = FILLER_200_WORDS) -> str:
    """The length-bias variant: the same answer plus contentless padding.

    Appended rather than prepended so the answer's own opening - which is what a
    reader anchors on - is unchanged, isolating length from first impression.
    The confidence probe tests the prepend direction separately.
    """
    return f"{answer.rstrip()}\n\n{filler}"


def prepend_confidence(answer: str, *, prefix: str = CONFIDENCE_PREFIX) -> str:
    """The confidence-bias variant: the same claims, asserted harder."""
    return f"{prefix}{answer.lstrip()}"


def jargonise(answer: str, *, substitutions: dict[str, str] | None = None) -> tuple[str, int]:
    """The jargon-bias variant: precise terms replaced by vaguer buzzwords.

    Returns the text and how many substitutions were made, because a variant
    with zero substitutions is not a jargon variant and must be excluded from
    the probe rather than counted as a pass. Matching is case-insensitive and
    whole-word.

    **Substitution happens in a single pass, and that is a correctness
    requirement rather than an optimisation.** Applying the table term by term
    lets a later key match text an earlier replacement just inserted: with
    ``write-ahead log -> best-practice durability layer`` applied first, the
    ``durability`` key then fires on the *replacement*, yielding "best-practice
    persistence guarantees layer" and a substitution count of 2 for one term in
    the original. The probe would then be comparing against text no
    substitution table describes. One alternation pass over the original string
    makes every replacement final.

    Longest-first alternation, so ``optimistic concurrency control`` wins over
    any shorter key inside it.
    """
    table = JARGON_SUBSTITUTIONS if substitutions is None else substitutions
    if not table:
        return answer, 0
    terms = sorted(table, key=len, reverse=True)
    pattern = re.compile(
        r"(?<!\w)(" + "|".join(re.escape(term) for term in terms) + r")(?!\w)",
        re.IGNORECASE,
    )
    lookup = {term.lower(): replacement for term, replacement in table.items()}
    count = 0

    def _replace(match: re.Match[str]) -> str:
        nonlocal count
        count += 1
        return lookup[match.group(0).lower()]

    return pattern.sub(_replace, answer), count


def word_count(text: str) -> int:
    return len(text.split())


def filler_problems(
    filler: str,
    *,
    concept_keys: Sequence[str],
    anchor_terms: Sequence[str] = (),
) -> list[str]:
    """Ways the filler could invalidate the length probe. Empty list means usable.

    **This is the probe's validity check, not a style check.** Plan section
    12.1 specifies *contentless* filler: if the padding mentions a concept the
    question expects, the grader may legitimately raise its coverage judgement,
    the score moves, and the probe reports length bias that is not there. Run
    per question, because what counts as content depends on which concepts that
    question expects.
    """
    problems: list[str] = []
    lowered = filler.lower()
    for key in concept_keys:
        for form in (key.lower(), key.replace("_", " ").lower()):
            if re.search(rf"(?<!\w){re.escape(form)}(?!\w)", lowered):
                problems.append(f"filler mentions expected concept {key!r} (as {form!r})")
                break
    for term in anchor_terms:
        if re.search(rf"(?<!\w){re.escape(term.lower())}(?!\w)", lowered):
            problems.append(f"filler mentions anchor term {term!r}")
    return problems


class _Scored(Protocol):
    """Anything with a ``score``. Narrow on purpose - see :func:`position_invariance`."""

    @property
    def score(self) -> float: ...


class _Scorer(Protocol):
    """The shape of ``scoring.score_answer``, without importing it."""

    def __call__(
        self, judgment: GradeJudgment, /, *, expected_concepts: Sequence[ExpectedConcept]
    ) -> _Scored: ...


def position_invariance(
    judgment: GradeJudgment,
    concepts: Sequence[ExpectedConcept],
    *,
    score_fn: _Scorer,
    trials: int = 8,
    seed: int = 1219,
) -> float:
    """Largest absolute score change over ``trials`` shuffles of ``concepts``.

    Plan section 12.1's position probe, and the assertion is ``== exactly 0``.
    Plan's words: *"The position-bias test is special: it must pass by
    construction, because scoring happens in code. It is a test that proves your
    architecture, not just your prompt."*

    **The model is deliberately not re-called.** The claim under test is that the
    *code-side arithmetic* does not depend on the order of the concept list -
    ``scoring.py`` indexes by concept key and sums over a mapping, and every
    product is a weight times one of 1.0/0.5/0.0, which is order-free in IEEE-754
    as well as in exact arithmetic. Re-asking the model with a shuffled list
    would measure the model's own nondeterminism and could never return exactly
    zero, which would turn a passing architectural guarantee into a failing
    empirical one.

    ``score_fn`` is injected rather than imported, so this module keeps no
    dependency on the scorer it is probing; the caller supplies
    ``scoring.score_answer``. The :class:`_Scorer` protocol is how that stays
    type-checked without the import.
    """
    baseline = score_fn(judgment, expected_concepts=concepts).score
    worst = 0.0
    for trial in range(trials):
        permuted = shuffled_concepts(concepts, seed=seed + trial)
        worst = max(worst, abs(score_fn(judgment, expected_concepts=permuted).score - baseline))
    return worst


def shuffled_concepts(concepts: Sequence[_T], *, seed: int) -> list[_T]:
    """The position-bias variant: the same concepts in a different order.

    Seeded so the exact permutation a result came from is recoverable, and
    guaranteed to be a real permutation: with two or more distinct concepts this
    retries until the order actually changed, because "shuffling changed
    nothing" would make the probe vacuously pass.
    """
    items: list[_T] = list(concepts)
    if len(items) < 2:
        return items
    rng = random.Random(seed)
    for _ in range(50):
        candidate = items[:]
        rng.shuffle(candidate)
        if candidate != items:
            return candidate
    return list(reversed(items))  # pragma: no cover - unreachable for n >= 2


__all__ = [
    "CONFIDENCE_PREFIX",
    "FILLER_200_WORDS",
    "FILLER_TARGET_WORDS",
    "FILLER_WORD_TOLERANCE",
    "JARGON_SUBSTITUTIONS",
    "filler_problems",
    "jargonise",
    "pad_with_filler",
    "position_invariance",
    "prepend_confidence",
    "shuffled_concepts",
    "word_count",
]
