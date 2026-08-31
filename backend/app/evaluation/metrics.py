"""Retrieval quality metrics, from first principles.

**What these measure.**  A retriever returns a *ranked list*.  These functions
answer three different questions about that list:

* ``recall_at_k``  - did we find the right questions at all, within the top K?
* ``reciprocal_rank`` - how near the top was the first right one?
* ``ndcg_at_k``   - how good is the whole ordering, when some results are more
  relevant than others?

They are separate because they disagree, and the disagreement is informative. A
system can find every relevant question (perfect recall) while burying the best
one at rank 9 (poor MRR).

**Pure functions, no I/O.**  No database, no model, no configuration.  Lists of
ids in, numbers out.  That is what makes them testable against hand-worked
examples, and it is why they live here rather than inside the evaluation script.

**Relevance grades.**  Labels are graded 0-2 (plan section 12.1):

    2 = directly answers the query - the question a person meant to find
    1 = genuinely related, a reasonable thing to return, but not the target
    0 = not relevant (never written down; absence means 0)

``recall_at_k`` and ``reciprocal_rank`` need a yes/no answer, so they take a
*set* of relevant ids - the caller decides where to cut, and
``EvalQuery.relevant_ids`` cuts at grade >= 1.  ``ndcg_at_k`` uses the grades
themselves, which is the whole reason for having them.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

#: The highest label a query may carry. Grades outside 0..MAX_GRADE are rejected
#: rather than clamped: a 5 in a dataset labelled 0-2 is a mistake, not a strong
#: opinion, and silently clamping it would hide the typo.
MAX_GRADE = 2


class EmptyRelevantSet(ValueError):
    """A query with no relevant questions cannot be scored.

    Recall would divide by zero and MRR would have nothing to look for. This is
    a *dataset* error, not a retrieval result: a query nobody can answer teaches
    nothing about the retriever. ``dataset.py`` rejects such queries at load, and
    these functions refuse them again rather than inventing a convention.
    """


def _check(relevant: object, k: int) -> None:
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")
    if not relevant:
        raise EmptyRelevantSet("no relevant ids: this query cannot be scored")


def recall_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    """Fraction of the relevant questions that appear in the top ``k``.

        recall@k = |{top k retrieved} INTERSECT relevant| / |relevant|

    With one relevant question this is the simple yes/no the name suggests -
    1.0 if it is in the top k, 0.0 if not. With three relevant questions and two
    of them in the top k, it is 0.67.

    Worked example::

        retrieved = [B, A, C]   relevant = {A}
        recall@3 = 1.0   (A is in the top 3)
        recall@1 = 0.0   (only B is in the top 1)

    Duplicate ids in ``retrieved`` are counted once: recall asks *which* were
    found, not how many times. A retriever returning the same id twice is a bug,
    but it must not be able to inflate its own score.

    Never exceeds 1.0, even when ``k`` is larger than the retrieved list.
    """
    _check(relevant, k)
    found = set(retrieved[:k]) & relevant
    return len(found) / len(relevant)


def first_relevant_rank(retrieved: Sequence[str], relevant: set[str]) -> int | None:
    """1-based position of the first relevant id, or None if there is none.

    Split out from ``reciprocal_rank`` because the raw rank is what a human
    reads in a per-query report: "the right answer was 7th" is legible in a way
    that "0.1429" is not.
    """
    if not relevant:
        raise EmptyRelevantSet("no relevant ids: this query cannot be scored")
    for position, item in enumerate(retrieved, start=1):
        if item in relevant:
            return position
    return None


def reciprocal_rank(retrieved: Sequence[str], relevant: set[str], k: int | None = None) -> float:
    """1 / (rank of the first relevant result), or 0.0 if none is found.

        rank 1  -> 1.0        rank 4  -> 0.25
        rank 2  -> 0.5        rank 10 -> 0.1
        rank 3  -> 0.333      none    -> 0.0

    Averaged over queries this is **MRR**, Mean Reciprocal Rank. It rewards
    putting *a* relevant result at the very top and cares about nothing else -
    moving the first hit from rank 2 to rank 1 gains 0.5, while moving it from
    rank 9 to rank 8 gains 0.014. That is the right shape for a system where a
    person looks at the first result and rarely scrolls.

    It deliberately ignores every relevant result after the first, which is
    exactly what ``ndcg_at_k`` is for.

    ``k`` truncates the list first, so a hit at rank 12 scores 0.0 for k=10.
    Passing ``k`` matters when comparing retrievers that returned lists of
    different lengths - otherwise the one that returned more results can score
    higher purely for having been asked for more.
    """
    if k is not None and k < 1:
        raise ValueError(f"k must be at least 1, got {k}")
    window = retrieved if k is None else retrieved[:k]
    rank = first_relevant_rank(window, relevant)
    return 0.0 if rank is None else 1.0 / rank


def dcg_at_k(retrieved: Sequence[str], grades: Mapping[str, int], k: int) -> float:
    """Discounted Cumulative Gain: total usefulness of the top ``k``, discounted by position.

    Two ideas, combined:

    * **Gain** - a more relevant result is worth more. Using the standard
      exponential form, ``gain = 2**grade - 1``, so grade 0 -> 0, grade 1 -> 1,
      grade 2 -> 3. A perfect answer is worth three partial ones, which is the
      point of grading at all.
    * **Discount** - a result further down is worth less, divided by
      ``log2(rank + 1)``: rank 1 keeps 100% of its gain, rank 2 keeps 63%,
      rank 3 keeps 50%, rank 10 keeps 29%. Logarithmic rather than linear
      because the difference between rank 1 and 2 matters far more to a reader
      than the difference between rank 19 and 20.

    Worked example, ``grades = {A: 2, B: 1}``, ``retrieved = [B, A, C]``::

        rank 1  B  gain 1  / log2(2) = 1/1.000 = 1.000
        rank 2  A  gain 3  / log2(3) = 3/1.585 = 1.893
        rank 3  C  gain 0  / log2(4) = 0
        DCG@3 = 2.893

    DCG on its own is not comparable between queries - a query with four
    relevant questions can score higher than one with a single relevant question
    no matter how well the second is served. ``ndcg_at_k`` fixes that.
    """
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")
    total = 0.0
    for position, item in enumerate(retrieved[:k], start=1):
        grade = grades.get(item, 0)
        if grade:
            total += (2**grade - 1) / math.log2(position + 1)
    return total


def ideal_dcg_at_k(grades: Mapping[str, int], k: int) -> float:
    """The DCG of the best ordering possible for these labels.

    Sort every graded question by grade, best first, and score that. This is the
    ceiling: no retriever can beat it, because no ordering puts more gain
    earlier. It is what turns DCG into a 0-1 number.
    """
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")
    best = sorted((g for g in grades.values() if g > 0), reverse=True)
    total = 0.0
    for position, grade in enumerate(best[:k], start=1):
        total += (2**grade - 1) / math.log2(position + 1)
    return total


def ndcg_at_k(retrieved: Sequence[str], grades: Mapping[str, int], k: int) -> float:
    """Normalised DCG: ``DCG@k / ideal DCG@k``. Between 0.0 and 1.0.

    **Why it exists.** Recall asks "did we find them"; MRR asks "was one of them
    first". Neither can say *how well ordered* a list of several relevant
    results is. Given one perfect answer and two partial ones, nDCG is the
    metric that rewards putting the perfect one first and prefers a partial hit
    at rank 2 over the same hit at rank 8.

    **Interpretation.** 1.0 means the ordering is as good as the labels allow -
    not that it is "100% accurate". 0.0 means nothing relevant was in the top k.
    A value between them is a fraction of the achievable ideal, so it is
    comparable across queries with different numbers of relevant questions,
    which raw DCG is not.

    Returns 0.0 when the ideal is 0 - which can only happen if every grade is 0,
    and ``_check`` has already rejected that.
    """
    _check({key for key, grade in grades.items() if grade > 0}, k)
    ideal = ideal_dcg_at_k(grades, k)
    if ideal == 0.0:  # pragma: no cover - unreachable after _check
        return 0.0
    return dcg_at_k(retrieved, grades, k) / ideal


__all__ = [
    "MAX_GRADE",
    "EmptyRelevantSet",
    "dcg_at_k",
    "first_relevant_rank",
    "ideal_dcg_at_k",
    "ndcg_at_k",
    "recall_at_k",
    "reciprocal_rank",
]
