"""Reciprocal Rank Fusion: combining two ranked lists into one.

**The problem.**  Vector search and lexical search both return "the best
questions for this query", ordered - but their scores are not comparable.  A
cosine similarity of 0.78 and a ``ts_rank_cd`` of 0.043 are numbers on unrelated
scales, and averaging them, or normalising them into [0, 1] first, means
inventing a conversion rate between two things that have none.  Worse, the
normalisation would depend on which documents happened to be returned, so adding
one document could reorder the others.

**The trick.**  Throw the scores away and keep only the *ranks*.  A document's
contribution from each list is ``1 / (k + rank)``, and its final score is the sum
over the lists it appeared in.  Ranks are 1-based and comparable across any two
retrievers by construction, so nothing has to be calibrated.

    score(d) = sum over each ranked list L containing d of  1 / (k + rank_L(d))

**A worked example**, with k = 60:

    vector : A B C D          lexical: C A E F
    A: 1/61 + 1/62 = 0.0325   <- top of one list, second of the other
    C: 1/63 + 1/61 = 0.0323   <- third and first
    B: 1/62         = 0.0161
    E: 1/62         = 0.0161  <- same score as B; the tie-break decides
    D: 1/63         = 0.0159
    F: 1/63         = 0.0159

Appearing in *both* lists is worth far more than being first in one - which is
the behaviour wanted, because agreement between two unrelated retrievers is real
evidence.

**Why k = 60.**  It is the value from Cormack, Clarke and Buettcher (2009), the
paper the method comes from, and the near-universal default since.  Its job is
damping: with k = 0 the top result would score 1.0 and the second 0.5, so
whichever list ranked something first would dominate everything else.  At 60 the
gap between rank 1 and rank 2 is about 1.6%, so a document needs *support* to
win rather than one lucky first place.  It is configurable
(``RETRIEVAL_RRF_K``); tuning it belongs to Day 10's evaluation, which can
measure the effect, and not to a guess made here.

**No candidate is discarded by fusion.**  The output contains the union of every
input list.  Truncation to the final K happens afterwards, in ``search.py``, and
is reported there.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

#: Cormack et al. (2009). See the module docstring.
DEFAULT_RRF_K = 60.0


@dataclass(frozen=True, slots=True)
class FusedItem:
    """One document's fused result, with the evidence that produced it."""

    key: str
    score: float
    #: source name -> 1-based rank in that source's list. Missing key means the
    #: source did not return this document at all.
    ranks: Mapping[str, int]

    @property
    def best_rank(self) -> int:
        return min(self.ranks.values())

    @property
    def sources(self) -> tuple[str, ...]:
        """Which retrievers found this, in a stable order."""
        return tuple(sorted(self.ranks))


def reciprocal_rank_fusion(
    rankings: Mapping[str, Sequence[str]],
    *,
    k: float = DEFAULT_RRF_K,
) -> list[FusedItem]:
    """Fuse named ranked lists of ids into one ranked list.

    ``rankings`` maps a source name (``"vector"``, ``"lexical"``) to that
    source's ids, best first. Sources may overlap, may be empty, and need not be
    the same length.

    **Ordering is total and deterministic**, which matters because a test that
    asserts on a ranking is worthless if ties resolve differently between runs,
    and because a candidate's position must not depend on dictionary iteration
    order. Three keys, in order:

    1. higher fused score first;
    2. then better (lower) best rank across the sources - a document some
       retriever put near the top outranks one everybody put in the middle;
    3. then the id, ascending, purely so that identical evidence always
       produces identical output.

    A duplicate id within one source's list keeps its first (best) rank; a later
    repeat is ignored rather than counted twice.
    """
    if k <= 0:
        raise ValueError(f"RRF k must be positive, got {k}")

    scores: dict[str, float] = {}
    ranks: dict[str, dict[str, int]] = {}

    for source in sorted(rankings):  # sorted: fusion must not depend on dict order
        for position, key in enumerate(rankings[source], start=1):
            per_source = ranks.setdefault(key, {})
            if source in per_source:
                continue  # a repeat within one list; the first occurrence stands
            per_source[source] = position
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + position)

    items = [FusedItem(key=key, score=scores[key], ranks=dict(ranks[key])) for key in scores]
    items.sort(key=lambda item: (-item.score, item.best_rank, item.key))
    return items


__all__ = ["DEFAULT_RRF_K", "FusedItem", "reciprocal_rank_fusion"]
