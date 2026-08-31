"""Stage 2: reordering the candidates Day 8's hybrid retrieval produced.

**What a cross-encoder is**, for a reader who has not met one.  Day 8's
embedding model is a *bi-encoder*: it turns each question into a vector on its
own, long before any query exists, and comparing a query to a question is then
just comparing two vectors.  That is what makes it fast enough to search the
whole bank - but it also means the question was summarised into 384 numbers
*without ever having seen the query*.

A **cross-encoder** reads the query and one candidate **together**, as a single
piece of text, and outputs one number: how relevant this candidate is to this
query.

    bi-encoder (Day 8)                cross-encoder (Day 9)
    ------------------                ---------------------
    query    -> [enc] -> vector       [query  SEP  candidate] -> [enc] -> score
    question -> [enc] -> vector                 (one model pass per candidate)
                  \\    /
                   cosine

Because the model attends over both texts at once, it can notice things a pair
of independent summaries cannot - that the query asks about cache *invalidation*
while this candidate only discusses *eviction*, for instance.  The price is that
nothing can be precomputed: it is one forward pass per candidate, every query.

**Why that forces a two-stage architecture.**  There is no index for a
cross-encoder, so scoring the whole bank means one model pass per question:

    100,000 questions -> 100,000 cross-encoder passes   (absurd)

    100,000 questions
        -> vector + lexical + RRF   (indexed, milliseconds)
        -> 40 candidates
        -> 40 cross-encoder passes  (a few hundred ms)
        -> best 8

Stage 1 optimises **recall** - get the right answer somewhere in the 40. Stage 2
optimises **precision** - get it to the top. Neither can do the other's job.

This module holds the *ordering* logic and knows nothing about which model
produces the scores; ``rerankers.py`` holds the implementations.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from app.retrieval.search import QuestionRef, RetrievedQuestion

#: Plan section 5.3: "I use the bi-encoder to go from 150 to 40 and the
#: cross-encoder to go from 40 to 8. Recall first, precision second."
DEFAULT_CANDIDATE_K = 40
DEFAULT_RERANK_FINAL_K = 8


@runtime_checkable
class Reranker(Protocol):
    """Anything that can score (query, document) pairs.

    Deliberately the narrowest possible interface: a list of texts in, a list of
    numbers out, same order, same length. It knows nothing about questions,
    databases or retrieval, which is what lets the deterministic stand-in be a
    dozen lines and the real model be swapped without touching this module.
    """

    @property
    def model_id(self) -> str:
        """Recorded on every result this reranker scores."""

    def score_pairs(self, query: str, documents: Sequence[str]) -> list[float]:
        """One relevance score per document, in the order given. Higher is better."""


def rerank_document(question: QuestionRef) -> str:
    """The text the cross-encoder is shown for one candidate.

    It is the same ``search_document`` Day 8 indexes - the question, its
    concepts, its taxonomy and its tags - and *not* the bare question text.

    **Why the same text and not just the question.** Day 8 found and fixed a bug
    where the vector arm searched "question + concepts" while the lexical arm
    searched "question" alone: fusing rankings over two different corpora ranks
    apples against oranges. Showing the reranker less than the retrievers saw
    would reintroduce exactly that mismatch one stage later - a candidate
    retrieved *because* of a concept key could then be demoted by a reranker
    that cannot see it. "thundering herd" appears in one item's
    ``expected_concepts`` and nowhere in its prose, and it is precisely the kind
    of query where stage 2 would otherwise undo stage 1's work.

    What is still excluded is what was excluded on Day 8 and for the same
    reasons: the id (opaque), ``difficulty_b`` (a number whose text means
    nothing to a language model - it is a filter, applied in SQL), the reference
    answer (it describes the *answer*, and is far longer than the question), and
    all review metadata and timestamps.

    Falls back to the bare text for a row ingested before Day 8's
    ``search_document`` column existed, which is the same COALESCE the
    ``tsv`` column applies.
    """
    return question.search_document or question.text


@dataclass(frozen=True, slots=True)
class RerankedQuestion:
    """A candidate after stage 2, carrying evidence from both stages.

    The Day 8 evidence is kept rather than replaced: being able to say "the
    reranker moved this from 7th to 1st" is most of the value of having a
    reranker at all, and it is what Day 10's ablation will measure.
    """

    candidate: RetrievedQuestion
    #: Raw model output. NOT a probability - see `rerankers.py`. Higher is more
    #: relevant. None when the reranker was unavailable and the hybrid order
    #: was passed through unchanged.
    rerank_score: float | None
    #: 1-based position after reranking.
    rerank_rank: int
    #: 1-based position before reranking, i.e. the RRF order.
    retrieval_rank: int

    @property
    def id(self) -> str:
        return self.candidate.id

    @property
    def question(self) -> QuestionRef:
        return self.candidate.question

    @property
    def rank_delta(self) -> int:
        """How far reranking moved this. Positive means promoted."""
        return self.retrieval_rank - self.rerank_rank


@dataclass(frozen=True, slots=True)
class RerankOutcome:
    """The reranked list, and an honest record of whether reranking happened."""

    query: str
    results: list[RerankedQuestion]
    #: False when the model was unavailable or disabled and the hybrid order was
    #: passed through. Never silently true.
    reranked: bool
    model_id: str | None = None
    #: Why reranking did not happen. Set if and only if `reranked` is False.
    fallback_reason: str | None = None
    candidates_scored: int = 0
    timings_ms: dict[str, float] = field(default_factory=dict)

    @property
    def truncated(self) -> int:
        """Candidates scored but not returned. Reported, never silent."""
        return max(0, self.candidates_scored - len(self.results))


def _passthrough(
    query: str,
    candidates: Sequence[RetrievedQuestion],
    final_k: int,
    reason: str,
) -> RerankOutcome:
    """Return the hybrid order unchanged, and say so.

    The fallback contract: a reranker that cannot run must degrade to Day 8's
    ranking, which is a perfectly good ranking, rather than fail the request.
    But it must never *look* like reranking happened - `reranked` is False,
    `fallback_reason` says why, and every `rerank_score` is None rather than 0.0,
    because a zero reads like a measurement and None does not.
    """
    return RerankOutcome(
        query=query,
        results=[
            RerankedQuestion(
                candidate=candidate,
                rerank_score=None,
                rerank_rank=position,
                retrieval_rank=position,
            )
            for position, candidate in enumerate(candidates[:final_k], start=1)
        ],
        reranked=False,
        fallback_reason=reason,
        candidates_scored=0,
    )


def rerank_candidates(
    reranker: Reranker,
    query: str,
    candidates: Sequence[RetrievedQuestion],
    *,
    final_k: int = DEFAULT_RERANK_FINAL_K,
) -> RerankOutcome:
    """Score every candidate against the query and return the best ``final_k``.

    **Ordering is total and deterministic**, by three keys:

    1. higher relevance score first;
    2. then the retrieval rank - where the reranker is indifferent between two
       candidates, Day 8's opinion breaks the tie, which is better than an
       arbitrary choice and better than the id, because it degrades toward the
       ranking that would have been served anyway;
    3. then the question id, so that identical evidence always produces
       identical output.

    An empty candidate list returns an empty result **without calling the
    model**: loading a cross-encoder to score nothing is pure waste.

    If the model raises, the hybrid order is passed through and the outcome
    records why. Retrieval degrading to "merely good" beats retrieval failing.
    """
    if not candidates:
        return RerankOutcome(
            query=query,
            results=[],
            reranked=True,
            model_id=reranker.model_id,
            candidates_scored=0,
            timings_ms={"score": 0.0, "sort": 0.0, "total": 0.0},
        )

    documents = [rerank_document(candidate.question) for candidate in candidates]

    started = time.perf_counter()
    try:
        scores = reranker.score_pairs(query, documents)
    except Exception as exc:  # noqa: BLE001 - any model failure degrades the same way
        return _passthrough(query, candidates, final_k, f"{type(exc).__name__}: {exc}")
    after_score = time.perf_counter()

    if len(scores) != len(documents):
        return _passthrough(
            query,
            candidates,
            final_k,
            f"{reranker.model_id} returned {len(scores)} scores for {len(documents)} candidates",
        )

    ordered = sorted(
        zip(candidates, scores, range(1, len(candidates) + 1), strict=True),
        key=lambda triple: (-triple[1], triple[2], triple[0].id),
    )
    results = [
        RerankedQuestion(
            candidate=candidate,
            rerank_score=score,
            rerank_rank=position,
            retrieval_rank=retrieval_rank,
        )
        for position, (candidate, score, retrieval_rank) in enumerate(ordered[:final_k], start=1)
    ]
    finished = time.perf_counter()

    return RerankOutcome(
        query=query,
        results=results,
        reranked=True,
        model_id=reranker.model_id,
        candidates_scored=len(candidates),
        timings_ms={
            "score": (after_score - started) * 1000,
            "sort": (finished - after_score) * 1000,
            "total": (finished - started) * 1000,
        },
    )


__all__ = [
    "DEFAULT_CANDIDATE_K",
    "DEFAULT_RERANK_FINAL_K",
    "RerankOutcome",
    "RerankedQuestion",
    "Reranker",
    "rerank_candidates",
    "rerank_document",
]
