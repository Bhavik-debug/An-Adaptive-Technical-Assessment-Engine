"""The retrieval layer: vector search, lexical search, and their fusion.

                            query
                              |
                  +-----------+-----------+
                  |                       |
            vector search           lexical search
       (meaning; bge + pgvector)  (words; tsvector + GIN)
                  |                       |
             ranked list             ranked list
                  +-----------+-----------+
                              |
                        RRF fusion
                              |
                     final ranked list

**Why both.**  They fail in opposite directions, which is the entire argument
for combining them.

* Lexical search cannot match *"How can a relational database speed up
  lookups?"* to a question about indexes, because they share almost no words.
* Vector search cannot reliably find an exact token - a rare identifier, an
  acronym like ``MVCC``, a spelling the model never saw - because embeddings
  blur exactly the detail you are asking it to be precise about.

Each covers the other's blind spot. Fusion is what lets a result that both
methods liked outrank one that only a single method loved.

This module deliberately stops at "ranked candidates". Cross-encoder reranking
is Day 9 and consumes this output unchanged; the ``/questions/search`` endpoint
is later still. Nothing here is exposed over HTTP.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field

from sqlalchemy import bindparam, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.retrieval.embedding import Embedder, query_text
from app.retrieval.rrf import DEFAULT_RRF_K, reciprocal_rank_fusion

#: Defaults, mirrored by RETRIEVAL_* settings. Candidate K is how many each
#: retriever proposes; final K is how many survive fusion. They are different
#: numbers on purpose - see `hybrid_search`.
DEFAULT_VECTOR_K = 30
DEFAULT_LEXICAL_K = 30
DEFAULT_FINAL_K = 10

# Both queries select the same six columns, written out rather than
# interpolated from a shared constant: an f-string that builds SQL is a pattern
# worth not having in the codebase at all, even where the interpolated value is
# a literal. `_to_ref` is the single place that reads them back.


@dataclass(frozen=True, slots=True)
class QuestionRef:
    """Just enough of a question to rank it and to show why it was returned."""

    id: str
    text: str
    topic_key: str
    subtopic_key: str
    difficulty_b: float
    tags: tuple[str, ...]
    #: The exact string Day 8 indexed - question + concepts + taxonomy + tags.
    #: Carried so that Day 9's cross-encoder can be shown the same text the
    #: retrievers searched; showing it less would let stage 2 demote a candidate
    #: that stage 1 found for a reason stage 2 cannot see. None for a row
    #: ingested before the column existed.
    search_document: str | None = None


@dataclass(frozen=True, slots=True)
class ScoredHit:
    """One retriever's opinion: a question, its rank in that list, its score."""

    question: QuestionRef
    rank: int
    score: float


@dataclass(frozen=True, slots=True)
class RetrievedQuestion:
    """A fused result, carrying the evidence from both retrievers.

    Every field except ``rrf_score`` is nullable on purpose: a question found
    only lexically has no vector rank, and saying so is more useful than
    substituting a zero that reads like a real measurement.
    """

    question: QuestionRef
    rrf_score: float
    vector_rank: int | None = None
    vector_similarity: float | None = None
    lexical_rank: int | None = None
    lexical_score: float | None = None

    @property
    def id(self) -> str:
        return self.question.id

    @property
    def sources(self) -> tuple[str, ...]:
        found = []
        if self.vector_rank is not None:
            found.append("vector")
        if self.lexical_rank is not None:
            found.append("lexical")
        return tuple(found)


@dataclass(frozen=True, slots=True)
class HybridSearchResult:
    """The final list, plus what it cost and what was thrown away getting there."""

    query: str
    results: list[RetrievedQuestion]
    vector_candidates: int
    lexical_candidates: int
    #: Distinct questions across both lists, before truncation to final K.
    fused_candidates: int
    timings_ms: dict[str, float] = field(default_factory=dict)

    @property
    def truncated(self) -> int:
        """Candidates fused but not returned. Reported, never silent."""
        return max(0, self.fused_candidates - len(self.results))


def _to_ref(row: object) -> QuestionRef:
    r = row  # a SQLAlchemy Row; attribute access is by column name
    return QuestionRef(
        id=r.id,  # type: ignore[attr-defined]
        text=r.text,  # type: ignore[attr-defined]
        topic_key=r.topic_key,  # type: ignore[attr-defined]
        subtopic_key=r.subtopic_key,  # type: ignore[attr-defined]
        difficulty_b=float(r.difficulty_b),  # type: ignore[attr-defined]
        tags=tuple(r.tags or ()),  # type: ignore[attr-defined]
        search_document=r.search_document,  # type: ignore[attr-defined]
    )


# ---------------------------------------------------------------------------
# vector search - meaning
# ---------------------------------------------------------------------------

# The inner query is what the HNSW index can answer: order by the distance
# operator, take a limit, nothing else. The outer query then applies a
# deterministic tie-break. Doing it in one query - `ORDER BY distance, id` -
# would ask for an ordering the index cannot supply, so PostgreSQL would sort
# every row instead of using the index at all.
_VECTOR_SQL = text(
    """
    SELECT * FROM (
        SELECT q.id, q.text, q.topic_key, q.subtopic_key, q.difficulty_b, q.tags,
               q.search_document,
               (q.embedding <=> :query_vector) AS distance
        FROM questions AS q
        WHERE q.embedding IS NOT NULL
        ORDER BY q.embedding <=> :query_vector
        LIMIT :limit
    ) AS hits
    ORDER BY distance ASC, id ASC
    """
).bindparams(bindparam("query_vector"))


async def vector_search(
    session: AsyncSession,
    embedder: Embedder,
    query: str,
    *,
    limit: int = DEFAULT_VECTOR_K,
) -> list[ScoredHit]:
    """Nearest questions to the query by meaning.

    ``<=>`` is pgvector's cosine *distance*: 0 means identical direction, 1
    means unrelated, 2 means opposite. Reported here as similarity
    (``1 - distance``) because "higher is better" is what every other score in
    this module means, and mixing the two conventions is how a ranking gets
    silently inverted.

    Rows with no embedding are excluded rather than treated as maximally
    distant - a question that has not been embedded is not a bad match, it is an
    unknown one, and returning it at the bottom would hide an ingest problem.

    Note that HNSW is an *approximate* index: it can miss a true nearest
    neighbour. At 60 rows PostgreSQL will usually choose a sequential scan
    anyway, which is exact; the index earns its place later, not now.
    """
    vector = embedder.embed_query(query_text(query))
    return await vector_search_by_vector(session, vector, limit=limit)


async def vector_search_by_vector(
    session: AsyncSession,
    vector: Sequence[float],
    *,
    limit: int = DEFAULT_VECTOR_K,
) -> list[ScoredHit]:
    """The database half of ``vector_search``, for a query already embedded.

    Split out so a caller can time embedding and SQL separately - they differ by
    two orders of magnitude, and reporting "vector search took 600 ms" when 599
    of them were loading a model would be a misleading measurement.
    """
    rows = (
        await session.execute(_VECTOR_SQL, {"query_vector": str(list(vector)), "limit": limit})
    ).all()
    return [
        ScoredHit(question=_to_ref(row), rank=position, score=1.0 - float(row.distance))
        for position, row in enumerate(rows, start=1)
    ]


# ---------------------------------------------------------------------------
# lexical search - words
# ---------------------------------------------------------------------------

# `websearch_to_tsquery` rather than `to_tsquery`: it accepts whatever a person
# types, including quotes, OR, and stray punctuation, and never raises a syntax
# error on user input. `to_tsquery('database index')` is a hard error, which
# would make the search endpoint fail on an ordinary two-word query.
#
# `ts_rank_cd` rather than `ts_rank`: the cover-density variant also accounts
# for how close the matched terms are to each other, so a question using the
# words together outranks one that mentions them in unrelated sentences. This is
# the plan's "BM25-ish" ranking (section 3, Day 8) - Postgres has no true BM25.
_LEXICAL_SQL = text(
    """
    SELECT q.id, q.text, q.topic_key, q.subtopic_key, q.difficulty_b, q.tags,
           q.search_document,
           ts_rank_cd(q.tsv, tsq) AS rank
    FROM questions AS q, websearch_to_tsquery('english', :query) AS tsq
    WHERE q.tsv @@ tsq
    ORDER BY rank DESC, q.id ASC
    LIMIT :limit
    """
)


async def lexical_search(
    session: AsyncSession,
    query: str,
    *,
    limit: int = DEFAULT_LEXICAL_K,
) -> list[ScoredHit]:
    """Questions matching the query's words, ranked by cover density.

    ``tsv`` is a GENERATED column on ``questions`` that PostgreSQL recomputes
    from ``text`` on every write, indexed with GIN - both created back in Day
    2's migration, so lexical search needed no new schema. Stemming means
    "indexes" matches "index"; stop words like "the" and "how" are dropped.

    An empty result is normal and correct: a query whose every term is a stop
    word, or matches nothing, produces no rows rather than arbitrary ones.
    """
    rows = (await session.execute(_LEXICAL_SQL, {"query": query, "limit": limit})).all()
    return [
        ScoredHit(question=_to_ref(row), rank=position, score=float(row.rank))
        for position, row in enumerate(rows, start=1)
    ]


# ---------------------------------------------------------------------------
# hybrid
# ---------------------------------------------------------------------------


async def hybrid_search(
    session: AsyncSession,
    embedder: Embedder,
    query: str,
    *,
    vector_k: int = DEFAULT_VECTOR_K,
    lexical_k: int = DEFAULT_LEXICAL_K,
    final_k: int = DEFAULT_FINAL_K,
    rrf_k: float = DEFAULT_RRF_K,
) -> HybridSearchResult:
    """Run both retrievers, fuse with RRF, return the top ``final_k``.

    **Candidate K vs final K.** Each retriever proposes ``*_k`` candidates
    (default 30) and only ``final_k`` (default 10) are returned. The gap is the
    point: a question ranked 22nd by vectors and 2nd lexically deserves to be
    fused, and it can only be fused if both retrievers were asked for more than
    the caller wants back. Making candidate K equal to final K would turn the
    hybrid into two independent top-10s stapled together.

    Candidates fused but not returned are counted in ``truncated`` rather than
    dropped silently.

    Both retrievers are run sequentially, not concurrently. Two round trips to a
    local database are a few milliseconds, and sharing one ``AsyncSession``
    across concurrent statements is not supported by SQLAlchemy - the correct
    fix is two sessions, which is complexity this earns no measurable time for.
    """
    started = time.perf_counter()
    query_vector = embedder.embed_query(query_text(query))
    after_embed = time.perf_counter()
    vector_hits = await vector_search_by_vector(session, query_vector, limit=vector_k)
    after_vector = time.perf_counter()
    lexical_hits = await lexical_search(session, query, limit=lexical_k)
    after_lexical = time.perf_counter()

    by_vector = {hit.question.id: hit for hit in vector_hits}
    by_lexical = {hit.question.id: hit for hit in lexical_hits}

    fused = reciprocal_rank_fusion(
        {
            "vector": [hit.question.id for hit in vector_hits],
            "lexical": [hit.question.id for hit in lexical_hits],
        },
        k=rrf_k,
    )

    results: list[RetrievedQuestion] = []
    for item in fused[:final_k]:
        vector_hit = by_vector.get(item.key)
        lexical_hit = by_lexical.get(item.key)
        # Either side may be absent, but not both: an id only exists in the
        # fused list because some retriever returned it.
        reference = vector_hit or lexical_hit
        assert reference is not None  # noqa: S101 - invariant of the fusion input
        results.append(
            RetrievedQuestion(
                question=reference.question,
                rrf_score=item.score,
                vector_rank=vector_hit.rank if vector_hit else None,
                vector_similarity=vector_hit.score if vector_hit else None,
                lexical_rank=lexical_hit.rank if lexical_hit else None,
                lexical_score=lexical_hit.score if lexical_hit else None,
            )
        )

    finished = time.perf_counter()
    return HybridSearchResult(
        query=query,
        results=results,
        vector_candidates=len(vector_hits),
        lexical_candidates=len(lexical_hits),
        fused_candidates=len(fused),
        timings_ms={
            # Split deliberately: the first `embed` of a process includes
            # loading the model (~1 s) and every later one is a few ms, so a
            # single "vector" number would be meaningless.
            "embed": (after_embed - started) * 1000,
            "vector_sql": (after_vector - after_embed) * 1000,
            "lexical_sql": (after_lexical - after_vector) * 1000,
            "fusion": (finished - after_lexical) * 1000,
            "total": (finished - started) * 1000,
        },
    )


def format_results(results: Sequence[RetrievedQuestion], *, width: int = 68) -> str:
    """A readable table of a result list, for scripts and manual verification."""
    lines = [
        f"{'#':>2}  {'id':<20} {'rrf':>7} {'vec':>5} {'sim':>6} {'lex':>5} {'rank':>7}  question",
        "-" * (58 + width),
    ]
    for position, hit in enumerate(results, start=1):
        snippet = hit.question.text[:width].replace("\n", " ")
        lines.append(
            f"{position:>2}  {hit.id:<20} {hit.rrf_score:>7.5f} "
            f"{_or_dash(hit.vector_rank):>5} {_or_dash(hit.vector_similarity, '.3f'):>6} "
            f"{_or_dash(hit.lexical_rank):>5} {_or_dash(hit.lexical_score, '.5f'):>7}  {snippet}"
        )
    return "\n".join(lines)


def _or_dash(value: float | int | None, spec: str = "") -> str:
    return "-" if value is None else format(value, spec)


__all__ = [
    "DEFAULT_FINAL_K",
    "DEFAULT_LEXICAL_K",
    "DEFAULT_VECTOR_K",
    "HybridSearchResult",
    "QuestionRef",
    "RetrievedQuestion",
    "ScoredHit",
    "format_results",
    "hybrid_search",
    "lexical_search",
    "vector_search",
    "vector_search_by_vector",
]
