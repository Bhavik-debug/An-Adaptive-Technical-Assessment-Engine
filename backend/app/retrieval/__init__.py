"""Retrieval: finding the right questions in the bank (Phase 2, Days 8-9).

Two stages, seven modules, each with one job.

**Stage 1 - candidate generation (Day 8).** Fast, indexed, broad; optimises
*recall*, because stage 2 can only reorder what this hands it.

* ``embedding``  - what text gets embedded, and the ``Embedder`` interface.
* ``embedders``  - the real model (fastembed/ONNX) and a deterministic stand-in.
* ``indexing``   - writing vectors for ingested questions, idempotently.
* ``rrf``        - Reciprocal Rank Fusion; pure, no I/O.
* ``search``     - vector search, lexical search, and hybrid over both.

**Stage 2 - reranking (Day 9).** Slower, more precise; optimises *precision*
over a small candidate set. Cannot be an index: one model pass per candidate.

* ``rerank``     - the ``Reranker`` interface and the ordering logic; no I/O.
* ``rerankers``  - the real cross-encoder and a deterministic stand-in.

``pipeline`` composes the two. ``search`` does not import ``rerank`` and
``rerank`` does not run queries, so candidate selection can change (Phase 3 adds
difficulty and coverage constraints) without touching the reranker.

Start with ``docs/retrieval.md`` if embeddings, vector similarity, pgvector,
HNSW or RRF are new - it explains each one from scratch.

**This file deliberately re-exports nothing.**  ``app/models/question.py`` needs
``EMBEDDING_DIM`` to declare the width of its ``vector`` column, so *models*
imports from *retrieval* - while ``retrieval/indexing.py`` and
``retrieval/search.py`` import the models.  Eager re-exports here would turn that
into a genuine import cycle: importing ``app.retrieval.embedding`` runs this
file first, which would pull in ``indexing``, which needs a half-built
``app.models.question``.  Keeping the package's ``__init__`` free of imports
makes ``embedding`` a leaf that anything may depend on.

Import from the module that owns the name::

    from app.retrieval.embedders import build_embedder
    from app.retrieval.pipeline import search_and_rerank
    from app.retrieval.search import hybrid_search

Nothing here is exposed over HTTP: Days 8 and 9 build the retrieval layer, and
the ``/questions/search`` endpoint is later.
"""
