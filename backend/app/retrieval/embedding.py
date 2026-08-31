"""What gets embedded, and the interface that turns text into a vector.

**What an embedding is**, for a reader who has not met one.  An embedding model
reads a piece of text and returns a fixed-length list of numbers - here 384 of
them - chosen so that texts *meaning* similar things land near each other.

    "Why do database indexes improve query performance?"
        -> [0.041, -0.118, 0.007, ...]   (384 numbers)
    "What is the purpose of an index in a relational database?"
        -> [0.038, -0.104, 0.011, ...]   (384 numbers, and close to the first)

Those two questions share almost no words, so keyword search relates them
poorly.  Their vectors are close, so vector search relates them well.  That is
the whole reason this module exists.  docs/retrieval.md explains it at length.

**The model.**  ``BAAI/bge-small-en-v1.5``: 384 dimensions, ~67 MB, runs on the
CPU in a few milliseconds per question.  It returns **L2-normalised** vectors -
every vector has length exactly 1 - which is why cosine distance is the right
operator and why cosine similarity is just the dot product.  Verified, not
assumed: see ``tests/unit/retrieval`` and the ``embeddings``-marked tests.

**Why the input text is built here and not at each call site.**  A vector is
only comparable with vectors built the same way.  If indexing embedded
"question text + concepts" and search embedded something else, every similarity
would be measured against a slightly different space and the system would be
quietly wrong rather than loudly broken.  One function, one recipe, one place to
change it - and ``text_fingerprint()`` records which recipe produced each stored
vector so a change to it invalidates the old ones instead of mixing them.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

#: The model this project embeds with. Recorded on every row so that changing
#: it invalidates the stored vectors rather than silently mixing two spaces.
EMBEDDING_MODEL_ID = "BAAI/bge-small-en-v1.5"

#: bge-small-en-v1.5 produces 384 numbers per text. This is the width of the
#: `vector` column in Postgres, so the two must agree exactly or every insert
#: fails - which is the correct, loud failure.
EMBEDDING_DIM = 384

#: Bumped whenever `document_text()` changes what it produces. It is part of the
#: fingerprint, so a recipe change re-embeds the bank on the next ingest rather
#: than leaving a mix of old and new vectors that nothing would detect.
TEXT_RECIPE_VERSION = 1

_WHITESPACE = re.compile(r"\s+")


@runtime_checkable
class Embedder(Protocol):
    """Anything that can turn text into vectors.

    Two implementations exist (``app/retrieval/embedders.py``): the real model,
    and a deterministic stand-in the test suite uses so that CI never downloads
    a 67 MB file. This mirrors the ``llm/providers`` split - real provider,
    offline stub - because it solves the same problem.
    """

    @property
    def model_id(self) -> str:
        """Recorded on every row this embedder produces."""

    @property
    def dimension(self) -> int: ...

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed question documents, in the order given."""

    def embed_query(self, text: str) -> list[float]:
        """Embed one search query."""


def normalise_whitespace(text: str) -> str:
    """Collapse runs of whitespace and strip. Applied to every text, both sides.

    Trivial, and it matters: the same question with a line break in a different
    place must produce the same fingerprint, or every reformatting of the JSONL
    would look like a content change and re-embed the bank.
    """
    return _WHITESPACE.sub(" ", text).strip()


def humanise_key(key: str) -> str:
    """``cache_invalidation`` -> ``cache invalidation``.

    Concept and taxonomy keys are snake_case identifiers. An embedding model
    tokenises ``cache_invalidation`` as an odd fragment it has rarely seen, but
    handles the two ordinary English words well. Feeding it identifiers instead
    of words throws away most of what the concept tags are worth.
    """
    return key.replace("_", " ").strip()


def document_text(
    *,
    text: str,
    topic_key: str,
    subtopic_key: str,
    concept_keys: Sequence[str],
    tags: Sequence[str],
) -> str:
    """The exact string that represents a question in vector space.

    Plan section 3, Day 8: *"embed ``text + concepts + tags``"*, plus the
    taxonomy, because "caching" and "binary search" are strong topical signals
    and they are already curated.

    **What is deliberately left out**, because an embedding of noise is noise:
    ``id`` (an opaque identifier), ``difficulty_b`` and ``discrimination_a``
    (numbers whose text form means nothing to a language model - they are
    *filters*, applied in SQL), ``reference_answer`` (it describes the answer,
    so embedding it would make questions match queries that are really about
    answers, and it is ten times longer than the question, so it would dominate),
    review metadata and timestamps.

    Deterministic: same inputs, same string, every time. No sorting is applied -
    the author's ordering of concepts and tags is meaningful and stable in the
    JSONL, and re-sorting here would only change fingerprints.
    """
    lines = [normalise_whitespace(text)]
    if concept_keys:
        lines.append("Concepts: " + ", ".join(humanise_key(k) for k in concept_keys))
    lines.append(f"Topic: {humanise_key(topic_key)} / {humanise_key(subtopic_key)}")
    if tags:
        lines.append("Tags: " + ", ".join(humanise_key(t) for t in tags))
    return "\n".join(lines)


def query_text(query: str) -> str:
    """The exact string a user's search query becomes before embedding.

    It shares ``normalise_whitespace`` with ``document_text`` and nothing else,
    and that asymmetry is deliberate rather than an oversight. A query is a
    question *about* the bank; wrapping it in "Concepts: ... Tags: ..." would
    invent metadata the user never supplied and push the vector away from the
    documents it should match. Retrieval models are trained on exactly this
    asymmetry - a short query against a longer passage.

    Note on bge specifically: the model card offers an optional query prefix
    ("Represent this sentence for searching relevant passages: ") for short
    queries, and says v1.5 does not generally need it. fastembed does not apply
    it either - ``query_embed`` and ``embed`` return identical vectors for this
    model, which was checked rather than assumed. Whether the prefix helps *this
    bank* is a question for Day 10's retrieval evaluation, which can measure it;
    guessing now would be optimising without evidence.
    """
    return normalise_whitespace(query)


def text_fingerprint(text: str, *, model_id: str) -> str:
    """A content hash of "this exact text, embedded by this exact model".

    Stored per row as ``questions.embedding_text_sha256``. It is what makes
    re-ingesting cheap and correct: a row whose fingerprint still matches does
    not need re-embedding, and a row whose question text, concepts, tags,
    taxonomy, model or text recipe changed no longer matches and is re-embedded
    automatically. Without it the only safe options are "re-embed everything
    every time" or "let the vectors silently go stale".
    """
    payload = f"v{TEXT_RECIPE_VERSION}\x00{model_id}\x00{text}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


__all__ = [
    "EMBEDDING_DIM",
    "EMBEDDING_MODEL_ID",
    "TEXT_RECIPE_VERSION",
    "Embedder",
    "document_text",
    "humanise_key",
    "normalise_whitespace",
    "query_text",
    "text_fingerprint",
]
