"""The two things that implement ``Embedder``: the real model, and a stand-in.

|                | `FastEmbedEmbedder`            | `HashingEmbedder`            |
|----------------|--------------------------------|------------------------------|
| understands    | meaning                        | word overlap only            |
| needs          | the `[embeddings]` extra + 67 MB | nothing                    |
| used by        | you, locally; ingest           | the test suite, CI           |
| deterministic  | yes (fixed weights, CPU)       | yes (hashing, no randomness) |

**Why a stand-in exists at all.**  The default test suite must be free, offline
and fast on a laptop and on someone else's fork.  Downloading a model in CI is
none of those.  So the tests drive the *whole* pipeline - ingest, vector column,
HNSW index, SQL, RRF, fusion - with an embedder whose output is fully
predictable, and a separate opt-in suite (`pytest -m embeddings`) checks that
the real model actually behaves the way the architecture assumes.

**The stand-in is not a fake result.**  It is a real bag-of-words vector, so
"shares words" really does mean "closer", which is enough to test that ranking,
fusion and SQL work.  It is *not* semantic, and no test may claim otherwise:
asserting that a hashed vector understands a paraphrase would be asserting
something false. That claim is tested against the real model or not at all.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.config import Settings
from app.retrieval.embedding import EMBEDDING_DIM, Embedder

_TOKEN = re.compile(r"[a-z0-9]+")

#: Where the downloaded weights go when `EMBEDDING_CACHE_DIR` is unset.
#: fastembed's own default is a temporary directory; a cache the operating
#: system is entitled to delete means downloading 67 MB again after a reboot.
DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[3] / ".model-cache"


class EmbeddingBackendUnavailable(RuntimeError):
    """The real embedding model cannot be loaded, with an actionable reason."""


# ---------------------------------------------------------------------------
# the deterministic stand-in
# ---------------------------------------------------------------------------


class HashingEmbedder:
    """A bag-of-words vector: no model, no download, no randomness.

    Each token is hashed to one of ``dimension`` slots and contributes there;
    the result is L2-normalised so that, exactly as with the real model, cosine
    similarity is the dot product and every vector has length 1. Two texts
    sharing many words score high, two sharing none score ~0.

    Hashing is SHA-256 rather than Python's ``hash()``, which is salted per
    process: a vector that changed between runs would make every stored
    embedding wrong after a restart.
    """

    def __init__(self, dimension: int = EMBEDDING_DIM, *, model_id: str = "hashing-v1") -> None:
        self._dimension = dimension
        self._model_id = model_id

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def dimension(self) -> int:
        return self._dimension

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * self._dimension
        for token in _TOKEN.findall(text.lower()):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            slot = int.from_bytes(digest[:4], "big") % self._dimension
            # The sign comes from a different byte, so unrelated tokens can
            # cancel instead of every vector drifting in one direction.
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[slot] += sign
        norm = math.sqrt(sum(v * v for v in vector))
        if norm == 0.0:
            # An empty or punctuation-only text. A zero vector has no direction,
            # so cosine distance against it is undefined; one fixed unit vector
            # keeps the maths valid and is deterministic.
            vector[0] = 1.0
            return vector
        return [v / norm for v in vector]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


# ---------------------------------------------------------------------------
# the real model
# ---------------------------------------------------------------------------


class FastEmbedEmbedder:
    """``BAAI/bge-small-en-v1.5`` through fastembed's ONNX runtime.

    Loading is deferred to the first call, so constructing one is free and an
    import of this module never pulls in onnxruntime. The model is loaded once
    per process and reused; loading costs a second or two, embedding costs a few
    milliseconds per text.

    The model returns L2-normalised vectors already, so nothing here rescales
    them - and ``tests/unit/retrieval`` asserts that assumption against the real
    model when the opt-in suite runs.
    """

    def __init__(
        self,
        model_id: str,
        *,
        cache_dir: Path | None = None,
        batch_size: int = 32,
        dimension: int = EMBEDDING_DIM,
    ) -> None:
        self._model_id = model_id
        self._cache_dir = cache_dir or DEFAULT_CACHE_DIR
        self._batch_size = batch_size
        self._dimension = dimension
        self._model: Any | None = None

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def dimension(self) -> int:
        return self._dimension

    def _load(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:  # pragma: no cover - depends on the install
            raise EmbeddingBackendUnavailable(
                "fastembed is not installed. Either install the optional extra:\n"
                '    pip install -e "./backend[embeddings]"\n'
                "or set EMBEDDING_BACKEND=hashing to run without a real model "
                "(retrieval will match words, not meaning)."
            ) from exc

        self._cache_dir.mkdir(parents=True, exist_ok=True)
        try:
            self._model = TextEmbedding(model_name=self._model_id, cache_dir=str(self._cache_dir))
        except Exception as exc:  # noqa: BLE001 - any load failure is the same story
            raise EmbeddingBackendUnavailable(
                f"could not load {self._model_id!r} ({type(exc).__name__}: {exc}). "
                "The first run downloads ~67 MB from HuggingFace and needs network "
                f"access; after that it is served from {self._cache_dir}."
            ) from exc
        return self._model

    def _check(self, vectors: list[list[float]], texts: Sequence[str]) -> list[list[float]]:
        if len(vectors) != len(texts):
            raise EmbeddingBackendUnavailable(
                f"{self._model_id} returned {len(vectors)} vectors for {len(texts)} texts"
            )
        for vector in vectors:
            if len(vector) != self._dimension:
                # Loud, immediately: a mismatched width would otherwise be
                # discovered as an opaque insert error against the vector column.
                raise EmbeddingBackendUnavailable(
                    f"{self._model_id} returned {len(vector)} dimensions, "
                    f"expected {self._dimension}"
                )
        return vectors

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        model = self._load()
        vectors = [
            [float(x) for x in vector]
            for vector in model.embed(list(texts), batch_size=self._batch_size)
        ]
        return self._check(vectors, texts)

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


# ---------------------------------------------------------------------------
# selection
# ---------------------------------------------------------------------------


def build_embedder(settings: Settings) -> Embedder:
    """The embedder this process should use, per ``EMBEDDING_BACKEND``."""
    if settings.embedding_backend == "hashing":
        return HashingEmbedder()
    cache_dir = Path(settings.embedding_cache_dir) if settings.embedding_cache_dir else None
    return FastEmbedEmbedder(
        settings.embedding_model,
        cache_dir=cache_dir,
        batch_size=settings.embedding_batch_size,
    )


@lru_cache(maxsize=1)
def _cached_real_embedder(model_id: str, cache_dir: str, batch_size: int) -> FastEmbedEmbedder:
    return FastEmbedEmbedder(model_id, cache_dir=Path(cache_dir), batch_size=batch_size)


def get_embedder(settings: Settings) -> Embedder:
    """``build_embedder``, but the real model is loaded at most once per process.

    Model load is the expensive part - a second or two - and it is pure setup
    with no per-caller state, so sharing one instance is free. The stand-in is
    cheap enough that caching it would only add a way to be surprised.
    """
    if settings.embedding_backend == "hashing":
        return HashingEmbedder()
    cache_dir = settings.embedding_cache_dir or str(DEFAULT_CACHE_DIR)
    return _cached_real_embedder(settings.embedding_model, cache_dir, settings.embedding_batch_size)


__all__ = [
    "DEFAULT_CACHE_DIR",
    "EmbeddingBackendUnavailable",
    "FastEmbedEmbedder",
    "HashingEmbedder",
    "build_embedder",
    "get_embedder",
]
