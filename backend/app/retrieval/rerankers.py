"""The two things that implement ``Reranker``: the real model, and a stand-in.

Exactly the split ``embedders.py`` uses, for exactly the same reason.

|                | `FastEmbedCrossEncoder`         | `LexicalOverlapReranker`     |
|----------------|---------------------------------|------------------------------|
| judges         | relevance, having read both      | shared words                 |
| needs          | the `[embeddings]` extra + ~1 GB | nothing                      |
| cost           | ~100 ms per candidate on CPU     | microseconds                 |
| used by        | you, locally                     | the test suite, CI           |
| deterministic  | yes (fixed weights, CPU)         | yes (no randomness)          |

**Why a stand-in exists.**  The default suite must be free, offline and fast, on
a laptop and on someone else's fork.  Downloading a gigabyte in CI is none of
those.  So the tests drive the whole two-stage pipeline - candidate generation,
scoring, ordering, tie-breaking, truncation, fallback - with a reranker whose
output is fully predictable, and a separate opt-in suite
(``pytest -m embeddings``) checks that the real model behaves as assumed.

**The stand-in is not a fake result, and it is not semantic.**  It scores real
word overlap, so "shares more words" genuinely means "ranks higher" - enough to
test every mechanism. It cannot judge relevance, and no test may claim it can.
That claim is made against the real model or not at all.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.config import Settings
from app.retrieval.embedders import DEFAULT_CACHE_DIR

_TOKEN = re.compile(r"[a-z0-9]+")

#: English words carrying no topical signal. Kept tiny and explicit: this list
#: exists so the stand-in is not dominated by "the", not to be a real stop list.
_STOP = frozenset(
    "a an and are as at be by do does for from how in is it of on or that the "
    "this to what when where which why with you your".split()
)


class RerankerUnavailable(RuntimeError):
    """The cross-encoder cannot be loaded or run, with an actionable reason."""


# ---------------------------------------------------------------------------
# the deterministic stand-in
# ---------------------------------------------------------------------------


class LexicalOverlapReranker:
    """Scores a pair by how much of the query's vocabulary the document contains.

    ``score = |query terms in document| / sqrt(document length)``

    The numerator is the signal - how much of what was asked for is present. The
    square-root denominator is a mild length penalty, so a long document does
    not win simply by containing more words; it is the same intuition behind
    every classical retrieval weighting, in its crudest form.

    Scores are unbounded above and never negative, which deliberately does *not*
    match the real model's range - nothing downstream may assume a range, and a
    stand-in that quietly shared one would hide that assumption.
    """

    def __init__(self, *, model_id: str = "lexical-overlap-v1") -> None:
        self._model_id = model_id

    @property
    def model_id(self) -> str:
        return self._model_id

    def score_pairs(self, query: str, documents: Sequence[str]) -> list[float]:
        terms = {t for t in _TOKEN.findall(query.lower()) if t not in _STOP}
        scores: list[float] = []
        for document in documents:
            tokens = _TOKEN.findall(document.lower())
            if not tokens or not terms:
                scores.append(0.0)
                continue
            present = sum(1 for term in terms if term in set(tokens))
            scores.append(present / math.sqrt(len(tokens)))
        return scores


# ---------------------------------------------------------------------------
# the real model
# ---------------------------------------------------------------------------


class FastEmbedCrossEncoder:
    """``BAAI/bge-reranker-base`` through fastembed's ONNX runtime, on the CPU.

    The plan's model (section 3, Day 9; section 5.3). ~110M parameters, no
    PyTorch, no API, no key. Reused from the same ``[embeddings]`` extra the Day
    8 embedder needs, so Day 9 added no dependency at all - only a second model
    file in the same cache directory.

    **Loading is deferred to the first call**, so constructing one is free and
    importing this module never pulls in onnxruntime. ``get_reranker()`` caches
    the loaded instance per process, because loading costs seconds and scoring
    costs milliseconds - reloading per query would make the reranker useless.

    **What the score is.** A raw model output - a logit. It is **not** a
    probability, is not bounded to [0, 1], and is routinely negative for
    irrelevant pairs. Higher means more relevant, and that is the only property
    anything in this codebase relies on. No sigmoid, no normalisation, no
    threshold is applied: a transformation would invent precision the number
    does not have, and thresholds need the evaluation data Day 10 produces.
    Scores from different queries are not comparable with each other either -
    the model ranks candidates *within* one query.
    """

    def __init__(
        self,
        model_id: str,
        *,
        cache_dir: Path | None = None,
        batch_size: int = 16,
    ) -> None:
        self._model_id = model_id
        self._cache_dir = cache_dir or DEFAULT_CACHE_DIR
        self._batch_size = batch_size
        self._model: Any | None = None

    @property
    def model_id(self) -> str:
        return self._model_id

    def _load(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            from fastembed.rerank.cross_encoder import TextCrossEncoder
        except ImportError as exc:  # pragma: no cover - depends on the install
            raise RerankerUnavailable(
                "fastembed is not installed. Either install the optional extra:\n"
                '    pip install -e "./backend[embeddings]"\n'
                "or set RERANK_BACKEND=overlap to run without a real model "
                "(candidates will be ordered by word overlap, not relevance), "
                "or RERANK_ENABLED=false to serve the hybrid order unchanged."
            ) from exc

        self._cache_dir.mkdir(parents=True, exist_ok=True)
        try:
            self._model = TextCrossEncoder(
                model_name=self._model_id, cache_dir=str(self._cache_dir)
            )
        except Exception as exc:  # noqa: BLE001 - any load failure is the same story
            raise RerankerUnavailable(
                f"could not load {self._model_id!r} ({type(exc).__name__}: {exc}). "
                "The first run downloads ~1 GB from HuggingFace and needs network "
                f"access; after that it is served from {self._cache_dir}."
            ) from exc
        return self._model

    def score_pairs(self, query: str, documents: Sequence[str]) -> list[float]:
        if not documents:
            return []
        model = self._load()
        scores = [
            float(score)
            for score in model.rerank(query, list(documents), batch_size=self._batch_size)
        ]
        if len(scores) != len(documents):  # pragma: no cover - library contract
            raise RerankerUnavailable(
                f"{self._model_id} returned {len(scores)} scores for {len(documents)} documents"
            )
        return scores


# ---------------------------------------------------------------------------
# selection
# ---------------------------------------------------------------------------


def build_reranker(settings: Settings) -> LexicalOverlapReranker | FastEmbedCrossEncoder:
    """A new reranker per ``RERANK_BACKEND``. Prefer ``get_reranker`` in a service."""
    if settings.rerank_backend == "overlap":
        return LexicalOverlapReranker()
    cache_dir = Path(settings.rerank_cache_dir) if settings.rerank_cache_dir else None
    return FastEmbedCrossEncoder(
        settings.rerank_model,
        cache_dir=cache_dir,
        batch_size=settings.rerank_batch_size,
    )


@lru_cache(maxsize=1)
def _cached_cross_encoder(model_id: str, cache_dir: str, batch_size: int) -> FastEmbedCrossEncoder:
    return FastEmbedCrossEncoder(model_id, cache_dir=Path(cache_dir), batch_size=batch_size)


def get_reranker(settings: Settings) -> LexicalOverlapReranker | FastEmbedCrossEncoder:
    """``build_reranker``, but the real model is loaded at most once per process.

    Model load is seconds; scoring is milliseconds. An instance holds only the
    loaded weights and no per-caller state, so sharing one is free - and not
    sharing one would mean paying the load on every query, which would make the
    two-stage architecture pointless. The stand-in is cheap enough that caching
    it would only add a way to be surprised.
    """
    if settings.rerank_backend == "overlap":
        return LexicalOverlapReranker()
    cache_dir = settings.rerank_cache_dir or str(DEFAULT_CACHE_DIR)
    return _cached_cross_encoder(settings.rerank_model, cache_dir, settings.rerank_batch_size)


__all__ = [
    "FastEmbedCrossEncoder",
    "LexicalOverlapReranker",
    "RerankerUnavailable",
    "build_reranker",
    "get_reranker",
]
