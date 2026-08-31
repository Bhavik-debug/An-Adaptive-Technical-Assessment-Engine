"""Turning ingested questions into stored vectors.

This is the second half of ingest, not a separate process:

    question-bank JSONL -> validate -> upsert rows -> embed -> store vectors

**Why it is not a separate script.**  Two commands that must both be run, in
order, to leave the system consistent is a system that will eventually be
inconsistent - somebody re-ingests an edited question and forgets to re-embed,
and search quietly returns results for the previous wording. So
``app/bank/ingest.py`` calls this in the *same transaction* as the upsert: the
run either leaves rows and vectors agreeing, or leaves nothing at all.

**Why it does not re-embed everything.**  Each row stores the model id and a
fingerprint of the exact text that was embedded. A row whose pair still matches
is already correct and is skipped. So re-ingesting an unchanged bank does no
model work at all, while editing one question re-embeds exactly that one -
without anybody having to remember which.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.bank.schema import BankItem
from app.models.question import Question
from app.retrieval.embedding import Embedder, document_text, text_fingerprint


@dataclass(slots=True)
class EmbedResult:
    """What the embed step did, in numbers a caller can assert on."""

    embedded: int = 0
    #: Rows already carrying a vector for this exact text and model.
    reused: int = 0
    #: Ids in the files with no matching row in the database.
    missing_rows: list[str] = field(default_factory=list)
    model_id: str = ""

    @property
    def summary(self) -> str:
        return (
            f"{self.embedded} embedded, {self.reused} already current "
            f"({self.model_id or 'no model'})"
        )


def document_text_for_item(item: BankItem) -> str:
    """The text embedded for one bank item.

    A thin adapter so that ``document_text`` never has to know about
    ``BankItem``: the recipe is about *fields*, and keeping it that way is what
    lets a query, a row and a file item all be handled by one function.
    """
    return document_text(
        text=item.text,
        topic_key=item.topic,
        subtopic_key=item.subtopic,
        concept_keys=item.concept_keys,
        tags=item.tags,
    )


async def embed_questions(
    session: AsyncSession,
    embedder: Embedder,
    items: Sequence[BankItem],
    *,
    force: bool = False,
) -> EmbedResult:
    """Write a vector for every item whose stored one is missing or stale.

    ``force=True`` re-embeds everything regardless of fingerprint. It exists for
    the case the fingerprint cannot detect: the *model file* changing underneath
    a name that stayed the same. Not the default, because on a large bank it is
    the expensive option and it is almost never the one you need.

    Does not commit - the caller owns the transaction, which is what keeps rows
    and vectors atomic together.
    """
    result = EmbedResult(model_id=embedder.model_id)
    if not items:
        return result

    wanted = {item.id: document_text_for_item(item) for item in items}
    fingerprints = {
        qid: text_fingerprint(doc, model_id=embedder.model_id) for qid, doc in wanted.items()
    }

    existing = {
        row.id: (row.embedding_model, row.embedding_text_sha256, row.embedding is not None)
        for row in (
            await session.execute(
                select(
                    Question.id,
                    Question.embedding_model,
                    Question.embedding_text_sha256,
                    Question.embedding,
                ).where(Question.id.in_(list(wanted)))
            )
        ).all()
    }

    stale: list[str] = []
    for qid in wanted:
        row = existing.get(qid)
        if row is None:
            result.missing_rows.append(qid)
            continue
        model_id, fingerprint, has_vector = row
        current = has_vector and model_id == embedder.model_id and fingerprint == fingerprints[qid]
        if current and not force:
            result.reused += 1
        else:
            stale.append(qid)

    if not stale:
        return result

    vectors = embedder.embed_documents([wanted[qid] for qid in stale])
    for qid, vector in zip(stale, vectors, strict=True):
        await session.execute(
            update(Question)
            .where(Question.id == qid)
            .values(
                embedding=vector,
                embedding_model=embedder.model_id,
                embedding_text_sha256=fingerprints[qid],
            )
        )
        result.embedded += 1

    await session.flush()
    return result


__all__ = ["EmbedResult", "document_text_for_item", "embed_questions"]
