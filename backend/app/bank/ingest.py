"""Projecting the dataset files onto Postgres.

The files are the source of truth; these tables are a *materialisation* of them.
That is the whole design, and it has three consequences worth stating:

* **Idempotent.**  Ingest is an upsert keyed on the readable id, so running it
  twice is running it once.  A dataset that can only be loaded into an empty
  database is a dataset nobody re-loads, and one that drifts.
* **Validated first, written second.**  Nothing reaches the database until the
  whole bank passes ``validate_bank``.  A partially-ingested bank is worse than
  no bank: the retrieval layer cannot tell the difference.
* **Topics before questions.**  ``questions.topic_key`` and
  ``questions.subtopic_key`` are foreign keys into ``topics``, so the taxonomy
  is written first, in parent-before-child order.

Deletions are deliberately *not* automatic. An item removed from the files is
reported, never dropped: rows in ``turns`` reference ``questions.id``, and a
silent cascade would erase interview history to tidy up a dataset edit.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.bank.loader import BankReport
from app.bank.schema import BankItem
from app.bank.taxonomy import Taxonomy
from app.models.question import Question
from app.models.taxonomy import Topic


@dataclass(slots=True)
class IngestResult:
    """What ingest did, in numbers a caller can assert on."""

    topics_written: int = 0
    questions_written: int = 0
    #: In the database but no longer in the files. Reported, never deleted.
    orphaned_question_ids: list[str] = field(default_factory=list)
    unreviewed_question_ids: list[str] = field(default_factory=list)

    @property
    def summary(self) -> str:
        return (
            f"{self.topics_written} topics, {self.questions_written} questions; "
            f"{len(self.unreviewed_question_ids)} not yet human-reviewed, "
            f"{len(self.orphaned_question_ids)} orphaned rows left in place"
        )


def question_row(item: BankItem) -> dict[str, object]:
    """One ``questions`` row from one bank item.

    ``tsv`` is absent on purpose: it is a generated column, recomputed by
    Postgres on every write of ``text``, so the search index cannot drift from
    the thing it indexes.
    """
    return {
        "id": item.id,
        "topic_key": item.topic,
        "subtopic_key": item.subtopic,
        "text": item.text,
        "difficulty_b": item.difficulty_b,
        "discrimination_a": item.discrimination_a,
        "expected_concepts": [c.model_dump() for c in item.expected_concepts],
        "reference_answer": item.reference_answer,
        "follow_up_seeds": [{"prompt": seed} for seed in item.follow_up_seeds] or None,
        "anchor_terms": item.anchor_terms or None,
        "time_estimate_s": item.time_estimate_s,
        "tags": item.tags or None,
        "source": item.source,
        "version": item.version,
    }


async def ingest_bank(
    session: AsyncSession,
    report: BankReport,
    taxonomy: Taxonomy,
    *,
    only_reviewed: bool = False,
) -> IngestResult:
    """Upsert the taxonomy and every item in ``report`` into the database.

    ``only_reviewed=True`` is the production posture: plan section 6.4 says an
    item with a wrong concept key silently corrupts every score derived from it,
    so a deployment that serves real candidates must serve reviewed items only.
    The default is False so that a local run can exercise the whole dataset.
    """
    if not report.ok:
        raise ValueError(
            f"refusing to ingest a bank with {len(report.errors)} validation errors; "
            "fix the dataset first"
        )

    result = IngestResult()

    # --- taxonomy, parents before children -------------------------------
    for node in taxonomy.rows():
        statement = pg_insert(Topic).values(
            key=node.key,
            parent_key=node.parent_key,
            display_name=node.display_name,
            domain=node.domain,
        )
        await session.execute(
            statement.on_conflict_do_update(
                index_elements=[Topic.key],
                set_={
                    "parent_key": statement.excluded.parent_key,
                    "display_name": statement.excluded.display_name,
                    "domain": statement.excluded.domain,
                },
            )
        )
        result.topics_written += 1

    # --- questions --------------------------------------------------------
    selected = [
        loaded
        for loaded in report.items
        if not only_reviewed or loaded.item.review_status == "reviewed"
    ]
    for loaded in selected:
        row = question_row(loaded.item)
        statement = pg_insert(Question).values(**row)
        await session.execute(
            statement.on_conflict_do_update(
                index_elements=[Question.id],
                set_={key: statement.excluded[key] for key in row if key != "id"},
            )
        )
        result.questions_written += 1
        if loaded.item.review_status != "reviewed":
            result.unreviewed_question_ids.append(loaded.item.id)

    await session.flush()

    file_ids = {loaded.item.id for loaded in report.items}
    existing = set((await session.execute(select(Question.id))).scalars().all())
    result.orphaned_question_ids = sorted(existing - file_ids)
    return result


__all__ = ["IngestResult", "ingest_bank", "question_row"]
