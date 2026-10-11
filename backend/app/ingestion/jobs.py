"""Resume and JD extraction as durable background jobs.

Same machinery as Day 22's grading and Day 23's follow-up wording: the row is
the job, a worker claims it with ``FOR UPDATE SKIP LOCKED`` and a lease, the
model is called with no transaction open, and the result is written only if the
worker still owns the attempt it claimed (fence on ``attempts``).  Retries use
the same bounded :class:`~app.interview.grading.RetryPolicy` and the same
transient/permanent classification.

A failed extraction leaves ``extracted`` NULL and ``status = 'failed'`` with a
short error code.  **Nothing is fabricated in its place**: a candidate whose
resume could not be parsed can still type a profile (``PATCH .../profile``).

The model only ever sees the *redacted* text stored on the row.  The parser
writes ``extracted`` and never ``profile`` (see ``app.models.ingestion``).
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Literal

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.bank.taxonomy import Taxonomy
from app.ingestion.extraction import (
    EXTRACTION_VERSION,
    JdExtraction,
    ResumeExtraction,
    jd_inputs,
    resume_inputs,
    verify_jd,
    verify_resume,
)
from app.interview.grading import RetryPolicy, classify_error
from app.llm.client import call_structured
from app.llm.tasks import TaskName
from app.models.ingestion import (
    INGEST_FAILED,
    INGEST_PENDING,
    INGEST_PROCESSING,
    INGEST_READY,
    JobDescription,
    Resume,
)

log = logging.getLogger(__name__)

Kind = Literal["resume", "jd"]
_MODELS: dict[Kind, type[Resume] | type[JobDescription]] = {
    "resume": Resume,
    "jd": JobDescription,
}
# Table names are fixed strings chosen here, never from input.
_CLAIM = {
    kind: sa.text(
        f"""
        UPDATE {table} AS t
           SET status = 'processing', attempts = t.attempts + 1,
               lease_expires_at = now() + make_interval(secs => :lease_s),
               next_attempt_at = NULL
         WHERE t.id = (
               SELECT id FROM {table}
                WHERE (status = 'pending'
                       AND (next_attempt_at IS NULL OR next_attempt_at <= now()))
                   OR (status = 'processing' AND lease_expires_at < now())
                ORDER BY created_at, id
                FOR UPDATE SKIP LOCKED
                LIMIT 1)
        RETURNING t.id, t.attempts, t.redacted_text
        """  # noqa: S608 - table names are the two literals below
    )
    for kind, table in (("resume", "resumes"), ("jd", "job_descriptions"))
}


@dataclass(frozen=True, slots=True)
class IngestionJob:
    kind: Kind
    row_id: uuid.UUID
    attempt: int
    redacted_text: str


#: (job, taxonomy) -> the verified suggestion.  Production: :func:`extract_with_model`.
ExtractFn = Callable[[IngestionJob, Taxonomy], Awaitable[dict[str, Any]]]


async def extract_with_model(job: IngestionJob, taxonomy: Taxonomy) -> dict[str, Any]:
    """One schema-constrained chokepoint call, then verification in code."""
    if job.kind == "resume":
        resume, _ = await call_structured(
            TaskName.RESUME_EXTRACTION,
            resume_inputs(job.redacted_text, taxonomy),
            ResumeExtraction,
        )
        return verify_resume(resume, text=job.redacted_text, taxonomy=taxonomy)
    jd, _ = await call_structured(
        TaskName.JD_EXTRACTION, jd_inputs(job.redacted_text, taxonomy), JdExtraction
    )
    return verify_jd(jd, text=job.redacted_text, taxonomy=taxonomy)


async def claim_next(
    maker: async_sessionmaker[AsyncSession], kind: Kind, policy: RetryPolicy
) -> IngestionJob | None:
    async with maker() as db, db.begin():
        row = (await db.execute(_CLAIM[kind], {"lease_s": policy.lease_s})).first()
    if row is None:
        return None
    return IngestionJob(
        kind=kind, row_id=row.id, attempt=row.attempts, redacted_text=row.redacted_text
    )


class IngestionOutcome:
    READY = "ingestion_ready"
    RETRY_SCHEDULED = "ingestion_retry_scheduled"
    FAILED = "ingestion_failed"
    STALE = "ingestion_stale"


async def _finish(
    maker: async_sessionmaker[AsyncSession], job: IngestionJob, values: dict[str, Any]
) -> bool:
    """Fenced write: only if this worker still owns the attempt it claimed."""
    model = _MODELS[job.kind]
    async with maker() as db, db.begin():
        result = await db.execute(
            sa.update(model)
            .where(
                model.id == job.row_id,
                model.status == INGEST_PROCESSING,
                model.attempts == job.attempt,
            )
            .values(lease_expires_at=None, **values)
        )
        return bool(result.rowcount)  # type: ignore[attr-defined]


async def process(
    maker: async_sessionmaker[AsyncSession],
    job: IngestionJob,
    *,
    taxonomy: Taxonomy,
    extract_fn: ExtractFn,
    policy: RetryPolicy,
) -> str:
    if job.attempt > policy.max_attempts:
        ok = await _finish(maker, job, {"status": INGEST_FAILED, "error": "retry_exhausted"})
        return IngestionOutcome.FAILED if ok else IngestionOutcome.STALE
    try:
        extracted = await asyncio.wait_for(extract_fn(job, taxonomy), timeout=policy.timeout_s)
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001 - classified and recorded, never swallowed
        code, transient = classify_error(exc)
        # The exception *type* only: a provider message can quote the document.
        log.warning(
            "extraction failed",
            extra={
                "ingestion.kind": job.kind,
                "ingestion.attempt": job.attempt,
                "ingestion.error_code": code,
                "ingestion.exception": type(exc).__name__,
            },
        )
        if transient and job.attempt < policy.max_attempts:
            ok = await _finish(
                maker,
                job,
                {
                    "status": INGEST_PENDING,
                    "error": code,
                    "next_attempt_at": sa.func.now()
                    + sa.func.make_interval(0, 0, 0, 0, 0, 0, policy.delay_after(job.attempt)),
                },
            )
            return IngestionOutcome.RETRY_SCHEDULED if ok else IngestionOutcome.STALE
        ok = await _finish(maker, job, {"status": INGEST_FAILED, "error": code})
        return IngestionOutcome.FAILED if ok else IngestionOutcome.STALE
    ok = await _finish(
        maker,
        job,
        {
            "status": INGEST_READY,
            "error": None,
            "extracted": extracted,
            "parser_version": EXTRACTION_VERSION,
            "parsed_at": sa.func.now(),
        },
    )
    return IngestionOutcome.READY if ok else IngestionOutcome.STALE


async def run_once_ingestion(
    maker: async_sessionmaker[AsyncSession],
    *,
    taxonomy: Taxonomy,
    extract_fn: ExtractFn,
    policy: RetryPolicy,
) -> str | None:
    """Claim and process at most one resume or JD job (resumes first)."""
    for kind in ("resume", "jd"):
        job = await claim_next(maker, kind, policy)
        if job is not None:
            return await process(
                maker, job, taxonomy=taxonomy, extract_fn=extract_fn, policy=policy
            )
    return None


__all__ = [
    "ExtractFn",
    "IngestionJob",
    "IngestionOutcome",
    "claim_next",
    "extract_with_model",
    "process",
    "run_once_ingestion",
]
