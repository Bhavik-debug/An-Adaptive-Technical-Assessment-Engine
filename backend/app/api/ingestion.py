"""Resumes, the editable profile, and job descriptions (Day 24).

Plan section 13.8::

    POST  /api/resumes                 -> 202 {resume_id, status}
    GET   /api/resumes/{id}            -> {suggestion, profile, status}
    PATCH /api/resumes/{id}/profile    -> the candidate corrects the extraction
    POST  /api/job-descriptions        -> 202 {jd_id, status}        (async; see docs)
    GET   /api/job-descriptions/{id}   -> {requirements, topic_weights, status}

**Upload format.** The resume is sent as the raw request body (``Content-Type:
application/pdf`` or ``text/plain``), not multipart: no form parser, no spooled
temporary file, and the size cap is enforced while reading the stream, before
the whole body is in memory.  The ``Content-Type`` header is *not* trusted -
the bytes are sniffed (``app.ingestion.files``).

Every route resolves the caller from the bearer token and filters every query
by that user id.  Another user's resume or JD is the same 404 as a missing one.
Error bodies carry a code and a fixed message; they never quote the document.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import uuid
from functools import lru_cache
from typing import Annotated, Any

import sqlalchemy as sa
from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.bank.taxonomy import Taxonomy, load_taxonomy
from app.db import get_sessionmaker
from app.deps import AppSettings, CurrentUser
from app.ingestion.files import DocumentRejected, extract_text, normalise
from app.ingestion.profile import CandidateProfile
from app.models.ingestion import INGEST_PENDING, JobDescription, Resume
from app.privacy import JD_KINDS, redact

router = APIRouter(prefix="/api", tags=["ingestion"])
Maker = Annotated[async_sessionmaker[AsyncSession], Depends(get_sessionmaker)]


@lru_cache(maxsize=1)
def taxonomy() -> Taxonomy:
    return load_taxonomy()


def _reject(
    code: str, message: str, http: int = status.HTTP_422_UNPROCESSABLE_ENTITY
) -> HTTPException:
    return HTTPException(http, detail={"code": code, "message": message})


_HTTP_FOR = {
    "file_too_large": status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
    "unsupported_type": status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
}


# ---------------------------------------------------------------------------
# Response shapes
# ---------------------------------------------------------------------------


class ResumeOut(BaseModel):
    resume_id: uuid.UUID
    status: str
    error: str | None
    document_kind: str
    page_count: int | None
    truncated: bool
    #: Counts of what was redacted before storage - never the values.
    redactions: dict[str, int]
    #: The parser's verified suggestion. Not trusted: seeds nothing until the
    #: candidate confirms it via PATCH .../profile.
    suggestion: dict[str, Any] | None
    #: What the candidate confirmed or typed. The only input to priors.
    profile: dict[str, Any] | None
    profile_confirmed: bool
    created_at: dt.datetime
    parsed_at: dt.datetime | None
    profile_updated_at: dt.datetime | None


class JobDescriptionIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1, max_length=200_000)

    @field_validator("text")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("must not be blank")
        return v


class JobDescriptionOut(BaseModel):
    jd_id: uuid.UUID
    status: str
    error: str | None
    redactions: dict[str, int]
    title: str | None
    requirements: list[dict[str, Any]]
    #: Computed in code from the requirement classifications (topic -> [0, 1]).
    topic_weights: dict[str, float]
    min_years_experience: int | None
    created_at: dt.datetime
    parsed_at: dt.datetime | None


def _resume_out(r: Resume) -> ResumeOut:
    return ResumeOut(
        resume_id=r.id,
        status=r.status,
        error=r.error,
        document_kind=r.document_kind,
        page_count=r.page_count,
        truncated=r.truncated,
        redactions=dict(r.redaction_counts or {}),
        suggestion=r.extracted,
        profile=r.profile,
        profile_confirmed=r.profile is not None,
        created_at=r.created_at,
        parsed_at=r.parsed_at,
        profile_updated_at=r.profile_updated_at,
    )


def _jd_out(j: JobDescription) -> JobDescriptionOut:
    extracted = j.extracted or {}
    return JobDescriptionOut(
        jd_id=j.id,
        status=j.status,
        error=j.error,
        redactions=dict(j.redaction_counts or {}),
        title=extracted.get("title"),
        requirements=list(extracted.get("requirements", [])),
        topic_weights=dict(extracted.get("topic_weights", {})),
        min_years_experience=extracted.get("min_years_experience"),
        created_at=j.created_at,
        parsed_at=j.parsed_at,
    )


async def _owned(db: AsyncSession, model: Any, row_id: uuid.UUID, user_id: uuid.UUID) -> Any:
    row = await db.scalar(
        sa.select(model)
        .where(model.id == row_id, model.user_id == user_id)
        .execution_options(populate_existing=True)
    )
    if row is None:
        what = "Resume" if model is Resume else "Job description"
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=f"{what} not found")
    return row


# ---------------------------------------------------------------------------
# Resumes
# ---------------------------------------------------------------------------


async def _read_capped(request: Request, cap: int) -> bytes:
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > cap:
        raise _reject("file_too_large", "The file is larger than the upload limit.", 413)
    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > cap:
            raise _reject("file_too_large", "The file is larger than the upload limit.", 413)
    return bytes(body)


@router.post(
    "/resumes",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Upload a resume (raw PDF or UTF-8 text body)",
    openapi_extra={
        "requestBody": {
            "required": True,
            "content": {
                "application/pdf": {"schema": {"type": "string", "format": "binary"}},
                "text/plain": {"schema": {"type": "string"}},
            },
        }
    },
)
async def upload_resume(
    request: Request, user: CurrentUser, maker: Maker, settings: AppSettings
) -> ResumeOut:
    data = await _read_capped(request, settings.resume_max_bytes)
    try:
        document = await extract_text(data, settings)
    except DocumentRejected as exc:
        raise _reject(exc.code, str(exc), _HTTP_FOR.get(exc.code, 422)) from None
    redacted = redact(document.text)
    resume = Resume(
        user_id=user.id,
        status=INGEST_PENDING,
        document_kind=document.kind,
        page_count=document.pages,
        truncated=document.truncated,
        source_sha256=hashlib.sha256(data).hexdigest(),
        redacted_text=redacted.text,
        redaction_counts=redacted.counts,
    )
    async with maker() as db:
        db.add(resume)
        await db.commit()
        await db.refresh(resume)
    return _resume_out(resume)


@router.get("/resumes/{resume_id}", summary="A resume's status, suggestion and profile")
async def get_resume(resume_id: uuid.UUID, user: CurrentUser, maker: Maker) -> ResumeOut:
    async with maker() as db:
        return _resume_out(await _owned(db, Resume, resume_id, user.id))


@router.patch(
    "/resumes/{resume_id}/profile",
    summary="Confirm or correct the profile (replaces it as a whole)",
)
async def edit_profile(
    resume_id: uuid.UUID, body: CandidateProfile, user: CurrentUser, maker: Maker
) -> ResumeOut:
    known = {n.key for n in taxonomy().subtopics()}
    unknown = body.unknown_subtopics(known)
    if unknown:
        raise _reject("unknown_subtopic", f"Unknown subtopic keys: {', '.join(unknown[:5])}")
    async with maker() as db:
        resume = await _owned(db, Resume, resume_id, user.id)
        # The whole profile is replaced: the request *is* the candidate's view.
        # The parser never writes this column, so nothing can later overwrite it.
        resume.profile = body.model_dump(mode="json")
        resume.profile_updated_at = sa.func.now()
        await db.commit()
        await db.refresh(resume)
        return _resume_out(resume)


# ---------------------------------------------------------------------------
# Job descriptions
# ---------------------------------------------------------------------------


@router.post(
    "/job-descriptions",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit a job description (pasted text)",
)
async def submit_jd(
    body: JobDescriptionIn, user: CurrentUser, maker: Maker, settings: AppSettings
) -> JobDescriptionOut:
    text = normalise(body.text)
    if len(text) < settings.jd_min_chars:
        raise _reject("too_short", "The job description is too short to extract requirements.")
    if len(text) > settings.jd_max_chars:
        raise _reject("too_long", "The job description is longer than the limit.", 413)
    redacted = redact(text, JD_KINDS)
    jd = JobDescription(
        user_id=user.id,
        status=INGEST_PENDING,
        redacted_text=redacted.text,
        redaction_counts=redacted.counts,
    )
    async with maker() as db:
        db.add(jd)
        await db.commit()
        await db.refresh(jd)
    return _jd_out(jd)


@router.get("/job-descriptions/{jd_id}", summary="A job description's requirements and weights")
async def get_jd(jd_id: uuid.UUID, user: CurrentUser, maker: Maker) -> JobDescriptionOut:
    async with maker() as db:
        return _jd_out(await _owned(db, JobDescription, jd_id, user.id))


__all__ = ["router", "taxonomy"]
