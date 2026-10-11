"""Starting a session: ``CREATED -> PREPARING -> ASKING`` with its first question.

One transaction, under the session lock::

    PREPARATION_STARTED {priors}            (Day 24: capped, from the confirmed profile)
    select the first item (Day 12, the blueprint's selection state)
    QUESTION_ASKED {question_id, text}

If the bank has nothing this blueprint allows, nothing is written and
:class:`~app.interview.flow.NoEligibleQuestion` is raised - a session must not
be left in ``PREPARING`` with no way forward.

**Repeated start.**  The first call starts the session.  A repeat while the
session is ``ASKING`` returns the question currently in front of the candidate
and writes nothing - so a client whose first response was lost can recover it.
A repeat in any other state is refused (``IllegalTransition``): there is no
question to hand back while an answer is being graded.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.blueprint import Blueprint
from app.ingestion.profile import parse_profile
from app.interview.cache import SessionStateCache
from app.interview.flow import stage_next_bank_question
from app.interview.fsm import EventType, IllegalTransition, InterviewState
from app.interview.priors import compute_priors, priors_payload
from app.interview.service import commit_and_cache
from app.interview.state import SessionState, fold
from app.interview.store import lock_session, read_events, stage_append
from app.models.ingestion import Resume


@dataclass(frozen=True, slots=True)
class StartResult:
    state: SessionState
    #: False when this call found the session already started (a replay).
    started: bool


async def _preparation(
    db: AsyncSession, resume_id: uuid.UUID | None, user_id: uuid.UUID, blueprint: Blueprint
) -> dict[str, Any]:
    """Day 24: the priors for this session, recorded in PREPARATION_STARTED.

    Read once, here, from the candidate-*confirmed* profile.  An unconfirmed
    resume (only a parser suggestion) seeds nothing.  Ownership is re-checked
    even though session creation checked it: the row is read by user_id.
    """
    if resume_id is None:
        return {"priors": {}, "prior_source": "none"}
    profile = parse_profile(
        await db.scalar(
            sa.select(Resume.profile).where(Resume.id == resume_id, Resume.user_id == user_id)
        )
    )
    priors = compute_priors(profile, blueprint)
    if not priors:
        return {"priors": {}, "prior_source": "no_profile"}
    return {"priors": priors_payload(priors), "prior_source": "resume_profile"}


async def start_session(
    maker: async_sessionmaker[AsyncSession],
    cache: SessionStateCache | None,
    *,
    session_id: uuid.UUID,
    user_id: uuid.UUID,
) -> StartResult:
    async with maker() as db:
        try:
            session = await lock_session(db, session_id=session_id, user_id=user_id)
            current = fold(session_id, await read_events(db, session_id))
            if current.state is not InterviewState.CREATED:
                if current.state is InterviewState.ASKING:
                    await db.rollback()
                    return StartResult(current, started=False)
                raise IllegalTransition(
                    current.state,
                    EventType.PREPARATION_STARTED.value,
                    "the session has already started and is not waiting for an answer",
                )
            blueprint = Blueprint.from_dict(session.blueprint)
            preparing = await stage_append(
                db,
                session_id=session_id,
                user_id=user_id,
                event_type=EventType.PREPARATION_STARTED,
                expected_seq=current.seq,
                payload=await _preparation(db, session.resume_id, user_id, blueprint),
            )
            asking = await stage_next_bank_question(
                db, preparing, user_id=user_id, blueprint=blueprint
            )
        except BaseException:
            await db.rollback()
            raise
        state = await commit_and_cache(db, cache, asking)
    return StartResult(state, started=True)


__all__ = ["StartResult", "start_session"]
