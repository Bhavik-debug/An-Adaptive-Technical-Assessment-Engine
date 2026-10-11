"""SQLAlchemy models.

Every model must be imported here. Alembic's autogenerate compares
``Base.metadata`` with the live database, and a model that was never imported is
not in the metadata - so it would silently be left out of every migration.
"""

from app.models.base import Base
from app.models.ingestion import JobDescription, Resume
from app.models.interview import FollowUp, InterviewEvent, InterviewSession, Turn
from app.models.question import Question
from app.models.reporting import Report
from app.models.skill import SkillState
from app.models.taxonomy import Topic
from app.models.user import User

__all__ = [
    "Base",
    "FollowUp",
    "InterviewEvent",
    "InterviewSession",
    "JobDescription",
    "Question",
    "Report",
    "Resume",
    "SkillState",
    "Topic",
    "Turn",
    "User",
]
