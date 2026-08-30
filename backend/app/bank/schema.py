"""What one question-bank item must look like.

Plan section 6.3.  The shape mirrors ``app.models.question.Question`` because
ingest is a projection of these files onto that table - if the two drift, ingest
is the thing that breaks, and it breaks late.  Everything the database cannot
express (a concept count, a weight range, an anchor term that has to occur in
the question text) is expressed here instead.

**``extra="forbid"`` is the most valuable line in this file.**  A misspelt field
in a hand-edited JSONL row is otherwise silently dropped, and the item quietly
loses its ``expected_concepts`` - the one field the entire grading pipeline is
built on.

**Why review status lives in the file and not in the table.**  Review is a fact
about the *dataset*, established in a pull request, and it must survive being
re-ingested into an empty database.  Storing it in Postgres would make it a
property of one deployment's rows.  Plan section 6.4 is explicit that unreviewed
items are never acceptable; ``review_status`` is what makes that checkable
rather than assumed.
"""

from __future__ import annotations

import datetime as dt
import re
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

# A readable id: 'sys-cache-002'. The bank is reviewed as a git diff, and a diff
# full of UUIDs is unreviewable (see the comment on Question.id).
ID_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)+-\d{3}$")
CONCEPT_KEY_RE = re.compile(r"^[a-z0-9]+(?:_[a-z0-9]+)*$")

ReviewStatus = Literal["drafted", "reviewed"]
Source = Literal["authored", "llm_drafted", "imported"]

#: Plan section 6.3: fewer than 3 concepts makes the score too coarse
#: (0/0.33/0.67/1); more than 6 and no candidate covers them all in the time
#: allowed, so the ceiling is unreachable and the item stops discriminating.
MIN_CONCEPTS = 3
MAX_CONCEPTS = 6


class ExpectedConcept(BaseModel):
    """One checklist entry the grader classifies an answer against.

    ``weight`` is an integer 1-3 on purpose: a 1-10 scale is false precision
    that cannot be justified item by item.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    key: Annotated[str, Field(min_length=2, max_length=64)]
    weight: Annotated[int, Field(ge=1, le=3)]
    # What "covered" looks like, for the grader and for the reviewer. A concept
    # whose hint cannot be written is a concept that is not observable.
    hint: Annotated[str, Field(min_length=8, max_length=300)]

    @model_validator(mode="after")
    def _key_is_vocabulary_shaped(self) -> ExpectedConcept:
        if not CONCEPT_KEY_RE.match(self.key):
            raise ValueError(
                f"concept key {self.key!r} must be lower_snake_case - keys are a shared "
                "vocabulary across questions, and a stray capital makes a second one"
            )
        return self


class BankItem(BaseModel):
    """One question-bank item. One line of one ``.jsonl`` file."""

    model_config = ConfigDict(extra="forbid")

    id: str
    # Validated against the taxonomy in loader.py: pydantic cannot see the
    # vocabulary, and a model that reached out to load a file would be untestable.
    topic: str
    subtopic: str

    text: Annotated[str, Field(min_length=40, max_length=1200)]

    # IRT (plan section 5.8). b is difficulty on the same scale as candidate
    # ability theta, so HIGHER b IS HARDER; a is discrimination.
    difficulty_b: Annotated[float, Field(ge=-3.0, le=3.0)]
    discrimination_a: Annotated[float, Field(ge=0.3, le=2.5)] = 1.0

    expected_concepts: Annotated[
        list[ExpectedConcept], Field(min_length=MIN_CONCEPTS, max_length=MAX_CONCEPTS)
    ]
    common_misconceptions: list[Annotated[str, Field(min_length=8)]] = Field(default_factory=list)
    reference_answer: Annotated[str, Field(min_length=200)]
    follow_up_seeds: list[Annotated[str, Field(min_length=8)]] = Field(default_factory=list)

    # Plan section 6.2: after an item is re-rendered by the LLM for a candidate,
    # the rendered text is checked to still contain these. A cheap deterministic
    # guard against the model quietly changing what was asked.
    anchor_terms: Annotated[list[str], Field(min_length=1, max_length=8)]

    time_estimate_s: Annotated[int, Field(ge=60, le=900)]
    tags: Annotated[list[str], Field(min_length=1, max_length=8)]

    source: Source
    review_status: ReviewStatus
    reviewed_by: str | None = None
    reviewed_at: dt.date | None = None
    version: Annotated[int, Field(ge=1)] = 1

    # -- cross-field rules ------------------------------------------------
    @model_validator(mode="after")
    def _id_is_readable(self) -> BankItem:
        if not ID_RE.match(self.id):
            raise ValueError(
                f"id {self.id!r} must look like 'sys-cache-002': lower-case segments "
                "separated by hyphens, ending in a three-digit number"
            )
        return self

    @model_validator(mode="after")
    def _concept_keys_are_distinct(self) -> BankItem:
        keys = [c.key for c in self.expected_concepts]
        duplicates = sorted({k for k in keys if keys.count(k) > 1})
        if duplicates:
            raise ValueError(f"expected_concepts repeats {duplicates} - two entries score twice")
        return self

    @model_validator(mode="after")
    def _reference_answer_is_real(self) -> BankItem:
        if not self.reference_answer.strip():
            raise ValueError("reference_answer is blank")
        return self

    @model_validator(mode="after")
    def _anchor_terms_occur_in_the_text(self) -> BankItem:
        haystack = self.text.lower()
        missing = [t for t in self.anchor_terms if t.lower() not in haystack]
        if missing:
            raise ValueError(
                f"anchor_terms {missing} do not appear in the question text - an anchor "
                "term that is not in the canonical text cannot guard a rendering of it"
            )
        return self

    @model_validator(mode="after")
    def _review_fields_agree(self) -> BankItem:
        """The honesty rule: 'reviewed' is a claim, and a claim needs a claimant.

        An item may not be marked reviewed without a named reviewer and a date,
        and an item that is *not* reviewed may not carry either - which stops a
        half-finished review from reading as a finished one.
        """
        if self.review_status == "reviewed":
            if not (self.reviewed_by and self.reviewed_by.strip()):
                raise ValueError("review_status is 'reviewed' but reviewed_by is empty")
            if self.reviewed_at is None:
                raise ValueError("review_status is 'reviewed' but reviewed_at is missing")
        elif self.reviewed_by is not None or self.reviewed_at is not None:
            raise ValueError(
                "review_status is 'drafted' but reviewed_by/reviewed_at are set - "
                "set the status to 'reviewed' or clear both fields"
            )
        return self

    # -- derived ----------------------------------------------------------
    @property
    def concept_keys(self) -> list[str]:
        return [c.key for c in self.expected_concepts]

    @property
    def total_concept_weight(self) -> int:
        return sum(c.weight for c in self.expected_concepts)


#: The field order the dataset is written in. JSONL has no schema of its own, so
#: a stable order is what makes a diff of two versions of an item readable.
FIELD_ORDER: tuple[str, ...] = tuple(BankItem.model_fields)

__all__ = [
    "CONCEPT_KEY_RE",
    "FIELD_ORDER",
    "ID_RE",
    "MAX_CONCEPTS",
    "MIN_CONCEPTS",
    "BankItem",
    "ExpectedConcept",
    "ReviewStatus",
    "Source",
]
