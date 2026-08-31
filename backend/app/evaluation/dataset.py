"""The retrieval evaluation dataset: queries with known-relevant question ids.

**Ground truth, and why it is a separate thing from the question bank.**

    QUESTION BANK  (data/question-bank/*.jsonl)
        what the system can retrieve - 60 questions

    EVALUATION SET (evals/datasets/retrieval_queries.jsonl)
        what we search FOR, and which questions we have decided are the
        right answers - used only to judge the retriever

They are never mixed. The bank is the product; this is the ruler held up
against it. A retriever evaluated on the bank alone could only be checked for
"did it return something", which is what Days 8 and 9 already established.

**Where the labels come from, honestly.** They were assigned by reading all 60
questions and deciding, per query, which ones a person asking that query would
want. That makes them *considered*, not *authoritative*: one author, no second
opinion, no inter-annotator agreement. See `docs/evaluation.md` for the full
list of limitations.

**Grades are 0-2** (plan section 12.1):

    2 = directly answers the query
    1 = genuinely related and reasonable to return, but not the target
    0 = not relevant, and therefore never written down

A query with no grade >= 1 is rejected: it cannot be scored, and a query nobody
can answer teaches nothing about the retriever.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from app.bank.paths import REPO_ROOT
from app.evaluation.metrics import MAX_GRADE

#: Plan section 13.9 fixes this layout: `evals/{datasets,suites,cache,reports}/`.
EVALS_DIR = REPO_ROOT / "evals"
DATASET_PATH = EVALS_DIR / "datasets" / "retrieval_queries.jsonl"
REPORTS_DIR = EVALS_DIR / "reports"

#: What retrieval situation a query is designed to exercise. Recorded so the
#: report can break results down by case - an average over mixed cases hides
#: precisely the thing an ablation is meant to expose.
QueryKind = Literal["semantic", "lexical", "hybrid", "rerank", "hard"]


class EvalQueryError(ValueError):
    """The evaluation dataset is malformed. Always fatal: it is the ruler."""


class EvalQuery(BaseModel):
    """One labelled query."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=3, max_length=32)
    #: Minimum 2 characters, not 5: a bare acronym like "MVCC" is a real query
    #: a person types, and one of the hardest cases in the set.
    query: Annotated[str, Field(min_length=2, max_length=300)]
    #: question id -> grade. At least one entry must be graded >= 1.
    relevant: dict[str, Annotated[int, Field(ge=0, le=MAX_GRADE)]]
    #: Which retrieval situation this query exercises.
    kind: QueryKind
    #: Why these ids were chosen. Written for the reviewer, not the machine -
    #: an unexplained label is one nobody can argue with, which is the opposite
    #: of what ground truth should be.
    note: Annotated[str, Field(min_length=10, max_length=400)]

    @model_validator(mode="after")
    def _has_something_relevant(self) -> EvalQuery:
        if not self.relevant_ids:
            raise ValueError(
                "no question graded >= 1: a query with no right answer cannot be scored"
            )
        return self

    @property
    def relevant_ids(self) -> set[str]:
        """The yes/no cut used by recall and MRR: grade >= 1."""
        return {qid for qid, grade in self.relevant.items() if grade >= 1}

    @property
    def primary_ids(self) -> set[str]:
        """Only the grade-2 questions - the ones the query is really about."""
        return {qid for qid, grade in self.relevant.items() if grade == MAX_GRADE}


@dataclass(frozen=True, slots=True)
class EvalDataset:
    """Every labelled query, plus the checks that span more than one of them."""

    queries: list[EvalQuery]
    path: Path

    def __len__(self) -> int:
        return len(self.queries)

    def __iter__(self) -> Iterator[EvalQuery]:
        return iter(self.queries)

    def by_kind(self) -> dict[str, list[EvalQuery]]:
        grouped: dict[str, list[EvalQuery]] = {}
        for query in self.queries:
            grouped.setdefault(query.kind, []).append(query)
        return grouped

    @property
    def labelled_question_ids(self) -> set[str]:
        return {qid for query in self.queries for qid in query.relevant_ids}


def load_eval_dataset(
    path: Path | None = None, *, known_question_ids: set[str] | None = None
) -> EvalDataset:
    """Read and validate the evaluation set.

    ``known_question_ids`` cross-checks every label against the actual bank.
    This is the check that matters most: a label pointing at a question id that
    does not exist is silently unreachable, so the query looks like a retrieval
    failure forever and no amount of tuning fixes it. Passing the real bank ids
    turns that into a load-time error.
    """
    target = DATASET_PATH if path is None else path
    try:
        raw_lines = target.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as exc:
        raise EvalQueryError(f"no evaluation dataset at {target}") from exc

    queries: list[EvalQuery] = []
    errors: list[str] = []
    seen: dict[str, int] = {}

    for line_no, raw in enumerate(raw_lines, start=1):
        if not raw.strip():
            continue
        where = f"{target.name}:{line_no}"
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            errors.append(f"{where}: not valid JSON ({exc.msg})")
            continue
        try:
            query = EvalQuery.model_validate(payload)
        except ValidationError as exc:
            detail = "; ".join(
                f"{'.'.join(str(p) for p in e['loc']) or '<query>'}: {e['msg']}"
                for e in exc.errors()
            )
            errors.append(f"{where}: {detail}")
            continue

        if query.id in seen:
            errors.append(
                f"{where}: duplicate query id '{query.id}' (first at line {seen[query.id]})"
            )
        else:
            seen[query.id] = line_no
        queries.append(query)

    if known_question_ids is not None:
        for query in queries:
            unknown = sorted(set(query.relevant) - known_question_ids)
            if unknown:
                errors.append(
                    f"{query.id}: labels point at question ids that are not in the bank: "
                    f"{', '.join(unknown)}"
                )

    if errors:
        raise EvalQueryError(f"{len(errors)} problem(s) in {target}:\n  " + "\n  ".join(errors))
    if not queries:
        raise EvalQueryError(f"{target} contains no queries")
    return EvalDataset(queries=queries, path=target)


__all__ = [
    "DATASET_PATH",
    "EVALS_DIR",
    "REPORTS_DIR",
    "EvalDataset",
    "EvalQuery",
    "EvalQueryError",
    "QueryKind",
    "load_eval_dataset",
]
