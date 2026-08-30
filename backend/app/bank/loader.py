"""Reading ``data/question-bank/*.jsonl``, and every check that spans two items.

``schema.py`` can tell you an item is well-formed.  It cannot tell you that its
id collides with another file's, that its subtopic does not belong to its topic,
or that it is a paraphrase of the question three lines above it.  Those are
properties of the *dataset*, and they live here.

**Every error is collected, none is raised.**  A validator that stops at the
first bad line turns a fifteen-minute fix into fifteen one-minute runs.  The
report is the product; the exit code is a summary of it.

**Two failure severities.**

* *errors* - the dataset is wrong.  Duplicate ids, an unknown subtopic, a
  near-verbatim duplicate.  These fail CI.
* *warnings* - the dataset is suspicious.  A concept key one character away from
  another key (``hash_collision`` vs ``hash_collisions`` are two vocabularies,
  and the second one silently never accumulates evidence), a family of items
  that overlap heavily without being duplicates.  These are for a human.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import ValidationError

from app.bank.paths import BANK_DIR
from app.bank.schema import BankItem
from app.bank.taxonomy import Taxonomy, load_taxonomy

#: Two questions whose word 4-grams overlap this much are the same question
#: wearing a different hat. Rejected: a duplicate item is asked twice in one
#: interview and counted as two independent pieces of evidence, which it is not.
DUPLICATE_ERROR_JACCARD = 0.60
#: Below that, but still high enough that a human should look.
DUPLICATE_WARN_JACCARD = 0.38

_WORD_RE = re.compile(r"[a-z0-9]+")
_SHINGLE = 4


@dataclass(frozen=True, slots=True)
class LoadedItem:
    """An item, plus where it came from - so every message can point at a line."""

    item: BankItem
    path: Path
    line: int
    domain: str

    @property
    def where(self) -> str:
        return f"{self.path.name}:{self.line}"


@dataclass(slots=True)
class BankReport:
    """The result of validating the whole bank."""

    items: list[LoadedItem] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    files: list[Path] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    @property
    def count(self) -> int:
        return len(self.items)

    # -- statistics, for the report the reviewer actually reads -----------
    def by_domain(self) -> Counter[str]:
        return Counter(loaded.domain for loaded in self.items)

    def by_topic(self) -> Counter[str]:
        return Counter(loaded.item.topic for loaded in self.items)

    def by_subtopic(self) -> Counter[str]:
        return Counter(loaded.item.subtopic for loaded in self.items)

    def by_review_status(self) -> Counter[str]:
        return Counter(loaded.item.review_status for loaded in self.items)

    def difficulty_histogram(self) -> Counter[str]:
        """Half-open bands of width 1.0 across the legal range of ``b``."""
        bands: Counter[str] = Counter()
        for loaded in self.items:
            low = int(loaded.item.difficulty_b // 1)
            bands[f"[{low:+d}, {low + 1:+d})"] += 1
        return bands

    def concept_vocabulary(self) -> Counter[str]:
        counts: Counter[str] = Counter()
        for loaded in self.items:
            counts.update(loaded.item.concept_keys)
        return counts

    def unreviewed(self) -> list[LoadedItem]:
        return [loaded for loaded in self.items if loaded.item.review_status != "reviewed"]


# ---------------------------------------------------------------------------
# reading
# ---------------------------------------------------------------------------


def bank_files(bank_dir: Path | None = None) -> list[Path]:
    """Every dataset file, in a stable order. ``taxonomy.json`` is not one."""
    target = BANK_DIR if bank_dir is None else bank_dir
    return sorted(target.glob("*.jsonl"))


def _format_validation_error(exc: ValidationError) -> str:
    parts = []
    for err in exc.errors():
        location = ".".join(str(p) for p in err["loc"]) or "<item>"
        parts.append(f"{location}: {err['msg']}")
    return "; ".join(parts)


def load_bank(
    bank_dir: Path | None = None, taxonomy: Taxonomy | None = None
) -> tuple[list[LoadedItem], list[str]]:
    """Parse every ``.jsonl`` file. Returns the items that parsed and the errors.

    Blank lines are skipped; a ``#`` comment line is not allowed, because JSONL
    has no comments and a tool that invents one produces files other tools
    cannot read.
    """
    tax = load_taxonomy() if taxonomy is None else taxonomy
    items: list[LoadedItem] = []
    errors: list[str] = []

    for path in bank_files(bank_dir):
        for line_no, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not raw_line.strip():
                continue
            where = f"{path.name}:{line_no}"
            try:
                payload = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                errors.append(f"{where}: not valid JSON ({exc.msg} at column {exc.colno})")
                continue
            if not isinstance(payload, dict):
                errors.append(f"{where}: a JSONL record must be a JSON object")
                continue
            try:
                item = BankItem.model_validate(payload)
            except ValidationError as exc:
                item_id = payload.get("id", "<no id>")
                errors.append(f"{where} ({item_id}): {_format_validation_error(exc)}")
                continue
            domain = tax.domain_of(item.subtopic) or "<unknown>"
            items.append(LoadedItem(item=item, path=path, line=line_no, domain=domain))

    return items, errors


# ---------------------------------------------------------------------------
# dataset-level checks
# ---------------------------------------------------------------------------


def _check_taxonomy(items: list[LoadedItem], tax: Taxonomy) -> list[str]:
    errors: list[str] = []
    for loaded in items:
        item = loaded.item
        topic_node = tax.get(item.topic)
        sub_node = tax.get(item.subtopic)
        if topic_node is None:
            errors.append(f"{loaded.where} ({item.id}): unknown topic '{item.topic}'")
        elif topic_node.level != "topic":
            errors.append(
                f"{loaded.where} ({item.id}): '{item.topic}' is a {topic_node.level}, "
                "not a topic - the topic field takes the middle level"
            )
        if sub_node is None:
            errors.append(f"{loaded.where} ({item.id}): unknown subtopic '{item.subtopic}'")
        elif sub_node.level != "subtopic":
            errors.append(
                f"{loaded.where} ({item.id}): '{item.subtopic}' is a {sub_node.level}, "
                "not a subtopic"
            )
        elif topic_node is not None and sub_node.parent_key != item.topic:
            errors.append(
                f"{loaded.where} ({item.id}): subtopic '{item.subtopic}' belongs to topic "
                f"'{sub_node.parent_key}', not '{item.topic}'"
            )
    return errors


def _check_unique_ids(items: list[LoadedItem]) -> list[str]:
    seen: dict[str, LoadedItem] = {}
    errors: list[str] = []
    for loaded in items:
        first = seen.get(loaded.item.id)
        if first is None:
            seen[loaded.item.id] = loaded
        else:
            errors.append(
                f"{loaded.where}: duplicate id '{loaded.item.id}', first seen at {first.where}"
            )
    return errors


def _check_one_domain_per_file(items: list[LoadedItem]) -> list[str]:
    """A file is the unit of review. Mixing domains in one makes it unreviewable."""
    per_file: dict[Path, set[str]] = {}
    for loaded in items:
        per_file.setdefault(loaded.path, set()).add(loaded.domain)
    return [
        f"{path.name}: contains items from more than one domain ({', '.join(sorted(domains))})"
        for path, domains in sorted(per_file.items())
        if len(domains) > 1
    ]


def _shingles(text: str) -> set[tuple[str, ...]]:
    words = _WORD_RE.findall(text.lower())
    if len(words) < _SHINGLE:
        return {tuple(words)}
    return {tuple(words[i : i + _SHINGLE]) for i in range(len(words) - _SHINGLE + 1)}


def _jaccard(a: set[tuple[str, ...]], b: set[tuple[str, ...]]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _check_near_duplicates(items: list[LoadedItem]) -> tuple[list[str], list[str]]:
    """Word-4-gram Jaccard over question text.

    Deliberately lexical rather than semantic: this runs in CI with no model and
    no embeddings, and it catches the failure that actually happens - the same
    question re-drafted with two words changed. Genuine semantic duplicates are
    what the Day 10 retrieval eval surfaces.
    """
    errors: list[str] = []
    warnings: list[str] = []
    shingles = [(loaded, _shingles(loaded.item.text)) for loaded in items]
    for i in range(len(shingles)):
        left, left_shingles = shingles[i]
        for j in range(i + 1, len(shingles)):
            right, right_shingles = shingles[j]
            score = _jaccard(left_shingles, right_shingles)
            if score >= DUPLICATE_ERROR_JACCARD:
                errors.append(
                    f"{left.item.id} ({left.where}) and {right.item.id} ({right.where}) are "
                    f"near-duplicates (4-gram Jaccard {score:.2f})"
                )
            elif score >= DUPLICATE_WARN_JACCARD:
                warnings.append(
                    f"{left.item.id} and {right.item.id} overlap heavily "
                    f"(4-gram Jaccard {score:.2f}) - check they test different things"
                )
    return errors, warnings


def _edit_distance_at_most_one(a: str, b: str) -> bool:
    """True when ``a`` becomes ``b`` with one insert, delete or substitution."""
    if abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        return sum(x != y for x, y in zip(a, b, strict=True)) == 1
    shorter, longer = (a, b) if len(a) < len(b) else (b, a)
    i = j = 0
    skipped = False
    while i < len(shorter) and j < len(longer):
        if shorter[i] == longer[j]:
            i += 1
            j += 1
            continue
        if skipped:
            return False
        skipped = True
        j += 1
    return True


def _check_concept_vocabulary(items: list[LoadedItem]) -> list[str]:
    """Near-identical concept keys are the most expensive typo in this project.

    ``hash_collision`` and ``hash_collisions`` are two vocabulary entries. Every
    score derived from either is computed over half the evidence, silently, and
    the per-concept report that is the whole point of the grader says a candidate
    has never been tested on one of them.
    """
    keys = sorted({key for loaded in items for key in loaded.item.concept_keys})
    warnings: list[str] = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            if _edit_distance_at_most_one(keys[i], keys[j]):
                warnings.append(
                    f"concept keys '{keys[i]}' and '{keys[j]}' differ by one character - "
                    "if they mean the same thing they must be one key"
                )
    return warnings


def validate_bank(bank_dir: Path | None = None, taxonomy: Taxonomy | None = None) -> BankReport:
    """Load and fully validate the bank. Never raises for a data problem."""
    tax = load_taxonomy() if taxonomy is None else taxonomy
    files = bank_files(bank_dir)
    items, errors = load_bank(bank_dir, tax)

    report = BankReport(items=items, errors=list(errors), files=files)
    if not files:
        report.errors.append(f"no *.jsonl files in {BANK_DIR if bank_dir is None else bank_dir}")
        return report

    report.errors.extend(_check_unique_ids(items))
    report.errors.extend(_check_taxonomy(items, tax))
    report.errors.extend(_check_one_domain_per_file(items))
    duplicate_errors, duplicate_warnings = _check_near_duplicates(items)
    report.errors.extend(duplicate_errors)
    report.warnings.extend(duplicate_warnings)
    report.warnings.extend(_check_concept_vocabulary(items))
    return report


__all__ = [
    "DUPLICATE_ERROR_JACCARD",
    "DUPLICATE_WARN_JACCARD",
    "BankReport",
    "LoadedItem",
    "bank_files",
    "load_bank",
    "validate_bank",
]
