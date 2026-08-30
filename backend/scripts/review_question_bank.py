"""The manual review workflow for the question bank.

    cd backend
    python scripts/review_question_bank.py --pending              # what is left to review
    python scripts/review_question_bank.py --show dsa-arrays-001  # read one item properly
    python scripts/review_question_bank.py --approve dsa-arrays-001 --reviewer manas

Plan section 6.4 budgets about ninety seconds per item and lists what to check.
This script exists because a JSONL line is one very long line, and reviewing a
dataset by squinting at wrapped JSON is how a wrong concept key gets approved.

``--approve`` rewrites exactly one line of one file, setting ``review_status``
to ``reviewed`` with the named reviewer and today's date, and re-validates
afterwards.  It refuses to approve anything the reviewer has not been shown, in
the only way a script can: it prints the item and asks for confirmation unless
``--yes`` is passed.

**It never marks an item reviewed on its own.** ``reviewed`` is a claim that a
person read the question; nothing here or in CI may make that claim for them.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import textwrap
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:  # so `python scripts/...` works without install
    sys.path.insert(0, str(BACKEND))

from app.bank.loader import LoadedItem, validate_bank  # noqa: E402
from app.bank.schema import FIELD_ORDER  # noqa: E402
from app.bank.taxonomy import load_taxonomy  # noqa: E402

CHECKLIST = (
    "1. technically correct",
    "2. unambiguous, answerable without hidden assumptions",
    "3. the listed concepts are what the question actually tests",
    "4. concepts are observable in an answer, and independent of each other",
    "5. difficulty b is honest against the last ten items you rated",
    "6. the reference answer is correct and sufficient to grade against",
    "7. topic and subtopic are right",
    "8. it is materially different from the other items in its subtopic",
    "9. it is answerable in the stated time estimate",
)


def _wrap(text: str, indent: str = "    ") -> str:
    return textwrap.fill(text, width=96, initial_indent=indent, subsequent_indent=indent)


def _show(loaded: LoadedItem) -> None:
    item = loaded.item
    print(f"\n{'=' * 100}")
    print(f"{item.id}   [{loaded.where}]   {loaded.domain} / {item.topic} / {item.subtopic}")
    print(
        f"b = {item.difficulty_b:+.1f}   a = {item.discrimination_a}   "
        f"{item.time_estimate_s}s   source={item.source}   review={item.review_status}"
    )
    print(f"{'=' * 100}\nQUESTION\n{_wrap(item.text)}\n")
    print("EXPECTED CONCEPTS")
    for concept in item.expected_concepts:
        print(f"    [{concept.weight}] {concept.key}")
        print(_wrap(concept.hint, indent="        "))
    if item.common_misconceptions:
        print("\nCOMMON MISCONCEPTIONS")
        for misconception in item.common_misconceptions:
            print(_wrap(f"- {misconception}"))
    print(f"\nREFERENCE ANSWER\n{_wrap(item.reference_answer)}")
    if item.follow_up_seeds:
        print("\nFOLLOW-UP SEEDS")
        for seed in item.follow_up_seeds:
            print(_wrap(f"- {seed}"))
    print(f"\nanchors: {', '.join(item.anchor_terms)}    tags: {', '.join(item.tags)}")
    print("\nCHECKLIST")
    for line in CHECKLIST:
        print(f"    {line}")


def _rewrite_line(loaded: LoadedItem, reviewer: str, on: dt.date) -> None:
    """Replace one line of one file, leaving every other byte untouched."""
    lines = loaded.path.read_text(encoding="utf-8").splitlines()
    payload = json.loads(lines[loaded.line - 1])
    if payload["id"] != loaded.item.id:  # pragma: no cover - defensive
        raise RuntimeError(f"{loaded.where} is no longer {loaded.item.id}; re-run without --yes")
    payload["review_status"] = "reviewed"
    payload["reviewed_by"] = reviewer
    payload["reviewed_at"] = on.isoformat()
    lines[loaded.line - 1] = json.dumps(
        {field: payload[field] for field in FIELD_ORDER}, ensure_ascii=False
    )
    loaded.path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--show", metavar="ID", help="print one item in a readable form")
    parser.add_argument("--pending", action="store_true", help="list items awaiting review")
    parser.add_argument("--approve", metavar="ID", help="mark one item reviewed")
    parser.add_argument("--reviewer", help="the human who reviewed it; required with --approve")
    parser.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    args = parser.parse_args()

    taxonomy = load_taxonomy()
    report = validate_bank(taxonomy=taxonomy)
    if not report.ok:
        print(f"{len(report.errors)} validation error(s); fix the dataset first", file=sys.stderr)
        for error in report.errors[:10]:
            print(f"  x {error}", file=sys.stderr)
        return 1

    index = {loaded.item.id: loaded for loaded in report.items}

    if args.pending:
        pending = report.unreviewed()
        print(f"{len(pending)} of {report.count} item(s) awaiting review")
        for loaded in pending:
            print(
                f"  {loaded.item.id:<24} {loaded.item.subtopic:<22} "
                f"b={loaded.item.difficulty_b:+.1f}  [{loaded.where}]"
            )
        return 0

    if args.show:
        loaded = index.get(args.show)
        if loaded is None:
            print(f"no item with id {args.show!r}", file=sys.stderr)
            return 1
        _show(loaded)
        return 0

    if args.approve:
        if not args.reviewer:
            print("--approve requires --reviewer: a claim needs a claimant", file=sys.stderr)
            return 1
        loaded = index.get(args.approve)
        if loaded is None:
            print(f"no item with id {args.approve!r}", file=sys.stderr)
            return 1
        if loaded.item.review_status == "reviewed":
            print(f"{loaded.item.id} is already reviewed by {loaded.item.reviewed_by}")
            return 0
        if not args.yes:
            _show(loaded)
            answer = input(f"\nmark {loaded.item.id} reviewed by {args.reviewer}? [y/N] ")
            if answer.strip().lower() not in {"y", "yes"}:
                print("not approved")
                return 1
        _rewrite_line(loaded, args.reviewer, dt.date.today())
        after = validate_bank(taxonomy=taxonomy)
        if not after.ok:
            print("the edit broke validation; revert it with git", file=sys.stderr)
            return 1
        print(f"{loaded.item.id} marked reviewed by {args.reviewer}")
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
