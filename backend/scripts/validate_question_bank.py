"""Validate the question bank, and print the report a reviewer reads.

    cd backend
    python scripts/validate_question_bank.py
    python scripts/validate_question_bank.py --require-reviewed   # the Phase 2 gate
    python scripts/validate_question_bank.py --table              # the review table

Exit code 0 when the dataset is valid, 1 when it is not.  Warnings never fail
the run on their own; ``--strict`` promotes them if you want them to.

All of the thinking lives in ``app/bank/loader.py`` - this file is the command
line around it.  The same function backs ``tests/unit/bank/test_committed_dataset.py``, so
CI and this script cannot disagree about what "valid" means.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:  # so `python scripts/...` works without install
    sys.path.insert(0, str(BACKEND))

from app.bank.loader import BankReport, validate_bank  # noqa: E402
from app.bank.taxonomy import TaxonomyError, load_taxonomy  # noqa: E402


def _print_summary(report: BankReport) -> None:
    print(f"files:  {len(report.files)} ({', '.join(p.name for p in report.files)})")
    print(f"items:  {report.count}")

    print("\nby domain")
    for name, count in sorted(report.by_domain().items()):
        print(f"  {name:<20} {count:>4}")

    print("\nby topic")
    for name, count in sorted(report.by_topic().items()):
        print(f"  {name:<20} {count:>4}")

    print("\ndifficulty b")
    for band, count in sorted(report.difficulty_histogram().items()):
        print(f"  {band:<20} {count:>4}  {'#' * count}")

    print("\nreview status")
    for status, count in sorted(report.by_review_status().items()):
        print(f"  {status:<20} {count:>4}")

    vocabulary = report.concept_vocabulary()
    reused = sum(1 for count in vocabulary.values() if count > 1)
    print(f"\nconcept vocabulary: {len(vocabulary)} keys, {reused} used by more than one item")
    top = vocabulary.most_common(8)
    if top:
        print("  most reused: " + ", ".join(f"{key} x{count}" for key, count in top))


def _print_table(report: BankReport) -> None:
    print(
        f"\n{'id':<24} {'domain':<16} {'topic':<16} {'subtopic':<22} "
        f"{'b':>5}  {'review':<9} concepts"
    )
    print("-" * 132)
    for loaded in sorted(report.items, key=lambda item: item.item.id):
        item = loaded.item
        print(
            f"{item.id:<24} {loaded.domain:<16} {item.topic:<16} {item.subtopic:<22} "
            f"{item.difficulty_b:>+5.1f}  {item.review_status:<9} "
            f"{', '.join(item.concept_keys)}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--bank-dir", type=Path, default=None, help="defaults to data/question-bank/"
    )
    parser.add_argument("--table", action="store_true", help="print the per-item review table")
    parser.add_argument(
        "--require-reviewed",
        action="store_true",
        help="fail unless every item is marked reviewed (the Phase 2 exit gate)",
    )
    parser.add_argument("--strict", action="store_true", help="treat warnings as errors")
    parser.add_argument(
        "--expect-count", type=int, default=None, help="fail unless the bank has exactly N items"
    )
    args = parser.parse_args()

    try:
        taxonomy = load_taxonomy()
    except TaxonomyError as exc:
        print(f"taxonomy: {exc}", file=sys.stderr)
        return 1

    report = validate_bank(args.bank_dir, taxonomy)

    _print_summary(report)
    if args.table:
        _print_table(report)

    if report.warnings:
        print(f"\n{len(report.warnings)} warning(s):")
        for warning in report.warnings:
            print(f"  ! {warning}")

    failures = list(report.errors)
    if args.strict:
        failures.extend(report.warnings)

    if args.expect_count is not None and report.count != args.expect_count:
        failures.append(f"expected {args.expect_count} items, found {report.count}")

    if args.require_reviewed:
        pending = report.unreviewed()
        if pending:
            failures.append(
                f"{len(pending)} item(s) are not marked reviewed: "
                + ", ".join(loaded.item.id for loaded in pending[:5])
                + (" ..." if len(pending) > 5 else "")
            )

    if failures:
        print(f"\n{len(failures)} error(s):", file=sys.stderr)
        for failure in failures:
            print(f"  x {failure}", file=sys.stderr)
        return 1

    print("\nok: the question bank is valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
