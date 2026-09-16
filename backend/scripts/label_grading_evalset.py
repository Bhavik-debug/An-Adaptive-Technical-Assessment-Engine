"""Hand-label the grading eval set, blind and in randomised order.

    python scripts/label_grading_evalset.py --labeller AB          # label
    python scripts/label_grading_evalset.py --labeller AB --pass b # second labeller
    python scripts/label_grading_evalset.py --status               # how far along
    python scripts/label_grading_evalset.py --anchors              # print the anchors

Plan section 12.2, step 3: *"Human-label all 120 on a 0-4 scale using written
anchors. ~45 s each = 1.5 h. Label in RANDOMISED order to avoid drift."*

**This is the one part of the pipeline a human has to do.** It is a script rather
than a spreadsheet because three properties have to be guaranteed rather than
remembered, and a spreadsheet guarantees none of them:

**1. Blind.** The labeller is shown the question and the answer. They are NOT
shown ``quality_intent``, ``concepts_included``, ``concepts_omitted``, the
expected-concept list, the row id, or any grader output. A labeller who can see
that a row was built as ``excellent`` will label it 4, and Day 19 would then be
measuring the generator against itself. This script never prints those fields.

**2. Randomised, and reproducibly so.** ``labelling_order`` shuffles with a
committed seed. In file order the five answers to one question sit together,
best first, and a labeller working down that list ranks them against each other
instead of judging each on its own - systematic anchoring, not noise.

**3. Anchored.** The 0-4 descriptors are printed before the first item and are
one keystroke away at any point. An unanchored 0-4 is a different scale by the
end of the afternoon than it was at the start.

Progress is saved after every single label, so the 1.5 hours can be done in as
many sittings as you like and a closed terminal costs nothing.
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:  # so `python scripts/...` works without install
    sys.path.insert(0, str(BACKEND))

from app.bank.loader import load_bank  # noqa: E402
from app.grading.evalset import (  # noqa: E402
    LABEL_ANCHORS,
    MAX_LABEL,
    MIN_LABEL,
    EvalAnswer,
    dump_answers,
    gate_eligible,
    label_distribution,
    label_sources,
    labelling_order,
    load_answers,
)

DATASET_DIR = BACKEND.parent / "evals" / "datasets" / "grading"
ANSWERS_PATH = DATASET_DIR / "answers.jsonl"

#: The committed labelling-order seed. Different from the build seed on purpose:
#: they answer different questions ("which 24 questions?" and "in what order were
#: they labelled?") and tying them together would mean a rebuild silently
#: reshuffled a labelling session that had already happened.
LABEL_SEED = 8021


def _anchors_block() -> str:
    lines = ["  The 0-4 scale:", ""]
    for score in range(MIN_LABEL, MAX_LABEL + 1):
        body = textwrap.fill(
            LABEL_ANCHORS[score], width=76, initial_indent="", subsequent_indent="        "
        )
        lines.append(f"    {score}   {body}")
        lines.append("")
    return "\n".join(lines)


def _show(answer: EvalAnswer, question_text: str, *, position: int, total: int) -> None:
    """Print one item for labelling. Blind: no intent, no concepts, no id."""
    print("\n" + "=" * 80)
    print(f"  {position} of {total}")
    print("=" * 80)
    print("\n  QUESTION\n")
    print(textwrap.fill(question_text, width=76, initial_indent="  ", subsequent_indent="  "))
    print("\n  ANSWER\n")
    print(textwrap.fill(answer.answer, width=76, initial_indent="  ", subsequent_indent="  "))
    print()


def _prompt(position: int, total: int) -> str:
    return f"  [{position}/{total}]  score 0-4  (a=anchors, s=skip, q=save and quit): "


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labeller", help="your initials; recorded with every label")
    parser.add_argument(
        "--pass",
        dest="which",
        choices=("a", "b"),
        default="a",
        help="'b' is the second labeller for the human-human ceiling (plan 12.2 step 4)",
    )
    parser.add_argument(
        "--source",
        choices=("human", "model"),
        default="human",
        help=(
            "who is labelling. 'human' is the only kind the Phase-4 QWK gate "
            "accepts; 'model' records a provisional label that is useful for "
            "developing the harness and is not the gate"
        ),
    )
    parser.add_argument(
        "--from-file",
        type=Path,
        help=(
            "apply labels non-interactively from a file of '<answer-id> <score>' "
            "lines. Same validation, same single write path - it exists so a "
            "labelling pass done elsewhere is applied by this script rather than "
            "by hand-editing the dataset"
        ),
    )
    parser.add_argument(
        "--worklist",
        type=Path,
        help=(
            "write the blind labelling worklist here and exit: shuffled order, "
            "question and answer only. No quality_intent, no concept split"
        ),
    )
    parser.add_argument("--limit", type=int, default=0, help="stop after N items this sitting")
    parser.add_argument("--status", action="store_true", help="print progress and exit")
    parser.add_argument("--anchors", action="store_true", help="print the anchors and exit")
    parser.add_argument("--seed", type=int, default=LABEL_SEED)
    parser.add_argument("--path", type=Path, default=ANSWERS_PATH)
    args = parser.parse_args()

    if args.anchors:
        print(_anchors_block())
        return 0

    if not args.path.exists():
        print(f"no dataset at {args.path}; build it first:")
        print("  python scripts/build_grading_evalset.py")
        return 1

    answers = load_answers(args.path)
    by_id = {a.id: a for a in answers}
    order = labelling_order(answers, seed=args.seed)

    score_field = "human_score" if args.which == "a" else "human_score_b"
    labeller_field = "labeller" if args.which == "a" else "labeller_b"
    source_field = "label_source" if args.which == "a" else "label_source_b"
    done = [aid for aid in order if getattr(by_id[aid], score_field) is not None]
    todo = [aid for aid in order if getattr(by_id[aid], score_field) is None]

    print(f"  dataset      {args.path}")
    print(f"  pass         {args.which}")
    print(f"  seed         {args.seed}  (the labelling order; committed, reproducible)")
    print(f"  labelled     {len(done)}/{len(answers)}")
    print(f"  sources      {label_sources(answers)}")
    print(f"  distribution {label_distribution(answers)}")
    print(f"  gate-eligible (human-labelled, 120/120): {gate_eligible(answers)}")

    if args.status:
        return 0

    loaded_bank, bank_errors = load_bank()
    if bank_errors:
        print("  the question bank does not load; cannot show questions.")
        return 1
    questions = {li.item.id: li.item.text for li in loaded_bank}

    if args.worklist:
        # Blind by construction: the only fields written are the id, the
        # question and the answer. A worklist carrying `quality_intent` would
        # make the whole labelling pass worthless.
        blocks = []
        for position, aid in enumerate(order, start=1):
            row = by_id[aid]
            blocks.append(
                f"### {position}/{len(order)}  {row.id}\n\n"
                f"QUESTION\n{questions.get(row.question_id, '(missing)')}\n\n"
                f"ANSWER\n{row.answer}\n"
            )
        args.worklist.write_text("\n".join(blocks), encoding="utf-8")
        print(f"\n  wrote {len(order)} items to {args.worklist} in seeded shuffled order")
        return 0

    if args.from_file:
        if not args.labeller:
            print("\n  --labeller is required (a score needs a claimant).")
            return 1
        applied = skipped = 0
        for number, line in enumerate(
            args.from_file.read_text(encoding="utf-8").splitlines(), start=1
        ):
            text = line.split("#", 1)[0].strip()
            if not text:
                continue
            try:
                aid, raw_score = text.split()
                score = int(raw_score)
            except ValueError:
                print(f"  {args.from_file.name} line {number}: expected '<id> <score>'")
                return 1
            if aid not in by_id:
                print(f"  {args.from_file.name} line {number}: unknown answer id {aid!r}")
                return 1
            if not MIN_LABEL <= score <= MAX_LABEL:
                print(f"  {args.from_file.name} line {number}: score {score} is off the scale")
                return 1
            if getattr(by_id[aid], score_field) is not None:
                skipped += 1
                continue
            by_id[aid] = by_id[aid].model_copy(
                update={
                    score_field: score,
                    labeller_field: args.labeller,
                    source_field: args.source,
                }
            )
            applied += 1
        dump_answers(by_id.values(), args.path)
        final_rows = list(by_id.values())
        print(f"\n  applied {applied}, left {skipped} already-labelled rows alone")
        print(f"  sources      {label_sources(final_rows)}")
        print(f"  distribution {label_distribution(final_rows)}")
        print(f"  gate-eligible: {gate_eligible(final_rows)}")
        return 0
    if not todo:
        print("\n  every answer in this pass is labelled.")
        return 0
    if not args.labeller:
        print("\n  --labeller is required to record a label (a score needs a claimant).")
        return 1

    question_text = questions

    print(f"\n  {len(todo)} to go. Progress is saved after every label.\n")
    print(_anchors_block())

    labelled_now = 0
    for aid in todo:
        if args.limit and labelled_now >= args.limit:
            print(f"\n  stopping at --limit {args.limit}.")
            break

        answer = by_id[aid]
        _show(
            answer,
            question_text.get(answer.question_id, "(question not found in the bank)"),
            position=len(done) + labelled_now + 1,
            total=len(answers),
        )

        while True:
            try:
                raw = input(_prompt(len(done) + labelled_now + 1, len(answers))).strip().lower()
            except (EOFError, KeyboardInterrupt):
                print("\n\n  saved. re-run to carry on.")
                return 0

            if raw == "q":
                print("\n  saved. re-run to carry on.")
                return 0
            if raw == "s":
                print("  skipped.")
                break
            if raw == "a":
                print()
                print(_anchors_block())
                continue
            if raw.isdigit() and MIN_LABEL <= int(raw) <= MAX_LABEL:
                # Written through the model so the score/labeller pairing rule is
                # enforced here rather than trusted.
                by_id[aid] = answer.model_copy(
                    update={
                        score_field: int(raw),
                        labeller_field: args.labeller,
                        source_field: args.source,
                    }
                )
                dump_answers(by_id.values(), args.path)
                labelled_now += 1
                break
            print(f"  enter {MIN_LABEL}-{MAX_LABEL}, or a / s / q.")

    final = list(by_id.values())
    print(f"\n  labelled this sitting  {labelled_now}")
    print(
        f"  labelled in total      {sum(1 for a in final if getattr(a, score_field) is not None)}"
        f"/{len(final)}"
    )
    print(f"  distribution           {label_distribution(final)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
