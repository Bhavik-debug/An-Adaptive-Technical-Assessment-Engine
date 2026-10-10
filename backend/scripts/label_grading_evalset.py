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
    blind_worklist,
    dump_answers,
    gate_eligible,
    label_agreement,
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


def _agreement_line(rows: list[EvalAnswer]) -> str | None:
    """Describe the pass-A-vs-retired comparison, naming which comparison it is.

    The label is computed from the data rather than hard-coded, because the whole
    point of ``label_agreement``'s ``sources`` key is that "human vs model" and
    "model vs model" are different claims and only one of them bears on the gate.
    Printing the wrong one is how a reproducibility number gets quoted as
    validation.
    """
    agreement = label_agreement(rows)
    if not agreement["n"]:
        return None
    sources = agreement["sources"]
    assert isinstance(sources, list)
    if sources == ["model"]:
        what = "model vs model (same family - reproducibility, NOT validation)"
    elif "human" in sources and "model" in sources:
        what = "human vs model"
    else:
        what = f"pass A ({'/'.join(sources)}) vs retired model"
    return f"  {what}\n    {agreement}"


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
        "--by-position",
        action="store_true",
        help=(
            "read --from-file as '<position> <score>', position being 1-based in "
            "the seeded labelling order. This is the form that pairs with a "
            "--worklist, which carries positions and no row ids because the ids "
            "leak quality_intent"
        ),
    )
    parser.add_argument(
        "--retire-model-labels",
        action="store_true",
        help=(
            "move every label_source='model' score out of human_score into "
            "model_score/model_labeller and clear the human fields, so a human "
            "pass has somewhere to write without destroying the model pass. "
            "Idempotent; refuses to overwrite an existing model_score"
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
    retired = sum(1 for a in answers if a.model_score is not None)
    if retired:
        print(f"  retired model pass  {retired}/{len(answers)}")
        if (line := _agreement_line(answers)) is not None:
            print(line)

    if args.status:
        return 0

    if args.retire_model_labels:
        # Day 18's first sitting filled `human_score` with model labels. The
        # human pass needs that field, and the model pass is worth keeping (see
        # EvalAnswer.model_score), so it is moved rather than overwritten.
        clash = sorted(
            a.id for a in answers if a.label_source == "model" and a.model_score is not None
        )
        if clash:
            print(f"\n  refusing: {len(clash)} rows already hold a model_score ({clash[:5]}).")
            print("  moving the current label there would destroy a retired pass.")
            return 1
        moved = 0
        for aid, row in list(by_id.items()):
            if row.label_source != "model" or row.human_score is None:
                continue
            by_id[aid] = row.model_copy(
                update={
                    "model_score": row.human_score,
                    "model_labeller": row.labeller,
                    "human_score": None,
                    "labeller": None,
                    "label_source": None,
                }
            )
            moved += 1
        dump_answers(by_id.values(), args.path)
        final_rows = list(by_id.values())
        print(f"\n  retired {moved} model labels into model_score/model_labeller")
        print(f"  sources      {label_sources(final_rows)}")
        print(f"  human_score is now free on {moved} rows; run a human pass to fill it")
        print(f"  gate-eligible: {gate_eligible(final_rows)}")
        return 0

    loaded_bank, bank_errors = load_bank()
    if bank_errors:
        print("  the question bank does not load; cannot show questions.")
        return 1
    questions = {li.item.id: li.item.text for li in loaded_bank}

    if args.worklist:
        # Built in `evalset.blind_worklist` rather than here, because "this
        # artefact is blind" is a property a test should assert rather than a
        # property a CLI f-string is trusted to have. It was trusted, and it
        # leaked the row id - which decodes `quality_intent` by arithmetic.
        args.worklist.write_text(
            blind_worklist(answers, questions=questions, seed=args.seed), encoding="utf-8"
        )
        print(f"\n  wrote {len(order)} items to {args.worklist} in seeded shuffled order")
        print("  positions only - no row ids, no intent. Apply with --from-file --by-position.")
        return 0

    if args.from_file:
        if not args.labeller:
            print("\n  --labeller is required (a score needs a claimant).")
            return 1
        shape = "<position> <score>" if args.by_position else "<id> <score>"
        applied = skipped = 0
        seen_keys: set[str] = set()
        for number, line in enumerate(
            args.from_file.read_text(encoding="utf-8").splitlines(), start=1
        ):
            text = line.split("#", 1)[0].strip()
            if not text:
                continue
            try:
                key, raw_score = text.split()
                score = int(raw_score)
            except ValueError:
                print(f"  {args.from_file.name} line {number}: expected {shape!r}")
                return 1
            # A repeated key means two scores were given for one answer and the
            # file order silently decides which wins. Refuse instead: a 120-line
            # hand-made file is exactly where that typo happens.
            if key in seen_keys:
                print(f"  {args.from_file.name} line {number}: {key!r} appears twice")
                return 1
            seen_keys.add(key)
            if args.by_position:
                # Positions are 1-based in the seeded order, which is a pure
                # function of the seed - so a worklist can stay free of row ids.
                if not key.isdigit() or not 1 <= int(key) <= len(order):
                    print(
                        f"  {args.from_file.name} line {number}: position {key!r} is not "
                        f"in 1..{len(order)}"
                    )
                    return 1
                aid = order[int(key) - 1]
            else:
                aid = key
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
        if skipped:
            # Loud, because the quiet version of this is how a human pass gets
            # reported as done having written nothing: every row already carried
            # a label, every row was skipped, and the exit code was 0. If the
            # intent was to replace a previous pass, retire it first.
            print(
                f"  NOTE {skipped} of {applied + skipped} scores were NOT written - those rows "
                f"already had a {score_field}."
            )
            if applied == 0:
                print("  nothing was written. --retire-model-labels frees human_score.")
        print(f"  sources      {label_sources(final_rows)}")
        print(f"  distribution {label_distribution(final_rows)}")
        if (line := _agreement_line(final_rows)) is not None:
            print(line)
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
