"""Build the 120-answer grading evaluation set. Plan section 12.2, Day 18.

    python scripts/build_grading_evalset.py --dry-run     # plan it, call nothing
    python scripts/build_grading_evalset.py               # build it (spends quota)
    python scripts/build_grading_evalset.py --questions 2 # a small smoke run
    python scripts/build_grading_evalset.py --validate    # check the committed file

**This spends quota.** It is one of three things in this repository that
deliberately does (the other two record LLM fixtures), it is run by hand, and it
is never run by a test or by CI.

**It resumes.** The build is keyed on ``(question_id, quality_intent)``, so a
re-run reads whatever is already in the output file and generates only what is
missing. That matters more than it sounds: the free NVIDIA endpoint times out
under concurrency often enough that a 120-call build rarely completes in one
pass, and a builder that discarded 116 good answers because 4 timed out would
make the dataset cost several times what it should. Partial output is written
and the exit code is non-zero, so a re-run converges and an incomplete file is
never mistaken for a finished one.

Generation is temperature 0 and marked cacheable, so with Redis up a re-run is
free at the provider too. With Redis down - the common local case - the response
cache is a no-op and a re-run pays again for anything not already in the file.
Resuming from the file is what makes the build affordable either way.

**Every call goes through ``call_structured``.** There is no direct provider
access here and there must never be: the eval-set builder is held to the same
rule as the interview path, because a builder with its own client would be the
first place the rule got quietly dropped.

The output is two committed artefacts under ``evals/datasets/grading/``:

* ``questions.jsonl`` - the 24 sampled bank items, so "which questions?" is
  answerable without re-running the sampler.
* ``answers.jsonl``   - the 120 records, ``human_score`` empty until labelled.

Labelling is a separate sitting and a separate script - see
``scripts/label_grading_evalset.py``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:  # so `python scripts/...` works without install
    sys.path.insert(0, str(BACKEND))

from app.bank.loader import load_bank  # noqa: E402
from app.bank.schema import BankItem  # noqa: E402
from app.config import Settings  # noqa: E402
from app.grading.answergen import (  # noqa: E402
    generate_answer,
    normalise_prose,
    prose_problems,
)
from app.grading.evalset import (  # noqa: E402
    QUALITY_LEVELS,
    TARGET_QUESTIONS,
    EvalAnswer,
    dump_answers,
    label_distribution,
    level_distribution,
    load_answers,
    stratified_questions,
    validate_dataset,
)
from app.llm.runtime import dispose_llm, get_response_cache, get_router, init_llm  # noqa: E402
from app.obs.logging import configure_logging  # noqa: E402

#: ``backend/`` -> repo root -> the committed dataset directory.
DOTENV = BACKEND.parent / ".env"
DATASET_DIR = BACKEND.parent / "evals" / "datasets" / "grading"
ANSWERS_PATH = DATASET_DIR / "answers.jsonl"
QUESTIONS_PATH = DATASET_DIR / "questions.jsonl"

#: The one seed. Chosen once, committed, and never tuned - "seed 1318" is
#: Day 13-18 and nothing more. Changing it reshuffles which 24 questions are in
#: the set, which would silently make a new dataset look like an edit to the old
#: one, so it is an argument with a default rather than a value anyone passes.
DEFAULT_SEED = 1318

#: How many generations run at once. Measured, not guessed: at 4 the free NVIDIA
#: endpoint returned `no response within 60s` for 3-4 calls out of 120 on every
#: attempt. The endpoint is shared and rate-limited, and a timeout wastes a whole
#: call, so more concurrency buys nothing here.
CONCURRENCY = 2


async def _build(
    questions: list[BankItem],
    seed: int,
    *,
    settings,
    have: dict[tuple[str, str], EvalAnswer],
) -> list[EvalAnswer]:
    """Generate the (question, level) pairs that are not already in ``have``.

    Returns only the newly generated rows; the caller merges. Keyed on
    ``(question_id, quality_intent)`` rather than on the row id, because that
    pair identifies a cell of the 24x5 design and stays meaningful even if the
    id scheme ever changes.
    """
    router = get_router()
    cache = get_response_cache()
    semaphore = asyncio.Semaphore(CONCURRENCY)
    # Deterministic ids: the Nth question's Mth level is always the same id, so
    # a rebuild diffs row by row instead of renumbering the whole file.
    jobs = [
        (index * len(QUALITY_LEVELS) + offset + 1, item, level)
        for index, item in enumerate(questions)
        for offset, level in enumerate(QUALITY_LEVELS)
        if (item.id, level) not in have
    ]
    if not jobs:
        return []
    print(f"  {len(have)} already present; generating {len(jobs)}")

    async def one(number: int, item: BankItem, level: str) -> EvalAnswer | None:
        async with semaphore:
            try:
                answer, included, omitted, meta = await generate_answer(
                    item, level, router=router, cache=cache, settings=settings
                )
            except Exception as exc:  # noqa: BLE001 - reported, not swallowed
                print(f"  [{number:3}/{len(jobs)}] {item.id:16} {level:16} FAILED  {exc}")
                return None

        text = normalise_prose(answer.answer)
        problems = prose_problems(text)
        flag = "  <- " + "; ".join(problems) if problems else ""
        print(
            f"  [{number:3}/{len(jobs)}] {item.id:16} {level:16} "
            f"{len(text.split()):4}w  {'cache' if meta.cache_hit else 'live '}{flag}"
        )
        return EvalAnswer(
            id=f"eval-{number:04d}",
            question_id=item.id,
            answer=text,
            quality_intent=level,  # type: ignore[arg-type]
            concepts_included=included,
            concepts_omitted=omitted,
            slice="synthetic",
            build_seed=seed,
            generator_model=meta.model,
        )

    results = await asyncio.gather(*(one(n, i, lvl) for n, i, lvl in jobs))
    return [row for row in results if row is not None]


def _print_report(
    answers: list[EvalAnswer],
    questions: dict[str, BankItem],
    *,
    expect_q: int,
    expect_a: int,
) -> list[str]:
    print()
    print(f"  answers            {len(answers)}")
    print(f"  questions          {len({a.question_id for a in answers})}")
    print(f"  level distribution {level_distribution(answers)}")
    print(f"  labelled           {sum(1 for a in answers if a.is_labelled)}/{len(answers)}")
    print(f"  label distribution {label_distribution(answers)}")
    words = [len(a.answer.split()) for a in answers]
    if words:
        ordered = sorted(words)
        print(
            f"  answer length      min {ordered[0]}  p50 {ordered[len(ordered) // 2]}  "
            f"max {ordered[-1]} words"
        )
    problems = validate_dataset(
        answers, questions=questions, expect_questions=expect_q, expect_answers=expect_a
    )
    print()
    if problems:
        print(f"  VALIDATION FAILED ({len(problems)} problem(s)):")
        for problem in problems:
            print(f"    - {problem}")
    else:
        print("  validation         OK")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--questions", type=int, default=TARGET_QUESTIONS, help="fewer, for a smoke run"
    )
    parser.add_argument("--dry-run", action="store_true", help="plan it; call nothing")
    parser.add_argument("--validate", action="store_true", help="check the committed file and exit")
    parser.add_argument("--out", type=Path, default=ANSWERS_PATH)
    args = parser.parse_args()

    settings = Settings(_env_file=DOTENV)  # type: ignore[call-arg]
    configure_logging(settings)

    loaded, errors = load_bank()
    if errors:
        print("the question bank does not load; fix it before building a dataset:")
        for error in errors[:10]:
            print(f"  - {error}")
        return 1

    domain_of = {li.item.id: li.domain for li in loaded}
    items = [li.item for li in loaded]
    questions = stratified_questions(
        items, seed=args.seed, count=args.questions, stratum_of=lambda i: domain_of[i.id]
    )
    by_id = {item.id: item for item in items}

    if args.validate:
        if not args.out.exists():
            print(f"no dataset at {args.out}")
            return 1
        answers = load_answers(args.out)
        expect_q = len({a.question_id for a in answers})
        problems = _print_report(answers, by_id, expect_q=expect_q, expect_a=len(answers))
        # Re-validate against the plan's fixed counts too, not only self-consistency.
        strict = validate_dataset(answers, questions=by_id)
        if strict and not problems:
            print("  against the plan's 24x5 target:")
            for problem in strict:
                print(f"    - {problem}")
        return 1 if problems else 0

    expected = args.questions * len(QUALITY_LEVELS)
    print(f"  seed               {args.seed}")
    print(f"  questions          {len(questions)}")
    print(f"  levels             {len(QUALITY_LEVELS)} {list(QUALITY_LEVELS)}")
    print(f"  answers to build   {expected}")
    print(f"  out                {args.out}")
    print()

    if args.dry_run:
        for item in questions:
            print(
                f"  {item.id:16} {item.topic:18} b={item.difficulty_b:+.1f}  "
                f"{len(item.expected_concepts)} concepts"
            )
        print()
        print("  dry run: no provider was called and nothing was written")
        return 0

    if "nvidia" not in settings.llm_providers:
        print(f"  provider order is {settings.llm_providers}; refusing to build a dataset")
        print("  from the stub - a committed dataset must come from a real model.")
        return 1

    # Resume: anything already written stays, including any labels it carries.
    existing = load_answers(args.out) if args.out.exists() else []
    wanted = {item.id for item in questions}
    have = {
        (row.question_id, row.quality_intent): row for row in existing if row.question_id in wanted
    }
    dropped = len(existing) - len(have)
    if dropped:
        print(f"  ignoring {dropped} existing rows for questions outside this sample")

    async def run() -> list[EvalAnswer]:
        init_llm(settings)
        try:
            return await _build(questions, args.seed, settings=settings, have=have)
        finally:
            await dispose_llm()

    fresh = asyncio.run(run())
    answers = sorted([*have.values(), *fresh], key=lambda row: row.id)

    # Written even when incomplete, so the next run resumes instead of repaying.
    # The exit code and the validation report are what say it is not finished.
    written = dump_answers(answers, args.out)
    problems = _print_report(answers, by_id, expect_q=len(questions), expect_a=expected)

    if len(answers) != expected or problems:
        print(f"\n  wrote {written} of {expected} answers to {args.out} - INCOMPLETE.")
        print("  re-run to generate the rest; what is already in the file is kept.")
        return 1

    QUESTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    QUESTIONS_PATH.write_text(
        "\n".join(
            json.dumps(item.model_dump(mode="json"), sort_keys=True, ensure_ascii=False)
            for item in questions
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\n  wrote {written} answers to {args.out}")
    print(f"  wrote {len(questions)} questions to {QUESTIONS_PATH}")
    print("\n  READ what was written before committing, then label it:")
    print("    python scripts/label_grading_evalset.py --labeller <your-initials>")
    return 0


if __name__ == "__main__":
    sys.exit(main())
