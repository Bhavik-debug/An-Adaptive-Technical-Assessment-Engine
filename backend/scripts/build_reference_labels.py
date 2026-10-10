"""Produce provisional reference labels for the 120-answer grading set.

    python scripts/build_reference_labels.py --dry-run     # what would run, no calls
    python scripts/build_reference_labels.py --pass blind  # the primary pass
    python scripts/build_reference_labels.py --pass concept_aware
    python scripts/build_reference_labels.py --adjudicate  # only the disagreements
    python scripts/build_reference_labels.py --validate    # check the committed file

Day 19, Part 2. Plan section 12.2 step 3 asks for 120 **human** labels; there are
none and this script does not produce any. It produces a third pass of
**autonomous, model-generated provisional reference labels** - see
``app/grading/reflabel.py`` for what that is and is not worth.

**Resumable, and that is not a convenience.** Every label is appended to the
artefact as soon as it validates, so a rate limit 80 rows in costs nothing but
the time already spent. Re-running skips rows that already have a label for the
requested pass. The free-tier quota this runs on is the reason the Day-18 build
script was written the same way.

**Isolation.** The only things put into a prompt are the question text, the
answer text and - for the concept-aware pass - the question's expected-concept
list. ``quality_intent``, the per-row concept split and both previous label
passes are never read into the inputs. The row id is not shown either. See
``reflabel_prompt.FORBIDDEN_IN_PROMPT`` and the test that renders every template
against real rows and greps for those strings.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:  # so `python scripts/...` works without install
    sys.path.insert(0, str(BACKEND))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(BACKEND.parent / ".env", override=False)

from app.bank.loader import load_bank  # noqa: E402
from app.bank.schema import BankItem  # noqa: E402
from app.config import Settings, get_settings  # noqa: E402
from app.grading.evalset import EvalAnswer, labelling_order, load_answers  # noqa: E402
from app.grading.reflabel import (  # noqa: E402
    REFERENCE_RUBRIC_VERSION,
    ReferenceJudgment,
    ReferenceLabel,
    ReferencePass,
    dump_reference_labels,
    labels_by_answer,
    load_reference_labels,
    needs_adjudication,
    reference_label_problems,
    score_distribution,
)

# Imported for its side effect as well as its value: importing the module is
# what registers all three templates with the prompt registry, and the
# adjudication template is reached through the registry by task name rather
# than by this binding.
from app.grading.reflabel_prompt import (  # noqa: E402, F401
    REFERENCE_LABEL_ADJUDICATE_PROMPT_V1,
    REFERENCE_LABEL_CONCEPTS_PROMPT_V1,
    REFERENCE_LABEL_PROMPT_V1,
)
from app.llm.client import call_structured  # noqa: E402
from app.llm.runtime import dispose_llm, init_llm  # noqa: E402
from app.llm.tasks import TaskName  # noqa: E402

DATASET_DIR = BACKEND.parent / "evals" / "datasets" / "grading"
ANSWERS_PATH = DATASET_DIR / "answers.jsonl"
LABELS_PATH = DATASET_DIR / "reference_labels.jsonl"

#: The labelling order. Same seed as Day 18's, so a row's position in the
#: labelling sequence is the same number it has always been - which is what makes
#: "did the labeller drift over the session?" answerable across passes.
LABEL_SEED = 8021

#: How many times a single row is retried before it is recorded as a failure.
#: A structured call that cannot produce a valid judgement after this many tries
#: is a datum, not a blocker: the row is written with ``last_error`` set and the
#: run carries on, because losing 119 labels to one malformed response would be
#: the worse outcome.
MAX_ATTEMPTS = 3

TASK_FOR_PASS: dict[str, TaskName] = {
    "blind": TaskName.EVAL_REFERENCE_LABEL,
    "concept_aware": TaskName.EVAL_REFERENCE_LABEL_CONCEPTS,
}
PROMPT_FOR_PASS = {
    "blind": REFERENCE_LABEL_PROMPT_V1,
    "concept_aware": REFERENCE_LABEL_CONCEPTS_PROMPT_V1,
}


def _render_concepts(item: BankItem) -> str:
    """The question's expected ideas, as reference material about the question.

    The *hint* is included and the *weight* is not. The weight is what the
    scorer multiplies by, and handing it to the labeller would invite it to
    reproduce the grader's arithmetic instead of forming its own judgement.

    Sorted by key, not by weight: weight order would rank the concepts, which is
    the same leak by a different route.
    """
    lines = []
    for concept in sorted(item.expected_concepts, key=lambda c: c.key):
        lines.append(f"- {concept.key.replace('_', ' ')}: {concept.hint}")
    return "\n".join(lines)


async def _label_one(
    answer: EvalAnswer,
    item: BankItem,
    *,
    pass_name: ReferencePass,
    settings: Settings,
) -> ReferenceLabel:
    """One row, with retries. Never raises; records the failure instead."""
    task = TASK_FOR_PASS[pass_name]

    inputs: dict[str, object] = {"question": item.text, "answer": answer.answer}
    if pass_name == "concept_aware":
        inputs["concepts"] = _render_concepts(item)

    last_error: str | None = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            judgment, meta = await call_structured(
                task, inputs, ReferenceJudgment, settings=settings
            )
        except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
            last_error = f"{type(exc).__name__}: {exc}"[:400]
            if attempt < MAX_ATTEMPTS:
                await asyncio.sleep(2.0 * attempt)
            continue
        return ReferenceLabel(
            answer_id=answer.id,
            question_id=answer.question_id,
            judgment=judgment,
            pass_name=pass_name,
            evaluator_model=meta.model,
            task=str(task),
            prompt_version=meta.prompt_version,
            prompt_fingerprint=meta.prompt_fingerprint,
            rubric_version=REFERENCE_RUBRIC_VERSION,
            temperature=meta.temperature,
            labelled_at=datetime.now(UTC).isoformat(timespec="seconds"),
            attempts=attempt,
            last_error=None,
        )

    # Exhausted. A placeholder row would be a fabricated label, so nothing is
    # returned - the caller records the id as a failure and the row stays absent.
    raise RuntimeError(last_error or "all attempts failed")


async def _run_pass(
    answers: Sequence[EvalAnswer],
    questions: dict[str, BankItem],
    *,
    pass_name: ReferencePass,
    settings: Settings,
    existing: list[ReferenceLabel],
    limit: int,
    concurrency: int = 1,
) -> tuple[list[ReferenceLabel], list[str]]:
    done = set(labels_by_answer(existing, pass_name=pass_name))
    order = labelling_order(answers, seed=LABEL_SEED)
    by_id = {a.id: a for a in answers}
    todo = [aid for aid in order if aid not in done]
    if limit:
        todo = todo[:limit]

    print(f"  pass {pass_name}: {len(done)} already labelled, {len(todo)} to do")
    if concurrency > 1:
        print(f"  concurrency {concurrency}")
    out = list(existing)
    failures: list[str] = []
    limiter = asyncio.Semaphore(max(1, concurrency))
    write_lock = asyncio.Lock()
    finished = 0

    async def one(answer_id: str) -> None:
        nonlocal finished
        answer = by_id[answer_id]
        async with limiter:
            try:
                label = await _label_one(
                    answer, questions[answer.question_id], pass_name=pass_name, settings=settings
                )
            except Exception as exc:  # noqa: BLE001
                async with write_lock:
                    finished += 1
                    failures.append(f"{answer_id}: {exc}")
                    print(
                        f"    [{finished}/{len(todo)}] {answer_id} FAILED: {str(exc)[:120]}",
                        flush=True,
                    )
                return
        async with write_lock:
            out.append(label)
            # Written after every row, under a lock: a rate limit costs the
            # current call rather than the run, and two coroutines cannot
            # serialise a list that is missing the other's row.
            dump_reference_labels(out, LABELS_PATH)
            finished += 1
            print(
                f"    [{finished}/{len(todo)}] {answer_id} -> {label.judgment.score} "
                f"(conf {label.judgment.confidence:.2f}, {label.attempts} att)",
                flush=True,
            )

    await asyncio.gather(*(one(answer_id) for answer_id in todo))
    return out, failures


async def _run_adjudication(
    answers: Sequence[EvalAnswer],
    questions: dict[str, BankItem],
    *,
    settings: Settings,
    existing: list[ReferenceLabel],
) -> tuple[list[ReferenceLabel], list[str]]:
    blind = labels_by_answer(existing, pass_name="blind")
    aware = labels_by_answer(existing, pass_name="concept_aware")
    contested = needs_adjudication(blind, aware)
    by_id = {a.id: a for a in answers}

    already = {
        label.answer_id
        for label in existing
        if label.pass_name == "blind" and label.adjudication is not None
    }
    todo = [aid for aid in contested if aid not in already]
    print(f"  adjudication: {len(contested)} contested, {len(todo)} to do")

    out = list(existing)
    failures: list[str] = []
    for position, answer_id in enumerate(todo, start=1):
        answer = by_id[answer_id]
        item = questions[answer.question_id]
        a, b = blind[answer_id], aware[answer_id]
        inputs: dict[str, object] = {
            "question": item.text,
            "answer": answer.answer,
            "score_a": str(a.judgment.score),
            "rationale_a": a.judgment.rationale,
            "score_b": str(b.judgment.score),
            "rationale_b": b.judgment.rationale,
        }
        try:
            judgment, meta = await call_structured(
                TaskName.EVAL_REFERENCE_LABEL_ADJUDICATE,
                inputs,
                ReferenceJudgment,
                settings=settings,
            )
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{answer_id}: {exc}")
            print(f"    [{position}/{len(todo)}] {answer_id} FAILED: {str(exc)[:120]}")
            continue

        out.append(
            ReferenceLabel(
                answer_id=answer_id,
                question_id=answer.question_id,
                judgment=judgment,
                pass_name="adjudication",
                evaluator_model=meta.model,
                task=str(TaskName.EVAL_REFERENCE_LABEL_ADJUDICATE),
                prompt_version=meta.prompt_version,
                prompt_fingerprint=meta.prompt_fingerprint,
                rubric_version=REFERENCE_RUBRIC_VERSION,
                temperature=meta.temperature,
                labelled_at=datetime.now(UTC).isoformat(timespec="seconds"),
            )
        )
        # The blind row carries the outcome, because the blind pass is the
        # primary reference label and `ReferenceLabel.score` reads the override
        # from there. `agreed` vs `revised` records whether the adjudicator
        # landed on the blind pass's number or moved it.
        revised = judgment.score != a.judgment.score
        out = [
            label.model_copy(
                update={
                    "adjudication": "revised" if revised else "agreed",
                    "adjudicated_score": judgment.score,
                }
            )
            if (label.pass_name == "blind" and label.answer_id == answer_id)
            else label
            for label in out
        ]
        dump_reference_labels(out, LABELS_PATH)
        print(
            f"    [{position}/{len(todo)}] {answer_id}  A={a.judgment.score} "
            f"B={b.judgment.score} -> {judgment.score} "
            f"({'revised' if revised else 'agreed'})"
        )
    return out, failures


def _report(labels: list[ReferenceLabel], answers: Sequence[EvalAnswer]) -> list[str]:
    print()
    for pass_name in ("blind", "concept_aware", "adjudication"):
        rows = [label for label in labels if label.pass_name == pass_name]
        if not rows:
            continue
        dist = score_distribution(rows)
        print(f"  {pass_name:<14} n={len(rows):<4} distribution {dist}")
    problems = reference_label_problems(
        labels, expected_answers=[a.id for a in answers], pass_name="blind"
    )
    print()
    if problems:
        for problem in problems:
            print(f"  PROBLEM  {problem}")
    else:
        print("  validation         OK")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pass",
        dest="which",
        choices=("blind", "concept_aware"),
        help="which labelling pass to run",
    )
    parser.add_argument(
        "--adjudicate",
        action="store_true",
        help="run the adjudicator on rows where the two passes differ by >= 2",
    )
    parser.add_argument("--limit", type=int, default=0, help="stop after N rows this run")
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="how many answers to label at once. 1 (the default) is sequential",
    )
    parser.add_argument("--dry-run", action="store_true", help="what would run; no calls")
    parser.add_argument("--validate", action="store_true", help="check the committed artefact")
    parser.add_argument("--path", type=Path, default=LABELS_PATH)
    args = parser.parse_args()

    answers = load_answers(ANSWERS_PATH)
    loaded, errors = load_bank()
    if errors:
        print("  the question bank does not load; refusing to label.")
        for error in errors[:5]:
            print(f"    {error}")
        return 1
    questions = {li.item.id: li.item for li in loaded}

    existing = load_reference_labels(args.path) if args.path.exists() else []
    print(f"  dataset        {ANSWERS_PATH}")
    print(f"  labels         {args.path}")
    print(f"  answers        {len(answers)}")
    print(f"  rubric         {REFERENCE_RUBRIC_VERSION}")
    print(f"  existing rows  {len(existing)}")

    if args.validate:
        problems = _report(existing, answers)
        return 1 if problems else 0

    if args.dry_run:
        order = labelling_order(answers, seed=LABEL_SEED)
        for pass_name in ("blind", "concept_aware"):
            done = set(labels_by_answer(existing, pass_name=pass_name))
            print(f"  {pass_name}: would label {len([a for a in order if a not in done])} rows")
        blind = labels_by_answer(existing, pass_name="blind")
        aware = labels_by_answer(existing, pass_name="concept_aware")
        if blind and aware:
            print(f"  adjudication: {len(needs_adjudication(blind, aware))} contested rows")
        print("\n  dry run: no calls made.")
        return 0

    if not args.which and not args.adjudicate:
        print("\n  nothing to do: pass --pass blind, --pass concept_aware or --adjudicate.")
        return 1

    settings = get_settings()

    async def run() -> tuple[list[ReferenceLabel], list[str]]:
        init_llm(settings)
        try:
            rows: list[ReferenceLabel] = existing
            failures: list[str] = []
            if args.which:
                rows, failures = await _run_pass(
                    answers,
                    questions,
                    pass_name=args.which,
                    settings=settings,
                    existing=rows,
                    limit=args.limit,
                    concurrency=args.concurrency,
                )
            if args.adjudicate:
                rows, more = await _run_adjudication(
                    answers, questions, settings=settings, existing=rows
                )
                failures = [*failures, *more]
            return rows, failures
        finally:
            await dispose_llm()

    labels, failures = asyncio.run(run())
    dump_reference_labels(labels, args.path)
    problems = _report(labels, answers)

    if failures:
        print(f"\n  {len(failures)} rows failed and were NOT labelled:")
        for failure in failures[:10]:
            print(f"    {failure}")
        print("  re-run to retry them; labelled rows are kept.")
        return 1
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
