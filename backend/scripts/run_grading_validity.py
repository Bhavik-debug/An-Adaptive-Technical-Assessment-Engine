"""The Day-19 grading-validity harness. Plan sections 12.1 and 3's Day-19 row.

    python scripts/run_grading_validity.py --grade                  # main pass, v1
    python scripts/run_grading_validity.py --retest 40 --runs 5     # test-retest
    python scripts/run_grading_validity.py --probe length           # one bias probe
    python scripts/run_grading_validity.py --probe all
    python scripts/run_grading_validity.py --grade --prompt v2      # prompt comparison
    python scripts/run_grading_validity.py --report                 # metrics only, no calls

**Every LLM call is persisted and every number is computed from the persisted
rows.** ``--report`` makes no calls at all: it reads
``evals/reports/grading/runs.jsonl`` and recomputes the whole report, so the
arithmetic is auditable and a reader can change the binning or re-slice a
subgroup without quota. Resumable for the same reason the Day-18 builder is: a
rate limit 90 answers into a 120-answer pass must cost nothing already paid for.

**What this harness measures, and against what.** The grader produces a
continuous [0,1] score; the reference labels are ordinal 0-4. ``score_to_band``
maps one onto the other and the mapping is a documented, tested decision rather
than an inline ``round`` - see ``app/grading/validity.py``.

**The eligibility rule this script will not break.** QWK here is computed
against *model-generated* provisional reference labels. It is reported as
``provisional (model-referenced)`` and the Phase-4 exit gate - QWK >= 0.65
against **human** labels - stays unmet. ``gate_eligible()`` is untouched and
still returns ``False``.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(BACKEND.parent / ".env", override=False)

from app.bank.loader import load_bank  # noqa: E402
from app.bank.schema import BankItem, ExpectedConcept  # noqa: E402
from app.config import Settings, get_settings  # noqa: E402
from app.grading.evalset import (  # noqa: E402
    EvalAnswer,
    gate_eligible,
    labelling_order,
    load_answers,
)
from app.grading.pipeline import grade_answer  # noqa: E402
from app.grading.probes import (  # noqa: E402
    FILLER_200_WORDS,
    filler_problems,
    jargonise,
    pad_with_filler,
    position_invariance,
    prepend_confidence,
    shuffled_concepts,
)
from app.grading.prompt_variants import GRADING_PROMPTS  # noqa: E402
from app.grading.reflabel import (  # noqa: E402
    labels_by_answer,
    load_reference_labels,
)
from app.grading.runs import (  # noqa: E402
    GradingFailure,
    GradingRun,
    RunVariant,
    dump_failures,
    dump_runs,
    load_failures,
    load_runs,
    runs_index,
    select,
)
from app.grading.scoring import score_answer  # noqa: E402
from app.grading.validity import (  # noqa: E402
    AgreementReport,
    agreement,
    retest_reliability,
    score_to_band,
)
from app.llm.runtime import dispose_llm, init_llm  # noqa: E402

DATASET_DIR = BACKEND.parent / "evals" / "datasets" / "grading"
ANSWERS_PATH = DATASET_DIR / "answers.jsonl"
REFERENCE_PATH = DATASET_DIR / "reference_labels.jsonl"
REPORT_DIR = BACKEND.parent / "evals" / "reports" / "grading"
RUNS_PATH = REPORT_DIR / "runs.jsonl"
FAILURES_PATH = REPORT_DIR / "failures.jsonl"
REPORT_PATH = BACKEND.parent / "evals" / "reports" / "grading_validity.md"

#: Plan section 12.1: "Test-retest: grade 40 answers 5x -> report sigma."
RETEST_ITEMS = 40
RETEST_RUNS = 5

#: How many answers each bias probe uses. The plan does not fix this; 24 is one
#: answer per question, drawn by taking the first of each question in the
#: committed labelling order, so the probe spans every question and difficulty
#: rather than clustering in whichever rows came first.
PROBE_ITEMS = 24

#: Plan's Phase-4 exit gate thresholds, transcribed. Named here so the report
#: cannot quote a threshold the gate does not actually set.
LENGTH_BIAS_MAX = 0.05
CONFIDENCE_BIAS_MAX = 0.03
RETEST_SIGMA_MAX = 0.04
QWK_TARGET = 0.65


def _probe_sample(answers: Sequence[EvalAnswer], *, count: int) -> list[EvalAnswer]:
    """``count`` answers spanning distinct questions, chosen reproducibly."""
    order = labelling_order(answers, seed=8021)
    by_id = {a.id: a for a in answers}
    picked: list[EvalAnswer] = []
    seen: set[str] = set()
    for answer_id in order:
        row = by_id[answer_id]
        if row.question_id in seen:
            continue
        seen.add(row.question_id)
        picked.append(row)
        if len(picked) >= count:
            break
    return picked


def _retest_sample(answers: Sequence[EvalAnswer], *, count: int) -> list[EvalAnswer]:
    """The first ``count`` answers in the committed labelling order."""
    order = labelling_order(answers, seed=8021)
    by_id = {a.id: a for a in answers}
    return [by_id[answer_id] for answer_id in order[:count]]


async def _grade_one(
    answer: EvalAnswer,
    item: BankItem,
    *,
    variant: RunVariant,
    run_index: int,
    settings: Settings,
    prompt_version: str = "v1",
    concepts: Sequence[ExpectedConcept] | None = None,
    answer_text: str | None = None,
    n_substitutions: int | None = None,
) -> GradingRun:
    # v1 is the registered active prompt, so it is passed as None and resolved
    # from the registry - that way the default path through this harness is the
    # same path a turn takes. v2/v3 are evaluation-only variants.
    grade_prompt, recheck_prompt = GRADING_PROMPTS[prompt_version]
    override = None if prompt_version == "v1" else grade_prompt
    recheck_override = None if prompt_version == "v1" else recheck_prompt
    result = await grade_answer(
        question_text=item.text,
        expected_concepts=list(concepts if concepts is not None else item.expected_concepts),
        answer_text=answer_text if answer_text is not None else answer.answer,
        common_misconceptions=list(item.common_misconceptions),
        settings=settings,
        prompt=override,
        recheck_prompt=recheck_override,
    )
    first = result.calls[0]
    return GradingRun(
        answer_id=answer.id,
        question_id=answer.question_id,
        variant=variant,
        run_index=run_index,
        score=result.score,
        concept_score=result.breakdown.concept_score,
        rubric_score=result.breakdown.rubric_score,
        penalty=result.breakdown.penalty,
        grader_confidence=result.grader_confidence,
        gate_fired=result.gate.triggered,
        gate_reason=(
            None
            if not result.gate.triggered
            else ",".join(
                filter(
                    None,
                    (
                        "signal_disagreement" if result.gate.signal_disagreement else None,
                        "low_confidence" if result.gate.low_confidence else None,
                    ),
                )
            )
        ),
        n_concepts=len(result.judgment.concepts),
        n_evidence_downgrades=len(result.verification.downgraded_keys),
        n_errors_dropped=result.verification.dropped_error_count,
        dropped_penalty=result.verification.dropped_penalty,
        n_errors_major=result.breakdown.major_errors,
        n_errors_minor=result.breakdown.minor_errors,
        prompt_version=first.prompt_version,
        prompt_fingerprint=first.prompt_fingerprint,
        model=first.model,
        temperature=first.temperature,
        n_calls=len(result.calls),
        graded_at=datetime.now(UTC).isoformat(timespec="seconds"),
        n_substitutions=n_substitutions,
        # The evidence-corrected judgment, so the score is re-derivable offline
        # and the position probe can re-score it without a second model call.
        judgment=result.judgment,
    )


async def _grade_many(
    jobs: list[tuple[EvalAnswer, BankItem, RunVariant, int, dict[str, object]]],
    *,
    settings: Settings,
    existing: list[GradingRun],
    prompt_version: str,
    concurrency: int = 1,
    limit: int = 0,
) -> tuple[list[GradingRun], list[GradingFailure]]:
    """Grade every outstanding job, writing each result as soon as it lands.

    ``concurrency`` bounds how many answers are in flight at once, defaulting to
    1 - sequential, the behaviour this started with - because the right value
    depends on a provider's rate limits rather than on this code.

    The confidence gate already fans out three re-grades per gated answer, so
    the real ceiling on provider calls is ``concurrency * 4``; that is printed
    rather than silently multiplied.
    """
    have = runs_index(existing)
    out = list(existing)
    failures: list[GradingFailure] = []
    todo = [job for job in jobs if (job[0].id, prompt_version, job[2], job[3]) not in have]
    if limit:
        # Caps the number of *new* calls, not the number of jobs considered.
        # Applying it to `jobs` instead would make `--limit 6` a no-op on a
        # resumed pass whose first six rows are already on disk.
        todo = todo[:limit]
    print(f"  {len(jobs) - len(todo)} already graded, {len(todo)} to do")
    if concurrency > 1:
        print(f"  concurrency {concurrency} (up to {concurrency * 4} calls in flight)")

    limiter = asyncio.Semaphore(max(1, concurrency))
    write_lock = asyncio.Lock()
    done = 0

    async def one(
        answer: EvalAnswer,
        item: BankItem,
        variant: RunVariant,
        run_index: int,
        extra: dict[str, object],
    ) -> None:
        nonlocal done
        async with limiter:
            try:
                run = await _grade_one(
                    answer,
                    item,
                    variant=variant,
                    run_index=run_index,
                    settings=settings,
                    prompt_version=prompt_version,
                    **extra,  # type: ignore[arg-type]
                )
            except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
                async with write_lock:
                    done += 1
                    failures.append(
                        GradingFailure(
                            answer_id=answer.id,
                            question_id=answer.question_id,
                            variant=variant,
                            run_index=run_index,
                            prompt_version=prompt_version,
                            error_type=type(exc).__name__,
                            message=str(exc)[:2000],
                            failed_at=datetime.now(UTC).isoformat(timespec="seconds"),
                        )
                    )
                    print(
                        f"    [{done}/{len(todo)}] {answer.id} {variant} FAILED: "
                        f"{str(exc)[:110]}",
                        flush=True,
                    )
                return
        async with write_lock:
            out.append(run)
            # After every completed row, under a lock: a rate limit then costs
            # one call rather than a whole pass, and two coroutines cannot write
            # a file that is missing the other's row.
            dump_runs(out, RUNS_PATH)
            done += 1
            print(
                f"    [{done}/{len(todo)}] {answer.id} {variant} r{run_index} "
                f"-> {run.score:.3f} (band {score_to_band(run.score)}"
                f"{', gated' if run.gate_fired else ''})",
                flush=True,
            )

    await asyncio.gather(*(one(*job) for job in todo))
    return out, failures


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _fmt_agreement(report: AgreementReport) -> list[str]:
    lines = [
        f"### {report.label}",
        "",
        f"- n = **{report.n}**",
        f"- QWK = **{report.qwk:+.4f}**",
        f"- Spearman rho = **{report.spearman:+.4f}**",
        f"- MAE = **{report.mae:.4f}** bands",
        f"- mean signed error (predicted - reference) = **{report.signed_error:+.4f}**",
        f"- exact agreement = **{report.exact}/{report.n}** ({report.exact_rate:.1%})",
        f"- within one band = **{report.adjacent}/{report.n}** ({report.adjacent_rate:.1%})",
        f"- reference distribution: {report.reference_distribution}",
        f"- predicted distribution: {report.predicted_distribution}",
        "",
        "Confusion matrix (rows = reference band, columns = predicted band):",
        "",
        "| ref \\ pred | 0 | 1 | 2 | 3 | 4 |",
        "|---|--:|--:|--:|--:|--:|",
    ]
    for band, row in enumerate(report.confusion):
        lines.append(f"| **{band}** | " + " | ".join(str(cell) for cell in row) + " |")
    lines.append("")
    return lines


def _status(ok: object) -> str:
    """PASS / FAIL / NOT MEASURED.

    Takes ``object`` because its callers read out of the untyped ``summary``
    dict, and because the three-way distinction is the point: ``None`` means the
    probe did not run, which must never be rendered as a pass.
    """
    if ok is None:
        return "**NOT MEASURED**"
    return "**PASS**" if ok else "**FAIL**"


def build_report(
    answers: Sequence[EvalAnswer],
    runs: Sequence[GradingRun],
    *,
    prompt_version: str,
) -> tuple[str, dict[str, object]]:
    """The whole Day-19 report, computed from persisted rows. No calls."""
    lines: list[str] = []
    summary: dict[str, object] = {}

    by_id = {a.id: a for a in answers}
    reference = (
        labels_by_answer(load_reference_labels(REFERENCE_PATH), pass_name="blind")
        if REFERENCE_PATH.exists()
        else {}
    )
    base = select(runs, prompt_version=prompt_version, variant="base", run_index=0)
    grader_model = next(iter({r.model for r in base.values()}), "n/a")
    generators = sorted({a.generator_model for a in answers if a.generator_model})
    reference_model = next(iter({r.evaluator_model for r in reference.values()}), "n/a")

    lines += [
        "# Grading validity - Day 19",
        "",
        "Generated by `backend/scripts/run_grading_validity.py --report` from",
        "`evals/reports/grading/runs.jsonl`. Every number here is recomputed from those",
        "persisted rows; the script makes no model calls in `--report` mode.",
        "",
        "## What is being compared",
        "",
        "| | |",
        "|---|---|",
        f"| Grader prompt | `{prompt_version}` |",
        f"| Grader model | `{grader_model}` |",
        "| Grader score | continuous [0,1] from `scoring.py`, mapped to 0-4 by "
        "`score_to_band` (equal-width bins) |",
        f"| Human labels | **none** - `gate_eligible()` = `{gate_eligible(answers)}` |",
        "",
        "> **Every QWK in this report is provisional and model-referenced.** The Phase-4",
        "> exit gate is QWK >= 0.65 against *human* labels. There are no human labels, so",
        "> that gate is **NOT MEASURED** and nothing here changes it.",
        "",
        "### Read this before any number below: the grader and the reference share a model",
        "",
        "The configured provider serves every capability tier from one model, so three",
        "roles that should be independent are filled by the same weights:",
        "",
        f"- answers **written** by: {', '.join(f'`{g}`' for g in generators)}",
        f"- answers **graded** by: `{grader_model}`",
        f"- reference labels **assigned** by: `{reference_model}`",
        "",
        "That makes the Day-19 reference comparison **partly circular**: it measures",
        "agreement between two different prompts to one model, and a shared blind spot is",
        "invisible to it. The Day-18 label passes were produced by a *different* model",
        "family (`claude-opus-5`), so despite their own contamination they are the **less",
        "circular** reference for grader validity - which is why all three are reported",
        "side by side below rather than one being promoted to primary.",
        "",
        "Neither circularity nor contamination is fixable with the infrastructure",
        "available; both are reported.",
        "",
    ]

    # ---- how the grader behaved -------------------------------------------
    if base:
        rows_b = list(base.values())
        gated_rows = [r for r in rows_b if r.gate_fired]
        downgraded = [r for r in rows_b if r.n_evidence_downgrades]
        total_calls = sum(r.n_calls for r in rows_b)
        band_counts = dict.fromkeys(range(5), 0)
        for r in rows_b:
            band_counts[score_to_band(r.score)] += 1
        lines += [
            "## How the grader behaved",
            "",
            "Not metrics about agreement - facts about the grader itself, which the "
            "metrics cannot be read without.",
            "",
            "| | |",
            "|---|---|",
            f"| Answers graded | **{len(rows_b)}** |",
            f"| Chokepoint calls | **{total_calls}** "
            f"({total_calls / len(rows_b):.2f} per answer) |",
            f"| Confidence gate fired | **{len(gated_rows)}/{len(rows_b)}** "
            f"({len(gated_rows) / len(rows_b):.1%}) |",
            f"| Evidence spans downgraded as fabricated "
            f"| **{sum(r.n_evidence_downgrades for r in rows_b)}** across "
            f"**{len(downgraded)}** answers ({len(downgraded) / len(rows_b):.1%}) |",
            f"| Errors the grader flagged | **{sum(r.n_errors_major for r in rows_b)}** major, "
            f"**{sum(r.n_errors_minor for r in rows_b)}** minor |",
            f"| Grader band distribution | {band_counts} |",
            "",
        ]
        gate_rate = len(gated_rows) / len(rows_b)
        summary["gate_fire_rate"] = gate_rate
        summary["evidence_downgrade_rows"] = len(downgraded)
        if gate_rate > 0.15:
            reasons: dict[str, int] = {}
            for r in gated_rows:
                reasons[r.gate_reason or "unknown"] = reasons.get(r.gate_reason or "unknown", 0) + 1
            lines += [
                f"**The confidence gate is firing on {gate_rate:.0%} of answers against the "
                "~8% plan section 7.3 budgets for.** Breakdown by trigger: "
                + ", ".join(f"`{k}` {v}" for k, v in sorted(reasons.items()))
                + ".",
                "",
                "`signal_disagreement` means the concept score and the rubric score "
                "differed by more than 0.25 - the checklist and the rubric told different "
                "stories about the same answer. At this rate that is a statement about the "
                "prompt rather than about a handful of hard items, and it is the direct "
                f"cause of the {total_calls / len(rows_b):.2f} calls per answer: every "
                "gated answer costs four.",
                "",
            ]
        if len(downgraded) / len(rows_b) > 0.2:
            lines += [
                f"**The evidence verifier downgraded a span on "
                f"{len(downgraded) / len(rows_b):.0%} of answers.** Each one is a span the "
                "model quoted as proof that a concept was covered, which did not appear in "
                "the answer. Plan section 7.4 ranks substring verification second among the "
                "reliability techniques; this is how much it is doing. Without it those "
                "concepts would have scored as covered on the strength of quotations that "
                "do not exist.",
                "",
            ]

    # ---- coverage and contract violations ---------------------------------
    failures = load_failures(FAILURES_PATH)
    v1_failures = [f for f in failures if f.prompt_version == prompt_version]
    ungraded = sorted({a.id for a in answers} - set(base))
    lines += [
        "## Coverage: which answers actually have a grade",
        "",
        f"- answers in the dataset: **{len(answers)}**",
        f"- answers with a `{prompt_version}` base grade: **{len(base)}**",
        f"- answers with **no** grade: **{len(ungraded)}**",
        "",
    ]
    if v1_failures:
        by_type: dict[str, list[str]] = {}
        for failure in v1_failures:
            by_type.setdefault(failure.error_type, []).append(failure.answer_id)
        lines += [
            "Grading attempts that produced no score, by kind:",
            "",
            "| Error | n | Answers |",
            "|---|--:|---|",
        ]
        for error_type, ids in sorted(by_type.items()):
            shown = ", ".join(sorted(set(ids))[:6])
            lines.append(f"| `{error_type}` | {len(set(ids))} | {shown} |")
        total_attempts = len(base) + len({f.answer_id for f in v1_failures})
        rate = len({f.answer_id for f in v1_failures}) / total_attempts if total_attempts else 0.0
        lines += [
            "",
            f"**Contract/schema violation rate: {rate:.1%}** of answers attempted "
            f"({len({f.answer_id for f in v1_failures})} of {total_attempts}). Plan "
            "section 12.1's fourth suite targets **< 0.5%** for schema violations after "
            "retry.",
            "",
            "`GradingContractError` is the dominant kind and it is **not** a schema",
            "failure: the model returned structurally valid JSON that invented concept",
            "keys, or failed to classify keys it was given. `grader.check_judgment`",
            "catches it, the pipeline does not retry it, and the answer therefore has no",
            "score. Every denominator in this report is the number of answers that *did*",
            "grade, which is stated on each line rather than assumed to be 120.",
            "",
        ]
        summary["contract_violation_rate"] = rate
        summary["ungraded_answers"] = len(ungraded)
    else:
        lines += ["No grading attempt failed at this prompt version.", ""]
        summary["contract_violation_rate"] = 0.0
        summary["ungraded_answers"] = len(ungraded)

    # ---- primary agreement -------------------------------------------------
    paired = sorted(set(base) & set(reference))
    if paired:
        # Three reference label sets, one table. None is promoted to "primary":
        # the Day-19 pass is better isolated from the construction intent, the
        # Day-18 passes are better distributed and come from a different model
        # family than the grader, and all three are model-generated. Which one
        # a reader should weight depends on which weakness matters to them, so
        # the report states all of it rather than choosing for them.
        references: list[tuple[str, dict[str, int], str]] = [
            (
                "Day-19 blind (this session)",
                {aid: reference[aid].score for aid in paired},
                "isolated from `quality_intent` by construction; **same model as the "
                "grader**; severe central tendency",
            ),
            (
                "Day-18 pass A",
                {
                    aid: score
                    for aid in sorted(base)
                    if (score := by_id[aid].human_score) is not None
                },
                "`claude-opus-5[1m]`; **different model family from the grader**; "
                "contaminated on 6 disclosed rows",
            ),
            (
                "Day-18 retired pass",
                {
                    aid: score
                    for aid in sorted(base)
                    if (score := by_id[aid].model_score) is not None
                },
                "`claude-opus-5`; different model family from the grader; the original "
                "Day-18 pass",
            ),
        ]

        lines += [
            "## Agreement with each reference label set",
            "",
            "| Reference | n | QWK | Spearman | MAE | exact | within 1 | ref. modal share |",
            "|---|--:|--:|--:|--:|--:|--:|--:|",
        ]
        reports: list[tuple[str, AgreementReport, str]] = []
        for name, label_map, caveat in references:
            ids = sorted(set(label_map) & set(base))
            if len(ids) < 2:
                continue
            rep = agreement(
                [label_map[aid] for aid in ids],
                [score_to_band(base[aid].score) for aid in ids],
                label=f"Grader ({prompt_version}) vs {name}",
            )
            modal = max(rep.reference_distribution.values()) / rep.n
            lines.append(
                f"| {name} | {rep.n} | {rep.qwk:+.4f} | {rep.spearman:+.4f} | {rep.mae:.3f} "
                f"| {rep.exact_rate:.1%} | {rep.adjacent_rate:.1%} | {modal:.1%} |"
            )
            reports.append((name, rep, caveat))

        lines += ["", "What each reference is, and what is wrong with it:", ""]
        for name, _, caveat in reports:
            lines.append(f"- **{name}** - {caveat}")
        lines += [
            "",
            "**The `ref. modal share` column is why the QWK values differ so much.** A",
            "reference that puts most rows in one band has little ordering information to",
            "agree about, and a quadratically weighted kappa against it is close to",
            "degenerate. That is a property of the label set, not of the grader.",
            "",
        ]

        primary = reports[0][1]
        summary["qwk_provisional_day19_blind"] = primary.qwk
        summary["spearman_day19_blind"] = primary.spearman
        summary["mae_bands_day19_blind"] = primary.mae
        summary["n_primary"] = primary.n
        for name, rep, _ in reports[1:]:
            key = name.lower().replace(" ", "_").replace("-", "")
            summary[f"qwk_{key}"] = rep.qwk

        raw_mae = sum(abs(reference[aid].score / 4 - base[aid].score) for aid in paired) / len(
            paired
        )
        summary["mae_continuous_day19_blind"] = raw_mae
        lines += [
            f"MAE on the raw continuous scale (Day-19 reference/4 vs grader score) = "
            f"**{raw_mae:.4f}** in [0,1] units, computed without binning, so the band "
            "mapping is not hiding error.",
            "",
            "### Full detail per reference",
            "",
        ]
        for _, rep, _ in reports:
            lines += _fmt_agreement(rep)
    else:
        lines += [
            "## Agreement with each reference label set",
            "",
            "**NOT MEASURED** - no answer has both a grader run and a reference label.",
            "",
        ]

    # ---- test-retest -------------------------------------------------------
    lines += ["## Test-retest reliability", ""]
    retest_ids = sorted(
        {
            run.answer_id
            for run in runs
            if run.prompt_version == prompt_version and run.variant == "base"
        }
    )
    all_runs_per_item: dict[str, list[float]] = {}
    for aid in retest_ids:
        all_runs_per_item[aid] = [
            run.score
            for run in sorted(
                (
                    run
                    for run in runs
                    if run.answer_id == aid
                    and run.prompt_version == prompt_version
                    and run.variant == "base"
                ),
                key=lambda run: run.run_index,
            )
        ]

    # Only items with the *same* number of runs go into one sigma, and the
    # deepest such group is the one reported.
    #
    # The obvious alternative - take every item with two or more runs and
    # truncate them all to the shortest - silently averages a 5-run item's
    # first two scores with a 2-run item's, and would report "5 runs" for a
    # sigma computed over 2. Grouping by depth and naming the depth keeps the
    # number and its label honest; items at other depths are counted as
    # excluded rather than quietly folded in.
    by_depth: dict[int, list[str]] = {}
    for aid, scores in all_runs_per_item.items():
        if len(scores) >= 2:
            by_depth.setdefault(len(scores), []).append(aid)
    per_item: dict[str, list[float]] = {}
    excluded_depths: dict[int, int] = {}
    if by_depth:
        depth = max(by_depth)
        per_item = {aid: all_runs_per_item[aid] for aid in sorted(by_depth[depth])}
        excluded_depths = {
            other: len(ids) for other, ids in sorted(by_depth.items()) if other != depth
        }
    if per_item:
        width = len(next(iter(per_item.values())))
        ids = sorted(per_item)
        matrix = [[per_item[aid][i] for aid in ids] for i in range(width)]
        rel = retest_reliability(matrix)
        gated_items = {
            run.answer_id
            for run in runs
            if run.variant == "base"
            and run.prompt_version == prompt_version
            and run.gate_fired
            and run.answer_id in set(ids)
        }
        ok = rel.mean_sigma < RETEST_SIGMA_MAX
        lines += [
            f"- items = **{rel.n_items}**, runs each = **{rel.n_runs}**",
            f"- mean sigma = **{rel.mean_sigma:.5f}** (target < {RETEST_SIGMA_MAX}) {_status(ok)}",
            f"- max sigma on any one item = **{rel.max_sigma:.5f}**",
            f"- items whose score varied at all = **{rel.n_items_varying}/{rel.n_items}**",
            f"- largest spread on one item = **{rel.max_range:.4f}** score units",
            f"- items where the confidence gate fired at least once = **{len(gated_items)}**",
            "",
            "The gate is the expected source of variance: a gated score is the median of"
            " three temperature-0.7 re-grades (plan section 7.3), so it is designed to vary."
            " An ungated grade runs at temperature 0 and should be stable.",
            "",
        ]
        if excluded_depths:
            lines += [
                "Items at other run depths were **excluded** rather than truncated into "
                "this sigma: "
                + ", ".join(
                    f"{count} item(s) with {d} runs" for d, count in excluded_depths.items()
                )
                + ". Mixing depths would report a sigma computed over the shallowest group "
                "under the deepest group's label.",
                "",
            ]
        if rel.n_runs < RETEST_RUNS:
            lines += [
                f"> **Deviation from the plan.** Plan section 12.1 specifies "
                f"*grade 40 answers 5x*. This sigma is over **{rel.n_items} items x "
                f"{rel.n_runs} runs**, because the repeated-grading pass was still "
                "incomplete when the report was generated. Re-run "
                "`--retest 40 --runs 5` and regenerate to close it.",
                "",
            ]
        summary["retest_mean_sigma"] = rel.mean_sigma
        summary["retest_items"] = rel.n_items
        summary["retest_runs"] = rel.n_runs
        summary["retest_pass"] = ok
    else:
        lines += ["**NOT MEASURED** - no answer has two or more runs at this prompt version.", ""]
        summary["retest_pass"] = None

    # ---- bias probes -------------------------------------------------------
    lines += [
        "## Bias probes",
        "",
        "| Probe | Method | Assertion | n | Result | Status |",
        "|---|---|---|--:|---|---|",
    ]
    probe_detail: list[str] = []

    def _paired_delta(variant: RunVariant) -> tuple[list[tuple[str, float, float]], list[str]]:
        rows: list[tuple[str, float, float]] = []
        skipped: list[str] = []
        var = select(runs, prompt_version=prompt_version, variant=variant, run_index=0)
        for aid in sorted(var):
            if aid not in base:
                skipped.append(f"{aid}: no base run")
                continue
            rows.append((aid, base[aid].score, var[aid].score))
        return rows, skipped

    def _gate_free(variant: RunVariant, rows: list[tuple[str, float, float]]) -> list[str]:
        """Answer ids in ``rows`` where the gate fired on neither side.

        **Why this subset exists.** A gated score is the median of three
        temperature-0.7 re-grades, so it is *designed* to move between
        invocations. If the gate fires on the base run or on the perturbed run,
        the observed delta mixes the bias being probed with sampling noise that
        has nothing to do with the perturbation - and on a probe whose threshold
        is 0.05 that noise can be larger than the effect.
        """
        var = select(runs, prompt_version=prompt_version, variant=variant, run_index=0)
        return [aid for aid, _, _ in rows if not base[aid].gate_fired and not var[aid].gate_fired]

    # length
    rows, _ = _paired_delta("filler")
    if rows:
        deltas = [after - before for _, before, after in rows]
        worst = max(abs(d) for d in deltas)
        mean_abs = sum(abs(d) for d in deltas) / len(deltas)
        ok = worst < LENGTH_BIAS_MAX
        lines.append(
            f"| Length | append {len(FILLER_200_WORDS.split())} words of contentless filler "
            f"| max abs delta < {LENGTH_BIAS_MAX} | {len(rows)} "
            f"| max {worst:.4f}, mean abs {mean_abs:.4f} | {_status(ok)} |"
        )
        summary["length_bias_max_abs_delta"] = worst
        summary["length_bias_pass"] = ok
        clean_ids = set(_gate_free("filler", rows))
        clean = [(aid, b, a) for aid, b, a in rows if aid in clean_ids]
        if clean and len(clean) < len(rows):
            clean_worst = max(abs(a - b) for _, b, a in clean)
            summary["length_bias_max_abs_delta_gate_free"] = clean_worst
            probe_detail += [
                f"**Length probe, excluding pairs where the confidence gate fired on "
                f"either side** ({len(clean)} of {len(rows)} pairs): max abs delta "
                f"**{clean_worst:.4f}**, {_status(clean_worst < LENGTH_BIAS_MAX)}.",
                "",
                "A gated score is a median of three temperature-0.7 re-grades, so it "
                "moves between invocations by design. On a probe with a 0.05 threshold "
                "that noise can exceed the effect being measured, so both figures are "
                "given and neither is presented as the only one.",
                "",
            ]
        probe_detail += [
            "### Length probe, per item",
            "",
            "| answer | base | padded | delta | gated? |",
            "|---|--:|--:|--:|---|",
            *(
                f"| {aid} | {before:.3f} | {after:.3f} | {after - before:+.4f} "
                f"| {'' if aid in clean_ids else 'yes'} |"
                for aid, before, after in rows
            ),
            "",
        ]
    else:
        lines.append(
            f"| Length | append filler | max abs delta < {LENGTH_BIAS_MAX} "
            f"| 0 | - | {_status(None)} |"
        )
        summary["length_bias_pass"] = None

    # confidence
    rows, _ = _paired_delta("confidence")
    if rows:
        deltas = [after - before for _, before, after in rows]
        worst = max(abs(d) for d in deltas)
        mean_abs = sum(abs(d) for d in deltas) / len(deltas)
        ok = worst < CONFIDENCE_BIAS_MAX
        lines.append(
            f'| Confidence | prepend "I\'m certain that..." '
            f"| max abs delta < {CONFIDENCE_BIAS_MAX} | {len(rows)} "
            f"| max {worst:.4f}, mean abs {mean_abs:.4f} | {_status(ok)} |"
        )
        summary["confidence_bias_max_abs_delta"] = worst
        summary["confidence_bias_pass"] = ok
        conf_clean_ids = set(_gate_free("confidence", rows))
        conf_clean = [(aid, b, a) for aid, b, a in rows if aid in conf_clean_ids]
        if conf_clean and len(conf_clean) < len(rows):
            conf_worst = max(abs(a - b) for _, b, a in conf_clean)
            summary["confidence_bias_max_abs_delta_gate_free"] = conf_worst
            probe_detail += [
                f"**Confidence probe, excluding gated pairs** ({len(conf_clean)} of "
                f"{len(rows)}): max abs delta **{conf_worst:.4f}**, "
                f"{_status(conf_worst < CONFIDENCE_BIAS_MAX)}.",
                "",
            ]
    else:
        lines.append(
            f"| Confidence | prepend certainty | max abs delta < {CONFIDENCE_BIAS_MAX} | 0 | - "
            f"| {_status(None)} |"
        )
        summary["confidence_bias_pass"] = None

    # jargon
    rows, _ = _paired_delta("jargon")
    var = select(runs, prompt_version=prompt_version, variant="jargon", run_index=0)
    usable = [(aid, b, a) for aid, b, a in rows if (var[aid].n_substitutions or 0) > 0]
    if usable:
        deltas = [after - before for _, before, after in usable]
        mean_delta = sum(deltas) / len(deltas)
        dropped = sum(1 for d in deltas if d < 0)
        ok = mean_delta < 0
        lines.append(
            f"| Jargon | replace precise terms with buzzwords | mean delta < 0 (score drops) "
            f"| {len(usable)} | mean {mean_delta:+.4f}, dropped on {dropped}/{len(usable)} "
            f"| {_status(ok)} |"
        )
        summary["jargon_mean_delta"] = mean_delta
        summary["jargon_pass"] = ok
        excluded = len(rows) - len(usable)
        if excluded:
            probe_detail += [
                f"*{excluded} jargon rows excluded: no substitution matched, so the variant was "
                "identical to the base answer and would have scored a vacuous pass.*",
                "",
            ]
    else:
        lines.append(
            f"| Jargon | buzzword substitution | mean delta < 0 | 0 | - | {_status(None)} |"
        )
        summary["jargon_pass"] = None

    # position - deterministic, no LLM
    rows, _ = _paired_delta("concepts_shuffled")
    if rows:
        deltas = [abs(after - before) for _, before, after in rows]
        worst = max(deltas)
        ok = worst == 0.0
        lines.append(
            f"| Position | shuffle `expected_concepts` order | delta == exactly 0 | {len(rows)} "
            f"| max abs delta {worst:.1e} | {_status(ok)} |"
        )
        summary["position_bias_max_abs_delta"] = worst
        summary["position_bias_pass"] = ok
    else:
        lines.append(
            f"| Position | shuffle concept order | delta == exactly 0 "
            f"| 0 | see note | {_status(None)} |"
        )
        summary["position_bias_pass"] = None

    lines += ["", *probe_detail]

    # ---- prompt comparison -------------------------------------------------
    versions = sorted({run.prompt_version for run in runs if run.variant == "base"})
    lines += ["## Prompt-version comparison", ""]
    if len(versions) < 2:
        lines += [
            f"**NOT MEASURED** - only prompt version(s) {versions} have been run.",
            "",
            "The plan's Day-19 row asks for 'prompt iteration v1 -> v3 against the metric'.",
            "Only `v1` existed before Day 19 (`GRADING_PROMPT_V1`, Day 16); see the report",
            "body for what was and was not done about that.",
            "",
        ]
    else:
        # The comparison is restricted to answers graded under EVERY version.
        #
        # Without that restriction v1 would be scored on 120 answers and the
        # variants on the 24 the quota allowed, and a difference in the *subset*
        # would be indistinguishable from a difference in the *prompt*. A paired
        # comparison on the common subset is the only one that measures what the
        # heading claims.
        slices = {
            version: select(runs, prompt_version=version, variant="base", run_index=0)
            for version in versions
        }
        common = sorted(set.intersection(*(set(sl) for sl in slices.values())) & set(reference))
        lines += [
            f"Paired on the **{len(common)} answers graded under every version**. "
            "v1 also has a full 120-answer pass; its figure on all 120 is in the "
            "agreement table above, and mixing the two would confound the prompt with "
            "the subset.",
            "",
            "| Prompt | n | QWK | Spearman | MAE | exact | gate fired |",
            "|---|--:|--:|--:|--:|--:|--:|",
        ]
        for version in versions:
            sl = slices[version]
            if not common:
                continue
            rep = agreement(
                [reference[aid].score for aid in common],
                [score_to_band(sl[aid].score) for aid in common],
                label=version,
            )
            n_gated = sum(1 for aid in common if sl[aid].gate_fired)
            lines.append(
                f"| `{version}` | {rep.n} | {rep.qwk:+.4f} | {rep.spearman:+.4f} "
                f"| {rep.mae:.4f} | {rep.exact_rate:.1%} | {n_gated} |"
            )
            summary[f"qwk_paired_{version}"] = rep.qwk
        lines += [
            "",
            "Same reference labels, same answers, same `score_to_band`, same metric code "
            "for every row - only the grading prompt differs.",
            "",
            f"**n = {len(common)}, which is small.** A QWK on 24 items has a wide "
            "confidence interval and the ordering of three such values should be treated "
            "as a direction to investigate, not as a result. The variants were not run on "
            "all 120 answers because the provider's free-tier quota was exhausted; "
            "finishing that is the first thing to do with fresh quota.",
            "",
        ]

    # ---- gate table --------------------------------------------------------
    lines += [
        "## Phase-4 exit gates",
        "",
        "| Gate | Requirement | Evidence | Status | Formally eligible? |",
        "|---|---|---|---|---|",
        f"| QWK (human reference) | >= {QWK_TARGET} on the 120-item set vs **human** labels "
        "| no human labels exist | **NOT MEASURED** | no |",
        f"| QWK (provisional, model reference) | development signal only "
        f"| {summary.get('qwk_provisional_day19_blind', float('nan')):+.4f} over n="
        f"{summary.get('n_primary', 0)} | informational | **no - not the gate** |",
        f"| Length bias | \\|delta\\| < {LENGTH_BIAS_MAX} "
        f"| {summary.get('length_bias_max_abs_delta', float('nan')):.4f} max "
        f"| {_status(summary.get('length_bias_pass'))} | yes |",
        f"| Position bias | delta == exactly 0 "
        f"| {summary.get('position_bias_max_abs_delta', float('nan')):.1e} max "
        f"| {_status(summary.get('position_bias_pass'))} | yes |",
        "| Injection ASR | 0% for score manipulation | Day 20, not run | **NOT MEASURED** | no |",
        "| CI QWK regression | build fails if QWK drops > 0.05 | Day 20, not wired "
        "| **NOT MEASURED** | no |",
        "",
        "Confidence and jargon probes are plan section 12.1 bias probes, not Phase-4 exit",
        "gates; they are reported above with the same PASS/FAIL discipline.",
        "",
    ]

    return "\n".join(lines) + "\n", summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grade", action="store_true", help="grade all 120 answers")
    parser.add_argument(
        "--grade-subset",
        type=int,
        default=0,
        metavar="N",
        help=(
            "grade N answers spanning N distinct questions, chosen by the same "
            "reproducible rule the bias probes use. Exists so a prompt-version "
            "comparison can run on a documented subset when a full 120-answer pass "
            "per version is out of quota; the subset is identical across versions, "
            "so the comparison stays paired"
        ),
    )
    parser.add_argument(
        "--retest", type=int, default=0, metavar="N", help="test-retest on N answers"
    )
    parser.add_argument(
        "--runs", type=int, default=RETEST_RUNS, help="runs per answer for --retest"
    )
    parser.add_argument(
        "--probe",
        choices=("length", "confidence", "jargon", "position", "all"),
        help="run a bias probe",
    )
    parser.add_argument("--probe-items", type=int, default=PROBE_ITEMS)
    parser.add_argument("--prompt", default="v1", help="which grading prompt version to report on")
    parser.add_argument("--limit", type=int, default=0, help="stop after N calls this run")
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help=(
            "how many answers to grade at once. 1 (the default) is sequential. The "
            "confidence gate fans out 3 re-grades per gated answer, so the provider "
            "sees up to 4x this many calls in flight"
        ),
    )
    parser.add_argument("--report", action="store_true", help="recompute the report; no calls")
    args = parser.parse_args()

    answers = load_answers(ANSWERS_PATH)
    loaded, errors = load_bank()
    if errors:
        print("  the question bank does not load; refusing to run.")
        return 1
    questions = {li.item.id: li.item for li in loaded}
    by_id = {a.id: a for a in answers}
    existing = load_runs(RUNS_PATH) if RUNS_PATH.exists() else []

    print(f"  answers        {len(answers)}")
    print(f"  runs on disk   {len(existing)}")
    print(f"  reference      {'present' if REFERENCE_PATH.exists() else 'MISSING'}")

    if args.report:
        text, summary = build_report(answers, existing, prompt_version=args.prompt)
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text(text, encoding="utf-8")
        print(f"\n  wrote {REPORT_PATH}")
        for key, value in summary.items():
            print(f"    {key:<34} {value}")
        return 0

    jobs: list[tuple[EvalAnswer, BankItem, RunVariant, int, dict[str, object]]] = []

    if args.grade:
        order = labelling_order(answers, seed=8021)
        for answer_id in order:
            answer = by_id[answer_id]
            jobs.append((answer, questions[answer.question_id], "base", 0, {}))

    if args.grade_subset:
        for answer in _probe_sample(answers, count=args.grade_subset):
            jobs.append((answer, questions[answer.question_id], "base", 0, {}))

    if args.retest:
        for answer in _retest_sample(answers, count=args.retest):
            for run_index in range(args.runs):
                jobs.append((answer, questions[answer.question_id], "base", run_index, {}))

    if args.probe in ("length", "all"):
        for answer in _probe_sample(answers, count=args.probe_items):
            item = questions[answer.question_id]
            bad = filler_problems(
                FILLER_200_WORDS,
                concept_keys=item.concept_keys,
                anchor_terms=item.anchor_terms,
            )
            if bad:
                # Refusing is the point: padding that mentions an expected
                # concept turns a length probe into a concept probe.
                print(f"    skipping {answer.id}: filler is not contentless here - {bad}")
                continue
            jobs.append(
                (
                    answer,
                    item,
                    "filler",
                    0,
                    {"answer_text": pad_with_filler(answer.answer)},
                )
            )

    if args.probe in ("confidence", "all"):
        for answer in _probe_sample(answers, count=args.probe_items):
            jobs.append(
                (
                    answer,
                    questions[answer.question_id],
                    "confidence",
                    0,
                    {"answer_text": prepend_confidence(answer.answer)},
                )
            )

    if args.probe in ("jargon", "all"):
        # Drawn from answers that actually *contain* a substitutable term,
        # rather than from the generic probe sample.
        #
        # This is a sampling fix, not a tuning one: on the generic sample only
        # 12 of 24 answers contain any term in the table, so half the calls
        # produce a variant identical to its base and contribute nothing but a
        # vacuous pass. The threshold, the direction of the assertion and the
        # substitution table are all unchanged - the probe simply stops being
        # run on answers the perturbation cannot touch.
        substitutable = [a for a in answers if jargonise(a.answer)[1] > 0]
        for answer in _probe_sample(substitutable, count=args.probe_items):
            text, made = jargonise(answer.answer)
            if made == 0:  # pragma: no cover - filtered above
                print(f"    skipping {answer.id}: no jargon substitution matched")
                continue
            jobs.append(
                (
                    answer,
                    questions[answer.question_id],
                    "jargon",
                    0,
                    {"answer_text": text, "n_substitutions": made},
                )
            )

    if args.probe in ("position", "all"):
        # Deterministic and code-side: NO model calls. Plan section 12.1 - "a
        # test that proves your architecture, not just your prompt". Every
        # stored judgment is re-scored with its concept list shuffled eight
        # ways; the assertion is that the score changes by exactly zero.
        print("\n  position probe (deterministic re-score, no LLM calls):")
        shuffled_rows = list(existing)
        have = runs_index(shuffled_rows)
        rescored = 0
        worst_overall = 0.0
        for answer in answers:
            base_key = (answer.id, args.prompt, "base", 0)
            baseline = have.get(base_key)
            if baseline is None or baseline.judgment is None:
                continue
            item = questions[answer.question_id]
            worst = position_invariance(
                baseline.judgment,
                item.expected_concepts,
                score_fn=score_answer,
                trials=8,
            )
            worst_overall = max(worst_overall, worst)
            rescored += 1
            shuffled = shuffled_concepts(item.expected_concepts, seed=1219)
            breakdown = score_answer(
                baseline.judgment,
                expected_concepts=shuffled,
            )
            shuffled_rows = [
                r
                for r in shuffled_rows
                if r.key != (answer.id, args.prompt, "concepts_shuffled", 0)
            ]
            shuffled_rows.append(
                baseline.model_copy(
                    update={
                        "variant": "concepts_shuffled",
                        "score": breakdown.score,
                        "concept_score": breakdown.concept_score,
                        "rubric_score": breakdown.rubric_score,
                        "penalty": breakdown.penalty,
                        "graded_at": datetime.now(UTC).isoformat(timespec="seconds"),
                    }
                )
            )
        if rescored:
            dump_runs(shuffled_rows, RUNS_PATH)
            print(f"    re-scored {rescored} stored judgments, 8 shuffles each")
            print(f"    largest absolute score change: {worst_overall:.1e}")
            print(f"    {'PASS - exactly zero' if worst_overall == 0.0 else 'FAIL'}")
            existing = shuffled_rows
        else:
            print("    no stored judgments to re-score; run --grade first")

    if not jobs:
        if not args.probe:
            print("\n  nothing to do. Pass --grade, --retest N, --probe X or --report.")
            return 1
        print("\n  no model-calling jobs queued.")
        return 0

    settings = get_settings()

    async def run() -> tuple[list[GradingRun], list[GradingFailure]]:
        init_llm(settings)
        try:
            return await _grade_many(
                jobs,
                settings=settings,
                existing=existing,
                prompt_version=args.prompt,
                concurrency=args.concurrency,
                limit=args.limit,
            )
        finally:
            await dispose_llm()

    runs_out, failures = asyncio.run(run())
    dump_runs(runs_out, RUNS_PATH)
    if failures:
        # Merged with what is already on disk, deduplicated by key: a resumed
        # pass re-attempts the same unreachable rows and must not triple-count
        # them.
        dump_failures([*load_failures(FAILURES_PATH), *failures], FAILURES_PATH)
    print(f"\n  {len(runs_out)} rows in {RUNS_PATH}")
    gated = sum(1 for r in runs_out if r.gate_fired)
    calls = sum(r.n_calls for r in runs_out)
    print(f"  gate fired on {gated} rows; {calls} chokepoint calls recorded in total")
    if failures:
        print(f"\n  {len(failures)} jobs failed:")
        for failure in failures[:10]:
            print(f"    {failure}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
