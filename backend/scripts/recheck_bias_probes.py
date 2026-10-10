"""Re-measure the Day-19 bias probes offline, under the current code - Day 20.

    python scripts/recheck_bias_probes.py            # print and write the report

**No model calls.** Every grading run from Day 19 is persisted in
``evals/reports/grading/runs.jsonl`` *with the judgment it was scored from*
(``GradingRun.judgment``), and both the evidence/error verification and the
scoring arithmetic are pure functions. So the probes can be re-derived exactly:
take each stored judgment, put it through the verification that exists *now*,
re-score it, and recompute the paired deltas.

That is what makes "rerun the probe after the fix" possible with an exhausted
provider quota, and it is a stronger form of reproducibility than re-calling
would be - a live re-run would mix the fix's effect with the model's own
nondeterminism, and this isolates the fix.

**What this can and cannot measure.** It measures any change to the
*deterministic* half of the pipeline: evidence verification, error-claim
verification, the scoring formula, the penalty constants. It cannot measure a
change to the *prompt* or to sampling, because those would produce different
judgments and the stored ones would no longer apply. Day 20's error-claim
verification is squarely in the first category; a prompt change would not be,
and this script would say so by changing nothing.
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Sequence
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app.bank.loader import load_bank  # noqa: E402
from app.bank.schema import BankItem  # noqa: E402
from app.grading.evalset import EvalAnswer, load_answers  # noqa: E402
from app.grading.evidence import verify_evidence  # noqa: E402
from app.grading.grader import graded_answer_text  # noqa: E402
from app.grading.probes import (  # noqa: E402
    jargonise,
    pad_with_filler,
    prepend_confidence,
)
from app.grading.runs import GradingRun, load_runs, select  # noqa: E402
from app.grading.scoring import score_answer  # noqa: E402

DATASET_DIR = BACKEND.parent / "evals" / "datasets" / "grading"
ANSWERS_PATH = DATASET_DIR / "answers.jsonl"
RUNS_PATH = BACKEND.parent / "evals" / "reports" / "grading" / "runs.jsonl"
REPORT_PATH = BACKEND.parent / "evals" / "reports" / "bias_probes_rechecked.md"

#: Plan section 12.1's thresholds, transcribed. Not adjusted.
LENGTH_BIAS_MAX = 0.05
CONFIDENCE_BIAS_MAX = 0.03

#: How each variant's graded text is rebuilt from the base answer. Must match
#: `run_grading_validity.py` exactly, or the verification would be run against
#: text the model was never shown and would invent fabrications.
TRANSFORMS: dict[str, Callable[[str], str]] = {
    "filler": pad_with_filler,
    "confidence": prepend_confidence,
    "jargon": lambda text: jargonise(text)[0],
}


def _rescore(run: GradingRun, *, text: str, item: BankItem) -> tuple[float, int, float]:
    """Re-verify and re-score one stored judgment. Returns (score, dropped, penalty).

    ``graded_answer_text`` is applied for the same reason the pipeline applies
    it: the model was shown the fence-neutralised text, and verifying against
    the raw answer would report spurious fabrications on any answer containing
    a sentinel.
    """
    if run.judgment is None:
        raise ValueError(f"{run.answer_id} {run.variant}: no stored judgment to re-score")
    verified = verify_evidence(run.judgment, answer_text=graded_answer_text(text))
    breakdown = score_answer(verified.judgment, expected_concepts=item.expected_concepts)
    return breakdown.score, verified.dropped_error_count, verified.dropped_penalty


def _probe(
    variant: str,
    base: dict[str, GradingRun],
    runs: Sequence[GradingRun],
    answers: dict[str, EvalAnswer],
    questions: dict[str, BankItem],
    prompt_version: str,
) -> list[tuple[str, float, float, int]]:
    """Paired (answer, delta_before, delta_after, claims_dropped) for one variant."""
    variants = select(runs, prompt_version=prompt_version, variant=variant, run_index=0)  # type: ignore[arg-type]
    transform = TRANSFORMS[variant]
    out: list[tuple[str, float, float, int]] = []
    for answer_id in sorted(variants):
        if answer_id not in base:
            continue
        answer = answers[answer_id]
        raw = answer.answer
        item = questions[answer.question_id]
        b_score, b_dropped, _ = _rescore(base[answer_id], text=raw, item=item)
        v_score, v_dropped, _ = _rescore(variants[answer_id], text=transform(raw), item=item)
        before = variants[answer_id].score - base[answer_id].score
        out.append((answer_id, before, v_score - b_score, b_dropped + v_dropped))
    return out


def main() -> int:
    if not RUNS_PATH.exists():
        print(f"no grading runs at {RUNS_PATH}")
        return 1
    answers = {a.id: a for a in load_answers(ANSWERS_PATH)}
    loaded, errors = load_bank()
    if errors:
        print("the question bank does not load; refusing to recheck.")
        return 1
    questions = {li.item.id: li.item for li in loaded}
    runs = load_runs(RUNS_PATH)
    base = select(runs, prompt_version="v1", variant="base", run_index=0)

    out: list[str] = []
    w = out.append
    w("# Bias probes, re-measured offline after the Day-20 fix")
    w("")
    w("Generated by `backend/scripts/recheck_bias_probes.py`. **No model calls.**")
    w("")
    w("Each row's stored judgment is put through the verification that exists now and")
    w("re-scored. `before` is the delta Day 19 recorded; `after` is the delta the same")
    w("judgments produce under the current rules. The difference is the fix's effect,")
    w("isolated from model nondeterminism - a live re-run would confound the two.")
    w("")

    summary: dict[str, float] = {}
    for variant, threshold, assertion in (
        ("filler", LENGTH_BIAS_MAX, "max |delta| < 0.05"),
        ("confidence", CONFIDENCE_BIAS_MAX, "max |delta| < 0.03"),
        ("jargon", 0.0, "mean delta < 0 (score drops)"),
    ):
        rows = _probe(variant, base, runs, answers, questions, "v1")
        if not rows:
            w(f"## {variant}: **NOT MEASURED** - no paired rows")
            w("")
            continue
        before = [b for _, b, _, _ in rows]
        after = [a for _, _, a, _ in rows]
        dropped = sum(d for *_, d in rows)
        name = {"filler": "Length", "confidence": "Confidence", "jargon": "Jargon"}[variant]
        w(f"## {name} probe ({assertion}), n = {len(rows)}")
        w("")
        w("| | before | after |")
        w("|---|--:|--:|")
        w(
            f"| max \\|delta\\| | {max(abs(x) for x in before):.4f} | "
            f"**{max(abs(x) for x in after):.4f}** |"
        )
        w(
            f"| mean \\|delta\\| | {sum(abs(x) for x in before) / len(before):.4f} | "
            f"**{sum(abs(x) for x in after) / len(after):.4f}** |"
        )
        w(
            f"| mean delta | {sum(before) / len(before):+.4f} | "
            f"**{sum(after) / len(after):+.4f}** |"
        )
        w(
            f"| rows at exactly 0 | {sum(1 for x in before if x == 0)} | "
            f"{sum(1 for x in after if x == 0)} |"
        )
        w("")
        w(f"Fabricated error claims dropped across the paired rows: **{dropped}**.")
        w("")
        if variant == "jargon":
            ok = sum(after) / len(after) < 0
            summary[f"{variant}_mean_delta"] = sum(after) / len(after)
        else:
            ok = max(abs(x) for x in after) < threshold
            summary[f"{variant}_max_abs_delta"] = max(abs(x) for x in after)
        w(f"**Status after the fix: {'PASS' if ok else 'FAIL'}** " f"(threshold {assertion}).")
        w("")
        w("| answer | before | after | change | claims dropped |")
        w("|---|--:|--:|--:|--:|")
        for answer_id, b, a, d in sorted(rows, key=lambda r: -abs(r[2])):
            w(f"| {answer_id} | {b:+.4f} | {a:+.4f} | {a - b:+.4f} | {d} |")
        w("")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(out) + "\n", encoding="utf-8")
    print("\n".join(out))
    print(f"\n  wrote {REPORT_PATH}")
    for key, value in summary.items():
        print(f"    {key:<34} {value:+.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
