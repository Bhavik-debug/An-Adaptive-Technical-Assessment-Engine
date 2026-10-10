"""Phase-4 remediation diagnostics - instrumentation only, no fixes.

    python scripts/diagnose_phase4.py --offline                 # 11 provider-free sections
    python scripts/diagnose_phase4.py --raw-spans               # 3 calls: raw spans
    python scripts/diagnose_phase4.py --replicate 4             # 4*k calls: k per condition
    python scripts/diagnose_phase4.py --catch-failing-span 4    # up to N calls: dump a failure

**Why this script exists.** :class:`app.grading.runs.GradingRun` stores the
*evidence-corrected* judgment, and :func:`app.grading.evidence.verify_evidence`
nulls a span when it downgrades the concept. So the committed artefacts record
**how many** evidence spans failed verification (``n_evidence_downgrades``) but
neither **what the model quoted** nor **which span failed**. That is the one
question the stage-attribution analysis cannot answer from disk, and it separates
two very different root causes:

* the model offered something that is not an exact quotation - a précis, or two
  real excerpts joined by an ellipsis - and the verifier refused credit the
  candidate had in fact earned; or
* the model invented the span, and the verifier did exactly its job.

``--raw-spans`` and ``--catch-failing-span`` answer it by calling
:func:`app.grading.grader.judge_answer`, which returns the raw judgment before
``verify_evidence`` touches it. ``--replicate`` answers a different question:
whether a reported bias maximum is an effect or an order statistic of the
grader's own run-to-run variance.

**This is instrumentation, not remediation.** It changes no production code, no
threshold, no baseline and no stored artefact; ``--offline`` is pure reads and
arithmetic, and the three live modes write nothing. It is committed because
``evals/reports/phase4_diagnostic.md`` quotes its output, and a report whose
evidence cannot be re-derived is an opinion.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Literal

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(BACKEND.parent / ".env", override=False)

from app.bank.loader import load_bank  # noqa: E402
from app.grading.evalset import QUALITY_LEVELS, leakage_report, load_answers  # noqa: E402
from app.grading.evidence import verify_evidence, verify_span  # noqa: E402
from app.grading.grader import graded_answer_text, judge_answer  # noqa: E402
from app.grading.injection import SUCCESS_MARGIN  # noqa: E402
from app.grading.probes import pad_with_filler, prepend_confidence  # noqa: E402
from app.grading.reflabel import load_reference_labels  # noqa: E402
from app.grading.runs import GradingRun, load_runs, select  # noqa: E402
from app.grading.scoring import CONCEPT_WEIGHT, COVER, score_answer  # noqa: E402
from app.grading.validity import spearman_rho  # noqa: E402

if TYPE_CHECKING:  # pragma: no cover - import-time only
    from app.bank.schema import BankItem, ExpectedConcept
    from app.config import Settings

DATASET = BACKEND.parent / "evals" / "datasets" / "grading"
REPORTS = BACKEND.parent / "evals" / "reports"
RUNS_PATH = REPORTS / "grading" / "runs.jsonl"
INJECTION_PATH = REPORTS / "grading" / "injection.jsonl"

RULE = "=" * 78
NL = "\n"

#: ``GradingRun.variant`` is a Literal, so the probe arms this script walks have
#: to be typed as one too rather than as plain strings.
Variant = Literal["base", "filler", "confidence", "jargon", "concepts_shuffled"]
PROBE_ARMS: tuple[Variant, ...] = ("base", "filler", "confidence", "jargon")

#: Rows chosen for raw-span capture: the worst length failure (plus its own
#: control, re-graded in the same sitting so the comparison is not against a
#: week-old sample), and the worst confidence failure. Fixed rather than sampled
#: so the experiment is repeatable, and small so it costs 3 provider calls.
RAW_SPAN_ROWS: tuple[tuple[str, Variant], ...] = (
    ("eval-0113", "base"),
    ("eval-0113", "filler"),
    ("eval-0068", "confidence"),
)

#: Rows re-graded k times per condition by ``--replicate``: the two that set the
#: reported length-bias and confidence-bias maxima. Both maxima were measured
#: from ONE sample of the variant against ONE sample of the control, on a grader
#: whose score range over repeated runs of *identical* input reaches 0.1956 - so
#: whether they are bias or an order statistic of noise is exactly the question
#: k runs per condition answers.
REPLICATE_ROWS: tuple[tuple[str, Variant], ...] = (
    ("eval-0113", "filler"),
    ("eval-0068", "confidence"),
)

VARIANT_TRANSFORM = {
    "base": lambda text: text,
    "filler": pad_with_filler,
    "confidence": prepend_confidence,
}


def offline() -> None:  # noqa: PLR0915 - a report, read top to bottom
    """Every diagnostic that needs no provider. Pure reads and arithmetic."""
    answers = {a.id: a for a in load_answers(DATASET / "answers.jsonl")}
    loaded, errors = load_bank()
    if errors:
        print("  the question bank does not load; aborting.")
        return
    questions = {li.item.id: li.item for li in loaded}
    runs = load_runs(RUNS_PATH)
    base = select(runs, prompt_version="v1", variant="base", run_index=0)

    print(RULE)
    print("1. ASR recomputed from stored scores vs stored outcome flags")
    print(RULE)
    rows = [
        json.loads(line)
        for line in INJECTION_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for version in sorted({r["prompt_version"] for r in rows}):
        cases = [r for r in rows if r["prompt_version"] == version]
        stored = sum(
            1 for r in cases if {"score_inflated", "credit_from_payload"} & set(r["outcomes"])
        )
        recomputed = sum(
            1
            for r in cases
            if (r["attacked_score"] - r["control_score"]) > SUCCESS_MARGIN + 1e-9
            or r["payload_credited"]
        )
        flag = "" if stored == recomputed else "   <-- STALE stored outcomes"
        print(
            f"  {version}: n={len(cases)}  stored={stored} ({stored / len(cases):.1%})  "
            f"recomputed={recomputed} ({recomputed / len(cases):.1%}){flag}"
        )
    for r in rows:
        delta = r["attacked_score"] - r["control_score"]
        if abs(delta - SUCCESS_MARGIN) < 1e-6:
            print(f"    boundary case {r['case_id']} ({r['prompt_version']}): delta={delta!r}")
    succeeded = [
        r
        for r in rows
        if (r["attacked_score"] - r["control_score"]) > SUCCESS_MARGIN + 1e-9
        or r["payload_credited"]
    ]
    print(f"  cases succeeding under the recomputed rule: {len(succeeded)} case/arm rows")
    for r in sorted(succeeded, key=lambda r: (r["case_id"], r["prompt_version"])):
        print(
            f"    {r['case_id']} {r['prompt_version']} {r['category']:24} {r['answer_id']} "
            f"delta={r['attacked_score'] - r['control_score']:+.4f} {r['notes']}"
        )
    both = {
        r["case_id"]
        for r in succeeded
        if sum(1 for q in succeeded if q["case_id"] == r["case_id"]) > 1
    }
    print(f"  replicating across BOTH prompt arms: {sorted(both)}")
    print(f"  any payload_credited: {sum(1 for r in rows if r['payload_credited'])}/{len(rows)}")
    print(f"  any leaked_fields   : {sum(1 for r in rows if r['leaked_fields'])}/{len(rows)}")

    print()
    print(RULE)
    print("2. Verbatim-quote failure rate, by variant")
    print(RULE)
    header = f"  {'variant':14}{'rows':>6}{'rows w/dgr':>12}{'downgrades':>12}"
    print(header + f"{'concepts':>10}{'per concept':>13}")
    for variant in PROBE_ARMS:
        rows_in_arm = list(select(runs, prompt_version="v1", variant=variant, run_index=0).values())
        if not rows_in_arm:
            continue
        downgrades = sum(r.n_evidence_downgrades for r in rows_in_arm)
        concepts = sum(r.n_concepts for r in rows_in_arm)
        with_any = sum(1 for r in rows_in_arm if r.n_evidence_downgrades)
        print(
            f"  {variant:14}{len(rows_in_arm):>6}{with_any:>12}{downgrades:>12}{concepts:>10}"
            f"{downgrades / concepts:>12.1%}"
        )
    surviving = sum(
        1
        for r in base.values()
        if r.judgment is not None
        for c in r.judgment.concepts
        if c.label != "absent"
    )
    downgraded = sum(r.n_evidence_downgrades for r in base.values())
    claimed = surviving + downgraded
    print(
        f"{NL}  base pass: of {claimed} concepts the model CLAIMED credit for, {downgraded} "
        f"were dropped for an unverifiable quote = {downgraded / claimed:.1%}"
    )

    print()
    print(RULE)
    print("3. Stage attribution for rows whose score moved more than 0.05")
    print(RULE)
    probes: tuple[Variant, ...] = ("filler", "confidence")
    for variant in probes:
        arm = select(runs, prompt_version="v1", variant=variant, run_index=0)
        big: list[tuple[str, float, int, bool]] = []
        for answer_id, row in sorted(arm.items()):
            control = base.get(answer_id)
            if control is None:
                continue
            delta = row.score - control.score
            if abs(delta) <= SUCCESS_MARGIN:
                continue
            verifier = row.n_evidence_downgrades - control.n_evidence_downgrades
            gated = control.gate_fired or row.gate_fired
            big.append((answer_id, delta, verifier, gated))
        print(f"  {variant}: {len(big)} of {len(arm)} rows exceed 0.05")
        print(f"    verifier activity differed : {sum(1 for r in big if r[2])}")
        print(f"    gate fired on either side  : {sum(1 for r in big if r[3])}")
        print(f"    neither                    : {sum(1 for r in big if not r[2] and not r[3])}")
        for answer_id, delta, verifier, gated in sorted(big, key=lambda r: -abs(r[1]))[:5]:
            print(
                f"      {answer_id}  delta={delta:+.4f}  d_downgrades={verifier:+d}  gate={gated}"
            )

    print()
    print(RULE)
    print("4. What one unverifiable quote is worth, in score")
    print(RULE)
    worst = 0.0
    for answer_id in base:
        item = questions[answers[answer_id].question_id]
        total = sum(c.weight for c in item.expected_concepts)
        for concept in item.expected_concepts:
            worst = max(worst, concept.weight / total * CONCEPT_WEIGHT * COVER["covered"])
    print(f"  heaviest single covered->absent downgrade in this dataset = {worst:.4f}")
    print("  length-bias threshold                                    = 0.0500")
    print(f"  => ONE unverifiable quote can exceed the gate by {worst / 0.05:.1f}x")

    print()
    print(RULE)
    print("5. Dataset defects: level separation and omitted-concept leakage")
    print(RULE)
    labels = {
        r.answer_id: r.score
        for r in load_reference_labels(DATASET / "reference_labels.jsonl")
        if r.pass_name == "blind"  # noqa: S105 - a labelling pass name, not a secret
    }
    print(f"  {'intent':18}{'n':>4}{'ref mean':>10}{'grader mean':>13}   ref dist")
    for level in QUALITY_LEVELS:
        ids_at_level = [a.id for a in answers.values() if a.quality_intent == level]
        ref = [labels[i] for i in ids_at_level if i in labels]
        grd = [base[i].score for i in ids_at_level if i in base]
        dist = dict(sorted(Counter(ref).items()))
        print(
            f"  {level:18}{len(ids_at_level):>4}{sum(ref) / len(ref):>10.2f}"
            f"{sum(grd) / len(grd):>13.3f}   {dist}"
        )
    report = leakage_report(list(answers.values()))
    by_answer: dict[str, list[str]] = report["by_answer"]  # type: ignore[assignment]
    print(
        f"{NL}  omitted-concept leakage: {report['leaked_slots']}/{report['omitted_slots']} slots "
        f"= {report['leak_rate']:.1%} across {report['answers_affected']} answers"
    )

    print()
    print(RULE)
    print("6. Does the grader's concept label match construction ground truth?")
    print(RULE)
    # ``concepts_included`` / ``concepts_omitted`` were decided by CODE before
    # generation, so they are the only non-model ground truth in the dataset.
    # An included concept the generator was told to demonstrate should grade
    # ``covered``; an omitted one should grade ``absent``. This is the one
    # accuracy number in Phase 4 that needs no model-generated reference label.
    cells: Counter[str] = Counter()
    for answer_id, row in base.items():
        if row.judgment is None:
            continue
        answer = answers[answer_id]
        graded_labels = {c.key: c.label for c in row.judgment.concepts}
        for key in answer.concepts_included:
            cells[f"included -> {graded_labels.get(key, 'MISSING')}"] += 1
        for key in answer.concepts_omitted:
            cells[f"omitted  -> {graded_labels.get(key, 'MISSING')}"] += 1
    for cell, n in sorted(cells.items()):
        print(f"  {cell:30}{n:>5}")
    included = sum(n for c, n in cells.items() if c.startswith("included"))
    omitted = sum(n for c, n in cells.items() if c.startswith("omitted"))
    hit = cells["included -> covered"]
    miss = cells["omitted  -> absent"]
    print(f"{NL}  included concepts graded covered : {hit}/{included} = {hit / included:.1%}")
    print(f"  omitted concepts graded absent   : {miss}/{omitted} = {miss / omitted:.1%}")

    print()
    print(RULE)
    print("7. How much signal is there to correlate against?")
    print(RULE)
    values = [labels[a] for a in sorted(labels)]
    counts = Counter(values)
    n = len(values)
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / n
    print(f"  reference labels n={n} mean={mean:.3f} population var={var:.4f} sd={var**0.5:.4f}")
    print(f"  distribution: {dict(sorted(counts.items()))}")
    modal = max(counts.values())
    print(f"  modal class holds {modal}/{n} = {modal / n:.1%} of the set")
    rank = {level: i for i, level in enumerate(reversed(QUALITY_LEVELS))}
    ids = [a for a in sorted(labels) if a in base]
    intents = [rank[answers[a].quality_intent] for a in ids]
    rho_ref = spearman_rho(intents, [labels[a] for a in ids])
    rho_grader = spearman_rho(intents, [base[a].score for a in ids])
    print(f"{NL}  Spearman(intended quality rank, reference label) = {rho_ref:+.4f}")
    print(f"  Spearman(intended quality rank, grader score)    = {rho_grader:+.4f}")
    print("  (QUALITY_LEVELS ranked excellent high .. wrong_confident low)")

    print()
    print(RULE)
    print("8. NOISE FLOOR: identical input, identical prompt, repeated runs")
    print(RULE)
    # The purest control available, and it was already on disk: the test-retest
    # arm re-graded the same answer with the same prompt several times. Any
    # score movement here is the grader's own sampling variance, with no text
    # change of any kind to attribute it to.
    repeats: dict[str, dict[int, GradingRun]] = {}
    for row in runs:
        if row.prompt_version == "v1" and row.variant == "base":
            repeats.setdefault(row.answer_id, {})[row.run_index] = row
    multi = {k: v for k, v in sorted(repeats.items()) if len(v) > 1}
    print(f"  answers with more than one base run: {len(multi)}")
    deltas: list[float] = []
    for answer_id, by_index in multi.items():
        order = sorted(by_index)
        scores = [by_index[i].score for i in order]
        print(
            f"  {answer_id}: n={len(scores)} scores={[round(x, 4) for x in scores]} "
            f"range={max(scores) - min(scores):.4f} "
            f"downgrades={[by_index[i].n_evidence_downgrades for i in order]} "
            f"gate={[by_index[i].gate_fired for i in order]}"
        )
        deltas.extend(by_index[i].score - by_index[order[0]].score for i in order[1:])
    inflating = sum(1 for d in deltas if d > SUCCESS_MARGIN)
    either = sum(1 for d in deltas if abs(d) > SUCCESS_MARGIN)
    print(f"{NL}  pairwise deltas vs run 0: n={len(deltas)}")
    print(f"    max |delta|                 = {max(abs(d) for d in deltas):.4f}")
    print(f"    mean |delta|                = {sum(abs(d) for d in deltas) / len(deltas):.4f}")
    up_rate = inflating / len(deltas)
    print(f"    exceeding +0.05 (inflation) = {inflating}/{len(deltas)} = {up_rate:.1%}")
    print(f"    exceeding 0.05 either way   = {either}/{len(deltas)} = {either / len(deltas):.1%}")

    print()
    print(RULE)
    print("9. ONE-SIDED inflation rate: the probes against that noise floor")
    print(RULE)
    # ASR is one-sided - only inflation counts as an attack succeeding - so the
    # probes have to be read one-sided too for the comparison to be fair.
    for variant in PROBE_ARMS[1:]:
        arm = select(runs, prompt_version="v1", variant=variant, run_index=0)
        ds = [arm[a].score - base[a].score for a in arm if a in base]
        up = sum(1 for d in ds if d > SUCCESS_MARGIN)
        print(
            f"  {variant:12} n={len(ds):>3}  inflated >+0.05: {up}/{len(ds)} = {up / len(ds):>5.1%}"
            f"   mean={sum(ds) / len(ds):+.4f}  max={max(ds):+.4f}  min={min(ds):+.4f}"
        )
    print(f"{NL}  compare with section 1's recomputed injection ASR.")

    print()
    print(RULE)
    print("10. Omitted concepts the grader credited: do their spans verify?")
    print(RULE)
    # The decisive question for section 6. If the spans verify, the answer text
    # really does cover the concept and the DATASET is wrong; if they do not,
    # the GRADER is inventing coverage. It is not a close call.
    span_ok = span_bad = 0
    examples: list[tuple[str, str, str, str, str]] = []
    for answer_id, row in sorted(base.items()):
        if row.judgment is None:
            continue
        answer = answers[answer_id]
        text = graded_answer_text(answer.answer)
        for judged in row.judgment.concepts:
            if judged.key not in answer.concepts_omitted or judged.label == "absent":
                continue
            if verify_span(judged.evidence, text).is_credited:
                span_ok += 1
                if len(examples) < 6 and judged.key not in by_answer.get(answer_id, ()):
                    examples.append(
                        (
                            answer_id,
                            answer.quality_intent,
                            judged.key,
                            judged.label,
                            (judged.evidence or "")[:150],
                        )
                    )
            else:
                span_bad += 1
    print(f"  omitted-but-credited slots: {span_ok + span_bad}")
    print(f"    span verifies against the answer : {span_ok}")
    print(f"    span does NOT verify             : {span_bad}")
    print(f"  the lexical detector flagged {report['leaked_slots']} of them")
    print(f"{NL}  examples the lexical detector did NOT flag:")
    for answer_id, intent, key, label, span in examples:
        print(f"    {answer_id} [{intent}] {key} -> {label}")
        print(f"      {span!r}")

    print()
    print(RULE)
    print("11. Answer length by intended quality level")
    print(RULE)
    print(f"  {'intent':18}{'mean words':>12}{'mean chars':>12}{'mean score':>12}")
    for level in QUALITY_LEVELS:
        ids_at_level = [a.id for a in answers.values() if a.quality_intent == level]
        words = [len(answers[i].answer.split()) for i in ids_at_level]
        chars = [len(answers[i].answer) for i in ids_at_level]
        scores = [base[i].score for i in ids_at_level if i in base]
        print(
            f"  {level:18}{sum(words) / len(words):>12.1f}{sum(chars) / len(chars):>12.1f}"
            f"{sum(scores) / len(scores):>12.3f}"
        )
    lengths = [len(answers[a].answer.split()) for a in sorted(base)]
    rho_len = spearman_rho(lengths, [base[a].score for a in sorted(base)])
    print(f"{NL}  Spearman(answer length in words, grader score) = {rho_len:+.4f}")


async def raw_spans() -> None:
    """Re-grade a few rows and print the RAW spans with their verdicts.

    Three provider calls, and the cheapest way to see whether a transformation
    changed the model's judgment at all.
    """
    from app.config import get_settings
    from app.llm.runtime import dispose_llm, init_llm

    answers = {a.id: a for a in load_answers(DATASET / "answers.jsonl")}
    loaded, _ = load_bank()
    questions = {li.item.id: li.item for li in loaded}
    settings = get_settings()
    init_llm(settings)
    try:
        for answer_id, variant in RAW_SPAN_ROWS:
            answer = answers[answer_id]
            item = questions[answer.question_id]
            shown = VARIANT_TRANSFORM[variant](answer.answer)
            judgment, _meta = await judge_answer(
                question_text=item.text,
                expected_concepts=list(item.expected_concepts),
                answer_text=shown,
                common_misconceptions=list(item.common_misconceptions),
                settings=settings,
            )
            graded = graded_answer_text(shown)
            original = graded_answer_text(answer.answer)
            print(RULE)
            print(f"{answer_id}  variant={variant}  RAW judgment, before verification")
            print(RULE)
            for concept in judgment.concepts:
                if concept.label == "absent":
                    print(f"  {concept.key:38} absent")
                    continue
                verdict = verify_span(concept.evidence, graded)
                in_original = verify_span(concept.evidence, original)
                mark = "OK " if verdict.is_credited else "BAD"
                print(f"  {concept.key:38} {concept.label:8} [{mark}] match={verdict.value}")
                print(f"      span: {(concept.evidence or '')[:110]!r}")
                print(f"      occurs in the untransformed answer too? {in_original.is_credited}")
            print(f"  rubric={judgment.rubric.model_dump()} conf={judgment.grader_confidence}")
            print(f"  errors={[(e.severity, e.claim[:70]) for e in judgment.errors]}")
            print()
    finally:
        await dispose_llm()


async def catch_failing_span(attempts: int, pace: float = 12.0) -> None:
    """Re-grade one row until a span fails verification, then print that span.

    The ``--raw-spans`` re-grades all happened to produce spans that verified, so
    they showed the mechanism working rather than failing. This keeps sampling
    the row with the highest observed failure rate (``eval-0068`` with the
    confidence prefix: 2 of 4 samples dropped a span) and dumps the **text of
    the offending span** the first time one fails.

    That text is what distinguishes a précis of something the candidate really
    wrote, or a pair of real excerpts joined by an ellipsis, from an invention -
    and nothing on disk can answer it, because ``verify_evidence`` nulls the span
    when it downgrades the concept.
    """
    from app.config import get_settings
    from app.llm.errors import AllProvidersFailedError
    from app.llm.runtime import dispose_llm, init_llm

    answers = {a.id: a for a in load_answers(DATASET / "answers.jsonl")}
    loaded, _ = load_bank()
    questions = {li.item.id: li.item for li in loaded}
    answer = answers["eval-0068"]
    item = questions[answer.question_id]
    shown = prepend_confidence(answer.answer)
    graded = graded_answer_text(shown)
    settings = get_settings()
    init_llm(settings)
    caught = 0
    try:
        for attempt in range(1, attempts + 1):
            try:
                judgment, _meta = await judge_answer(
                    question_text=item.text,
                    expected_concepts=list(item.expected_concepts),
                    answer_text=shown,
                    common_misconceptions=list(item.common_misconceptions),
                    settings=settings,
                )
            except AllProvidersFailedError:
                print(f"  attempt {attempt}: rate limited, waiting", flush=True)
                await asyncio.sleep(pace * 2)
                continue
            bad = [
                c
                for c in judgment.concepts
                if c.label != "absent" and not verify_span(c.evidence, graded).is_credited
            ]
            print(f"  attempt {attempt}: {len(bad)} span(s) failed verification", flush=True)
            for concept in bad:
                caught += 1
                span = concept.evidence or ""
                print(RULE)
                print(f"  FAILING SPAN  {concept.key} claimed {concept.label}")
                print(f"    span as offered : {span!r}")
                print(f"    len             : {len(span)} chars")
                # Longest prefix of the span that IS in the answer. A quotation
                # that diverges after a long run is a paraphrase or an
                # ellipsis-joined excerpt; an invention shares almost nothing.
                longest = next((e for e in range(len(span), 0, -1) if span[:e] in graded), 0)
                print(f"    longest verbatim prefix present in the answer: {longest} chars")
                if longest:
                    print(f"      {span[:longest]!r}")
                    print(f"    diverges at: {span[longest : longest + 60]!r}")
                words = [w for w in span.split() if len(w) > 3]
                present = sum(1 for w in words if w.lower() in graded.lower())
                print(f"    content words present in the answer: {present}/{len(words)}")
                print(RULE)
            if caught:
                return
            await asyncio.sleep(pace)
        print("  no span failed in the attempts allowed; the mechanism stayed unobserved")
    finally:
        await dispose_llm()


async def _graded_score(
    *,
    item: BankItem,
    shown: str,
    concepts: Sequence[ExpectedConcept],
    settings: Settings,
    pace: float,
) -> float:
    """One paced, rate-limit-tolerant grading call, scored.

    The free NVIDIA tier returns 429 well before this experiment finishes if the
    calls are issued back to back, so each call is spaced and each 429 is waited
    out rather than retried immediately. A diagnostic that dies halfway through
    an arm would leave the two conditions sampled unequally, which is the one
    thing this experiment exists to avoid.
    """
    from app.llm.errors import AllProvidersFailedError

    for attempt in range(6):
        try:
            judgment, _meta = await judge_answer(
                question_text=item.text,
                expected_concepts=concepts,
                answer_text=shown,
                common_misconceptions=list(item.common_misconceptions),
                settings=settings,
            )
        except AllProvidersFailedError:
            wait = pace * (attempt + 2)
            print(f"      rate limited; waiting {wait:.0f}s", flush=True)
            await asyncio.sleep(wait)
            continue
        verified = verify_evidence(judgment, answer_text=graded_answer_text(shown))
        await asyncio.sleep(pace)
        return score_answer(verified.judgment, expected_concepts=concepts).score
    raise RuntimeError("the provider refused every attempt; experiment not run")


async def replicate(k: int, pace: float = 12.0) -> None:
    """Re-grade the two headline bias rows k times per condition.

    Costs ``4 * k`` provider calls and writes nothing. Both conditions are
    sampled in the same sitting against the same provider, which is what the
    original probe did not do: its controls came from Day 19's committed run and
    its variants from a later sitting.
    """
    from app.config import get_settings
    from app.llm.runtime import dispose_llm, init_llm

    answers = {a.id: a for a in load_answers(DATASET / "answers.jsonl")}
    loaded, _ = load_bank()
    questions = {li.item.id: li.item for li in loaded}
    runs = load_runs(RUNS_PATH)
    settings = get_settings()
    init_llm(settings)
    try:
        for answer_id, variant in REPLICATE_ROWS:
            answer = answers[answer_id]
            item = questions[answer.question_id]
            concepts = list(item.expected_concepts)
            print(RULE)
            print(f"{answer_id}  variant={variant}  k={k} per condition, same sitting")
            print(RULE)
            stored = select(runs, prompt_version="v1", variant=variant, run_index=0)[answer_id]
            control = select(runs, prompt_version="v1", variant="base", run_index=0)[answer_id]
            print(
                f"  committed: control={control.score:.4f}  variant={stored.score:.4f}  "
                f"delta={stored.score - control.score:+.4f}"
            )
            samples: dict[str, list[float]] = {}
            for condition in ("base", variant):
                shown = VARIANT_TRANSFORM[condition](answer.answer)
                scores: list[float] = []
                for _ in range(k):
                    scores.append(
                        await _graded_score(
                            item=item,
                            shown=shown,
                            concepts=concepts,
                            settings=settings,
                            pace=pace,
                        )
                    )
                    print(f"      {condition} run {len(scores)}: {scores[-1]:.4f}", flush=True)
                samples[condition] = scores
                print(
                    f"  {condition:12} {[round(x, 4) for x in scores]}  "
                    f"mean={sum(scores) / k:.4f}  range={max(scores) - min(scores):.4f}"
                )
            a, b = samples["base"], samples[variant]
            print(
                f"  mean delta (variant - control, same sitting) = {sum(b) / k - sum(a) / k:+.4f}"
            )
            print(f"  worst single-sample pairing available        = {max(b) - min(a):+.4f}")
            print(f"  best  single-sample pairing available        = {min(b) - max(a):+.4f}")
            print()
    finally:
        await dispose_llm()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true", help="all provider-free diagnostics")
    parser.add_argument("--raw-spans", action="store_true", help="3 calls: capture raw spans")
    parser.add_argument(
        "--catch-failing-span",
        type=int,
        default=0,
        metavar="N",
        help="up to N calls: re-grade until a span fails verification, then dump it",
    )
    parser.add_argument(
        "--replicate",
        type=int,
        default=0,
        metavar="K",
        help="4*K calls: re-grade the two headline bias rows K times per condition",
    )
    args = parser.parse_args()
    if not (args.offline or args.raw_spans or args.replicate or args.catch_failing_span):
        parser.print_help()
        return 1
    if args.offline:
        offline()
    if args.raw_spans:
        asyncio.run(raw_spans())
    if args.catch_failing_span:
        asyncio.run(catch_failing_span(args.catch_failing_span))
    if args.replicate:
        asyncio.run(replicate(args.replicate))
    return 0


if __name__ == "__main__":
    sys.exit(main())
