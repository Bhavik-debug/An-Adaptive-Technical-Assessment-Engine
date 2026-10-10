"""Pass D and the whole chain - plan sections 7.2 and 7.3.

Day 16 built Pass A (assemble) and Pass B (one classification call). Day 17 adds
the rest, and this module is where all four meet::

    PASS A  code   fetch expected_concepts[] + misconceptions[]      grader.py
    PASS B  LLM    one call, temp 0.0: labels + evidence spans       grader.py
             code  verify every span is really in the answer         evidence.py
    PASS C  code   concept blend, rubric blend, penalty, clamp        scoring.py
    PASS D  code   if the signals disagree or the model was unsure,   here
                   re-grade n=3 at temp 0.7 and take the MEDIAN

**The gate is control flow, and control flow is code's.** Nothing in this file
asks a model whether to re-grade, how many samples to take, or which score to
keep. The trigger is two comparisons over numbers this project computed; the
aggregation is ``statistics.median`` over scores this project computed. The
model's only contribution, three extra times, is another set of classifications.

**Why a median and why exactly three.** Plan section 7.3 is self-consistency: ask
the identical question several times and keep the middle answer. Median rather
than mean because the median is robust to one wild outlier, which is precisely
the failure mode being defended against - a mean of three would let a single
0.05 drag a pair of 0.8s down to 0.55. Three rather than five because the plan
says three, and because odd counts have an unambiguous median that is *one of
the observed values* rather than a number nobody produced.

**The original score does not vote.** Plan section 7.3 says "re-grade n = 3 ...
recompute all three scores in code, take the MEDIAN score" - three re-grades,
three scores, one median. Letting the original in would make four values and
force the median to average the two middle ones, producing a number no single
grading pass produced and giving away the robustness the median was chosen for.
The initial score is kept on the outcome for the trace, not for the arithmetic.
"""

from __future__ import annotations

import asyncio
import logging
import statistics
from collections.abc import Sequence
from dataclasses import dataclass

from app.bank.schema import ExpectedConcept
from app.config import Settings
from app.grading.evidence import VerifiedJudgment, verify_evidence
from app.grading.grader import build_grading_inputs, check_judgment
from app.grading.schema import GradeJudgment
from app.grading.scoring import ScoreBreakdown, score_answer
from app.llm.cache import ResponseCache
from app.llm.client import CallMeta, call_structured
from app.llm.errors import LLMError
from app.llm.prompts import PromptTemplate
from app.llm.router import ProviderRouter
from app.llm.tasks import TaskName
from app.llm.types import ANONYMOUS_TRACE, TraceCtx

log = logging.getLogger(__name__)

#: Plan section 7.3, trigger 1: the two independent signals disagree by more
#: than this, so at least one of them is probably wrong.
GATE_DISAGREEMENT_THRESHOLD = 0.25

#: Plan section 7.3, trigger 2: the model said it was unsure. Strictly below -
#: a confidence of exactly 0.6 does not fire, matching the plan's `< 0.6`.
GATE_CONFIDENCE_FLOOR = 0.6

#: Plan section 7.3: n = 3. The sampling settings are NOT here - temperature 0.7
#: and top_p 0.95 live in the routing table on ``TaskName.GRADE_RECHECK``, which
#: is the one place this project keeps sampling decisions (plan section 13.6).
REGRADE_SAMPLES = 3


@dataclass(frozen=True, slots=True)
class ConfidenceGateOutcome:
    """Why the gate did or did not fire, and what it changed.

    Every field is here so that "why is this answer's score 0.41?" is answerable
    from one object. A gate that fires on ~8% of answers (plan section 7.3) and
    leaves no record of which 8% is a gate nobody can audit or cost.
    """

    triggered: bool
    #: The two triggers, kept apart. They mean different things - "the checklist
    #: and the rubric told different stories" is a prompt problem, "the model
    #: said it was unsure" is an item problem - and a single boolean would lose
    #: the distinction exactly when Day 19 wants to count them.
    signal_disagreement: bool
    low_confidence: bool
    disagreement: float
    grader_confidence: float
    #: The score before the gate. Never part of the median; see the module
    #: docstring.
    initial_score: float
    #: One per successful re-grade, in the order they were sampled.
    regrade_scores: tuple[float, ...]
    samples_requested: int
    #: Fewer than requested means a re-grade call failed. The gate degrades
    #: rather than failing the whole grade - a graded answer with two samples is
    #: worth more than an exception.
    samples_failed: int

    @property
    def regraded(self) -> bool:
        return bool(self.regrade_scores)


@dataclass(frozen=True, slots=True)
class GradeResult:
    """One graded answer: the number, and everything that justifies it."""

    #: The final score in [0, 1]. This is what feeds ``update_ability``.
    score: float
    #: The judgment the score was computed from - already evidence-corrected,
    #: and, when the gate fired, the re-grade that produced the median.
    judgment: GradeJudgment
    breakdown: ScoreBreakdown
    verification: VerifiedJudgment
    gate: ConfidenceGateOutcome
    #: Every chokepoint call this grade cost: one, or four when the gate fired.
    #: Summing ``cost_usd`` across these is the per-grade cost.
    calls: tuple[CallMeta, ...]

    @property
    def grader_confidence(self) -> float:
        """What ``ability.k_factor`` wants for ``f_conf``."""
        return self.judgment.grader_confidence


def gate_triggers(breakdown: ScoreBreakdown, grader_confidence: float) -> tuple[bool, bool]:
    """Plan section 7.3's trigger, as two booleans. Pure.

    ::

        Trigger if:  |concept_score - rubric_score| > 0.25
                or:  grader_confidence < 0.6

    Both comparisons are strict, as written. Returned separately rather than
    or-ed together so the caller records *which* condition fired.
    """
    disagreement = breakdown.signal_disagreement > GATE_DISAGREEMENT_THRESHOLD
    unsure = grader_confidence < GATE_CONFIDENCE_FLOOR
    return disagreement, unsure


def _pick_median(scores: Sequence[float]) -> int:
    """Index of the sample whose score is the median.

    The returned judgment must be the one that produced the returned score, or a
    report would show evidence that does not explain its own number. With an odd
    count ``statistics.median`` returns one of the observed values, so this is an
    exact lookup; with an even count (only reachable when a re-grade failed) it
    returns their average, and the closest sample is the honest choice.
    """
    target = statistics.median(scores)
    return min(range(len(scores)), key=lambda i: abs(scores[i] - target))


async def grade_answer(
    *,
    question_text: str,
    expected_concepts: Sequence[ExpectedConcept],
    answer_text: str,
    common_misconceptions: Sequence[str] = (),
    trace: TraceCtx = ANONYMOUS_TRACE,
    router: ProviderRouter | None = None,
    cache: ResponseCache | None = None,
    settings: Settings | None = None,
    prompt: PromptTemplate | None = None,
    recheck_prompt: PromptTemplate | None = None,
) -> GradeResult:
    """Grade one answer end to end. Plan section 7.2, all four passes.

    Returns a :class:`GradeResult` whose ``score`` is computed entirely by the
    code in ``scoring.py``. The model is called once, or four times when the gate
    fires, and every call goes through ``call_structured``.

    Raises ``GradingInputError`` before calling anything if the request is not
    gradeable, and whatever the chokepoint raises if the *first* grading call
    cannot produce a valid judgment. A failing **re-grade** is different and does
    not raise: the gate degrades to the samples it got (see
    :class:`ConfidenceGateOutcome`).
    """
    inputs = build_grading_inputs(
        question_text=question_text,
        expected_concepts=expected_concepts,
        answer_text=answer_text,
        common_misconceptions=common_misconceptions,
    )
    # The exact string the model is shown, read back out of the assembled inputs
    # rather than recomputed with `graded_answer_text`. Same value either way -
    # but taking it from the dict that becomes the prompt means verification is
    # checking against the literal text that was sent, with no second call that
    # could ever disagree.
    answer_shown = inputs["answer"]

    judgment, meta = await call_structured(
        TaskName.GRADE_ANSWER,
        inputs,
        GradeJudgment,
        trace=trace,
        router=router,
        cache=cache,
        settings=settings,
        prompt=prompt,
    )
    check_judgment(
        judgment,
        expected_concepts=expected_concepts,
        common_misconceptions=common_misconceptions,
    )

    verification = verify_evidence(judgment, answer_text=answer_shown)
    breakdown = score_answer(verification.judgment, expected_concepts=expected_concepts)
    calls: list[CallMeta] = [meta]

    disagreement, unsure = gate_triggers(breakdown, judgment.grader_confidence)
    if not (disagreement or unsure):
        return GradeResult(
            score=breakdown.score,
            judgment=verification.judgment,
            breakdown=breakdown,
            verification=verification,
            gate=ConfidenceGateOutcome(
                triggered=False,
                signal_disagreement=False,
                low_confidence=False,
                disagreement=breakdown.signal_disagreement,
                grader_confidence=judgment.grader_confidence,
                initial_score=breakdown.score,
                regrade_scores=(),
                samples_requested=0,
                samples_failed=0,
            ),
            calls=tuple(calls),
        )

    log.info(
        "grading confidence gate fired; re-grading %d times",
        REGRADE_SAMPLES,
        extra={
            "grading.gate_disagreement": disagreement,
            "grading.gate_low_confidence": unsure,
            "grading.disagreement": breakdown.signal_disagreement,
            "grading.grader_confidence": judgment.grader_confidence,
            "grading.initial_score": breakdown.score,
        },
    )

    samples = await _regrade(
        inputs,
        expected_concepts=expected_concepts,
        common_misconceptions=common_misconceptions,
        answer_shown=answer_shown,
        trace=trace,
        router=router,
        cache=cache,
        settings=settings,
        # The gate is self-consistency: the re-grade must ask the *identical*
        # question (see GRADING_RECHECK_PROMPT_V1's docstring). When the caller
        # is evaluating a prompt variant, "identical" means identical to the
        # variant under test, so the override is carried through rather than
        # silently reverting the re-grades to the active v1 text.
        recheck_prompt=recheck_prompt,
    )
    calls.extend(meta for _, _, _, meta in samples)
    failed = REGRADE_SAMPLES - len(samples)

    if not samples:
        # Every re-grade failed. Keeping the initial grade is the only honest
        # option left: it is a real, validated, evidence-checked grade, and the
        # outcome records that the gate fired and got nothing, so a session full
        # of these is visible rather than indistinguishable from a calm one.
        log.warning("grading confidence gate fired but every re-grade failed; keeping the first")
        final_verification, final_breakdown = verification, breakdown
        regrade_scores: tuple[float, ...] = ()
    else:
        regrade_scores = tuple(sample[2].score for sample in samples)
        chosen = _pick_median(regrade_scores)
        final_verification, final_breakdown = samples[chosen][1], samples[chosen][2]

    return GradeResult(
        score=final_breakdown.score,
        judgment=final_verification.judgment,
        breakdown=final_breakdown,
        verification=final_verification,
        gate=ConfidenceGateOutcome(
            triggered=True,
            signal_disagreement=disagreement,
            low_confidence=unsure,
            disagreement=breakdown.signal_disagreement,
            grader_confidence=judgment.grader_confidence,
            initial_score=breakdown.score,
            regrade_scores=regrade_scores,
            samples_requested=REGRADE_SAMPLES,
            samples_failed=failed,
        ),
        calls=tuple(calls),
    )


async def _regrade(
    inputs: dict[str, str],
    *,
    expected_concepts: Sequence[ExpectedConcept],
    common_misconceptions: Sequence[str],
    answer_shown: str,
    trace: TraceCtx,
    router: ProviderRouter | None,
    cache: ResponseCache | None,
    settings: Settings | None,
    recheck_prompt: PromptTemplate | None = None,
) -> list[tuple[GradeJudgment, VerifiedJudgment, ScoreBreakdown, CallMeta]]:
    """Sample ``GRADE_RECHECK`` n times and score each one, in code.

    The three calls run concurrently: they are independent by construction -
    that is what makes them samples rather than a conversation - and three
    sequential provider round-trips would triple the latency of the 8% of turns
    that need this, for nothing.

    Each sample is verified and scored **before** anything is compared, which is
    the point plan section 7.3 makes with "recompute all three scores in code".
    Nothing compares raw judgments, and no model is shown another model's answer.

    A sample that fails is dropped with a warning rather than propagated: one bad
    re-grade should not turn a gradeable answer into an exception, and the count
    of failures is reported to the caller.
    """
    identical = [
        call_structured(
            TaskName.GRADE_RECHECK,
            inputs,
            GradeJudgment,
            trace=trace,
            router=router,
            cache=cache,
            settings=settings,
            prompt=recheck_prompt,
        )
        for _ in range(REGRADE_SAMPLES)
    ]
    results = await asyncio.gather(*identical, return_exceptions=True)

    samples: list[tuple[GradeJudgment, VerifiedJudgment, ScoreBreakdown, CallMeta]] = []
    for index, result in enumerate(results):
        if isinstance(result, BaseException):
            # Narrow deliberately: an LLMError is a re-grade that did not come
            # back and is survivable, while a ValueError from `check_judgment`
            # or `score_answer` means this sample described a different question
            # and is also survivable - but a CancelledError or a KeyboardInterrupt
            # is not ours to swallow.
            if isinstance(result, LLMError | ValueError):
                log.warning("grading re-grade sample %d failed: %s", index, result)
                continue
            raise result
        judgment, meta = result
        try:
            check_judgment(
                judgment,
                expected_concepts=expected_concepts,
                common_misconceptions=common_misconceptions,
            )
        except ValueError as exc:
            log.warning("grading re-grade sample %d did not match the question: %s", index, exc)
            continue
        verification = verify_evidence(judgment, answer_text=answer_shown)
        breakdown = score_answer(verification.judgment, expected_concepts=expected_concepts)
        samples.append((judgment, verification, breakdown, meta))
    return samples


__all__ = [
    "GATE_CONFIDENCE_FLOOR",
    "GATE_DISAGREEMENT_THRESHOLD",
    "REGRADE_SAMPLES",
    "ConfidenceGateOutcome",
    "GradeResult",
    "gate_triggers",
    "grade_answer",
]
