"""Pass A and Pass B of plan section 7.2: assemble the request, make the call.

    PASS A - DETERMINISTIC   Fetch expected_concepts[] + misconceptions[]
                             for this question. No model involved.
    PASS B - ONE LLM CALL    Classification only. For each concept the model
    (temp 0.0)               returns a LABEL and a VERBATIM EVIDENCE SPAN.

Day 16 builds those two and stops there.  Pass C (the arithmetic) and Pass D
(the confidence gate) are Day 17 and are deliberately absent - see the module's
``__all__`` and the Day-16 report.  What exists today returns a
:class:`~app.grading.schema.GradeJudgment`, which carries **no number that could
be mistaken for a score**.

Everything in this module is either input assembly or a call to
``call_structured``.  There is no provider here, no HTTP, no retry, no cache and
no model name: those are the chokepoint's job, and the whole reason plan section
13.3 exists is so that a grader does not reimplement them.

**What is deliberately not an argument.**  No candidate name, email, session id
or resume text is a parameter of :func:`judge_answer`.  The grader needs the
question, the checklist and the answer; anything else would be candidate data
travelling to a provider for no grading benefit, and plan section 14.1 forbids
it.  ``trace`` carries identifiers for correlation, and it goes to the *span*,
never into a prompt.
"""

from __future__ import annotations

from collections.abc import Sequence

from app.bank.schema import MAX_CONCEPTS, MIN_CONCEPTS, ExpectedConcept
from app.config import Settings
from app.grading.prompt import isolate_candidate_text
from app.grading.schema import GradeJudgment
from app.llm.cache import ResponseCache
from app.llm.client import CallMeta, call_structured
from app.llm.router import ProviderRouter
from app.llm.tasks import TaskName
from app.llm.types import ANONYMOUS_TRACE, TraceCtx

#: A ceiling on the answer we will grade, in characters. Roughly 2k tokens of
#: English, which is several times the longest answer a 5-7 minute spoken or
#: typed response produces. It exists so that a paste of a whole textbook - or a
#: bug in an upstream transcriber - fails here with a clear message instead of
#: silently blowing the task's input budget and returning a truncated grade.
MAX_ANSWER_CHARS = 8000


class GradingInputError(ValueError):
    """The request we were asked to grade is not gradeable.

    Raised *before* any model is called, because every one of these conditions
    is knowable from the arguments and paying a provider to discover them would
    be both slower and more expensive than checking.
    """


class GradingContractError(ValueError):
    """The model returned a valid ``GradeJudgment`` about the wrong question.

    Schema validation proves the *shape* is right. It cannot prove the model
    classified the concepts it was actually asked about, because the legal set
    of keys is a property of the question, not of the type. That check is here.

    Worth being precise about what this catches: a model that invents a concept
    key, drops one, or reports a misconception that was never in the list has
    produced an object that would score perfectly well and mean nothing. Failing
    loudly is the only safe response - the alternative is a silent mismatch that
    shows up as an unexplainable ability estimate weeks later.
    """


def render_expected_concepts(concepts: Sequence[ExpectedConcept]) -> str:
    """The checklist as prompt text, in a canonical order.

    **Sorted by key, and that is the point.**  Phase 4's exit gate requires that
    shuffling ``expected_concepts`` changes the score by *exactly* zero.  The
    scoring arithmetic (Day 17) is order-free on its own, but that alone would
    not be enough: if the prompt listed the concepts in caller order, a shuffle
    would produce different prompt text, a different cache key, a different
    fixture key, and - since models are order-sensitive - possibly different
    labels.  Sorting here makes the *prompt itself* identical under a shuffle,
    so the gate holds for the whole pipeline rather than only for the last step.

    **Weights are not shown.**  The model classifies; the weights decide what a
    classification is worth, and that decision is code's. Telling the grader
    which concepts are worth three points invites it to think about the total,
    which is precisely the reasoning this design removes.
    """
    lines = [
        f"- {concept.key}: {concept.hint}" for concept in sorted(concepts, key=lambda c: c.key)
    ]
    return "\n".join(lines)


def render_misconceptions(misconceptions: Sequence[str]) -> str:
    """The misconception list as prompt text, sorted for the same reason."""
    cleaned = sorted({m.strip() for m in misconceptions if m.strip()})
    if not cleaned:
        return "(none recorded for this question)"
    return "\n".join(f"- {m}" for m in cleaned)


def graded_answer_text(answer_text: str) -> str:
    """The exact text the model is shown inside the answer fence.

    **This is the string evidence spans must be verified against**, and that is
    why it is a named function rather than two lines inside
    :func:`build_grading_inputs`.

    The model quotes from what it was sent, not from what the caller passed in.
    Those differ: the answer is stripped, and ``isolate_candidate_text`` replaces
    any fence sentinel with a visible marker. A candidate whose answer genuinely
    contained ``</candidate_answer>`` would therefore have had their text
    rewritten before the model saw it, and a perfectly honest span quoted across
    that point would not appear in the *original* string. Verifying against the
    original would mark it fabricated and silently delete the credit.

    One function, used by both the prompt builder and the verifier, is what
    stops those two ever drifting apart. Day 17 added it; Day 16 had the same
    two lines inline.
    """
    return isolate_candidate_text(answer_text.strip())


def build_grading_inputs(
    *,
    question_text: str,
    expected_concepts: Sequence[ExpectedConcept],
    answer_text: str,
    common_misconceptions: Sequence[str] = (),
) -> dict[str, str]:
    """Pass A: everything the prompt needs, as plain strings.

    Separate from :func:`judge_answer` because it is pure, and because the cache
    key is computed from exactly this dict - so "did two requests hash the same?"
    is a question a test can answer without a router, a provider or an event
    loop. That is what the order-independence test uses.
    """
    question = question_text.strip()
    if not question:
        raise GradingInputError("question_text is empty; there is nothing to grade against")

    answer = answer_text.strip()
    if not answer:
        raise GradingInputError(
            "answer_text is empty. An unanswered question is a policy decision for the "
            "interview flow, not a grading one - do not spend a model call to be told "
            "that nothing is covered"
        )
    if len(answer) > MAX_ANSWER_CHARS:
        raise GradingInputError(
            f"answer_text is {len(answer)} characters, over the {MAX_ANSWER_CHARS} "
            "limit; truncate or split it upstream rather than grading a prefix silently"
        )

    if not MIN_CONCEPTS <= len(expected_concepts) <= MAX_CONCEPTS:
        raise GradingInputError(
            f"expected {MIN_CONCEPTS}-{MAX_CONCEPTS} expected_concepts, got "
            f"{len(expected_concepts)}; the contract bounds the concepts list to the "
            "same range, so a question outside it could not be answered validly"
        )
    keys = [concept.key for concept in expected_concepts]
    duplicates = sorted({key for key in keys if keys.count(key) > 1})
    if duplicates:
        raise GradingInputError(f"expected_concepts repeats {duplicates}")

    return {
        "question": question,
        "expected_concepts": render_expected_concepts(expected_concepts),
        "misconceptions": render_misconceptions(common_misconceptions),
        # The one input that is untrusted text. Isolated through the same
        # function the evidence verifier uses, so that no caller can forget and
        # no second code path exists.
        "answer": graded_answer_text(answer),
    }


def check_judgment(
    judgment: GradeJudgment,
    *,
    expected_concepts: Sequence[ExpectedConcept],
    common_misconceptions: Sequence[str] = (),
) -> None:
    """Assert the judgment is about the question we asked. Raises, or returns None.

    Kept separate from the pydantic contract because it is a *relational* rule:
    it depends on the item, and a type cannot know the item. Kept out of the
    chokepoint's retry loop for the same reason - ``call_structured`` validates
    a schema, and teaching it about question banks would make it domain-aware.

    The prompt already names the exact keys and says not to deviate, so in
    practice this fires on a genuinely confused model rather than on ordinary
    variation. It is the guarantee, not the mechanism.
    """
    asked = {concept.key for concept in expected_concepts}
    answered = set(judgment.labels_by_key)
    if answered != asked:
        invented = sorted(answered - asked)
        missing = sorted(asked - answered)
        detail = []
        if invented:
            detail.append(f"invented {invented}")
        if missing:
            detail.append(f"did not classify {missing}")
        raise GradingContractError(
            "the judgment does not match the concepts that were asked about: " + "; ".join(detail)
        )

    allowed = {m.strip() for m in common_misconceptions}
    reported = {m.strip() for m in judgment.misconceptions_hit}
    unknown = sorted(reported - allowed)
    if unknown:
        raise GradingContractError(
            f"misconceptions_hit contains entries that were not in the question's list: "
            f"{unknown}; they must be copied from the list supplied in the prompt"
        )


async def judge_answer(
    *,
    question_text: str,
    expected_concepts: Sequence[ExpectedConcept],
    answer_text: str,
    common_misconceptions: Sequence[str] = (),
    trace: TraceCtx = ANONYMOUS_TRACE,
    router: ProviderRouter | None = None,
    cache: ResponseCache | None = None,
    settings: Settings | None = None,
) -> tuple[GradeJudgment, CallMeta]:
    """Pass B: one classification call through the chokepoint.

    Returns the validated judgment and the call's metadata. It does **not**
    return a score, and there is no argument by which it could be asked for one.

    ``router``, ``cache`` and ``settings`` pass straight through to
    ``call_structured`` and exist for the same reason they exist on
    ``probe_llm``: a test - or the offline replay suite - can point this at the
    stub provider without monkeypatching the process-wide runtime.

    Raises :class:`GradingInputError` before calling anything if the request is
    not gradeable, ``SchemaValidationFailedError`` if the model could not
    produce a valid object in the allowed attempts, and
    :class:`GradingContractError` if it produced a valid object about the wrong
    concepts.
    """
    inputs = build_grading_inputs(
        question_text=question_text,
        expected_concepts=expected_concepts,
        answer_text=answer_text,
        common_misconceptions=common_misconceptions,
    )
    judgment, meta = await call_structured(
        TaskName.GRADE_ANSWER,
        inputs,
        GradeJudgment,
        trace=trace,
        router=router,
        cache=cache,
        settings=settings,
    )
    check_judgment(
        judgment,
        expected_concepts=expected_concepts,
        common_misconceptions=common_misconceptions,
    )
    return judgment, meta


__all__ = [
    "MAX_ANSWER_CHARS",
    "GradingContractError",
    "GradingInputError",
    "build_grading_inputs",
    "check_judgment",
    "graded_answer_text",
    "judge_answer",
    "render_expected_concepts",
    "render_misconceptions",
]
