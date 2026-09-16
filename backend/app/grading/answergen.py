"""Controlled answer synthesis - plan section 12.2, step 2.

    Generate 5 answers per question at CONTROLLED quality levels -
    excellent / good / partial / superficial / wrong-but-confident.
    Produce these with an LLM GIVEN THE CONCEPT KEY, instructing it to
    include or omit named concepts.
    -> You now have GROUND TRUTH BY CONSTRUCTION for concept coverage.

**This is eval-set construction, not interview machinery.** Nothing in
``app/grading``'s grading path imports it, no interview turn will ever call it,
and it exists to be run by hand from ``scripts/build_grading_evalset.py`` on the
days a dataset is built. It is here rather than under ``scripts/`` because it
defines a task, a prompt and a schema, and those belong with the package that
owns them - a script should be an entry point, not a place where contracts live.

**Why the LLM writes the prose and code decides everything else.** The plan is
specific about the division: the model is *given the concept keys* and told which
to demonstrate and which to stay silent about. So the model chooses wording; code
chooses the experiment. The split matters because the resulting file is a
measuring instrument - if the model also chose which concepts to include, the
"ground truth by construction" would be the model's report of its own behaviour,
which is exactly the circularity the whole dataset exists to avoid.

**The generator must not become a grader, and vice versa.** They are different
tasks with different prompts and different temperatures, and the generator is
never shown the grading rubric. A generator that knew how it would be marked
would write to the rubric, and Day 19 would measure that agreement instead of the
grader's validity.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from app.bank.schema import BankItem, ExpectedConcept
from app.config import Settings
from app.grading.evalset import QualityIntent
from app.llm.cache import ResponseCache
from app.llm.client import CallMeta, call_structured
from app.llm.prompts import PromptTemplate, register_prompt
from app.llm.router import ProviderRouter
from app.llm.tasks import TaskName
from app.llm.types import ANONYMOUS_TRACE, TraceCtx

#: How each quality level is described *to the generator*. These are
#: instructions for writing, deliberately phrased in terms of what the answer
#: should DO - they are not the labelling anchors a human reads (those are
#: ``evalset.LABEL_ANCHORS``) and they are not a rubric.
#:
#: Keeping the two sets of words apart is the point. If the generator were given
#: the labeller's anchors, the dataset would be a test of whether one model can
#: hit a target another model reads back - a much easier task than the one being
#: measured, and a meaningless one.
LEVEL_BRIEFS: dict[QualityIntent, str] = {
    "excellent": (
        "A strong senior engineer's answer. Demonstrate every concept you are "
        "told to include, with concrete mechanisms, at least one quantity or "
        "bound, and an explicit trade-off with the condition under which you "
        "would choose differently. Organised, and calibrated about what you are "
        "unsure of."
    ),
    "good": (
        "A solid answer from a competent engineer. Demonstrate the concepts you "
        "are told to include, correctly and with real mechanisms, but stay at "
        "one level of detail: no quantities, and at most a passing mention of "
        "any trade-off. Nothing wrong in it, just less depth."
    ),
    "partial": (
        "A half-formed answer. Demonstrate the included concepts only "
        "superficially - name the mechanism and assert that it works, without "
        "explaining how or what it costs. Vague in places. Nothing factually "
        "wrong."
    ),
    "superficial": (
        "A weak answer that sounds like it is participating. Use correct "
        "vocabulary and general statements that would be true of any system. "
        "Touch the included concepts by NAME ONLY, with no mechanism and nothing "
        "a colleague could act on. Somewhat wordy relative to its content. "
        "Nothing outright false."
    ),
    "wrong_confident": (
        "A confidently incorrect answer. Assertive, fluent, and containing at "
        "least two clear factual errors about the subject matter - claims that "
        "would lead to a broken system if acted on. Where you touch an included "
        "concept, get it wrong while sounding certain. Never hedge."
    ),
}

#: How many of a question's concepts each level is told to demonstrate, as a
#: fraction of the question's concept count. Code turns these into an explicit
#: include/omit split *before* the model is asked anything, which is what makes
#: the split ground truth rather than a report.
#:
#: The descent is deliberate: coverage is the signal the grader's concept
#: classifier is being tested on, so the dataset has to actually vary it. On a
#: four-concept question these give 4 / 3 / 2 / 1 / 2 included concepts, so four
#: distinct coverage counts appear across the five levels rather than the three
#: an evenly-spaced table would have produced.
#:
#: ``wrong_confident`` is given the same share as ``partial`` rather than zero,
#: on purpose and not by oversight: a wrong answer that also omitted everything
#: would confound "wrong" with "empty", and the whole point of that level is to
#: isolate a grader that rewards confident prose. Pairing it with ``partial``
#: means the two differ *only* in correctness, which is the comparison worth
#: having.
#:
#: **What the split does and does not claim.** ``concepts_omitted`` is a strong
#: claim: the generator was told to say nothing about those, so a grader marking
#: one ``covered`` is a false positive, full stop. ``concepts_included`` is
#: weaker and varies by level - ``excellent`` should read as ``covered``, but
#: ``superficial`` names the concept without demonstrating it (so ``partial`` is
#: the honest label) and ``wrong_confident`` addresses it incorrectly (so
#: ``absent`` may well be right). Day 19 must treat the omitted set as the
#: precision ground truth and read the included set per level, not as a blanket
#: "should be covered". This is written down here because it is exactly the kind
#: of thing a later analysis gets wrong by assuming.
LEVEL_COVERAGE: dict[QualityIntent, float] = {
    "excellent": 1.0,
    "good": 0.75,
    "partial": 0.5,
    "superficial": 0.25,
    "wrong_confident": 0.5,
}

#: Bounds on the generated prose, in words. The job of the ceiling is to catch a
#: *runaway* generation - a model that ignored the brief and returned an essay -
#: not to enforce a house style.
#:
#: **260 was the first guess and it was wrong.** The Day-18 build produced 8
#: answers of 265-349 words out of 120, all of them `excellent` or `superficial`
#: (the two longest briefs) and all of them perfectly good answers that simply
#: overshot a 200-word target. Raising the ceiling to 400 was the honest fix:
#: the alternative was either discarding 8 valid rows to make a threshold I had
#: invented come out clean, or regenerating them - which at temperature 0 means
#: changing the prompt, which changes every fingerprint and rebuilds all 120.
#: The observed distribution is reported in the dataset README rather than
#: hidden behind a bound that was tuned to it.
MIN_WORDS = 25
MAX_WORDS = 400


class SyntheticAnswer(BaseModel):
    """What the generator returns. One field, because one field is all we want.

    No ``quality``, no ``concepts_covered``, no self-assessment of any kind. The
    model is not asked to report what it did: code already knows, because code
    decided. A field asking the model which concepts it included would be the
    model grading its own homework, and it would be the field everything
    downstream quietly trusted.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    answer: str = Field(
        min_length=100,
        max_length=4000,
        description=(
            "The candidate's answer, in first person, as speech transcribed into "
            "prose. No headings, no bullet lists, no markdown, no preamble such "
            "as 'Here is an answer'. Just what the candidate said."
        ),
    )


_SYSTEM = """
You are writing SYNTHETIC INTERVIEW ANSWERS for a grader's test set. You are not
grading anything, and you are not being graded.

Your job is to write what a candidate of a specified quality level would say to a
technical interview question, demonstrating exactly the concepts you are told to
demonstrate and staying silent about the ones you are told to omit.

## The rules that make this dataset usable

1. **Demonstrate every concept in "Concepts to demonstrate".** Show it in
   substance, at the depth the quality brief allows - do not merely name it
   unless the brief says to.

2. **Say nothing about any concept in "Concepts to leave out".** Not a mention,
   not a gesture, not a synonym, not an implication a reader would score as
   partial credit. If leaving one out makes the answer feel incomplete, that is
   correct and intended - it is what makes the answer a test case.

3. **Write in the candidate's voice.** First person, conversational, the way
   someone talks in an interview. Contractions are good. Perfect prose is not -
   real answers meander slightly.

4. **No formatting.** No markdown, no headings, no numbered lists, no bullet
   points. Continuous prose only. A list is a giveaway that a model wrote it.

5. **Never mention the question, the concepts, the quality level, this task, or
   that you are an AI.** The output must read as an answer and nothing else.

6. **Length follows the brief, not the level.** Do not pad a weak answer to look
   thorough or clip a strong one to look concise, beyond what the brief says.

You return a single JSON object with one field: `answer`.
""".strip()


_USER = """
## The interview question

$question

## Quality level to write at

$level_brief

## Concepts to demonstrate

$concepts_included

## Concepts to leave out entirely

$concepts_omitted

## Length

Roughly $word_target words.

Write the answer now.
""".strip()


#: Registered from here, like the grading prompt, so that ``llm/`` stays free of
#: domain text - see :func:`app.llm.prompts.register_prompt`.
ANSWER_SYNTHESIS_PROMPT_V1 = PromptTemplate(
    task=TaskName.EVAL_ANSWER_SYNTHESIS,
    version="v1",
    system=_SYSTEM,
    user=_USER,
    required_inputs=(
        "question",
        "level_brief",
        "concepts_included",
        "concepts_omitted",
        "word_target",
    ),
)

register_prompt(ANSWER_SYNTHESIS_PROMPT_V1)


def split_concepts(
    item: BankItem, level: QualityIntent
) -> tuple[list[ExpectedConcept], list[ExpectedConcept]]:
    """Decide, in code, which concepts this answer will demonstrate.

    Returns ``(included, omitted)``. **This is the ground truth**, and it is
    computed before any model is called.

    Deterministic, and deterministic in a specific way worth spelling out: the
    concepts are taken **heaviest-first**, by ``(-weight, key)``. Not randomly.
    A random split would make "which concepts were planted?" vary between two
    builds of the same dataset, and it would also let an unlucky draw hand the
    ``excellent`` and ``partial`` answers to the same subset. Heaviest-first
    means a partial answer covers what a partial answer plausibly *would* cover -
    the central ideas - and it makes the concept score at each level predictable
    enough that a Day-19 result which departs from it is interesting rather than
    noise.
    """
    ordered = sorted(item.expected_concepts, key=lambda c: (-c.weight, c.key))
    share = LEVEL_COVERAGE[level]
    # At least one, never all unless the level says 1.0: a level that silently
    # collapsed to "everything" would stop being a distinct condition.
    take = max(1, round(len(ordered) * share))
    if share < 1.0:
        take = min(take, len(ordered) - 1)
    return ordered[:take], ordered[take:]


def word_target(level: QualityIntent) -> int:
    """Roughly how long an answer at this level should be.

    **Deliberately not monotone in quality.** ``superficial`` is the *longest*
    brief after ``excellent``, because the failure mode this dataset most needs
    to contain is the fluent, wordy, empty answer - plan section 7.4 technique 5
    requires the grader to be immune to verbosity, and a set whose bad answers
    were all short would let a grader pass the length-bias probe by accident
    while simply rewarding length.
    """
    return {
        "excellent": 200,
        "good": 150,
        "partial": 110,
        "superficial": 170,
        "wrong_confident": 130,
    }[level]


def build_inputs(
    item: BankItem, level: QualityIntent
) -> tuple[dict[str, str], list[str], list[str]]:
    """Assemble the prompt inputs, and return the ground truth alongside them.

    Returned together on purpose: the inputs and the include/omit lists must
    describe the same split, and handing a caller two things it has to keep in
    step is how they stop being in step.
    """
    included, omitted = split_concepts(item, level)
    inputs = {
        "question": item.text.strip(),
        "level_brief": LEVEL_BRIEFS[level],
        # Hints are shown so the generator knows what "demonstrating" this
        # concept looks like - the same hint text the grader is shown, because
        # both need to mean the same thing by a concept key.
        "concepts_included": "\n".join(f"- {c.key}: {c.hint}" for c in included)
        or "(none - do not demonstrate any of the expected concepts)",
        "concepts_omitted": "\n".join(f"- {c.key}: {c.hint}" for c in omitted)
        or "(none - every concept may appear)",
        "word_target": str(word_target(level)),
    }
    return inputs, [c.key for c in included], [c.key for c in omitted]


async def generate_answer(
    item: BankItem,
    level: QualityIntent,
    *,
    trace: TraceCtx = ANONYMOUS_TRACE,
    router: ProviderRouter | None = None,
    cache: ResponseCache | None = None,
    settings: Settings | None = None,
) -> tuple[SyntheticAnswer, list[str], list[str], CallMeta]:
    """One controlled answer, through the chokepoint.

    Returns the answer, the concept keys it was told to include, the ones it was
    told to omit, and the call metadata.

    Like every other LLM call in this project it goes through
    ``call_structured`` - which also means the eval-set build inherits the
    response cache, so re-running the builder after a crash re-reads rather than
    re-pays for what already succeeded.
    """
    inputs, included, omitted = build_inputs(item, level)
    answer, meta = await call_structured(
        TaskName.EVAL_ANSWER_SYNTHESIS,
        inputs,
        SyntheticAnswer,
        trace=trace,
        router=router,
        cache=cache,
        settings=settings,
    )
    return answer, included, omitted, meta


#: Typographic characters models like to emit, and the plain equivalents a person
#: typing an answer into a textarea would actually produce.
#:
#: **Why this is not cosmetic.** Evidence-span verification is exact substring
#: matching (Day 17), so if a stored answer contains U+2011 NON-BREAKING HYPHEN
#: and the grader re-quotes it as a plain "-", the span fails the check, the
#: concept is downgraded to `absent`, and Day 19 measures a fabrication that never
#: happened. The first smoke run of this generator produced 9 non-breaking hyphens
#: and an em dash in five answers, so this is an observed problem rather than a
#: hypothetical one. Normalising at generation time keeps the *dataset* clean;
#: nothing here touches a real candidate's text at grading time, where
#: transforming the input would be exactly the unsafe normalisation Day 17
#: refused.
PROSE_REPLACEMENTS: dict[str, str] = {
    "‑": "-",  # non-breaking hyphen
    "‐": "-",  # hyphen
    "‒": "-",  # figure dash
    "–": "-",  # en dash
    "—": " - ",  # em dash, which is used as a clause break
    "‘": "'",
    "’": "'",
    "“": '"',
    "”": '"',
    "…": "...",
    " ": " ",  # non-breaking space
    "​": "",  # zero-width space
}


def normalise_prose(text: str) -> str:
    """Flatten typographic punctuation to what a person would type.

    Applied to *generated* answers before they are written to the dataset, and
    nowhere else. Collapses the substitutions in ``PROSE_REPLACEMENTS`` and then
    normalises whitespace runs, so the committed file contains only characters a
    candidate could plausibly have produced in a web form.
    """
    out = text
    for fancy, plain in PROSE_REPLACEMENTS.items():
        out = out.replace(fancy, plain)
    # An em dash became " - ", which can leave doubled spaces.
    return " ".join(out.split())


def prose_problems(
    text: str, *, min_words: int = MIN_WORDS, max_words: int = MAX_WORDS
) -> list[str]:
    """Deterministic checks on generated prose. Returns the reasons to reject it.

    Cheap, and worth doing before a row reaches a committed file: a generation
    that came back as a bulleted list or with a "Sure! Here's an answer:"
    preamble is not a candidate answer, and would quietly make the dataset
    measure a grader's reaction to formatting.
    """
    problems: list[str] = []
    stripped = text.strip()
    words = len(stripped.split())

    if words < min_words:
        problems.append(f"too short: {words} words")
    if words > max_words:
        problems.append(f"too long: {words} words")
    if any(
        line.lstrip().startswith(("- ", "* ", "#", "1. ", "2. ")) for line in stripped.splitlines()
    ):
        problems.append("contains markdown list or heading formatting")
    lowered = stripped.lower()
    for giveaway in ("here is", "here's an answer", "as an ai", "certainly!", "sure!"):
        if lowered.startswith(giveaway):
            problems.append(f"starts with a model preamble: {giveaway!r}")
    leftover = sorted({ch for ch in stripped if ch in PROSE_REPLACEMENTS})
    if leftover:
        problems.append(f"un-normalised typographic characters: {leftover}")
    return problems


__all__ = [
    "ANSWER_SYNTHESIS_PROMPT_V1",
    "LEVEL_BRIEFS",
    "LEVEL_COVERAGE",
    "MAX_WORDS",
    "MIN_WORDS",
    "PROSE_REPLACEMENTS",
    "SyntheticAnswer",
    "build_inputs",
    "generate_answer",
    "normalise_prose",
    "prose_problems",
    "split_concepts",
    "word_target",
]
