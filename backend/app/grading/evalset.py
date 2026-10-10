"""The grading evaluation set - plan section 12.2, built on Day 18.

    1. Sample 24 questions, stratified across topic x difficulty.
    2. Generate 5 answers per question at CONTROLLED quality levels -
       excellent / good / partial / superficial / wrong-but-confident.
       Produce these with an LLM GIVEN THE CONCEPT KEY, instructing it to
       include or omit named concepts.
       -> You now have GROUND TRUTH BY CONSTRUCTION for concept coverage.
    3. Human-label all 120 on a 0-4 scale using written anchors.
       Label in RANDOMISED order to avoid drift.

This module owns steps 1 and 3's *bookkeeping* - the record shape, the stratified
sample, the randomised labelling order, and every validation rule.  It is pure:
no LLM, no network, no clock.  Step 2 lives in ``answergen.py`` because it is the
one part that calls a model, and ``scripts/build_grading_evalset.py`` drives both.

**The two ground truths, and why there are two.**  Plan section 12.2 is explicit
that the construction is what makes 120 labels affordable:

    Because you know which concepts were planted, you can measure per-concept
    classification accuracy (precision/recall on `covered` vs `absent`)
    independently of the aggregate score. If the aggregate QWK is disappointing,
    this tells you whether the problem is the classifier or the scoring formula -
    which is the difference between a two-hour fix and a week of flailing.

So every record carries both:

* ``concepts_included`` / ``concepts_omitted`` - **ground truth by
  construction**, decided by code before any model was asked for anything. This
  is the label for the *classifier*.
* ``human_score`` - an independent 0-4 holistic judgement, collected blind and in
  randomised order. This is the label for the *scorer*, and it is what Day 19's
  QWK is computed against.

**Neither is derived from the grader.**  That is the whole point, and it is worth
stating as a rule rather than a hope: nothing in this module imports
``app.grading.scoring`` or ``app.grading.pipeline``, and a test asserts it. A
ground truth computed from the thing being measured measures nothing.

``quality_intent`` is *not* a third label.  It is the instruction the generator
was given, and the interesting Day-19 question is precisely how often a human
reading the answer blind agrees with the level it was built at.
"""

from __future__ import annotations

import json
import random
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.bank.schema import BankItem

#: Plan section 12.2, named exactly. The order is the designed quality ordering,
#: best first, and it is load-bearing: ``QUALITY_LEVELS.index`` is the ordinal a
#: Day-19 rank correlation would use if it wanted the *intent* rather than the
#: human label.
QualityIntent = Literal["excellent", "good", "partial", "superficial", "wrong_confident"]

QUALITY_LEVELS: tuple[QualityIntent, ...] = (
    "excellent",
    "good",
    "partial",
    "superficial",
    "wrong_confident",
)

#: Plan section 3, Day 18: 24 questions x 5 levels = 120 answers.
TARGET_QUESTIONS = 24
TARGET_ANSWERS = TARGET_QUESTIONS * len(QUALITY_LEVELS)

#: Plan section 12.2 step 5 adds "25 REAL answers from your own pilot sessions".
#: There are no pilot sessions before Phase 5, so today produces one slice and
#: the field exists so the second can be added without a schema migration - and
#: so that "report the two slices separately" stays possible.
Slice = Literal["synthetic", "real"]

#: Who produced a 0-4 label. **Not decoration.**
#:
#: Plan section 12.2 step 3 says "human-label all 120", and the Phase-4 exit gate
#: is QWK >= 0.65 *against human labels*. A label produced any other way may be
#: perfectly useful for developing the Day-19 harness and is still not the thing
#: the gate is written about, so the difference has to be machine-checkable
#: rather than remembered - otherwise a QWK computed over model labels reads
#: exactly like a QWK computed over human ones, and the gate quietly stops
#: meaning what it says.
#:
#: ``model``  - assigned by an LLM reading the answer against ``LABEL_ANCHORS``.
#:             Independent of the *grader* (different vendor, different prompt,
#:             no concept checklist, no rubric dimensions, no evidence spans, no
#:             code-side arithmetic), but not a human.
#: ``human``  - assigned by a person. The only kind the exit gate accepts.
LabelSource = Literal["human", "model"]

#: The 0-4 anchors the human labeller reads. Written out here rather than left to
#: the labeller's judgement for the same reason the *grading* rubric is anchored
#: (plan section 7.4, technique 4): an unanchored 0-4 is a different scale for
#: every labeller and every hour of the afternoon, and the drift shows up later
#: as a disappointing QWK that nobody can attribute.
#:
#: These describe the ANSWER, not the concepts. The labeller is deliberately not
#: shown the concept checklist - see ``labelling_order``.
LABEL_ANCHORS: dict[int, str] = {
    0: (
        "Wrong, or empty of content. Actively misleading claims, or nothing that "
        "addresses the question. A confident answer that is wrong belongs here, "
        "not at 1 - confidence is not partial credit."
    ),
    1: (
        "Barely relevant. Names the topic and little else: no mechanism, no "
        "specifics, nothing a colleague could act on. Padding and restated "
        "question text count as nothing."
    ),
    2: (
        "Partially correct. Some real content and at least one substantive point, "
        "with obvious gaps or vagueness. Would need follow-up before you trusted "
        "it."
    ),
    3: (
        "Solid. Covers the main ground correctly with specifics, and a colleague "
        "could act on it. May miss a subtlety or a trade-off."
    ),
    4: (
        "Excellent. Correct, specific, well organised, and aware of what its own "
        "choices cost. What you would hope a strong candidate says."
    ),
}

MIN_LABEL = 0
MAX_LABEL = 4


class EvalAnswer(BaseModel):
    """One record. Field names follow plan section 12.2's sample record.

    ``extra="forbid"`` for the same reason ``BankItem`` uses it: this file is
    hand-inspectable and hand-editable, and a misspelt field in a JSONL row is
    otherwise silently dropped - taking the label with it.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^eval-\d{4}$")
    question_id: str
    answer: str = Field(min_length=20, max_length=8000)
    quality_intent: QualityIntent

    #: GROUND TRUTH BY CONSTRUCTION. Decided by code before generation; the
    #: generator was told to demonstrate exactly these and to stay silent about
    #: the others. Together they must be the question's full concept set.
    concepts_included: list[str]
    concepts_omitted: list[str]

    #: The independent holistic label, 0-4, collected blind. ``None`` until a
    #: human has labelled it - and a null here is a loud, countable fact rather
    #: than a zero that would quietly corrupt a correlation.
    human_score: int | None = Field(default=None, ge=MIN_LABEL, le=MAX_LABEL)
    labeller: str | None = None
    #: Provenance for ``human_score``. See :data:`LabelSource` - the Phase-4 gate
    #: only accepts ``"human"``, and :func:`gate_eligible` is what enforces it.
    label_source: LabelSource | None = None

    #: Plan section 12.2 step 4: a second labeller on 40 items gives the
    #: human-human ceiling. Declared now, collected later - the Phase-4 exit gate
    #: defers reporting that ceiling to Phase 9.
    human_score_b: int | None = Field(default=None, ge=MIN_LABEL, le=MAX_LABEL)
    labeller_b: str | None = None
    label_source_b: LabelSource | None = None

    #: The **retired Day-18 model pass**, kept rather than overwritten.
    #:
    #: Day 18's first labelling sitting filled ``human_score`` with labels an LLM
    #: assigned against :data:`LABEL_ANCHORS`, flagged honestly as
    #: ``label_source: "model"``. The human pass needs ``human_score`` back, and
    #: the choice at that point is to destroy the model pass or to move it here.
    #:
    #: It is moved here, for two reasons. The 120 model labels cost real quota to
    #: produce and are the only thing a Day-19 harness could be developed against
    #: before human labels existed, so throwing them away would be throwing away
    #: a working artefact. More importantly, holding both passes on one row makes
    #: **"how well does a model label agree with a human one?"** a question this
    #: dataset can answer - see :func:`label_agreement`. That is worth knowing
    #: before anyone proposes model labelling as a way to skip the 1.5 hours on a
    #: future set, and it is unanswerable once the column is gone.
    #:
    #: These fields are **never** gate-eligible and :func:`gate_eligible` does not
    #: look at them. They are a preserved measurement, not a ground truth.
    model_score: int | None = Field(default=None, ge=MIN_LABEL, le=MAX_LABEL)
    #: Which model assigned :attr:`model_score`. Same pairing rule as the human
    #: labels: a score needs a claimant, and "some model" is not a claimant when
    #: the question later becomes whether that model drifted.
    model_labeller: str | None = None

    slice: Slice = "synthetic"

    #: Reproducibility. Which builder run produced this row, and the seed that
    #: chose the questions and the concept split.
    build_seed: int | None = None
    #: The model that wrote the prose. Not the grader, and recorded so that a
    #: dataset generated by one model is never silently compared with one
    #: generated by another.
    generator_model: str | None = None

    @model_validator(mode="after")
    def _label_and_labeller_agree(self) -> EvalAnswer:
        """A score needs a claimant, mirroring ``BankItem``'s review rule.

        An unattributed label cannot be audited, re-checked, or excluded when
        one labeller turns out to have drifted.
        """
        if (self.human_score is None) != (self.labeller is None):
            raise ValueError(
                f"{self.id}: human_score and labeller must be set together "
                f"(got score={self.human_score!r}, labeller={self.labeller!r})"
            )
        if (self.human_score is None) != (self.label_source is None):
            raise ValueError(
                f"{self.id}: human_score and label_source must be set together; an "
                "unattributed label cannot be told apart from a human one, which is "
                "the distinction the Phase-4 gate rests on"
            )
        if (self.human_score_b is None) != (self.labeller_b is None):
            raise ValueError(f"{self.id}: human_score_b and labeller_b must be set together")
        if (self.human_score_b is None) != (self.label_source_b is None):
            raise ValueError(f"{self.id}: human_score_b and label_source_b must be set together")
        if (self.model_score is None) != (self.model_labeller is None):
            raise ValueError(
                f"{self.id}: model_score and model_labeller must be set together; a "
                "retired label whose author is unrecorded cannot be compared with "
                "anything later"
            )
        return self

    @model_validator(mode="after")
    def _concepts_are_disjoint(self) -> EvalAnswer:
        overlap = sorted(set(self.concepts_included) & set(self.concepts_omitted))
        if overlap:
            raise ValueError(
                f"{self.id}: {overlap} is both included and omitted; the construction "
                "ground truth would be ambiguous for exactly those concepts"
            )
        return self

    @property
    def is_labelled(self) -> bool:
        return self.human_score is not None

    @property
    def is_human_labelled(self) -> bool:
        """The stricter question, and the one the Phase-4 gate actually asks."""
        return self.human_score is not None and self.label_source == "human"


def stratified_questions(
    items: Sequence[BankItem],
    *,
    seed: int,
    count: int = TARGET_QUESTIONS,
    stratum_of: Callable[[BankItem], str] | None = None,
) -> list[BankItem]:
    """Plan section 12.2 step 1: sample ``count`` questions, stratified.

    Stratified across **domain x difficulty**, which is this bank's usable
    reading of the plan's "topic x difficulty". The bank's six topics are very
    unevenly populated - ``databases`` and ``dsa`` hold 20 each while ``caching``
    holds 3 - so stratifying on topic directly would either take every caching
    item or none, and the sample would be a statement about the bank's authoring
    history rather than about its subject matter. Domain is the level at which
    the bank is balanced by construction (20/20/20), and difficulty is stratified
    *within* each domain so the sample spans easy to hard rather than clustering
    wherever a domain happens to sit.

    ``stratum_of`` names the stratum for an item. The caller passes the real
    domain from ``LoadedItem.domain``; the default is the item id's prefix, which
    happens to map one-to-one onto domain in this bank but is a heuristic rather
    than a fact and is only here so the function is usable from a test that has
    ``BankItem`` and nothing else.

    Deterministic given ``seed``: same seed, same 24 questions, forever. The
    dataset is a committed artefact and "which questions are in it?" must have
    one answer.
    """
    if count <= 0:
        raise ValueError("count must be positive")
    if len(items) < count:
        raise ValueError(f"need at least {count} bank items to sample from, got {len(items)}")

    name = stratum_of or _domain_of
    by_domain: dict[str, list[BankItem]] = {}
    for item in items:
        by_domain.setdefault(name(item), []).append(item)

    domains = sorted(by_domain)
    per_domain = count // len(domains)
    remainder = count - per_domain * len(domains)

    rng = random.Random(seed)
    chosen: list[BankItem] = []
    for index, domain in enumerate(domains):
        # The remainder goes to the largest domains first, so a bank that does
        # not divide evenly still produces a stable, explainable split.
        take = per_domain + (1 if index < remainder else 0)
        chosen.extend(_spread_by_difficulty(by_domain[domain], take, rng))

    # Sorted by id so the committed manifest diffs cleanly; the sampling itself
    # was random, and this only fixes the order it is written down in.
    return sorted(chosen, key=lambda item: item.id)


def _domain_of(item: BankItem) -> str:
    """Fallback stratum: the item id's prefix.

    ``BankItem`` carries ``topic``/``subtopic`` but not the domain - domain is a
    property of which file the item lives in, which only ``LoadedItem`` knows.
    In this bank the prefixes (``db``/``dsa``/``sys``) do map one-to-one onto the
    three domains, but that is a coincidence of the authoring convention rather
    than a guarantee, which is why the real caller passes ``stratum_of``.
    """
    return item.id.split("-", 1)[0]


def _spread_by_difficulty(
    items: Sequence[BankItem], take: int, rng: random.Random
) -> list[BankItem]:
    """Take ``take`` items spanning the difficulty range, not clustered in it.

    Sort by ``difficulty_b``, cut into ``take`` contiguous bands, and draw one
    from each. That guarantees the sample spans the domain's full difficulty
    range while still being random *within* a band - which is what makes a
    Day-19 result about the grader rather than about an accidentally easy slice.
    """
    if take > len(items):
        raise ValueError(f"cannot take {take} items from a pool of {len(items)}")
    ordered = sorted(items, key=lambda item: (item.difficulty_b, item.id))
    picked: list[BankItem] = []
    for band in range(take):
        start = band * len(ordered) // take
        end = max(start + 1, (band + 1) * len(ordered) // take)
        picked.append(rng.choice(ordered[start:end]))
    return picked


def labelling_order(answers: Sequence[EvalAnswer], *, seed: int) -> list[str]:
    """Plan section 12.2 step 3: the order to label in, RANDOMISED.

    Returns answer ids, never the answers themselves, so a caller cannot
    accidentally iterate the dataset in file order and think it shuffled.

    **Why the order matters enough to be a seeded function rather than a
    ``shuffle`` at the call site.** In file order, the five answers to one
    question sit together, best first. A labeller working down that list is not
    judging each answer on its own - they are ranking five answers against each
    other, and the fifth one is scored relative to the first four. That is
    anchoring, it is systematic rather than noisy, and it would inflate the
    apparent agreement between the human labels and the construction intent for
    a reason that has nothing to do with either. Randomising breaks it, and
    seeding means the exact order a labeller saw is recoverable a month later
    when someone asks whether drift explains a result.
    """
    ids = [answer.id for answer in answers]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate answer ids; refusing to build a labelling order")
    rng = random.Random(seed)
    rng.shuffle(ids)
    return ids


# ---------------------------------------------------------------------------
# Reading and writing
# ---------------------------------------------------------------------------


def load_answers(path: Path) -> list[EvalAnswer]:
    """Read the dataset. Every row validates, or the whole read fails.

    Failing the whole read is deliberate: a dataset with one silently-skipped row
    is a dataset whose count nobody can trust, and the count is the first thing
    every later report quotes.
    """
    answers: list[EvalAnswer] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            answers.append(EvalAnswer.model_validate_json(line))
        except ValueError as exc:
            raise ValueError(f"{path.name} line {number}: {exc}") from exc
    return answers


def dump_answers(answers: Iterable[EvalAnswer], path: Path) -> int:
    """Write the dataset, one JSON object per line, in id order.

    Sorted, and with ``sort_keys``, because this file is reviewed as a git diff:
    a row whose fields reordered between runs would show as a change to every
    line and hide the one label that actually moved.
    """
    ordered = sorted(answers, key=lambda answer: answer.id)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(answer.model_dump(mode="json"), sort_keys=True, ensure_ascii=False)
        for answer in ordered
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(ordered)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class EvalSetError(ValueError):
    """The dataset as a whole is wrong, even though every row parsed."""


def validate_dataset(
    answers: Sequence[EvalAnswer],
    *,
    questions: dict[str, BankItem],
    expect_questions: int = TARGET_QUESTIONS,
    expect_answers: int = TARGET_ANSWERS,
    require_labels: bool = False,
    require_human_labels: bool = False,
) -> list[str]:
    """Every whole-dataset rule, checked at once. Returns the list of problems.

    Returns rather than raises so a report can show *all* of them; the script
    turns a non-empty list into a non-zero exit. Plan section 12.2 fixes the
    counts, and everything else here exists because it is a way the set could be
    quietly wrong while still having 120 rows in it.

    ``require_labels`` is off by default because generation and labelling are
    two different sittings - the plan budgets 1.5 hours of human time for the
    second one - and a dataset that is built but not yet labelled is a real,
    valid intermediate state that the validator should be able to describe
    rather than reject.
    """
    problems: list[str] = []

    if len(answers) != expect_answers:
        problems.append(f"expected {expect_answers} answers, found {len(answers)}")

    ids = [answer.id for answer in answers]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        problems.append(f"duplicate answer ids: {duplicates}")

    question_ids = sorted({answer.question_id for answer in answers})
    if len(question_ids) != expect_questions:
        problems.append(
            f"expected {expect_questions} distinct questions, found {len(question_ids)}"
        )

    unknown = [qid for qid in question_ids if qid not in questions]
    if unknown:
        problems.append(f"answers reference questions that are not in the bank: {unknown}")

    # Exactly one answer per (question, level), and every level present.
    seen: dict[tuple[str, str], int] = {}
    for answer in answers:
        seen[(answer.question_id, answer.quality_intent)] = (
            seen.get((answer.question_id, answer.quality_intent), 0) + 1
        )
    for qid in question_ids:
        levels = [level for level in QUALITY_LEVELS if seen.get((qid, level))]
        if len(levels) != len(QUALITY_LEVELS):
            missing = sorted(set(QUALITY_LEVELS) - set(levels))
            problems.append(f"{qid}: missing quality levels {missing}")
        repeated = sorted(level for level in QUALITY_LEVELS if seen.get((qid, level), 0) > 1)
        if repeated:
            problems.append(f"{qid}: more than one answer at levels {repeated}")

    # The construction ground truth must describe the question it belongs to.
    for answer in answers:
        item = questions.get(answer.question_id)
        if item is None:
            continue
        planted = set(answer.concepts_included) | set(answer.concepts_omitted)
        expected = set(item.concept_keys)
        if planted != expected:
            problems.append(
                f"{answer.id}: included+omitted {sorted(planted)} does not match "
                f"{answer.question_id}'s concepts {sorted(expected)}"
            )

    if require_labels:
        unlabelled = sorted(answer.id for answer in answers if not answer.is_labelled)
        if unlabelled:
            problems.append(
                f"{len(unlabelled)} answers have no human_score (first few: {unlabelled[:5]})"
            )

    if require_human_labels:
        not_human = sorted(answer.id for answer in answers if not answer.is_human_labelled)
        if not_human:
            problems.append(
                f"{len(not_human)} answers are not human-labelled (first few: "
                f"{not_human[:5]}); the Phase-4 QWK gate is written against human "
                "labels and a model label is not a substitute for one"
            )

    return problems


def gate_eligible(answers: Sequence[EvalAnswer]) -> bool:
    """Is this set fit to compute the **Phase-4 exit-gate** QWK against?

    One function, so that no Day-19 report has to remember the rule. The gate is
    "QWK >= 0.65 on the 120-item labelled set", and plan section 12.2 step 3 is
    explicit that those labels are human. A QWK over model labels is a perfectly
    good development signal and is *not* the gate - and the difference is
    invisible in the number itself, which is exactly why it is checked here
    rather than described in prose.
    """
    return len(answers) == TARGET_ANSWERS and all(a.is_human_labelled for a in answers)


def label_sources(answers: Sequence[EvalAnswer]) -> dict[str, int]:
    """How many labels came from where. ``unlabelled`` counts the empty ones."""
    counts: dict[str, int] = {"human": 0, "model": 0, "unlabelled": 0}
    for answer in answers:
        counts[answer.label_source or "unlabelled"] += 1
    return counts


def intent_agreement(answers: Sequence[EvalAnswer]) -> dict[str, dict[str, object]]:
    """Per level: how the blind labels actually landed. Only labelled rows count.

    **What this is for.** ``quality_intent`` is an instruction given to the
    generator, not a label. Whether the generator *obeyed* it is an open question
    that only a blind reading can answer, and this is where that answer lives.

    A level whose labels sit on top of a neighbouring level's is not a distinct
    experimental condition, however carefully its brief was written - and a
    Day-19 report that treated it as one would be measuring a distinction the
    data does not contain.
    """
    out: dict[str, dict[str, object]] = {}
    for level in QUALITY_LEVELS:
        scores = [a.human_score for a in answers if a.quality_intent == level and a.is_labelled]
        if not scores:
            out[level] = {"n": 0}
            continue
        ordered = sorted(scores)  # type: ignore[type-var]
        out[level] = {
            "n": len(ordered),
            "mean": sum(ordered) / len(ordered),  # type: ignore[arg-type]
            "median": ordered[len(ordered) // 2],
            "min": ordered[0],
            "max": ordered[-1],
            "distribution": {
                score: sum(1 for value in ordered if value == score)
                for score in range(MIN_LABEL, MAX_LABEL + 1)
            },
        }
    return out


def unachieved_intent(answers: Sequence[EvalAnswer], *, solid_from: int = 3) -> list[str]:
    """Ids of ``wrong_confident`` answers a blind labeller rated **solid**.

    The sharpest single measure of whether the construction worked, and the one
    that produced the Day-18 dataset's worst finding: the ``wrong_confident``
    brief asks for "at least two clear factual errors ... claims that would lead
    to a broken system if acted on", and a row scoring 3 or 4 is one where the
    generator simply did not comply and wrote a correct answer instead.

    These rows are **not** wrong, so they are not bad data - they are just not
    the condition they were labelled as. A Day-19 analysis that used
    ``quality_intent == "wrong_confident"`` as "answers containing errors" would
    be wrong about most of this slice, which is exactly the assumption this
    function exists to stop anyone making.
    """
    return sorted(
        answer.id
        for answer in answers
        if answer.quality_intent == "wrong_confident"
        and answer.human_score is not None
        and answer.human_score >= solid_from
    )


def leaked_omitted_concepts(answer: EvalAnswer) -> list[str]:
    """Omitted concepts whose **name appears literally** in the answer text.

    A measured weakness of the Day-18 set, not a hypothetical one: 21 of the 192
    omitted-concept slots - 10.9% - trip this check. The generator was told to
    stay silent about those concepts and mostly did, but "say nothing about
    ``repeatable_read``" is a harder instruction than it looks when the question
    is *about* isolation levels.

    **Why this matters to Day 19.** ``concepts_omitted`` is the precision ground
    truth: a grader marking an omitted concept ``covered`` is supposed to be a
    false positive. Where the concept is named in the answer, that inference is
    unsafe - the grader may be reading the text correctly and the *construction*
    is what slipped. Day 19 should report precision twice, once over all slots
    and once excluding the flagged ones, and say which it is quoting.

    **What this check is and is not.** It matches the concept key literally,
    with underscores read as spaces. So it is a *lower bound on mentions*: it
    cannot produce a false alarm, and it will miss a leak that paraphrases. It
    is also not a claim that the concept was *demonstrated* - ``eval-0019``
    names ``repeatable_read`` only to say the distinction "kind of blurs", which
    a grader could fairly call ``partial`` and could fairly call ``absent``.
    Treat a hit as "this slot is unreliable as precision ground truth", nothing
    more.
    """
    lowered = answer.answer.lower()
    return [
        key for key in answer.concepts_omitted if key in lowered or key.replace("_", " ") in lowered
    ]


def leakage_report(answers: Sequence[EvalAnswer]) -> dict[str, object]:
    """Whole-set leakage, for a Day-19 report to quote rather than recompute."""
    slots = sum(len(a.concepts_omitted) for a in answers)
    flagged = {a.id: leaked for a in answers if (leaked := leaked_omitted_concepts(a))}
    leaked_slots = sum(len(v) for v in flagged.values())
    return {
        "omitted_slots": slots,
        "leaked_slots": leaked_slots,
        "leak_rate": (leaked_slots / slots) if slots else 0.0,
        "answers_affected": len(flagged),
        "by_answer": flagged,
    }


def level_distribution(answers: Sequence[EvalAnswer]) -> dict[str, int]:
    return {level: sum(1 for a in answers if a.quality_intent == level) for level in QUALITY_LEVELS}


def label_distribution(answers: Sequence[EvalAnswer]) -> dict[int, int]:
    """How many answers sit at each 0-4 human label. Unlabelled rows are absent."""
    counts = dict.fromkeys(range(MIN_LABEL, MAX_LABEL + 1), 0)
    for answer in answers:
        if answer.human_score is not None:
            counts[answer.human_score] += 1
    return counts


def blind_worklist(
    answers: Sequence[EvalAnswer],
    *,
    questions: dict[str, str],
    seed: int,
) -> str:
    """The labelling worklist, as text: shuffled, and blind by construction.

    Lives here rather than in the script because "this artefact is blind" is a
    property worth asserting in a test, and a f-string buried in a CLI is
    trusted instead. It was trusted, and it was wrong - see below.

    Each item carries its **1-based position in the seeded order**, the question
    and the answer. Nothing else: no ``quality_intent``, no concept split, no
    expected-concept list, no grader output, and **no row id**.

    **Why the row id is excluded.** Answer ids are allocated sequentially per
    question x level, so ``(N - 1) mod 5`` recovers ``quality_intent`` exactly:
    ``eval-0110`` -> 4 -> ``wrong_confident``, ``eval-0048`` -> 2 ->
    ``partial``. An earlier version of the worklist printed the id, which meant
    the one artefact advertised as blind handed the labeller the construction
    intent one subtraction away from plain text. The interactive prompt never
    showed it; the file did, and the file is what a labelling pass done away
    from the terminal actually reads.

    Positions are sufficient to apply scores afterwards, because the order is a
    pure function of ``seed``. Nothing has to be stored to map a position back
    to a row, so there is no key file to leak either.
    """
    order = labelling_order(answers, seed=seed)
    by_id = {answer.id: answer for answer in answers}
    blocks = []
    for position, answer_id in enumerate(order, start=1):
        row = by_id[answer_id]
        blocks.append(
            f"### {position}/{len(order)}\n\n"
            f"QUESTION\n{questions.get(row.question_id, '(missing)')}\n\n"
            f"ANSWER\n{row.answer}\n"
        )
    return "\n".join(blocks)


def model_label_distribution(answers: Sequence[EvalAnswer]) -> dict[int, int]:
    """The same histogram over the retired Day-18 model pass.

    Reported next to :func:`label_distribution` rather than instead of it: the
    two distributions being different shapes is the first thing anyone should
    see before treating one as a stand-in for the other.
    """
    counts = dict.fromkeys(range(MIN_LABEL, MAX_LABEL + 1), 0)
    for answer in answers:
        if answer.model_score is not None:
            counts[answer.model_score] += 1
    return counts


def label_agreement(answers: Sequence[EvalAnswer]) -> dict[str, object]:
    """How closely the current pass-A label and the retired pass agree, row by row.

    Only rows carrying **both** a ``human_score`` and a ``model_score`` count.

    **``sources`` is part of the answer, not metadata.** This function does not
    require pass A to be human, because the dataset's history is that a model
    pass was retired here and a *second* pass written into ``human_score``, and a
    comparison that silently returned ``n: 0`` in that state would be worse than
    useless - it would read as "no disagreement found". So the sources of both
    sides are reported alongside the numbers, and the caller is expected to say
    which comparison it is quoting:

    * ``{"human", "model"}`` - **human versus model.** The useful one. It answers
      whether model labelling could stand in for the 1.5 hours of human time on a
      future set.
    * ``{"model", "model"}`` - **model versus model.** Two passes by the same
      model family, which is near-self-consistency, not two opinions converging.
      A high agreement here is evidence of reproducibility and is *not* evidence
      that either pass is correct. Both passes can be wrong in the same direction
      for the same reason, and nothing in this number would show it.

    It is not a validation of either pass either way - there is no third opinion
    here to adjudicate with. ``mean_signed_error`` is the useful half, because a
    pass that is uniformly one point generous is a different problem from one
    that is noisy: the first is a correctable bias, the second is not.

    Signed as *retired minus current*: a positive mean means the retired pass was
    the more generous of the two.
    """
    pairs: list[tuple[int, int]] = [
        (answer.human_score, answer.model_score)
        for answer in answers
        if answer.human_score is not None and answer.model_score is not None
    ]
    if not pairs:
        return {"n": 0}
    sources = sorted(
        {
            source
            for answer in answers
            if answer.human_score is not None
            and answer.model_score is not None
            and (source := answer.label_source) is not None
        }
        | {"model"}  # the retired side is a model pass by construction
    )
    deltas = [retired - current for current, retired in pairs]
    return {
        "n": len(pairs),
        "sources": sources,
        "exact": sum(1 for d in deltas if d == 0),
        "within_one": sum(1 for d in deltas if abs(d) <= 1),
        "retired_higher": sum(1 for d in deltas if d > 0),
        "retired_lower": sum(1 for d in deltas if d < 0),
        "mean_signed_error": sum(deltas) / len(deltas),
        "mean_absolute_error": sum(abs(d) for d in deltas) / len(deltas),
        "max_absolute_error": max(abs(d) for d in deltas),
    }


__all__ = [
    "LABEL_ANCHORS",
    "MAX_LABEL",
    "MIN_LABEL",
    "QUALITY_LEVELS",
    "TARGET_ANSWERS",
    "TARGET_QUESTIONS",
    "EvalAnswer",
    "EvalSetError",
    "LabelSource",
    "QualityIntent",
    "Slice",
    "blind_worklist",
    "dump_answers",
    "gate_eligible",
    "intent_agreement",
    "label_agreement",
    "label_distribution",
    "label_sources",
    "labelling_order",
    "leakage_report",
    "leaked_omitted_concepts",
    "level_distribution",
    "load_answers",
    "model_label_distribution",
    "stratified_questions",
    "unachieved_intent",
    "validate_dataset",
]
