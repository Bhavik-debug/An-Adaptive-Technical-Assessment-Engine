"""The grading evaluation set - plan section 12.2, built on Day 18.

Two kinds of test live here and they are worth telling apart.

Most are about the *machinery*: the record shape, the stratified sample, the
randomised labelling order, the validation rules. Those are ordinary unit tests
over pure functions.

The rest are about the **committed artefact** - they open
``evals/datasets/grading/answers.jsonl`` and assert the file on disk is the file
the plan asked for. Those are the ones that matter on the day somebody edits a
row by hand, and they are why the dataset is validated as data rather than
trusted as output.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.bank.loader import load_bank
from app.bank.schema import BankItem, ExpectedConcept
from app.grading.answergen import (
    LEVEL_BRIEFS,
    LEVEL_COVERAGE,
    PROSE_REPLACEMENTS,
    normalise_prose,
    prose_problems,
    split_concepts,
    word_target,
)
from app.grading.evalset import (
    LABEL_ANCHORS,
    MAX_LABEL,
    MIN_LABEL,
    QUALITY_LEVELS,
    TARGET_ANSWERS,
    TARGET_QUESTIONS,
    EvalAnswer,
    dump_answers,
    gate_eligible,
    intent_agreement,
    label_distribution,
    label_sources,
    labelling_order,
    leakage_report,
    leaked_omitted_concepts,
    level_distribution,
    load_answers,
    stratified_questions,
    unachieved_intent,
    validate_dataset,
)

DATASET_DIR = Path(__file__).resolve().parents[3].parent / "evals" / "datasets" / "grading"
ANSWERS_PATH = DATASET_DIR / "answers.jsonl"
QUESTIONS_PATH = DATASET_DIR / "questions.jsonl"


def concept(key: str, weight: int = 1) -> ExpectedConcept:
    return ExpectedConcept(key=key, weight=weight, hint="a hint that is long enough")


def fake_item(item_id: str, *, b: float = 0.0, n_concepts: int = 4) -> BankItem:
    keys = [f"concept_{i}" for i in range(n_concepts)]
    text = (
        f"A question about {item_id} that is long enough to satisfy the bank schema's "
        "minimum length requirement for question text, which is forty characters."
    )
    return BankItem(
        id=item_id,
        topic="databases",
        subtopic="indexing",
        text=text,
        difficulty_b=b,
        expected_concepts=[concept(k, weight=(i % 3) + 1) for i, k in enumerate(keys)],
        reference_answer="x" * 220,
        anchor_terms=["question"],
        time_estimate_s=300,
        tags=["tag"],
        source="authored",
        review_status="drafted",
    )


def answer_row(number: int, question_id: str, level: str, **overrides) -> EvalAnswer:
    payload: dict[str, object] = {
        "id": f"eval-{number:04d}",
        "question_id": question_id,
        "answer": "An answer long enough to pass the minimum length rule for a record.",
        "quality_intent": level,
        "concepts_included": ["concept_0"],
        "concepts_omitted": ["concept_1", "concept_2", "concept_3"],
    }
    payload.update(overrides)
    return EvalAnswer.model_validate(payload)


# ---------------------------------------------------------------------------
# The plan's numbers
# ---------------------------------------------------------------------------


class TestThePlansDesign:
    def test_twenty_four_questions_times_five_levels_is_one_hundred_and_twenty(self):
        assert TARGET_QUESTIONS == 24
        assert len(QUALITY_LEVELS) == 5
        assert TARGET_ANSWERS == 120

    def test_the_five_levels_are_the_plans_five(self):
        """Plan section 12.2: excellent / good / partial / superficial /
        wrong-but-confident."""
        assert QUALITY_LEVELS == (
            "excellent",
            "good",
            "partial",
            "superficial",
            "wrong_confident",
        )

    def test_every_level_has_a_brief_a_coverage_share_and_a_length(self):
        for level in QUALITY_LEVELS:
            assert level in LEVEL_BRIEFS
            assert level in LEVEL_COVERAGE
            assert word_target(level) > 0

    def test_the_label_scale_is_zero_to_four_with_written_anchors(self):
        """Plan section 12.2 step 3: "a 0-4 scale using written anchors"."""
        assert sorted(LABEL_ANCHORS) == list(range(MIN_LABEL, MAX_LABEL + 1))
        for score, text in LABEL_ANCHORS.items():
            assert len(text) > 60, score


# ---------------------------------------------------------------------------
# Ground truth must not come from the thing being measured
# ---------------------------------------------------------------------------


class TestTheGroundTruthIsIndependentOfTheGrader:
    """The one property that decides whether Day 19 measures anything.

    A dataset whose labels were produced by the grader would give a QWK of 1.0
    and mean nothing. These assert the independence structurally rather than
    hoping for it.
    """

    @staticmethod
    def _imports(module) -> set[str]:
        tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
        names: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names.update(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                names.add(node.module)
        return names

    def test_the_evalset_module_cannot_see_the_scorer_or_the_pipeline(self):
        import app.grading.evalset as evalset

        forbidden = {
            "app.grading.scoring",
            "app.grading.pipeline",
            "app.grading.evidence",
            "app.grading.grader",
        }
        assert not (self._imports(evalset) & forbidden)

    def test_the_generator_cannot_see_the_scorer_or_the_pipeline(self):
        import app.grading.answergen as answergen

        forbidden = {"app.grading.scoring", "app.grading.pipeline", "app.grading.grader"}
        assert not (self._imports(answergen) & forbidden)

    def test_the_generator_is_never_shown_the_grading_rubric(self):
        """A generator that knew how it would be marked would write to the
        rubric, and Day 19 would measure that agreement instead of validity."""
        from app.grading.answergen import ANSWER_SYNTHESIS_PROMPT_V1
        from app.grading.rubric import RUBRIC_KEYS, render_rubric

        text = ANSWER_SYNTHESIS_PROMPT_V1.system + ANSWER_SYNTHESIS_PROMPT_V1.user
        assert render_rubric() not in text
        for key in RUBRIC_KEYS:
            assert key not in text

    def test_the_generator_is_never_shown_the_labelling_anchors(self):
        from app.grading.answergen import ANSWER_SYNTHESIS_PROMPT_V1

        text = ANSWER_SYNTHESIS_PROMPT_V1.system + ANSWER_SYNTHESIS_PROMPT_V1.user
        for anchor in LABEL_ANCHORS.values():
            assert anchor not in text

    def test_the_generator_is_not_asked_what_it_did(self):
        """Code decided the concept split. A model reporting its own coverage
        would be the field everything downstream quietly trusted."""
        from app.grading.answergen import SyntheticAnswer

        assert set(SyntheticAnswer.model_fields) == {"answer"}


# ---------------------------------------------------------------------------
# Stratified sampling
# ---------------------------------------------------------------------------


class TestStratifiedSampling:
    def _pool(self) -> list[BankItem]:
        items = []
        for prefix, count in (("db", 20), ("dsa", 20), ("sys", 20)):
            for n in range(count):
                items.append(fake_item(f"{prefix}-topic-{n:03d}", b=-1.5 + n * 0.15))
        return items

    def test_it_returns_the_requested_count(self):
        assert len(stratified_questions(self._pool(), seed=1318)) == TARGET_QUESTIONS

    def test_it_spreads_evenly_across_strata(self):
        picked = stratified_questions(self._pool(), seed=1318)
        by_prefix: dict[str, int] = {}
        for item in picked:
            by_prefix[item.id.split("-")[0]] = by_prefix.get(item.id.split("-")[0], 0) + 1
        assert by_prefix == {"db": 8, "dsa": 8, "sys": 8}

    def test_it_is_deterministic_for_a_seed(self):
        a = stratified_questions(self._pool(), seed=1318)
        b = stratified_questions(self._pool(), seed=1318)
        assert [i.id for i in a] == [i.id for i in b]

    def test_a_different_seed_gives_a_different_sample(self):
        a = stratified_questions(self._pool(), seed=1318)
        b = stratified_questions(self._pool(), seed=99)
        assert [i.id for i in a] != [i.id for i in b]

    def test_it_spans_the_difficulty_range_rather_than_clustering(self):
        """A sample drawn only from the easy end would make a Day-19 result a
        statement about easy questions."""
        pool = self._pool()
        picked = stratified_questions(pool, seed=1318)
        bs = [i.difficulty_b for i in picked]
        span = max(bs) - min(bs)
        pool_span = max(i.difficulty_b for i in pool) - min(i.difficulty_b for i in pool)
        assert span > 0.7 * pool_span

    def test_it_never_repeats_a_question(self):
        picked = stratified_questions(self._pool(), seed=1318)
        assert len({i.id for i in picked}) == len(picked)

    def test_it_refuses_a_pool_that_is_too_small(self):
        with pytest.raises(ValueError, match="at least"):
            stratified_questions(self._pool()[:10], seed=1318)

    def test_a_custom_stratum_function_is_honoured(self):
        pool = self._pool()
        picked = stratified_questions(pool, seed=1318, stratum_of=lambda i: "one-big-stratum")
        assert len(picked) == TARGET_QUESTIONS


# ---------------------------------------------------------------------------
# Controlled construction
# ---------------------------------------------------------------------------


class TestConceptSplitIsGroundTruthByConstruction:
    def test_included_and_omitted_partition_the_question(self):
        item = fake_item("db-x-001")
        for level in QUALITY_LEVELS:
            included, omitted = split_concepts(item, level)
            keys = {c.key for c in included} | {c.key for c in omitted}
            assert keys == set(item.concept_keys)
            assert not ({c.key for c in included} & {c.key for c in omitted})

    def test_excellent_includes_everything(self):
        item = fake_item("db-x-001")
        included, omitted = split_concepts(item, "excellent")
        assert len(included) == len(item.expected_concepts)
        assert omitted == []

    def test_every_other_level_omits_at_least_one(self):
        """A level that silently covered everything would stop being a distinct
        experimental condition."""
        item = fake_item("db-x-001")
        for level in QUALITY_LEVELS:
            if level == "excellent":
                continue
            _, omitted = split_concepts(item, level)
            assert omitted, level

    def test_every_level_includes_at_least_one(self):
        item = fake_item("db-x-001")
        for level in QUALITY_LEVELS:
            included, _ = split_concepts(item, level)
            assert included, level

    def test_coverage_descends_with_quality(self):
        item = fake_item("db-x-001")
        counts = [
            len(split_concepts(item, lvl)[0])
            for lvl in ("excellent", "good", "partial", "superficial")
        ]
        assert counts == sorted(counts, reverse=True)
        assert len(set(counts)) == 4, "the four levels must be four distinct conditions"

    def test_wrong_confident_matches_partials_coverage(self):
        """They differ only in correctness, which is the comparison worth
        having: a wrong answer that also omitted everything would confound
        'wrong' with 'empty'."""
        item = fake_item("db-x-001")
        assert len(split_concepts(item, "wrong_confident")[0]) == len(
            split_concepts(item, "partial")[0]
        )

    def test_the_split_is_deterministic(self):
        item = fake_item("db-x-001")
        for level in QUALITY_LEVELS:
            first = [c.key for c in split_concepts(item, level)[0]]
            second = [c.key for c in split_concepts(item, level)[0]]
            assert first == second

    def test_the_split_takes_the_heaviest_concepts_first(self):
        """A partial answer should cover the central ideas, not a random subset -
        and a random subset would also vary between builds of one dataset."""
        item = fake_item("db-x-001")
        included, omitted = split_concepts(item, "partial")
        assert min(c.weight for c in included) >= max(c.weight for c in omitted)

    def test_it_works_for_the_banks_five_concept_items(self):
        item = fake_item("db-x-002", n_concepts=5)
        for level in QUALITY_LEVELS:
            included, omitted = split_concepts(item, level)
            assert len(included) + len(omitted) == 5


class TestProseNormalisation:
    @pytest.mark.parametrize("fancy", sorted(PROSE_REPLACEMENTS))
    def test_every_typographic_character_is_flattened(self, fancy):
        assert fancy not in normalise_prose(f"before {fancy} after")

    def test_it_is_idempotent(self):
        text = "I'd use write‑through — never stale…"
        once = normalise_prose(text)
        assert normalise_prose(once) == once

    def test_a_normalised_answer_has_no_prose_problems(self):
        text = normalise_prose("I’d use write‑through caching " * 10)
        assert "un-normalised" not in " ".join(prose_problems(text))

    def test_prose_problems_flags_markdown_and_preambles(self):
        assert any("markdown" in p for p in prose_problems("- a bullet\n- another " * 10))
        assert any("preamble" in p for p in prose_problems("Here is an answer. " * 10))

    def test_prose_problems_flags_length(self):
        assert any("too short" in p for p in prose_problems("short"))
        assert any("too long" in p for p in prose_problems("word " * 500))


# ---------------------------------------------------------------------------
# The record
# ---------------------------------------------------------------------------


class TestEvalAnswerRecord:
    def test_a_valid_row_validates(self):
        row = answer_row(1, "db-x-001", "partial")
        assert row.is_labelled is False

    def test_an_unknown_field_is_refused(self):
        """A misspelt field in a hand-edited JSONL row would otherwise be
        silently dropped, taking the label with it."""
        with pytest.raises(ValidationError, match="Extra inputs"):
            answer_row(1, "db-x-001", "partial", humanscore=3)

    def test_the_id_must_be_shaped_like_an_eval_id(self):
        with pytest.raises(ValidationError):
            answer_row(1, "db-x-001", "partial", id="something-else")

    @pytest.mark.parametrize("score", [-1, 5, 9])
    def test_a_label_off_the_scale_is_refused(self, score):
        with pytest.raises(ValidationError):
            answer_row(1, "db-x-001", "partial", human_score=score, labeller="AB")

    def test_a_score_without_a_labeller_is_refused(self):
        """A score needs a claimant - an unattributed label cannot be audited or
        excluded when one labeller turns out to have drifted."""
        with pytest.raises(ValidationError, match="must be set together"):
            answer_row(1, "db-x-001", "partial", human_score=3)

    def test_a_labeller_without_a_score_is_refused(self):
        with pytest.raises(ValidationError, match="must be set together"):
            answer_row(1, "db-x-001", "partial", labeller="AB")

    def test_the_second_pass_follows_the_same_rule(self):
        with pytest.raises(ValidationError, match="must be set together"):
            answer_row(1, "db-x-001", "partial", human_score_b=3)

    def test_a_concept_cannot_be_both_included_and_omitted(self):
        with pytest.raises(ValidationError, match="both included and omitted"):
            answer_row(
                1,
                "db-x-001",
                "partial",
                concepts_included=["concept_0"],
                concepts_omitted=["concept_0"],
            )

    def test_an_unknown_quality_level_is_refused(self):
        with pytest.raises(ValidationError):
            answer_row(1, "db-x-001", "brilliant")

    def test_a_labelled_row_reports_itself_as_labelled(self):
        row = answer_row(
            1, "db-x-001", "partial", human_score=2, labeller="AB", label_source="human"
        )
        assert row.is_labelled is True


class TestRoundTrip:
    def test_rows_survive_a_write_and_a_read(self, tmp_path):
        rows = [answer_row(n, "db-x-001", QUALITY_LEVELS[n % 5]) for n in range(1, 6)]
        path = tmp_path / "answers.jsonl"
        assert dump_answers(rows, path) == 5
        assert load_answers(path) == sorted(rows, key=lambda r: r.id)

    def test_the_file_is_written_in_a_stable_diffable_order(self, tmp_path):
        rows = [answer_row(n, "db-x-001", QUALITY_LEVELS[n % 5]) for n in (3, 1, 2)]
        path = tmp_path / "answers.jsonl"
        dump_answers(rows, path)
        first = path.read_text(encoding="utf-8")
        dump_answers(list(reversed(rows)), path)
        assert path.read_text(encoding="utf-8") == first

    def test_one_bad_row_fails_the_whole_read(self, tmp_path):
        """A dataset with one silently-skipped row is a dataset whose count
        nobody can trust, and the count is the first thing every report quotes."""
        path = tmp_path / "answers.jsonl"
        path.write_text(
            json.dumps(answer_row(1, "db-x-001", "good").model_dump(mode="json"))
            + '\n{"id": "eval-0002", "broken": true}\n',
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="line 2"):
            load_answers(path)


# ---------------------------------------------------------------------------
# Randomised labelling order
# ---------------------------------------------------------------------------


class TestLabellingOrder:
    def _rows(self) -> list[EvalAnswer]:
        rows = []
        n = 1
        for q in range(24):
            for level in QUALITY_LEVELS:
                rows.append(answer_row(n, f"db-x-{q:03d}", level))
                n += 1
        return rows

    def test_it_contains_every_answer_exactly_once(self):
        rows = self._rows()
        order = labelling_order(rows, seed=8021)
        assert sorted(order) == sorted(r.id for r in rows)

    def test_it_is_actually_shuffled(self):
        rows = self._rows()
        assert labelling_order(rows, seed=8021) != [r.id for r in rows]

    def test_it_is_reproducible_for_a_seed(self):
        rows = self._rows()
        assert labelling_order(rows, seed=8021) == labelling_order(rows, seed=8021)

    def test_a_different_seed_gives_a_different_order(self):
        rows = self._rows()
        assert labelling_order(rows, seed=8021) != labelling_order(rows, seed=1)

    def test_it_separates_the_five_answers_to_one_question(self):
        """The anchoring this exists to prevent: in file order the five answers
        to a question sit together, best first, and a labeller ranks them
        against each other instead of judging each on its own."""
        rows = self._rows()
        order = labelling_order(rows, seed=8021)
        by_id = {r.id: r for r in rows}
        adjacent_same_question = sum(
            1
            for a, b in zip(order, order[1:], strict=False)
            if by_id[a].question_id == by_id[b].question_id
        )
        # In file order this would be 96 of 119. Shuffled, it should be a
        # handful; the bound is loose because it is a property of chance.
        assert adjacent_same_question < 15

    def test_it_refuses_duplicate_ids(self):
        rows = self._rows()
        with pytest.raises(ValueError, match="duplicate"):
            labelling_order([*rows, rows[0]], seed=8021)


# ---------------------------------------------------------------------------
# Whole-dataset validation
# ---------------------------------------------------------------------------


class TestValidateDataset:
    def _full(self) -> tuple[list[EvalAnswer], dict[str, BankItem]]:
        questions = {f"db-x-{q:03d}": fake_item(f"db-x-{q:03d}") for q in range(24)}
        rows = []
        n = 1
        for qid, item in questions.items():
            for level in QUALITY_LEVELS:
                included, omitted = split_concepts(item, level)
                rows.append(
                    answer_row(
                        n,
                        qid,
                        level,
                        concepts_included=[c.key for c in included],
                        concepts_omitted=[c.key for c in omitted],
                    )
                )
                n += 1
        return rows, questions

    def test_a_correct_dataset_has_no_problems(self):
        rows, questions = self._full()
        assert validate_dataset(rows, questions=questions) == []

    def test_a_missing_answer_is_reported(self):
        rows, questions = self._full()
        assert validate_dataset(rows[:-1], questions=questions)

    def test_a_duplicate_id_is_reported(self):
        rows, questions = self._full()
        rows[1] = rows[1].model_copy(update={"id": rows[0].id})
        assert any("duplicate" in p for p in validate_dataset(rows, questions=questions))

    def test_a_missing_level_is_reported(self):
        rows, questions = self._full()
        rows[0] = rows[0].model_copy(update={"quality_intent": rows[1].quality_intent})
        problems = validate_dataset(rows, questions=questions)
        assert any("missing quality levels" in p for p in problems)
        assert any("more than one answer at levels" in p for p in problems)

    def test_an_unknown_question_is_reported(self):
        rows, questions = self._full()
        rows[0] = rows[0].model_copy(update={"question_id": "not-in-bank-001"})
        assert any("not in the bank" in p for p in validate_dataset(rows, questions=questions))

    def test_a_concept_split_that_does_not_match_the_question_is_reported(self):
        rows, questions = self._full()
        rows[0] = rows[0].model_copy(update={"concepts_included": ["invented_concept"]})
        assert any("does not match" in p for p in validate_dataset(rows, questions=questions))

    def test_unlabelled_rows_are_only_a_problem_when_labels_are_required(self):
        rows, questions = self._full()
        assert validate_dataset(rows, questions=questions) == []
        problems = validate_dataset(rows, questions=questions, require_labels=True)
        assert any("no human_score" in p for p in problems)

    def test_distributions_are_reported(self):
        rows, _ = self._full()
        assert level_distribution(rows) == dict.fromkeys(QUALITY_LEVELS, 24)
        assert label_distribution(rows) == dict.fromkeys(range(5), 0)


# ---------------------------------------------------------------------------
# The committed artefact
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not ANSWERS_PATH.exists(), reason="dataset not built in this checkout")
class TestTheCommittedDataset:
    """Assertions about the file on disk, not about the code that wrote it."""

    @pytest.fixture(scope="class")
    def dataset(self) -> list[EvalAnswer]:
        return load_answers(ANSWERS_PATH)

    @pytest.fixture(scope="class")
    def bank(self) -> dict[str, BankItem]:
        loaded, errors = load_bank()
        assert not errors
        return {li.item.id: li.item for li in loaded}

    def test_it_has_exactly_one_hundred_and_twenty_answers(self, dataset):
        assert len(dataset) == TARGET_ANSWERS

    def test_it_covers_exactly_twenty_four_questions(self, dataset):
        assert len({row.question_id for row in dataset}) == TARGET_QUESTIONS

    def test_every_question_has_all_five_levels(self, dataset):
        assert level_distribution(dataset) == dict.fromkeys(QUALITY_LEVELS, TARGET_QUESTIONS)

    def test_it_passes_every_whole_dataset_rule(self, dataset, bank):
        assert validate_dataset(dataset, questions=bank) == []

    def test_every_answer_is_prose_a_candidate_could_have_typed(self, dataset):
        """No markdown, no model preambles, no typographic characters that would
        break evidence-span verification."""
        offenders = {row.id: prose_problems(row.answer) for row in dataset}
        assert {k: v for k, v in offenders.items() if v} == {}

    def test_no_answer_is_a_duplicate_of_another(self, dataset):
        """Five answers to one question must be five different answers."""
        texts = [row.answer for row in dataset]
        assert len(set(texts)) == len(texts)

    def test_every_row_records_its_build_seed_and_generator(self, dataset):
        for row in dataset:
            assert row.build_seed is not None, row.id
            assert row.generator_model, row.id

    def test_the_generator_was_not_the_stub(self, dataset):
        """A dataset invented by the offline stub would be semantically
        meaningless, and `structured_mode` would have said so at build time."""
        for row in dataset:
            assert "stub" not in (row.generator_model or "").lower(), row.id

    def test_every_row_is_the_synthetic_slice(self, dataset):
        """Plan section 12.2 step 5's 25 real answers need pilot sessions, which
        do not exist before Phase 5. When they are added, this test changes."""
        assert {row.slice for row in dataset} == {"synthetic"}

    def test_the_committed_questions_file_matches_the_answers(self, dataset):
        assert QUESTIONS_PATH.exists()
        ids = {
            json.loads(line)["id"]
            for line in QUESTIONS_PATH.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        assert ids == {row.question_id for row in dataset}

    def test_the_labelling_order_is_reproducible_from_the_committed_file(self, dataset):
        """The seed is committed, so the exact order a labeller saw is
        recoverable a month later when someone asks whether drift explains a
        result."""
        first = labelling_order(dataset, seed=8021)
        second = labelling_order(load_answers(ANSWERS_PATH), seed=8021)
        assert first == second

    def test_no_row_carries_a_grader_score(self, dataset):
        """The dataset holds human labels and construction truth. A grader score
        in here would be the start of a circular evaluation."""
        forbidden = {"score", "grader_score", "predicted", "rating"}
        for row in dataset:
            assert not (forbidden & set(row.model_dump())), row.id


class TestOmittedConceptLeakage:
    """A measured weakness of the Day-18 set, pinned so it cannot drift silently.

    ``concepts_omitted`` is the precision ground truth for Day 19, and it is
    strong but not perfect: the generator was told to stay silent about those
    concepts and mostly was, but "say nothing about `repeatable_read`" is a hard
    instruction when the question is about isolation levels.
    """

    def test_a_named_omitted_concept_is_flagged(self):
        row = answer_row(
            1,
            "db-x-001",
            "partial",
            answer="I would think about concept 1 here, which matters a great deal.",
            concepts_included=["concept_0"],
            concepts_omitted=["concept_1", "concept_2"],
        )
        assert leaked_omitted_concepts(row) == ["concept_1"]

    def test_underscores_are_read_as_spaces(self):
        row = answer_row(
            1,
            "db-x-001",
            "partial",
            answer="You have to think about repeatable read semantics here, mostly.",
            concepts_included=["concept_0"],
            concepts_omitted=["repeatable_read"],
        )
        assert leaked_omitted_concepts(row) == ["repeatable_read"]

    def test_a_clean_answer_is_not_flagged(self):
        row = answer_row(
            1,
            "db-x-001",
            "partial",
            answer="A perfectly ordinary answer that names nothing it should not.",
            concepts_included=["concept_0"],
            concepts_omitted=["concept_1", "concept_2"],
        )
        assert leaked_omitted_concepts(row) == []

    def test_it_never_flags_an_included_concept(self):
        """It is a check on the *omitted* set; an included concept appearing is
        the entire point of including it."""
        row = answer_row(
            1,
            "db-x-001",
            "partial",
            answer="Here I discuss concept 0 at length and in detail, thoroughly.",
            concepts_included=["concept_0"],
            concepts_omitted=["concept_1"],
        )
        assert leaked_omitted_concepts(row) == []

    def test_the_report_totals_add_up(self):
        rows = [
            answer_row(
                1,
                "db-x-001",
                "partial",
                answer="This mentions concept 1 which was meant to be omitted here.",
                concepts_included=["concept_0"],
                concepts_omitted=["concept_1", "concept_2"],
            ),
            answer_row(
                2,
                "db-x-001",
                "good",
                answer="This one mentions nothing it should not, and is clean.",
                concepts_included=["concept_0"],
                concepts_omitted=["concept_1", "concept_2"],
            ),
        ]
        report = leakage_report(rows)
        assert report["omitted_slots"] == 4
        assert report["leaked_slots"] == 1
        assert report["answers_affected"] == 1
        assert report["leak_rate"] == pytest.approx(0.25)

    def test_an_empty_set_does_not_divide_by_zero(self):
        assert leakage_report([])["leak_rate"] == 0.0


@pytest.mark.skipif(not ANSWERS_PATH.exists(), reason="dataset not built in this checkout")
class TestTheCommittedDatasetsLeakage:
    def test_the_leak_rate_is_recorded_and_bounded(self):
        """At build time this was 21/192 = 10.9%.

        The bound is deliberately loose. The purpose is not to pass - it is to
        make the number visible, and to fail loudly if a rebuild ever makes the
        construction dramatically less controlled than the set Day 19 is
        reported against.
        """
        report = leakage_report(load_answers(ANSWERS_PATH))
        assert report["omitted_slots"] == 192
        assert report["leak_rate"] < 0.25, (
            f"omitted-concept leakage is {report['leak_rate']:.1%}; the construction "
            "is no longer controlled enough to serve as precision ground truth"
        )


class TestLabelProvenance:
    """``label_source`` is what keeps "labelled" and "gate-eligible" apart.

    The Phase-4 exit gate is QWK >= 0.65 against **human** labels. A QWK over
    model labels is a useful development signal and reads identically in a
    report, so the distinction has to be enforced in the data rather than
    remembered in prose.
    """

    def test_a_score_without_a_source_is_refused(self):
        with pytest.raises(ValidationError, match="label_source must be set together"):
            answer_row(1, "db-x-001", "partial", human_score=3, labeller="AB")

    def test_a_source_without_a_score_is_refused(self):
        with pytest.raises(ValidationError, match="label_source must be set together"):
            answer_row(1, "db-x-001", "partial", label_source="human")

    def test_the_second_pass_follows_the_same_rule(self):
        with pytest.raises(ValidationError, match="label_source_b must be set together"):
            answer_row(1, "db-x-001", "partial", human_score_b=3, labeller_b="CD")

    @pytest.mark.parametrize("source", ["human", "model"])
    def test_a_fully_attributed_label_validates(self, source):
        row = answer_row(
            1, "db-x-001", "partial", human_score=3, labeller="AB", label_source=source
        )
        assert row.is_labelled is True

    def test_only_a_human_label_is_human_labelled(self):
        model = answer_row(
            1, "db-x-001", "partial", human_score=3, labeller="m", label_source="model"
        )
        human = answer_row(
            2, "db-x-001", "good", human_score=3, labeller="AB", label_source="human"
        )
        assert model.is_labelled and not model.is_human_labelled
        assert human.is_labelled and human.is_human_labelled

    def test_an_unknown_source_is_refused(self):
        with pytest.raises(ValidationError):
            answer_row(
                1, "db-x-001", "partial", human_score=3, labeller="AB", label_source="guessed"
            )

    def test_gate_eligibility_needs_both_the_count_and_the_provenance(self):
        rows = [
            answer_row(
                n,
                f"db-x-{n:03d}",
                "good",
                human_score=3,
                labeller="m",
                label_source="model",
            )
            for n in range(1, TARGET_ANSWERS + 1)
        ]
        assert len(rows) == TARGET_ANSWERS
        assert gate_eligible(rows) is False, "model labels must never satisfy the gate"

        human = [r.model_copy(update={"label_source": "human"}) for r in rows]
        assert gate_eligible(human) is True
        assert gate_eligible(human[:-1]) is False, "119 labelled rows is not the 120-item set"

    def test_validate_dataset_can_demand_human_labels(self):
        rows = [
            answer_row(1, "db-x-001", "good", human_score=3, labeller="m", label_source="model")
        ]
        problems = validate_dataset(
            rows,
            questions={"db-x-001": fake_item("db-x-001")},
            expect_questions=1,
            expect_answers=1,
            require_human_labels=True,
        )
        assert any("not human-labelled" in p for p in problems)

    def test_label_sources_counts_every_row(self):
        rows = [
            answer_row(1, "db-x-001", "good", human_score=3, labeller="m", label_source="model"),
            answer_row(2, "db-x-001", "partial", human_score=2, labeller="A", label_source="human"),
            answer_row(3, "db-x-001", "superficial"),
        ]
        assert label_sources(rows) == {"human": 1, "model": 1, "unlabelled": 1}


class TestIntentAgreement:
    """Whether the generator obeyed its brief - answerable only from blind labels."""

    def test_it_reports_per_level_statistics(self):
        rows = [
            answer_row(
                1, "db-x-001", "excellent", human_score=4, labeller="m", label_source="model"
            ),
            answer_row(
                2, "db-x-001", "excellent", human_score=3, labeller="m", label_source="model"
            ),
            answer_row(3, "db-x-001", "good", human_score=3, labeller="m", label_source="model"),
        ]
        report = intent_agreement(rows)
        assert report["excellent"]["n"] == 2
        assert report["excellent"]["mean"] == pytest.approx(3.5)
        assert report["excellent"]["min"] == 3
        assert report["excellent"]["max"] == 4
        assert report["good"]["n"] == 1
        assert report["partial"]["n"] == 0

    def test_unlabelled_rows_are_excluded_rather_than_counted_as_zero(self):
        rows = [
            answer_row(1, "db-x-001", "good", human_score=3, labeller="m", label_source="model"),
            answer_row(2, "db-x-001", "good"),
        ]
        assert intent_agreement(rows)["good"]["n"] == 1

    def test_a_correct_wrong_confident_answer_is_flagged(self):
        rows = [
            answer_row(
                1,
                "db-x-001",
                "wrong_confident",
                human_score=3,
                labeller="m",
                label_source="model",
            ),
            answer_row(
                2,
                "db-x-001",
                "wrong_confident",
                human_score=0,
                labeller="m",
                label_source="model",
            ),
        ]
        assert unachieved_intent(rows) == ["eval-0001"]

    def test_other_levels_are_never_flagged(self):
        """It is a check on one brief, not a general quality complaint."""
        rows = [
            answer_row(n, "db-x-001", lvl, human_score=4, labeller="m", label_source="model")
            for n, lvl in enumerate(("excellent", "good", "partial", "superficial"), start=1)
        ]
        assert unachieved_intent(rows) == []


@pytest.mark.skipif(not ANSWERS_PATH.exists(), reason="dataset not built in this checkout")
class TestTheCommittedDatasetsLabels:
    @pytest.fixture(scope="class")
    def dataset(self) -> list[EvalAnswer]:
        return load_answers(ANSWERS_PATH)

    def test_every_answer_is_labelled(self, dataset):
        assert all(row.is_labelled for row in dataset)
        assert sum(1 for row in dataset if row.is_labelled) == TARGET_ANSWERS

    def test_every_label_is_attributed_and_sourced(self, dataset):
        for row in dataset:
            assert row.labeller, row.id
            assert row.label_source in ("human", "model"), row.id

    def test_the_set_is_not_yet_gate_eligible(self, dataset):
        """The Day-18 labels are model-assigned. Plan section 12.2 step 3 asks
        for human ones and the Phase-4 gate is written against them, so this
        must stay False until a human pass happens. If it ever goes green
        without a human labelling session, something has quietly relabelled the
        dataset.
        """
        assert gate_eligible(dataset) is False
        assert label_sources(dataset)["human"] == 0

    def test_the_construction_produced_a_real_quality_gradient(self, dataset):
        """The means must descend across the five levels in the designed order.

        This is the check that the dataset is a measuring instrument at all: if
        a blind labeller cannot see the gradient the construction intended,
        there is nothing for Day 19 to measure.
        """
        report = intent_agreement(dataset)
        means = [report[level]["mean"] for level in QUALITY_LEVELS]
        assert means == sorted(means, reverse=True), means
        assert means[0] - means[-1] > 0.75, "the gradient is too flat to be usable"

    def test_the_wrong_confident_level_largely_did_not_comply(self, dataset):
        """A measured defect, pinned so it cannot be forgotten.

        18 of 24 ``wrong_confident`` answers read as solid to a blind labeller:
        the generator was asked for at least two clear factual errors and mostly
        wrote a correct answer instead. The bound is loose and deliberately
        asserts the *defect*, not its absence - if a rebuild fixes it this test
        fails, and the documentation gets corrected rather than silently
        drifting out of date.
        """
        flagged = unachieved_intent(dataset)
        assert len(flagged) >= 12, (
            "wrong_confident compliance improved; re-measure and update the "
            "dataset README, which currently documents 18/24 as non-compliant"
        )

    def test_the_label_distribution_is_compressed_and_that_is_recorded(self, dataset):
        """Day 19 must know this before quoting a QWK.

        87% of the labels are 3 or 4 and no answer scored 1. A quadratically
        weighted agreement statistic over a distribution this narrow is weak and
        unstable, and that is a fact about the dataset rather than about any
        grader measured on it.
        """
        distribution = label_distribution(dataset)
        assert sum(distribution.values()) == TARGET_ANSWERS
        top_two = distribution[3] + distribution[4]
        assert top_two / TARGET_ANSWERS > 0.8, "the compression has changed; re-document it"
