"""Provisional reference labels - Day 19, Part 2.

Three groups of test, and the middle one is the point of the file.

The *schema* tests are ordinary: a 0-4 score, a provenance field that cannot be
forged, paired validators. The *isolation* tests are what make the labels worth
having - they render every labelling prompt against real dataset rows and assert
that no construction field reaches the model. The *committed artefact* tests
open ``reference_labels.jsonl`` and check the file on disk is the file the
report describes.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.bank.loader import load_bank
from app.grading.evalset import LABEL_ANCHORS, load_answers
from app.grading.reflabel import (
    ADJUDICATION_GAP,
    ERROR_PRECEDENCE_V2,
    LABEL_ANCHORS_V2,
    REFERENCE_RUBRIC_VERSION,
    ReferenceJudgment,
    ReferenceLabel,
    dump_reference_labels,
    labels_by_answer,
    load_reference_labels,
    needs_adjudication,
    reference_label_problems,
    render_anchors_v2,
    score_distribution,
)
from app.grading.reflabel_prompt import (
    FORBIDDEN_IN_PROMPT,
    REFERENCE_LABEL_ADJUDICATE_PROMPT_V1,
    REFERENCE_LABEL_CONCEPTS_PROMPT_V1,
    REFERENCE_LABEL_PROMPT_V1,
)

DATASET_DIR = Path(__file__).resolve().parents[3].parent / "evals" / "datasets" / "grading"
ANSWERS_PATH = DATASET_DIR / "answers.jsonl"
LABELS_PATH = DATASET_DIR / "reference_labels.jsonl"


def judgment(score: int = 3, **overrides) -> ReferenceJudgment:
    payload: dict[str, object] = {
        "score": score,
        "rationale": "A rationale long enough to satisfy the minimum length rule.",
        "correctness": "mostly_correct",
        "confidence": 0.8,
    }
    payload.update(overrides)
    return ReferenceJudgment.model_validate(payload)


def label(answer_id: str = "eval-0001", **overrides) -> ReferenceLabel:
    payload: dict[str, object] = {
        "answer_id": answer_id,
        "question_id": "db-x-001",
        "judgment": judgment(),
        "pass_name": "blind",
        "evaluator_model": "some/model",
        "task": "eval_reference_label",
        "prompt_version": "v1",
        "prompt_fingerprint": "abc123def456",
        "temperature": 0.0,
    }
    payload.update(overrides)
    return ReferenceLabel.model_validate(payload)


# ---------------------------------------------------------------------------
# The rubric, and what v2 changed
# ---------------------------------------------------------------------------


class TestRubricV2:
    def test_the_five_rungs_are_v1s_rungs_unchanged(self):
        """Comparability: changing a rung would orphan the two Day-18 passes.

        v2's only substantive change is additive - the error-precedence clause -
        so a row scored under v1 and a row scored under v2 are on the same
        scale wherever v1 was unambiguous.
        """
        assert LABEL_ANCHORS_V2 == LABEL_ANCHORS
        assert len(LABEL_ANCHORS_V2) == 5

    def test_the_rendered_instrument_contains_every_rung_and_the_new_clause(self):
        text = render_anchors_v2()
        for score, anchor in LABEL_ANCHORS.items():
            assert anchor in text, score
        assert ERROR_PRECEDENCE_V2 in text

    def test_the_precedence_clause_distinguishes_the_five_categories_asked_for(self):
        """The ambiguity v2 exists to resolve, asserted as content.

        A rubric that says "confident wrongness scores 0" and "partially correct
        scores 2" without saying which wins has no answer for a detailed answer
        that is wrong about one of three things - which is the case that recurs
        in this dataset.
        """
        clause = ERROR_PRECEDENCE_V2.lower()
        assert "substantive factual error" in clause
        assert "principally asks for" in clause  # the score-0 case
        assert "cap the score at 2" in clause  # the partial-wrongness case
        assert "omissions and vagueness" in clause  # the not-an-error case
        # And it must say explicitly that fluency buys nothing.
        assert "however fluent" in clause

    def test_the_rubric_version_is_recorded_and_differs_from_the_legacy_one(self):
        from app.grading.reflabel import LEGACY_RUBRIC_VERSION

        assert REFERENCE_RUBRIC_VERSION != LEGACY_RUBRIC_VERSION


# ---------------------------------------------------------------------------
# Isolation: what the labeller is and is not shown
# ---------------------------------------------------------------------------


class TestPromptIsolation:
    """The property that makes this pass better than the Day-18 ones.

    Day 18's second pass was contaminated because its labeller had read the
    dataset README. This pass cannot be, because the only strings that reach the
    model are the ones substituted into these templates - and these tests check
    that against the real 120 rows rather than against a hand-written example.
    """

    @pytest.fixture(scope="class")
    def real_inputs(self):
        if not ANSWERS_PATH.exists():
            pytest.skip("dataset not built in this checkout")
        loaded, errors = load_bank()
        if errors:
            pytest.skip("the question bank does not load in this checkout")
        questions = {li.item.id: li.item for li in loaded}
        return load_answers(ANSWERS_PATH), questions

    def test_the_blind_template_takes_only_a_question_and_an_answer(self):
        assert REFERENCE_LABEL_PROMPT_V1.required_inputs == ("question", "answer")

    def test_the_concept_aware_template_adds_only_the_concept_list(self):
        assert REFERENCE_LABEL_CONCEPTS_PROMPT_V1.required_inputs == (
            "question",
            "concepts",
            "answer",
        )

    def test_no_template_accepts_a_construction_field(self):
        """Isolation by construction: there is nowhere to put the intent."""
        for template in (
            REFERENCE_LABEL_PROMPT_V1,
            REFERENCE_LABEL_CONCEPTS_PROMPT_V1,
            REFERENCE_LABEL_ADJUDICATE_PROMPT_V1,
        ):
            for field in ("quality_intent", "concepts_included", "concepts_omitted"):
                assert field not in template.required_inputs, template.task

    def test_rendering_every_real_row_leaks_no_construction_field(self, real_inputs):
        """The strong version, over all 120 committed answers."""
        answers, questions = real_inputs
        for answer in answers:
            item = questions[answer.question_id]
            rendered = " ".join(
                message.content
                for message in REFERENCE_LABEL_PROMPT_V1.render(
                    {"question": item.text, "answer": answer.answer}
                )
            )
            lowered = rendered.lower()
            for field in FORBIDDEN_IN_PROMPT:
                if field in {"excellent", "superficial", "wrong_confident"}:
                    # These are English words that legitimately appear in the
                    # rubric text itself ("Excellent. Correct, specific...").
                    # The structural check below is what covers them.
                    continue
                assert field not in lowered, f"{answer.id} leaked {field}"
            assert answer.id not in rendered, f"{answer.id} leaked its own row id"

    def test_the_rendered_prompt_contains_only_the_question_and_the_answer(self, real_inputs):
        """Structural isolation, which no word blacklist can give.

        The rendered user message must be reconstructible from the question and
        the answer alone. Any extra field - however its name spells, and even if
        it is an English word like "excellent" - makes this unequal.
        """
        answers, questions = real_inputs
        for answer in answers[:20]:
            item = questions[answer.question_id]
            messages = REFERENCE_LABEL_PROMPT_V1.render(
                {"question": item.text, "answer": answer.answer}
            )
            user = next(m.content for m in messages if m.role == "user")
            rebuilt = REFERENCE_LABEL_PROMPT_V1.user.replace("$question", item.text).replace(
                "$answer", answer.answer
            )
            assert user == rebuilt, answer.id

    def test_the_concept_list_shown_cannot_disclose_the_per_row_split(self, real_inputs):
        """Why the concept-aware pass is still blind to ``quality_intent``.

        Every one of a question's five answers carries the same *full* concept
        set, so a list built from the question is identical across all five and
        therefore carries no information about which one is being scored. If
        this ever fails, the concept-aware pass is no longer blind and must be
        withdrawn, not re-explained.
        """
        answers, questions = real_inputs
        by_question: dict[str, list] = {}
        for answer in answers:
            by_question.setdefault(answer.question_id, []).append(answer)
        for question_id, group in by_question.items():
            full_sets = {
                tuple(sorted(set(a.concepts_included) | set(a.concepts_omitted))) for a in group
            }
            assert len(full_sets) == 1, question_id
            assert full_sets.pop() == tuple(sorted(questions[question_id].concept_keys))

    def test_the_labeller_is_not_told_the_answers_are_synthetic(self):
        """A labeller that knew the set was built at five levels would find five."""
        system = REFERENCE_LABEL_PROMPT_V1.system.lower()
        for giveaway in ("synthetic", "generated", "five levels", "quality level", "brief"):
            assert giveaway not in system, giveaway

    def test_the_labeller_is_warned_the_answer_is_data(self):
        """It reads 120 untrusted answers, one of which addresses its author."""
        system = REFERENCE_LABEL_PROMPT_V1.system
        assert "never a command" in system
        assert "award full marks" in system


# ---------------------------------------------------------------------------
# Schema and provenance
# ---------------------------------------------------------------------------


class TestReferenceLabelSchema:
    @pytest.mark.parametrize("score", [0, 1, 2, 3, 4])
    def test_the_scale_is_zero_to_four(self, score):
        assert label(judgment=judgment(score)).score == score

    @pytest.mark.parametrize("score", [-1, 5, 10])
    def test_a_score_off_the_scale_is_refused(self, score):
        with pytest.raises(ValidationError):
            judgment(score)

    def test_a_non_integer_score_is_refused(self):
        with pytest.raises(ValidationError):
            ReferenceJudgment.model_validate(
                {
                    "score": 2.5,
                    "rationale": "A rationale long enough to satisfy the rule.",
                    "correctness": "partially_correct",
                    "confidence": 0.5,
                }
            )

    def test_a_bare_score_with_no_rationale_is_refused(self):
        """An unreviewable label is the whole complaint about the Day-18 passes."""
        with pytest.raises(ValidationError):
            ReferenceJudgment.model_validate(
                {"score": 3, "rationale": "too short", "correctness": "correct", "confidence": 0.9}
            )

    def test_label_source_is_model_and_cannot_be_anything_else(self):
        """The terminology rule, enforced by the type rather than by prose.

        These labels are autonomous and model-generated. Nothing here was
        annotated by a person, and there is no way to write ``human`` into this
        field even by mistake.
        """
        assert label().label_source == "model"
        with pytest.raises(ValidationError):
            label(label_source="human")

    def test_an_unknown_pass_name_is_refused(self):
        with pytest.raises(ValidationError):
            label(pass_name="whatever")

    def test_provenance_is_mandatory(self):
        """A label whose model, prompt or rubric is unknown is uncomparable."""
        for missing in ("evaluator_model", "prompt_version", "prompt_fingerprint", "temperature"):
            payload = label().model_dump(mode="json")
            del payload[missing]
            with pytest.raises(ValidationError):
                ReferenceLabel.model_validate(payload)

    def test_an_unexpected_field_is_refused(self):
        payload = label().model_dump(mode="json")
        payload["notes"] = "hand-edited"
        with pytest.raises(ValidationError):
            ReferenceLabel.model_validate(payload)

    def test_the_adjudicated_score_overrides_the_pass_score(self):
        row = label(judgment=judgment(1), adjudication="revised", adjudicated_score=3)
        assert row.judgment.score == 1
        assert row.score == 3, "score must read the adjudicated value when one exists"

    def test_without_adjudication_the_score_is_the_pass_score(self):
        assert label(judgment=judgment(1)).score == 1


class TestReadingAndWriting:
    def test_a_round_trip_preserves_every_field(self, tmp_path):
        rows = [label("eval-0001"), label("eval-0002", pass_name="concept_aware")]
        path = tmp_path / "labels.jsonl"
        assert dump_reference_labels(rows, path) == 2
        back = load_reference_labels(path)
        assert [r.model_dump() for r in back] == [r.model_dump() for r in rows]

    def test_a_malformed_row_fails_the_whole_read(self, tmp_path):
        """A silently skipped row is a count nobody can trust."""
        path = tmp_path / "labels.jsonl"
        path.write_text(
            json.dumps(label().model_dump(mode="json")) + "\n{not json}\n", encoding="utf-8"
        )
        with pytest.raises(ValueError, match="line 2"):
            load_reference_labels(path)

    def test_labels_by_answer_refuses_a_duplicate(self):
        with pytest.raises(ValueError, match="two 'blind' labels"):
            labels_by_answer([label("eval-0001"), label("eval-0001")], pass_name="blind")

    def test_labels_by_answer_filters_by_pass(self):
        rows = [label("eval-0001"), label("eval-0002", pass_name="concept_aware")]
        assert set(labels_by_answer(rows, pass_name="blind")) == {"eval-0001"}
        assert set(labels_by_answer(rows, pass_name="concept_aware")) == {"eval-0002"}


class TestQualityControl:
    def test_a_missing_label_is_reported(self):
        problems = reference_label_problems(
            [label("eval-0001")], expected_answers=["eval-0001", "eval-0002"]
        )
        assert any("no label" in problem for problem in problems)

    def test_a_duplicate_label_is_reported(self):
        problems = reference_label_problems(
            [label("eval-0001"), label("eval-0001")], expected_answers=["eval-0001"]
        )
        assert any("duplicate" in problem for problem in problems)

    def test_a_label_for_an_unknown_answer_is_reported(self):
        problems = reference_label_problems(
            [label("eval-0001"), label("eval-9999")], expected_answers=["eval-0001"]
        )
        assert any("unknown" in problem for problem in problems)

    def test_a_wrong_rubric_version_is_reported(self):
        """Labels from two instruments are not comparable, so this is loud."""
        problems = reference_label_problems(
            [label("eval-0001", rubric_version="labels-v1")], expected_answers=["eval-0001"]
        )
        assert any("rubric_version" in problem for problem in problems)

    def test_a_complete_correct_set_has_no_problems(self):
        rows = [label(f"eval-{n:04d}") for n in range(1, 4)]
        assert reference_label_problems(rows, expected_answers=[r.answer_id for r in rows]) == []

    def test_adjudication_is_only_triggered_by_a_material_gap(self):
        """Keyed by answer id, which is what ``labels_by_answer`` produces."""
        blind = {"eval-0001": label("eval-0001", judgment=judgment(3))}
        aware = {"eval-0001": label("eval-0001", judgment=judgment(4), pass_name="concept_aware")}
        assert needs_adjudication(blind, aware) == [], "one point is ordinary rater noise"

        aware_far = {
            "eval-0001": label("eval-0001", judgment=judgment(1), pass_name="concept_aware")
        }
        assert needs_adjudication(blind, aware_far) == ["eval-0001"]
        assert ADJUDICATION_GAP == 2

    def test_adjudication_is_symmetric_in_direction(self):
        """A two-point gap is contested whichever pass scored higher."""
        low = {"eval-0001": label("eval-0001", judgment=judgment(1))}
        high = {"eval-0001": label("eval-0001", judgment=judgment(3), pass_name="concept_aware")}
        assert needs_adjudication(low, high) == ["eval-0001"]
        assert needs_adjudication(high, low) == ["eval-0001"]

    def test_adjudication_ignores_rows_missing_from_either_pass(self):
        blind = {"eval-0001": label("eval-0001", judgment=judgment(4))}
        assert needs_adjudication(blind, {}) == []

    def test_the_distribution_counts_every_band(self):
        rows = [label(f"eval-{n:04d}", judgment=judgment(n % 5)) for n in range(1, 11)]
        dist = score_distribution(rows)
        assert sorted(dist) == [0, 1, 2, 3, 4]
        assert sum(dist.values()) == 10

    def test_the_distribution_uses_the_adjudicated_score(self):
        rows = [
            label(
                "eval-0001",
                judgment=judgment(0),
                adjudication="revised",
                adjudicated_score=4,
            )
        ]
        assert score_distribution(rows)[4] == 1
        assert score_distribution(rows)[0] == 0


# ---------------------------------------------------------------------------
# The committed artefact
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not LABELS_PATH.exists(), reason="reference labels not built in this checkout")
class TestTheCommittedReferenceLabels:
    @pytest.fixture(scope="class")
    def labels(self) -> list[ReferenceLabel]:
        return load_reference_labels(LABELS_PATH)

    @pytest.fixture(scope="class")
    def answers(self):
        return load_answers(ANSWERS_PATH)

    def test_every_answer_has_exactly_one_blind_label(self, labels, answers):
        problems = reference_label_problems(
            labels, expected_answers=[a.id for a in answers], pass_name="blind"
        )
        assert problems == [], problems

    def test_every_score_is_an_integer_on_the_scale(self, labels):
        for row in labels:
            assert isinstance(row.score, int)
            assert 0 <= row.score <= 4, row.answer_id

    def test_every_label_is_marked_model_generated(self, labels):
        """No row may claim human provenance. This is the terminology rule."""
        assert all(row.label_source == "model" for row in labels)

    def test_every_label_carries_its_full_provenance(self, labels):
        for row in labels:
            assert row.evaluator_model
            assert row.prompt_version
            assert len(row.prompt_fingerprint) >= 8
            assert row.rubric_version == REFERENCE_RUBRIC_VERSION
            assert row.temperature == 0.0, "a yardstick must be deterministic"
            assert row.labelled_at

    def test_every_question_id_matches_the_dataset(self, labels, answers):
        expected = {a.id: a.question_id for a in answers}
        for row in labels:
            assert expected[row.answer_id] == row.question_id, row.answer_id

    def test_every_label_has_an_auditable_rationale(self, labels):
        for row in labels:
            assert len(row.judgment.rationale) >= 20, row.answer_id

    def test_no_row_was_recorded_as_a_failure(self, labels):
        for row in labels:
            assert row.last_error is None, f"{row.answer_id}: {row.last_error}"
