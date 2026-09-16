"""The grading contract - plan section 7.2, Pass B.

The tests that matter most here are the *negative* ones. A schema that accepts
good input is ordinary; this schema exists to reject a specific list of things,
and each rejection is a defence that plan section 7.1 or 7.4 asks for by name.
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from app.bank.schema import CONCEPT_KEY_RE
from app.grading.rubric import RUBRIC_KEYS
from app.grading.schema import (
    MAX_ERRORS,
    MAX_EVIDENCE_CHARS,
    ConceptJudgment,
    GradeJudgment,
    RubricScores,
)
from app.llm.structured import schema_spec

from .conftest import judgment_payload


class TestTheModelCannotEmitAScore:
    """Plan section 7.1: *the fix is to stop asking the model for a number.*

    These four tests are the architectural claim of the whole phase, expressed
    as assertions. If any of them ever goes red, the separation the project is
    built on has been lost.
    """

    def test_the_contract_has_no_score_field(self):
        forbidden = {"score", "rating", "grade", "points", "overall", "percentage", "mark"}
        assert forbidden.isdisjoint(GradeJudgment.model_fields)

    @pytest.mark.parametrize("field", ["score", "rating", "overall_score", "grade"])
    def test_a_volunteered_score_is_rejected_not_ignored(self, field):
        """Not stripped, not ignored - rejected.

        Silently dropping an extra key would be worse than it sounds: the model
        would have spent its attention deciding a number, and we would never
        know it had. ``extra='forbid'`` turns that into a validation failure the
        chokepoint reports and retries.
        """
        payload = judgment_payload()
        payload[field] = 0.9
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            GradeJudgment.model_validate(payload)

    def test_the_provider_facing_schema_also_closes_the_object(self):
        """Belt and braces: on a provider that constrains decoding, the extra
        key cannot even be generated, so the retry never has to happen."""
        spec = schema_spec(GradeJudgment)
        assert spec.schema["additionalProperties"] is False
        for definition in spec.schema["$defs"].values():
            assert definition["additionalProperties"] is False

    def test_no_field_invites_hidden_reasoning(self):
        """Plan section 7: evidence is a quotation, not the model's thinking.

        Nothing in the contract asks the model to explain itself, so nothing
        stores a chain of thought - which is both a cost decision and a
        retention one.
        """
        reasoning_like = {"reasoning", "rationale", "explanation", "thinking", "analysis", "notes"}
        assert reasoning_like.isdisjoint(GradeJudgment.model_fields)
        assert reasoning_like.isdisjoint(ConceptJudgment.model_fields)


class TestEvidenceIsStructurallyRequired:
    """Plan section 7.2: *the model cannot claim a concept is covered without
    pointing at specific text.* Day 17 checks the span is real; this is the
    half that makes an empty claim unrepresentable."""

    @pytest.mark.parametrize("label", ["covered", "partial"])
    def test_credit_without_a_citation_is_refused(self, label):
        with pytest.raises(ValidationError, match="no evidence span"):
            ConceptJudgment.model_validate({"key": "eviction", "label": label, "evidence": None})

    @pytest.mark.parametrize("label", ["covered", "partial"])
    def test_a_blank_citation_does_not_count_as_one(self, label):
        """Whitespace is the obvious way round a null check, so it is closed."""
        with pytest.raises(ValidationError, match="no evidence span"):
            ConceptJudgment.model_validate({"key": "eviction", "label": label, "evidence": "   "})

    def test_an_absent_concept_may_not_carry_a_citation(self):
        """A quote attached to 'absent' is a contradiction, and it would be a
        span Day 17 then has to decide what to do with. Refuse it here."""
        with pytest.raises(ValidationError, match="labelled 'absent' but carries"):
            ConceptJudgment.model_validate(
                {"key": "eviction", "label": "absent", "evidence": "LRU"}
            )

    def test_absent_with_null_is_the_valid_shape(self):
        judgment = ConceptJudgment.model_validate(
            {"key": "eviction", "label": "absent", "evidence": None}
        )
        assert judgment.evidence is None

    def test_an_evidence_span_is_a_quotation_not_an_essay(self):
        payload = judgment_payload()
        payload["concepts"][0]["evidence"] = "x" * (MAX_EVIDENCE_CHARS + 1)
        with pytest.raises(ValidationError):
            GradeJudgment.model_validate(payload)


class TestTheJudgmentIsWellFormed:
    def test_a_valid_judgment_validates(self):
        judgment = GradeJudgment.model_validate(judgment_payload())
        assert judgment.labels_by_key == {
            "cache_invalidation": "covered",
            "eviction": "covered",
            "stampede": "partial",
        }
        assert judgment.grader_confidence == pytest.approx(0.82)

    @pytest.mark.parametrize("label", ["Covered", "unknown", "yes", "", "COVERED"])
    def test_a_label_outside_the_three_is_refused(self, label):
        payload = judgment_payload()
        payload["concepts"][0]["label"] = label
        with pytest.raises(ValidationError):
            GradeJudgment.model_validate(payload)

    def test_a_repeated_concept_key_is_refused(self):
        """Two votes about one concept, and whichever is read last wins."""
        payload = judgment_payload()
        payload["concepts"][1]["key"] = payload["concepts"][0]["key"]
        with pytest.raises(ValidationError, match="repeats"):
            GradeJudgment.model_validate(payload)

    @pytest.mark.parametrize("key", ["Cache Invalidation", "cache-invalidation", "_leading"])
    def test_a_key_that_could_not_be_a_bank_key_is_refused(self, key):
        """The grammar is enforced as a schema ``pattern``, not as a validator,
        so it also reaches the provider-facing JSON Schema - see the comment on
        ``ConceptJudgment.key``."""
        payload = judgment_payload()
        payload["concepts"][0]["key"] = key
        with pytest.raises(ValidationError, match="string_pattern_mismatch"):
            GradeJudgment.model_validate(payload)

    def test_the_key_grammar_reaches_the_provider_facing_schema(self):
        key_schema = schema_spec(GradeJudgment).schema["$defs"]["ConceptJudgment"]["properties"][
            "key"
        ]
        assert key_schema["pattern"] == CONCEPT_KEY_RE.pattern

    @pytest.mark.parametrize("confidence", [-0.1, 1.1, 2.0])
    def test_confidence_outside_zero_to_one_is_refused(self, confidence):
        """``ability.k_factor`` raises on a confidence outside [0, 1], so a
        value that got past here would blow up inside the Day-11 estimator
        instead - two layers away from the thing that produced it."""
        with pytest.raises(ValidationError):
            GradeJudgment.model_validate(judgment_payload(grader_confidence=confidence))

    def test_a_missing_required_field_is_refused(self):
        payload = judgment_payload()
        del payload["grader_confidence"]
        with pytest.raises(ValidationError, match="grader_confidence"):
            GradeJudgment.model_validate(payload)

    def test_the_concept_list_is_bounded_like_the_bank_is(self):
        """A bank item carries 3-6 concepts, so a judgment about one cannot
        legally carry 0 or 20."""
        with pytest.raises(ValidationError):
            GradeJudgment.model_validate(judgment_payload(concepts=[]))

    def test_the_error_list_is_bounded(self):
        errors = [{"claim": f"wrong claim number {i}", "severity": "minor"} for i in range(20)]
        with pytest.raises(ValidationError):
            GradeJudgment.model_validate(judgment_payload(errors=errors))
        fine = errors[:MAX_ERRORS]
        assert len(GradeJudgment.model_validate(judgment_payload(errors=fine)).errors) == MAX_ERRORS

    def test_a_judgment_is_frozen(self):
        """Nothing downstream may edit a judgment in place: a grade that was
        mutated after validation is a grade whose trace no longer describes it."""
        judgment = GradeJudgment.model_validate(judgment_payload())
        with pytest.raises(ValidationError):
            judgment.grader_confidence = 1.0


class TestTheRubricMatchesTheRubric:
    def test_the_contract_names_exactly_the_four_anchored_dimensions(self):
        """The prompt and the schema are written in two different files; this
        is the test that stops them drifting apart."""
        assert tuple(RubricScores.model_fields) == RUBRIC_KEYS

    @pytest.mark.parametrize("level", [-1, 5, 10])
    def test_a_level_off_the_anchored_scale_is_refused(self, level):
        payload = judgment_payload()
        payload["rubric"]["structure"] = level
        with pytest.raises(ValidationError):
            GradeJudgment.model_validate(payload)

    @pytest.mark.parametrize("level", [0, 4])
    def test_both_ends_of_the_scale_are_reachable(self, level):
        """0 and 4 are real classifications, not reserved values - a scale whose
        ends nobody may use is a three-point scale wearing a five-point label."""
        payload = judgment_payload()
        payload["rubric"] = dict.fromkeys(RUBRIC_KEYS, level)
        assert GradeJudgment.model_validate(payload).rubric.as_mapping() == dict.fromkeys(
            RUBRIC_KEYS, level
        )

    def test_a_missing_dimension_is_refused_rather_than_defaulted(self):
        """A defaulted dimension would silently score as whatever the default
        was, and the Day-19 QWK number would be measuring that default."""
        payload = judgment_payload()
        del payload["rubric"]["specificity"]
        with pytest.raises(ValidationError, match="specificity"):
            GradeJudgment.model_validate(payload)

    def test_as_mapping_is_what_day_17_will_divide(self):
        rubric = GradeJudgment.model_validate(judgment_payload()).rubric
        assert rubric.as_mapping() == {
            "structure": 3,
            "tradeoff_awareness": 2,
            "specificity": 3,
            "communication": 3,
        }
        assert len(rubric.as_mapping()) == len(RUBRIC_KEYS)


class TestOrderIndependence:
    """Phase 4's exit gate: *shuffling ``expected_concepts`` changes the score by
    exactly 0.* The scorer is Day 17, but the property is won here - by the
    judgment meaning a mapping rather than a sequence."""

    def test_two_judgments_differing_only_in_order_canonicalise_equal(self):
        forward = GradeJudgment.model_validate(judgment_payload())
        payload = judgment_payload()
        payload["concepts"] = list(reversed(payload["concepts"]))
        backward = GradeJudgment.model_validate(payload)

        assert forward != backward
        assert forward.canonical() == backward.canonical()

    def test_canonical_judgments_serialise_to_identical_bytes(self):
        payload = judgment_payload()
        shuffled = judgment_payload(concepts=list(reversed(payload["concepts"])))
        a = GradeJudgment.model_validate(payload).canonical()
        b = GradeJudgment.model_validate(shuffled).canonical()
        assert json.dumps(a.model_dump(mode="json"), sort_keys=True) == json.dumps(
            b.model_dump(mode="json"), sort_keys=True
        )

    def test_labels_by_key_does_not_depend_on_order(self):
        payload = judgment_payload()
        shuffled = judgment_payload(concepts=list(reversed(payload["concepts"])))
        assert (
            GradeJudgment.model_validate(payload).labels_by_key
            == GradeJudgment.model_validate(shuffled).labels_by_key
        )

    def test_canonical_sorts_errors_and_misconceptions_too(self):
        payload = judgment_payload(
            errors=[
                {"claim": "zebra claim", "severity": "minor"},
                {"claim": "alpha claim", "severity": "minor"},
            ],
            misconceptions_hit=["zzz misconception", "aaa misconception"],
        )
        canonical = GradeJudgment.model_validate(payload).canonical()
        assert [e.claim for e in canonical.errors] == ["alpha claim", "zebra claim"]
        assert canonical.misconceptions_hit == ["aaa misconception", "zzz misconception"]

    def test_canonical_changes_nothing_but_order(self):
        judgment = GradeJudgment.model_validate(judgment_payload())
        canonical = judgment.canonical()
        assert canonical.labels_by_key == judgment.labels_by_key
        assert canonical.evidence_by_key == judgment.evidence_by_key
        assert canonical.rubric == judgment.rubric
        assert canonical.grader_confidence == judgment.grader_confidence
