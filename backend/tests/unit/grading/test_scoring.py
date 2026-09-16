"""Pass C, the arithmetic - plan section 7.2.

This is the module an examiner will actually interrogate, so the tests are
written to be read: the headline is the plan's own worked example, checked
intermediate by intermediate rather than only at the total, because a total can
be right for two compensating wrong reasons.

Everything here is pure. No provider, no event loop, no fixtures beyond data.
"""

from __future__ import annotations

import itertools

import pytest

from app.bank.schema import ExpectedConcept
from app.grading.rubric import MAX_LEVEL, RUBRIC_KEYS
from app.grading.schema import GradeJudgment
from app.grading.scoring import (
    CONCEPT_WEIGHT,
    COVER,
    MAJOR_ERROR_PENALTY,
    MAX_PENALTY,
    MINOR_ERROR_PENALTY,
    RUBRIC_WEIGHT,
    ScoringError,
    clamp,
    concept_coverage,
    error_penalty,
    rubric_blend,
    score_answer,
)

from .conftest import CONCEPTS, judgment_payload

# ---------------------------------------------------------------------------
# The plan's own worked example
# ---------------------------------------------------------------------------

#: Plan section 7.2's example, transcribed: weights 3/2/2/1, total 8.
PLAN_CONCEPTS = (
    ExpectedConcept(key="cache_invalidation", weight=3, hint="how a stale row is removed"),
    ExpectedConcept(key="stampede", weight=2, hint="the thundering-herd problem on a miss"),
    ExpectedConcept(key="consistency_model", weight=2, hint="what staleness is acceptable"),
    ExpectedConcept(key="eviction", weight=1, hint="an eviction policy and its bound"),
)

#: Plan section 7.2's Pass B output, transcribed verbatim.
PLAN_JUDGMENT = {
    "concepts": [
        {
            "key": "cache_invalidation",
            "label": "covered",
            "evidence": "I'd use write-through so the cache never serves a stale row",
        },
        {"key": "stampede", "label": "absent", "evidence": None},
        {
            "key": "consistency_model",
            "label": "partial",
            "evidence": "some staleness is probably fine",
        },
        {"key": "eviction", "label": "covered", "evidence": "LRU with a memory cap"},
    ],
    "errors": [{"claim": "Redis is ACID compliant", "severity": "major"}],
    "misconceptions_hit": [],
    "rubric": {"structure": 3, "tradeoff_awareness": 2, "specificity": 3, "communication": 3},
    "grader_confidence": 0.82,
}


class TestThePlansWorkedExample:
    """Plan section 7.2 works one example all the way through. If this class is
    green, the implementation is the plan's arithmetic and not something like
    it. Every intermediate is asserted, because a correct total built from two
    compensating errors is the bug this catches."""

    @pytest.fixture
    def breakdown(self):
        judgment = GradeJudgment.model_validate(PLAN_JUDGMENT)
        return score_answer(judgment, expected_concepts=PLAN_CONCEPTS)

    def test_concept_score(self, breakdown):
        """(3x1.0 + 2x0.0 + 2x0.5 + 1x1.0) / 8 = 5.0 / 8 = 0.625"""
        assert breakdown.concept_score == pytest.approx(0.625)
        assert breakdown.total_concept_weight == 8

    def test_rubric_score(self, breakdown):
        """(3 + 2 + 3 + 3) / 16 = 11/16 = 0.6875"""
        assert breakdown.rubric_score == pytest.approx(0.6875)

    def test_penalty(self, breakdown):
        """One major error: 0.15"""
        assert breakdown.penalty == pytest.approx(0.15)
        assert (breakdown.major_errors, breakdown.minor_errors) == (1, 0)

    def test_raw(self, breakdown):
        """0.65x0.625 + 0.35x0.6875 - 0.15 = 0.4969"""
        assert breakdown.raw == pytest.approx(0.4969, abs=5e-5)

    def test_final_score(self, breakdown):
        """The plan prints 0.50. The scorer does not round - it returns
        0.496875 - and 0.50 is that number displayed to two places.

        Rounding in the scorer would be a quiet loss of precision on the value
        that feeds `update_ability`, and the plan's own formula line says
        `score = clamp(raw, 0.0, 1.0)` with no rounding step. Display rounding
        belongs to the report (Day 25).
        """
        assert breakdown.score == pytest.approx(0.496875)
        assert round(breakdown.score, 2) == 0.50
        assert breakdown.was_clamped is False

    def test_the_gate_does_not_fire_on_it(self, breakdown):
        """Plan section 7.2: |0.625 - 0.6875| = 0.0625 < 0.25, confidence 0.82 > 0.6."""
        assert breakdown.signal_disagreement == pytest.approx(0.0625)
        assert breakdown.signal_disagreement < 0.25


# ---------------------------------------------------------------------------
# The constants are the plan's constants
# ---------------------------------------------------------------------------


class TestTheConstantsAreTranscribedNotInvented:
    def test_the_cover_map(self):
        assert COVER == {"covered": 1.0, "partial": 0.5, "absent": 0.0}

    def test_the_blend_weights_sum_to_one(self):
        assert (CONCEPT_WEIGHT, RUBRIC_WEIGHT) == (0.65, 0.35)
        assert CONCEPT_WEIGHT + RUBRIC_WEIGHT == pytest.approx(1.0)

    def test_concept_coverage_holds_the_majority_share(self):
        """Plan section 7.2: coverage is the objective signal, so it dominates."""
        assert CONCEPT_WEIGHT > RUBRIC_WEIGHT

    def test_the_penalty_constants(self):
        assert (MAJOR_ERROR_PENALTY, MINOR_ERROR_PENALTY, MAX_PENALTY) == (0.15, 0.05, 0.30)


# ---------------------------------------------------------------------------
# Concept coverage
# ---------------------------------------------------------------------------


class TestConceptCoverage:
    def _judgment(self, labels: dict[str, str]) -> GradeJudgment:
        return GradeJudgment.model_validate(
            judgment_payload(
                concepts=[
                    {
                        "key": key,
                        "label": label,
                        "evidence": None if label == "absent" else f"quote for {key}",
                    }
                    for key, label in labels.items()
                ]
            )
        )

    def test_all_covered_is_one(self):
        judgment = self._judgment(dict.fromkeys(("a_one", "b_two", "c_three"), "covered"))
        concepts = [
            ExpectedConcept(key=k, weight=w, hint="a hint that is long enough")
            for k, w in (("a_one", 3), ("b_two", 2), ("c_three", 1))
        ]
        assert concept_coverage(judgment, concepts) == (1.0, 6)

    def test_all_absent_is_zero(self):
        judgment = self._judgment(dict.fromkeys(("a_one", "b_two", "c_three"), "absent"))
        concepts = [
            ExpectedConcept(key=k, weight=1, hint="a hint that is long enough")
            for k in ("a_one", "b_two", "c_three")
        ]
        assert concept_coverage(judgment, concepts)[0] == 0.0

    def test_all_partial_is_exactly_one_half(self):
        judgment = self._judgment(dict.fromkeys(("a_one", "b_two", "c_three"), "partial"))
        concepts = [
            ExpectedConcept(key=k, weight=w, hint="a hint that is long enough")
            for k, w in (("a_one", 3), ("b_two", 2), ("c_three", 1))
        ]
        assert concept_coverage(judgment, concepts)[0] == 0.5

    def test_weights_actually_weight(self):
        """Covering the weight-3 concept must beat covering the weight-1 one."""
        concepts = [
            ExpectedConcept(key=k, weight=w, hint="a hint that is long enough")
            for k, w in (("a_one", 3), ("b_two", 2), ("c_three", 1))
        ]
        heavy = self._judgment({"a_one": "covered", "b_two": "absent", "c_three": "absent"})
        light = self._judgment({"a_one": "absent", "b_two": "absent", "c_three": "covered"})
        assert concept_coverage(heavy, concepts)[0] == pytest.approx(3 / 6)
        assert concept_coverage(light, concepts)[0] == pytest.approx(1 / 6)

    def test_partial_is_worth_exactly_half_a_covered(self):
        concepts = [
            ExpectedConcept(key=k, weight=2, hint="a hint that is long enough")
            for k in ("a_one", "b_two", "c_three")
        ]
        covered = self._judgment({"a_one": "covered", "b_two": "absent", "c_three": "absent"})
        partial = self._judgment({"a_one": "partial", "b_two": "absent", "c_three": "absent"})
        assert concept_coverage(partial, concepts)[0] == pytest.approx(
            concept_coverage(covered, concepts)[0] / 2
        )

    def test_a_judgment_about_other_concepts_is_refused(self):
        judgment = GradeJudgment.model_validate(judgment_payload())
        other = [
            ExpectedConcept(key=k, weight=1, hint="a hint that is long enough")
            for k in ("x_one", "y_two", "z_three")
        ]
        with pytest.raises(ScoringError, match="does not classify exactly"):
            concept_coverage(judgment, other)

    def test_duplicate_expected_keys_are_refused(self):
        judgment = GradeJudgment.model_validate(judgment_payload())
        duped = [CONCEPTS[0], CONCEPTS[0], CONCEPTS[1]]
        with pytest.raises(ScoringError, match="duplicate keys"):
            concept_coverage(judgment, duped)


# ---------------------------------------------------------------------------
# Rubric and penalty
# ---------------------------------------------------------------------------


class TestRubricNormalisation:
    def _with_rubric(self, **levels: int) -> GradeJudgment:
        return GradeJudgment.model_validate(judgment_payload(rubric=levels))

    def test_all_zero_is_zero(self):
        assert rubric_blend(self._with_rubric(**dict.fromkeys(RUBRIC_KEYS, 0))) == 0.0

    def test_all_max_is_one(self):
        assert rubric_blend(self._with_rubric(**dict.fromkeys(RUBRIC_KEYS, MAX_LEVEL))) == 1.0

    def test_all_mid_is_one_half(self):
        assert rubric_blend(self._with_rubric(**dict.fromkeys(RUBRIC_KEYS, 2))) == 0.5

    def test_the_denominator_is_max_level_times_dimension_count(self):
        judgment = self._with_rubric(
            structure=4, tradeoff_awareness=0, specificity=0, communication=0
        )
        assert rubric_blend(judgment) == pytest.approx(4 / (MAX_LEVEL * len(RUBRIC_KEYS)))

    def test_every_dimension_contributes_equally(self):
        """No dimension is secretly weighted; the plan blends them flat."""
        scores = []
        for key in RUBRIC_KEYS:
            levels = dict.fromkeys(RUBRIC_KEYS, 0)
            levels[key] = MAX_LEVEL
            scores.append(rubric_blend(self._with_rubric(**levels)))
        assert len(set(scores)) == 1


class TestErrorPenalty:
    def _with_errors(self, majors: int, minors: int) -> GradeJudgment:
        errors = [{"claim": f"major claim {i}", "severity": "major"} for i in range(majors)]
        errors += [{"claim": f"minor claim {i}", "severity": "minor"} for i in range(minors)]
        return GradeJudgment.model_validate(judgment_payload(errors=errors))

    def test_no_errors_is_no_penalty(self):
        assert error_penalty(self._with_errors(0, 0)) == (0.0, 0, 0)

    @pytest.mark.parametrize(
        ("majors", "minors", "expected"),
        [
            (1, 0, 0.15),
            (0, 1, 0.05),
            (1, 1, 0.20),
            (0, 3, 0.15),
            (2, 0, 0.30),
        ],
    )
    def test_the_arithmetic(self, majors, minors, expected):
        assert error_penalty(self._with_errors(majors, minors))[0] == pytest.approx(expected)

    @pytest.mark.parametrize(("majors", "minors"), [(3, 0), (2, 5), (0, 9), (5, 5)])
    def test_the_cap_holds(self, majors, minors):
        """The checklist, not the penalty, is meant to decide a score - so a long
        list of small mistakes cannot drive a good answer to zero."""
        assert error_penalty(self._with_errors(majors, minors))[0] == MAX_PENALTY

    def test_a_major_costs_three_times_a_minor(self):
        assert error_penalty(self._with_errors(1, 0))[0] == pytest.approx(
            3 * error_penalty(self._with_errors(0, 1))[0]
        )

    def test_the_counts_are_reported(self):
        assert error_penalty(self._with_errors(2, 1))[1:] == (2, 1)


# ---------------------------------------------------------------------------
# Clamping
# ---------------------------------------------------------------------------


class TestClamping:
    def test_clamp_is_the_identity_inside_the_range(self):
        for value in (0.0, 0.25, 0.5, 1.0):
            assert clamp(value) == value

    def test_clamp_bounds_both_ends(self):
        assert clamp(-3.0) == 0.0
        assert clamp(7.5) == 1.0

    def test_a_bad_answer_with_errors_cannot_go_negative(self):
        """All absent, rubric 0, two majors: raw = -0.30, score = 0.0."""
        judgment = GradeJudgment.model_validate(
            judgment_payload(
                concepts=[{"key": c.key, "label": "absent", "evidence": None} for c in CONCEPTS],
                errors=[
                    {"claim": "a major claim", "severity": "major"},
                    {"claim": "another major claim", "severity": "major"},
                ],
                rubric=dict.fromkeys(RUBRIC_KEYS, 0),
            )
        )
        breakdown = score_answer(judgment, expected_concepts=CONCEPTS)
        assert breakdown.raw == pytest.approx(-0.30)
        assert breakdown.score == 0.0
        assert breakdown.was_clamped is True

    def test_a_perfect_answer_is_exactly_one(self):
        judgment = GradeJudgment.model_validate(
            judgment_payload(
                concepts=[
                    {"key": c.key, "label": "covered", "evidence": f"quote for {c.key}"}
                    for c in CONCEPTS
                ],
                errors=[],
                rubric=dict.fromkeys(RUBRIC_KEYS, MAX_LEVEL),
            )
        )
        breakdown = score_answer(judgment, expected_concepts=CONCEPTS)
        assert breakdown.score == 1.0
        assert breakdown.was_clamped is False

    def test_the_score_is_always_a_legal_ability_input(self):
        """``app.ability.update_ability`` rejects a score outside [0, 1]; the
        clamp is what guarantees it never sees one."""
        for labels in itertools.product(("covered", "partial", "absent"), repeat=len(CONCEPTS)):
            judgment = GradeJudgment.model_validate(
                judgment_payload(
                    concepts=[
                        {
                            "key": c.key,
                            "label": label,
                            "evidence": None if label == "absent" else f"quote for {c.key}",
                        }
                        for c, label in zip(CONCEPTS, labels, strict=True)
                    ],
                    errors=[{"claim": "a major claim", "severity": "major"}],
                )
            )
            score = score_answer(judgment, expected_concepts=CONCEPTS).score
            assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# Determinism and order independence
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_the_same_inputs_give_the_same_breakdown(self):
        judgment = GradeJudgment.model_validate(PLAN_JUDGMENT)
        first = score_answer(judgment, expected_concepts=PLAN_CONCEPTS)
        second = score_answer(judgment, expected_concepts=PLAN_CONCEPTS)
        assert first == second

    def test_shuffling_the_judgments_concepts_changes_nothing(self):
        """Phase 4's position-bias gate, at the arithmetic.

        Not `approx`: **exactly** equal. The gate says the score must move by
        exactly 0, and it does because every product is a small integer weight
        times one of 1.0 / 0.5 / 0.0 - all exact in binary - so no summation
        order can differ by even one ulp.
        """
        base = score_answer(
            GradeJudgment.model_validate(PLAN_JUDGMENT), expected_concepts=PLAN_CONCEPTS
        )
        for permutation in itertools.permutations(PLAN_JUDGMENT["concepts"]):
            shuffled = GradeJudgment.model_validate(
                {**PLAN_JUDGMENT, "concepts": list(permutation)}
            )
            assert score_answer(shuffled, expected_concepts=PLAN_CONCEPTS).score == base.score

    def test_shuffling_the_expected_concepts_changes_nothing(self):
        base = score_answer(
            GradeJudgment.model_validate(PLAN_JUDGMENT), expected_concepts=PLAN_CONCEPTS
        )
        for permutation in itertools.permutations(PLAN_CONCEPTS):
            judgment = GradeJudgment.model_validate(PLAN_JUDGMENT)
            assert score_answer(judgment, expected_concepts=permutation).score == base.score

    def test_shuffling_the_errors_changes_nothing(self):
        payload = judgment_payload(
            errors=[
                {"claim": "a major claim", "severity": "major"},
                {"claim": "a minor claim", "severity": "minor"},
            ]
        )
        forward = score_answer(GradeJudgment.model_validate(payload), expected_concepts=CONCEPTS)
        reversed_payload = judgment_payload(errors=list(reversed(payload["errors"])))
        backward = score_answer(
            GradeJudgment.model_validate(reversed_payload), expected_concepts=CONCEPTS
        )
        assert forward.score == backward.score

    def test_scoring_does_not_mutate_its_arguments(self):
        judgment = GradeJudgment.model_validate(PLAN_JUDGMENT)
        before = judgment.model_dump()
        score_answer(judgment, expected_concepts=PLAN_CONCEPTS)
        assert judgment.model_dump() == before


class TestTheScorerIsIndependentOfTheModel:
    """Plan section 7.4 technique 3: score computed in code, never by the model.

    The strongest available statement of that: the scorer's signature contains
    no provider, no router, no task and no prompt, and its module imports
    nothing from ``app.llm``.
    """

    def test_the_scoring_module_imports_nothing_from_the_llm_layer(self):
        import ast
        from pathlib import Path

        import app.grading.scoring as scoring

        tree = ast.parse(Path(scoring.__file__).read_text(encoding="utf-8"))
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
        assert not any(name.startswith("app.llm") for name in imported), imported

    def test_a_score_field_on_the_judgment_could_not_reach_the_scorer_anyway(self):
        """Belt and braces: the contract refuses the key, and the scorer reads
        only `concepts`, `errors` and `rubric` - so even a hypothetical leak
        would be arithmetically inert."""
        from app.grading.schema import GradeJudgment as Contract

        assert "score" not in Contract.model_fields
