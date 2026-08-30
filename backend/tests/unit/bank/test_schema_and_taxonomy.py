"""The rules an item must satisfy, and the rules the taxonomy must satisfy.

These are the constraints from plan section 6.3 written as executable
assertions. Each test names a way the dataset could go wrong, because a schema
test that only checks the happy path proves nothing about the schema.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

import pytest
from pydantic import ValidationError

from app.bank.schema import BankItem
from app.bank.taxonomy import TaxonomyError, parse_taxonomy

VALID_TAXONOMY: dict[str, Any] = {
    "domains": [
        {
            "key": "cs_fundamentals",
            "display_name": "CS Fundamentals",
            "topics": [
                {
                    "key": "dsa",
                    "display_name": "Data structures",
                    "subtopics": [{"key": "arrays", "display_name": "Arrays"}],
                }
            ],
        }
    ]
}


def item(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "id": "dsa-arrays-001",
        "topic": "dsa",
        "subtopic": "arrays",
        "text": "Explain how a two-pointer scan removes duplicates from a sorted array in place.",
        "difficulty_b": 0.5,
        "discrimination_a": 1.0,
        "expected_concepts": [
            {"key": "two_pointer_technique", "weight": 3, "hint": "read index and write index"},
            {"key": "time_complexity", "weight": 2, "hint": "one pass, so linear time"},
            {"key": "in_place_mutation", "weight": 1, "hint": "no second array is allocated"},
        ],
        "common_misconceptions": [],
        "reference_answer": "A strong answer keeps a write index and a read index. " * 6,
        "follow_up_seeds": [],
        "anchor_terms": ["two-pointer", "sorted array"],
        "time_estimate_s": 180,
        "tags": ["dsa"],
        "source": "llm_drafted",
        "review_status": "drafted",
        "reviewed_by": None,
        "reviewed_at": None,
        "version": 1,
    }
    base.update(overrides)
    return base


class TestTheItemSchema:
    def test_a_well_formed_item_validates(self):
        assert BankItem.model_validate(item()).id == "dsa-arrays-001"

    def test_fewer_than_three_concepts_is_rejected(self):
        """Plan 6.3: two concepts make the score 0/0.5/1, which is too coarse to be evidence."""
        concepts = item()["expected_concepts"][:2]
        with pytest.raises(ValidationError):
            BankItem.model_validate(item(expected_concepts=concepts))

    def test_more_than_six_concepts_is_rejected(self):
        """A ceiling nobody can reach in the time allowed stops discriminating."""
        concepts = [
            {"key": f"concept_{i}", "weight": 1, "hint": "an observable claim"} for i in range(7)
        ]
        with pytest.raises(ValidationError):
            BankItem.model_validate(item(expected_concepts=concepts))

    @pytest.mark.parametrize("b", [-3.5, 3.5])
    def test_difficulty_outside_the_theta_scale_is_rejected(self, b):
        with pytest.raises(ValidationError):
            BankItem.model_validate(item(difficulty_b=b))

    @pytest.mark.parametrize("b", [-3.0, 0.0, 3.0])
    def test_the_endpoints_of_the_range_are_allowed(self, b):
        assert BankItem.model_validate(item(difficulty_b=b)).difficulty_b == b

    def test_a_blank_reference_answer_is_rejected(self):
        with pytest.raises(ValidationError):
            BankItem.model_validate(item(reference_answer="   "))

    def test_a_weight_outside_one_to_three_is_rejected(self):
        """1-10 would be false precision that cannot be justified item by item."""
        concepts = item()["expected_concepts"]
        concepts[0] = {**concepts[0], "weight": 5}
        with pytest.raises(ValidationError):
            BankItem.model_validate(item(expected_concepts=concepts))

    def test_a_repeated_concept_key_is_rejected(self):
        """Otherwise one concept is scored twice and silently outweighs the others."""
        concepts = item()["expected_concepts"]
        concepts[1] = {**concepts[1], "key": concepts[0]["key"]}
        with pytest.raises(ValidationError):
            BankItem.model_validate(item(expected_concepts=concepts))

    def test_a_concept_key_that_is_not_snake_case_is_rejected(self):
        concepts = item()["expected_concepts"]
        concepts[0] = {**concepts[0], "key": "TwoPointer"}
        with pytest.raises(ValidationError):
            BankItem.model_validate(item(expected_concepts=concepts))

    def test_an_anchor_term_missing_from_the_text_is_rejected(self):
        """Plan 6.2: anchors guard a re-rendered question.

        An anchor that is not in the canonical text guards nothing.
        """
        with pytest.raises(ValidationError, match="anchor_terms"):
            BankItem.model_validate(item(anchor_terms=["red-black tree"]))

    def test_an_unknown_field_is_rejected_rather_than_dropped(self):
        """A typo in a hand-edited row must be loud, not a silently missing field."""
        with pytest.raises(ValidationError):
            BankItem.model_validate(item(expcted_concepts=[]))

    def test_an_unreadable_id_is_rejected(self):
        with pytest.raises(ValidationError, match="sys-cache-002"):
            BankItem.model_validate(item(id="7f3a19c2"))


class TestTheHonestyRule:
    """'Reviewed' is a claim about a human, and the schema will not let it be free."""

    def test_reviewed_without_a_reviewer_is_rejected(self):
        with pytest.raises(ValidationError, match="reviewed_by"):
            BankItem.model_validate(
                item(review_status="reviewed", reviewed_at=dt.date(2026, 8, 30))
            )

    def test_reviewed_without_a_date_is_rejected(self):
        with pytest.raises(ValidationError, match="reviewed_at"):
            BankItem.model_validate(item(review_status="reviewed", reviewed_by="manas"))

    def test_drafted_with_a_reviewer_is_rejected(self):
        """A half-finished review must not read as a finished one."""
        with pytest.raises(ValidationError):
            BankItem.model_validate(item(review_status="drafted", reviewed_by="manas"))

    def test_a_fully_reviewed_item_validates(self):
        parsed = BankItem.model_validate(
            item(
                source="authored",
                review_status="reviewed",
                reviewed_by="manas",
                reviewed_at=dt.date(2026, 8, 30),
            )
        )
        assert parsed.review_status == "reviewed"


class TestTheTaxonomy:
    def test_a_valid_taxonomy_parses_into_three_levels(self):
        tax = parse_taxonomy(VALID_TAXONOMY)
        assert [n.key for n in tax.domains()] == ["cs_fundamentals"]
        assert [n.key for n in tax.topics()] == ["dsa"]
        assert [n.key for n in tax.subtopics()] == ["arrays"]
        assert tax.domain_of("arrays") == "cs_fundamentals"
        assert tax.is_subtopic_of("arrays", "dsa")
        assert not tax.is_subtopic_of("arrays", "cs_fundamentals")

    def test_rows_are_ordered_parents_before_children(self):
        """The insert order `topics.parent_key` requires."""
        levels = [node.level for node in parse_taxonomy(VALID_TAXONOMY).rows()]
        assert levels == ["domain", "topic", "subtopic"]

    def test_a_key_reused_at_two_levels_is_rejected(self):
        """topics.key is one primary key across all three levels."""
        clashing = {
            "domains": [
                {
                    "key": "caching",
                    "display_name": "Caching",
                    "topics": [
                        {
                            "key": "caching",
                            "display_name": "Caching",
                            "subtopics": [{"key": "ttl", "display_name": "TTL"}],
                        }
                    ],
                }
            ]
        }
        with pytest.raises(TaxonomyError, match="defined twice"):
            parse_taxonomy(clashing)

    def test_a_topic_with_no_subtopics_is_rejected(self):
        """Theta is stored at subtopic level; a childless topic can never be measured."""
        empty = {
            "domains": [
                {
                    "key": "backend",
                    "display_name": "Backend",
                    "topics": [{"key": "databases", "display_name": "Databases", "subtopics": []}],
                }
            ]
        }
        with pytest.raises(TaxonomyError, match="no subtopics"):
            parse_taxonomy(empty)

    def test_a_key_that_is_not_snake_case_is_rejected(self):
        bad = {
            "domains": [
                {
                    "key": "CS Fundamentals",
                    "display_name": "CS Fundamentals",
                    "topics": [
                        {
                            "key": "dsa",
                            "display_name": "DSA",
                            "subtopics": [{"key": "arrays", "display_name": "Arrays"}],
                        }
                    ],
                }
            ]
        }
        with pytest.raises(TaxonomyError):
            parse_taxonomy(bad)
