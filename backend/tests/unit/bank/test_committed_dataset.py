"""The CI gate on the real, committed question bank.

Plan section 3, Phase 2 exit gate: *"reviewed items in data/question-bank/*.jsonl,
passing CI validation"*.  This file is that sentence, executable - it runs the
same ``validate_bank`` the ``validate_question_bank.py`` script does, so CI and
the command line cannot disagree about what "valid" means.

**What this file deliberately does not test.**  Whether a question is any good.
Schema validity is not content quality: a technically wrong question with three
plausible concept keys passes every assertion here. That judgement is the human
review recorded in ``review_status``, and ``test_review_status_is_honest``
exists to make sure the dataset never *claims* a review that did not happen.
"""

from __future__ import annotations

import pytest

from app.bank.loader import validate_bank
from app.bank.paths import BANK_DIR, TAXONOMY_PATH
from app.bank.schema import MAX_CONCEPTS, MIN_CONCEPTS
from app.bank.taxonomy import load_taxonomy


@pytest.fixture(scope="module")
def taxonomy():
    assert TAXONOMY_PATH.exists(), f"no taxonomy at {TAXONOMY_PATH}"
    return load_taxonomy()


@pytest.fixture(scope="module")
def report(taxonomy):
    return validate_bank(BANK_DIR, taxonomy)


class TestTheDatasetIsValid:
    def test_it_has_no_validation_errors(self, report):
        assert report.ok, "\n".join(report.errors[:20])

    def test_it_is_not_empty(self, report):
        assert report.count > 0
        assert report.files

    def test_every_item_has_between_three_and_six_concepts(self, report):
        for loaded in report.items:
            count = len(loaded.item.expected_concepts)
            assert MIN_CONCEPTS <= count <= MAX_CONCEPTS, f"{loaded.item.id} has {count}"

    def test_every_difficulty_is_on_the_theta_scale(self, report):
        for loaded in report.items:
            assert -3.0 <= loaded.item.difficulty_b <= 3.0, loaded.item.id

    def test_every_item_has_a_substantive_reference_answer(self, report):
        for loaded in report.items:
            assert loaded.item.reference_answer.strip(), loaded.item.id

    def test_ids_are_unique(self, report):
        ids = [loaded.item.id for loaded in report.items]
        assert len(ids) == len(set(ids))

    def test_every_taxonomy_reference_resolves(self, report, taxonomy):
        for loaded in report.items:
            assert taxonomy.is_subtopic_of(loaded.item.subtopic, loaded.item.topic), loaded.item.id


class TestTheDatasetIsHonest:
    def test_review_status_is_honest(self, report):
        """An item may only claim review if it names a reviewer and a date.

        The schema enforces this per item; this asserts it over the whole bank,
        because "150 reviewed items" is a gate someone will be tempted to meet
        with a search-and-replace.
        """
        for loaded in report.items:
            if loaded.item.review_status == "reviewed":
                assert loaded.item.reviewed_by, loaded.item.id
                assert loaded.item.reviewed_at is not None, loaded.item.id
            else:
                assert loaded.item.reviewed_by is None, loaded.item.id

    def test_llm_drafted_items_are_never_marked_reviewed_without_a_reviewer(self, report):
        drafted = [item.item for item in report.items if item.item.source == "llm_drafted"]
        for item in drafted:
            if item.review_status == "reviewed":
                assert item.reviewed_by, f"{item.id} claims review with no reviewer"


class TestCoverage:
    """Not correctness - shape. A bank concentrated in one place cannot drive a CAT."""

    def test_it_spans_more_than_one_domain(self, report):
        assert len(report.by_domain()) >= 2

    def test_difficulty_is_not_all_one_value(self, report):
        values = {loaded.item.difficulty_b for loaded in report.items}
        assert len(values) > 5, "item selection needs a spread of b to choose from"

    def test_no_subtopic_carries_more_than_a_quarter_of_the_bank(self, report):
        worst_key, worst_count = report.by_subtopic().most_common(1)[0]
        assert worst_count <= max(3, report.count // 4), f"{worst_key} dominates the bank"
