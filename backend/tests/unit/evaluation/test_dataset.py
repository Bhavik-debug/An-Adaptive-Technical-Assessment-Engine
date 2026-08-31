"""The evaluation dataset: its schema, and the committed file itself.

The committed-file checks matter more than they look. This dataset is the ruler
the whole retrieval system is measured with; a label pointing at a question id
that does not exist would make a query look like a permanent retrieval failure
that no amount of work could fix.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from app.bank.loader import validate_bank
from app.bank.taxonomy import load_taxonomy
from app.evaluation.dataset import (
    DATASET_PATH,
    EvalQuery,
    EvalQueryError,
    load_eval_dataset,
)


def query(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "id": "eq-001",
        "query": "how do database indexes work",
        "relevant": {"db-index-001": 2},
        "kind": "semantic",
        "note": "the canonical indexing question",
    }
    base.update(overrides)
    return base


def write(tmp_path, rows: list[dict[str, Any]], name: str = "queries.jsonl"):
    path = tmp_path / name
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return path


class TestTheSchema:
    def test_a_well_formed_query_validates(self):
        assert EvalQuery.model_validate(query()).id == "eq-001"

    def test_a_one_word_acronym_query_is_allowed(self):
        """'MVCC' is a real query someone types, and one of the hardest cases."""
        assert EvalQuery.model_validate(query(query="MVCC")).query == "MVCC"

    def test_a_query_with_nothing_relevant_is_rejected(self):
        """It cannot be scored, and it teaches nothing about the retriever."""
        with pytest.raises(ValueError, match="no question graded"):
            EvalQuery.model_validate(query(relevant={}))

    def test_a_query_whose_labels_are_all_zero_is_rejected(self):
        with pytest.raises(ValueError, match="no question graded"):
            EvalQuery.model_validate(query(relevant={"db-index-001": 0}))

    @pytest.mark.parametrize("grade", [-1, 3, 5])
    def test_a_grade_outside_zero_to_two_is_rejected(self, grade):
        """A 5 in a 0-2 dataset is a typo, not a strong opinion."""
        with pytest.raises(ValueError):
            EvalQuery.model_validate(query(relevant={"db-index-001": grade}))

    def test_an_unknown_kind_is_rejected(self):
        with pytest.raises(ValueError):
            EvalQuery.model_validate(query(kind="vibes"))

    def test_an_unknown_field_is_rejected_rather_than_dropped(self):
        with pytest.raises(ValueError):
            EvalQuery.model_validate(query(relevent={"db-index-001": 2}))

    def test_a_note_is_required(self):
        """An unexplained label is one nobody can argue with."""
        with pytest.raises(ValueError):
            EvalQuery.model_validate(query(note="short"))

    def test_relevant_ids_cuts_at_grade_one(self):
        parsed = EvalQuery.model_validate(query(relevant={"a": 2, "b": 1, "c": 0}))
        assert parsed.relevant_ids == {"a", "b"}

    def test_primary_ids_keeps_only_grade_two(self):
        parsed = EvalQuery.model_validate(query(relevant={"a": 2, "b": 1}))
        assert parsed.primary_ids == {"a"}


class TestLoading:
    def test_a_valid_file_loads(self, tmp_path):
        dataset = load_eval_dataset(write(tmp_path, [query()]))
        assert len(dataset) == 1

    def test_blank_lines_are_skipped(self, tmp_path):
        path = tmp_path / "q.jsonl"
        path.write_text(json.dumps(query()) + "\n\n\n", encoding="utf-8")
        assert len(load_eval_dataset(path)) == 1

    def test_a_missing_file_is_reported_clearly(self, tmp_path):
        with pytest.raises(EvalQueryError, match="no evaluation dataset"):
            load_eval_dataset(tmp_path / "absent.jsonl")

    def test_an_empty_file_is_rejected(self, tmp_path):
        path = tmp_path / "q.jsonl"
        path.write_text("\n", encoding="utf-8")
        with pytest.raises(EvalQueryError, match="no queries"):
            load_eval_dataset(path)

    def test_malformed_json_is_reported_with_its_line(self, tmp_path):
        path = tmp_path / "q.jsonl"
        path.write_text(json.dumps(query()) + "\n{nope}\n", encoding="utf-8")
        with pytest.raises(EvalQueryError, match="q.jsonl:2"):
            load_eval_dataset(path)

    def test_duplicate_query_ids_are_rejected(self, tmp_path):
        path = write(tmp_path, [query(), query()])
        with pytest.raises(EvalQueryError, match="duplicate query id"):
            load_eval_dataset(path)

    def test_every_error_is_collected_not_just_the_first(self, tmp_path):
        path = write(tmp_path, [query(id="eq-001", relevant={}), query(id="eq-002", kind="x")])
        with pytest.raises(EvalQueryError, match="2 problem"):
            load_eval_dataset(path)

    def test_a_label_pointing_at_an_unknown_question_is_rejected(self, tmp_path):
        """The check that matters most: an unreachable label is a permanent fake failure."""
        path = write(tmp_path, [query(relevant={"db-does-not-exist": 2})])
        with pytest.raises(EvalQueryError, match="not in the bank"):
            load_eval_dataset(path, known_question_ids={"db-index-001"})

    def test_labels_are_not_checked_when_no_bank_is_supplied(self, tmp_path):
        assert len(load_eval_dataset(write(tmp_path, [query()]))) == 1


class TestTheCommittedDataset:
    """The real ruler, checked against the real bank."""

    @pytest.fixture(scope="class")
    def question_ids(self):
        report = validate_bank(taxonomy=load_taxonomy())
        assert report.ok, "the question bank must validate before the eval set can be checked"
        return {loaded.item.id for loaded in report.items}

    @pytest.fixture(scope="class")
    def dataset(self, question_ids):
        return load_eval_dataset(known_question_ids=question_ids)

    def test_it_exists_and_loads(self, dataset):
        assert DATASET_PATH.exists()
        assert len(dataset) > 0

    def test_every_label_points_at_a_real_question(self, dataset, question_ids):
        """Enforced by the loader; asserted here so the failure names this test."""
        for item in dataset:
            assert set(item.relevant) <= question_ids, item.id

    def test_every_query_has_at_least_one_primary_answer(self, dataset):
        """A query with only partial matches has no clear right answer to measure."""
        for item in dataset:
            assert item.primary_ids, f"{item.id} has no grade-2 label"

    def test_query_ids_are_unique(self, dataset):
        ids = [item.id for item in dataset]
        assert len(ids) == len(set(ids))

    def test_query_texts_are_unique(self, dataset):
        """Two identical queries would double-weight whatever they measure."""
        texts = [item.query.lower() for item in dataset]
        assert len(texts) == len(set(texts))

    def test_it_covers_more_than_one_kind_of_retrieval_situation(self, dataset):
        """An all-semantic set would make the lexical arm look worthless by construction."""
        assert len(dataset.by_kind()) >= 4

    def test_it_covers_all_three_domains(self, dataset):
        prefixes = {qid.split("-")[0] for qid in dataset.labelled_question_ids}
        assert {"db", "dsa", "sys"} <= prefixes

    def test_it_exercises_a_large_share_of_the_bank(self, dataset, question_ids):
        """A set concentrated on ten questions would measure those ten, not the bank."""
        assert len(dataset.labelled_question_ids) >= len(question_ids) // 2
