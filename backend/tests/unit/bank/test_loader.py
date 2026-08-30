"""The dataset-level checks: the ones no single item can fail on its own.

Every test writes a small bank to a temporary directory rather than asserting
against the real one, so a change to the committed dataset cannot make a
validator test pass or fail for the wrong reason.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from app.bank.loader import load_bank, validate_bank
from app.bank.taxonomy import Taxonomy, parse_taxonomy

TAXONOMY: dict[str, Any] = {
    "domains": [
        {
            "key": "cs_fundamentals",
            "display_name": "CS Fundamentals",
            "topics": [
                {
                    "key": "dsa",
                    "display_name": "Data structures",
                    "subtopics": [
                        {"key": "arrays", "display_name": "Arrays"},
                        {"key": "graphs", "display_name": "Graphs"},
                    ],
                }
            ],
        },
        {
            "key": "backend",
            "display_name": "Backend",
            "topics": [
                {
                    "key": "databases",
                    "display_name": "Databases",
                    "subtopics": [{"key": "indexing", "display_name": "Indexing"}],
                }
            ],
        },
    ]
}


@pytest.fixture
def taxonomy() -> Taxonomy:
    return parse_taxonomy(TAXONOMY)


def make(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "id": "dsa-arrays-001",
        "topic": "dsa",
        "subtopic": "arrays",
        "text": (
            "Describe how a two-pointer scan removes duplicates from a sorted array in place, "
            "and give the time and space complexity of your approach."
        ),
        "difficulty_b": 0.5,
        "discrimination_a": 1.0,
        "expected_concepts": [
            {"key": "two_pointer_technique", "weight": 3, "hint": "a read index and a write index"},
            {"key": "time_complexity", "weight": 2, "hint": "one pass, so linear time"},
            {"key": "in_place_mutation", "weight": 1, "hint": "no second array is allocated"},
        ],
        "common_misconceptions": [],
        "reference_answer": "A strong answer keeps a write index and a read index. " * 6,
        "follow_up_seeds": [],
        "anchor_terms": ["sorted array"],
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


def write(directory: Path, name: str, items: list[dict[str, Any]]) -> Path:
    path = directory / name
    path.write_text(
        "\n".join(json.dumps(item) for item in items) + "\n",
        encoding="utf-8",
    )
    return path


class TestReading:
    def test_a_valid_file_loads(self, tmp_path, taxonomy):
        write(tmp_path, "dsa.jsonl", [make()])
        items, errors = load_bank(tmp_path, taxonomy)
        assert errors == []
        assert [loaded.item.id for loaded in items] == ["dsa-arrays-001"]
        assert items[0].domain == "cs_fundamentals"

    def test_blank_lines_are_skipped(self, tmp_path, taxonomy):
        (tmp_path / "dsa.jsonl").write_text(json.dumps(make()) + "\n\n\n", encoding="utf-8")
        items, errors = load_bank(tmp_path, taxonomy)
        assert errors == []
        assert len(items) == 1

    def test_a_malformed_line_is_reported_with_its_line_number(self, tmp_path, taxonomy):
        (tmp_path / "dsa.jsonl").write_text(json.dumps(make()) + "\n{not json}\n", encoding="utf-8")
        items, errors = load_bank(tmp_path, taxonomy)
        assert len(items) == 1, "the good line must still load"
        assert len(errors) == 1
        assert "dsa.jsonl:2" in errors[0]

    def test_every_error_is_collected_rather_than_the_first_raised(self, tmp_path, taxonomy):
        """Fifteen one-minute runs is the failure mode this avoids."""
        write(
            tmp_path,
            "dsa.jsonl",
            [make(difficulty_b=9.0), make(id="dsa-arrays-002", reference_answer=" ")],
        )
        _, errors = load_bank(tmp_path, taxonomy)
        assert len(errors) == 2


class TestDatasetLevelRules:
    def test_duplicate_ids_across_files_are_an_error(self, tmp_path, taxonomy):
        write(tmp_path, "dsa.jsonl", [make()])
        write(
            tmp_path,
            "extra.jsonl",
            [make(subtopic="graphs", text=make()["text"] + " Extend it to a graph.")],
        )
        report = validate_bank(tmp_path, taxonomy)
        assert any("duplicate id" in error for error in report.errors)

    def test_a_subtopic_from_another_topic_is_an_error(self, tmp_path, taxonomy):
        write(tmp_path, "dsa.jsonl", [make(subtopic="indexing")])
        report = validate_bank(tmp_path, taxonomy)
        assert any("belongs to topic" in error for error in report.errors)

    def test_an_unknown_subtopic_is_an_error(self, tmp_path, taxonomy):
        write(tmp_path, "dsa.jsonl", [make(subtopic="quantum")])
        report = validate_bank(tmp_path, taxonomy)
        assert any("unknown subtopic" in error for error in report.errors)

    def test_a_topic_key_used_where_a_subtopic_belongs_is_an_error(self, tmp_path, taxonomy):
        write(tmp_path, "dsa.jsonl", [make(subtopic="dsa")])
        report = validate_bank(tmp_path, taxonomy)
        assert any("is a topic, not a subtopic" in error for error in report.errors)

    def test_mixing_domains_in_one_file_is_an_error(self, tmp_path, taxonomy):
        """A file is the unit of review; a mixed file is not reviewable."""
        write(
            tmp_path,
            "mixed.jsonl",
            [
                make(),
                make(
                    id="db-index-001",
                    topic="databases",
                    subtopic="indexing",
                    text=(
                        "Explain why a B-tree index speeds reads on that column, and what "
                        "the same index costs on every write."
                    ),
                    anchor_terms=["B-tree index"],
                ),
            ],
        )
        report = validate_bank(tmp_path, taxonomy)
        assert any("more than one domain" in error for error in report.errors)

    def test_a_near_verbatim_duplicate_is_an_error(self, tmp_path, taxonomy):
        original = make()
        twin = make(id="dsa-arrays-002", text=original["text"].replace("Describe", "Explain"))
        write(tmp_path, "dsa.jsonl", [original, twin])
        report = validate_bank(tmp_path, taxonomy)
        assert any("near-duplicates" in error for error in report.errors)

    def test_two_genuinely_different_questions_are_not_flagged(self, tmp_path, taxonomy):
        other = make(
            id="dsa-graphs-001",
            subtopic="graphs",
            text=(
                "You need the shortest path in edges through an unweighted graph. Explain why "
                "breadth-first search is correct here and depth-first search is not."
            ),
            anchor_terms=["unweighted graph"],
        )
        write(tmp_path, "dsa.jsonl", [make(), other])
        report = validate_bank(tmp_path, taxonomy)
        assert report.ok, report.errors
        assert report.warnings == []

    def test_concept_keys_one_character_apart_produce_a_warning(self, tmp_path, taxonomy):
        """The most expensive typo in the project: two vocabularies where there should be one."""
        singular = make()
        plural = make(
            id="dsa-graphs-001",
            subtopic="graphs",
            text=(
                "You need the shortest path in edges through an unweighted graph. Explain why "
                "breadth-first search is correct here and depth-first search is not."
            ),
            anchor_terms=["unweighted graph"],
            expected_concepts=[
                {"key": "two_pointer_techniques", "weight": 3, "hint": "note the stray plural"},
                {
                    "key": "graph_traversal",
                    "weight": 2,
                    "hint": "the frontier expands level by level",
                },
                {
                    "key": "shortest_path",
                    "weight": 1,
                    "hint": "first visit is the minimum-edge path",
                },
            ],
        )
        write(tmp_path, "dsa.jsonl", [singular, plural])
        report = validate_bank(tmp_path, taxonomy)
        assert report.ok, report.errors
        assert any("differ by one character" in warning for warning in report.warnings)

    def test_an_empty_directory_is_an_error_rather_than_a_vacuous_pass(self, tmp_path, taxonomy):
        report = validate_bank(tmp_path, taxonomy)
        assert not report.ok
        assert any("no *.jsonl" in error for error in report.errors)
