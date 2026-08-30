"""The controlled vocabulary: domain -> topic -> subtopic (plan section 9.1).

Three levels, no more.  Two is too coarse to be actionable ("you are weak at
backend" helps nobody); four is unmaintainable, and every leaf ends up with
``n = 1``.

**Why the taxonomy is a data file and not a Python dict.**  It is ingested into
the ``topics`` table, which the ``questions`` table has foreign keys into.  A
file can be diffed in the pull request that adds a subtopic, next to the
questions that needed it - which is exactly the review this dataset lives or
dies by.

**Why keys are globally unique across levels.**  ``topics.key`` is a single
primary key for a self-referencing table, so the topic ``caching`` and a
subtopic ``caching`` could not coexist.  That constraint is enforced here, at
load, rather than discovered by a foreign-key violation during ingest.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.bank.paths import TAXONOMY_PATH

KEY_PATTERN = "lower_snake_case"


class TaxonomyError(ValueError):
    """The taxonomy file is malformed. Always fatal: nothing can validate without it."""


@dataclass(frozen=True, slots=True)
class TopicNode:
    """One row of the ``topics`` table, before it is a row.

    ``level`` is derived, not stored in the database: a node with no parent is a
    domain, a node whose parent is a domain is a topic, and so on.  It exists
    here because every validation message wants to say which level was wrong.
    """

    key: str
    display_name: str
    domain: str
    parent_key: str | None
    level: str  # "domain" | "topic" | "subtopic"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise TaxonomyError(message)


def _valid_key(key: object) -> bool:
    return (
        isinstance(key, str)
        and bool(key)
        and key == key.lower()
        and key.replace("_", "").isalnum()
        and not key.startswith("_")
        and not key.endswith("_")
    )


@dataclass(frozen=True, slots=True)
class Taxonomy:
    """An immutable, indexed view of the taxonomy file."""

    nodes: dict[str, TopicNode]

    # -- lookups ----------------------------------------------------------
    def get(self, key: str) -> TopicNode | None:
        return self.nodes.get(key)

    def domains(self) -> list[TopicNode]:
        return [n for n in self.nodes.values() if n.level == "domain"]

    def topics(self) -> list[TopicNode]:
        return [n for n in self.nodes.values() if n.level == "topic"]

    def subtopics(self) -> list[TopicNode]:
        return [n for n in self.nodes.values() if n.level == "subtopic"]

    def children_of(self, key: str) -> list[TopicNode]:
        return [n for n in self.nodes.values() if n.parent_key == key]

    def is_subtopic_of(self, subtopic_key: str, topic_key: str) -> bool:
        node = self.nodes.get(subtopic_key)
        return node is not None and node.level == "subtopic" and node.parent_key == topic_key

    def domain_of(self, key: str) -> str | None:
        node = self.nodes.get(key)
        return None if node is None else node.domain

    def rows(self) -> list[TopicNode]:
        """Every node, parents before children - the order ``topics`` must be inserted in."""
        order = {"domain": 0, "topic": 1, "subtopic": 2}
        return sorted(self.nodes.values(), key=lambda n: (order[n.level], n.key))


def _parse(raw: object, source: str) -> Taxonomy:
    _require(isinstance(raw, dict), f"{source}: top level must be an object")
    assert isinstance(raw, dict)  # narrowed by _require
    domains = raw.get("domains")
    _require(
        isinstance(domains, list) and bool(domains), f"{source}: 'domains' must be a non-empty list"
    )
    assert isinstance(domains, list)

    nodes: dict[str, TopicNode] = {}

    def add(key: object, display: object, domain: str, parent: str | None, level: str) -> str:
        _require(_valid_key(key), f"{source}: {level} key {key!r} is not {KEY_PATTERN}")
        assert isinstance(key, str)
        _require(
            isinstance(display, str) and bool(display.strip()),
            f"{source}: {level} '{key}' has no display_name",
        )
        assert isinstance(display, str)
        _require(
            key not in nodes,
            f"{source}: key '{key}' is defined twice - topics.key is a single primary key "
            "across all three levels, so a name may be used at only one of them",
        )
        nodes[key] = TopicNode(
            key=key, display_name=display.strip(), domain=domain, parent_key=parent, level=level
        )
        return key

    for domain_raw in domains:
        _require(isinstance(domain_raw, dict), f"{source}: each domain must be an object")
        assert isinstance(domain_raw, dict)
        domain_key_raw = domain_raw.get("key")
        _require(
            _valid_key(domain_key_raw),
            f"{source}: domain key {domain_key_raw!r} is not {KEY_PATTERN}",
        )
        assert isinstance(domain_key_raw, str)
        domain_key = add(
            domain_key_raw, domain_raw.get("display_name"), domain_key_raw, None, "domain"
        )

        topics = domain_raw.get("topics")
        _require(
            isinstance(topics, list) and bool(topics),
            f"{source}: domain '{domain_key}' has no topics",
        )
        assert isinstance(topics, list)
        for topic_raw in topics:
            _require(isinstance(topic_raw, dict), f"{source}: each topic must be an object")
            assert isinstance(topic_raw, dict)
            topic_key = add(
                topic_raw.get("key"), topic_raw.get("display_name"), domain_key, domain_key, "topic"
            )
            subtopics = topic_raw.get("subtopics")
            _require(
                isinstance(subtopics, list) and bool(subtopics),
                f"{source}: topic '{topic_key}' has no subtopics",
            )
            assert isinstance(subtopics, list)
            for sub_raw in subtopics:
                _require(isinstance(sub_raw, dict), f"{source}: each subtopic must be an object")
                assert isinstance(sub_raw, dict)
                add(
                    sub_raw.get("key"),
                    sub_raw.get("display_name"),
                    domain_key,
                    topic_key,
                    "subtopic",
                )

    return Taxonomy(nodes=nodes)


def parse_taxonomy(raw: Any, source: str = "<memory>") -> Taxonomy:
    """Validate an already-decoded taxonomy document."""
    return _parse(raw, source)


def load_taxonomy(path: Path | None = None) -> Taxonomy:
    """Read and validate the taxonomy file. Raises ``TaxonomyError`` if it is malformed."""
    target = TAXONOMY_PATH if path is None else path
    try:
        raw = json.loads(target.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:  # pragma: no cover - operator error
        raise TaxonomyError(f"no taxonomy at {target}") from exc
    except json.JSONDecodeError as exc:
        raise TaxonomyError(f"{target}: invalid JSON: {exc}") from exc
    return _parse(raw, str(target))
