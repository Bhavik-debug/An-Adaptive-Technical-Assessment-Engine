"""The four bias probes' perturbations - Day 19, plan section 12.1.

**The perturbation is the experiment.** A length probe whose filler happens to
mention an expected concept measures concept detection, not length bias - and it
would *fail*, correctly, while being reported as a bias finding. So the filler
is checked against every concept key and anchor term in the committed bank here,
rather than inspected by eye once and trusted.
"""

from __future__ import annotations

import pytest

from app.bank.loader import load_bank
from app.grading.probes import (
    CONFIDENCE_PREFIX,
    FILLER_200_WORDS,
    FILLER_TARGET_WORDS,
    FILLER_WORD_TOLERANCE,
    JARGON_SUBSTITUTIONS,
    filler_problems,
    jargonise,
    pad_with_filler,
    prepend_confidence,
    shuffled_concepts,
    word_count,
)

ANSWER = (
    "The problem is a lost update. Two transactions read the same row, each adds "
    "ten, and the second write overwrites the first. Use select for update to "
    "take a row lock, or a version column checked in the where clause."
)


class TestTheFiller:
    def test_it_is_about_two_hundred_words(self):
        """Plan section 12.1 specifies 200. Pinned so it cannot drift."""
        count = word_count(FILLER_200_WORDS)
        assert abs(count - FILLER_TARGET_WORDS) <= FILLER_WORD_TOLERANCE, count

    def test_it_mentions_no_concept_or_anchor_from_the_real_bank(self):
        """The probe's validity condition, checked against the whole bank.

        If this fails the length probe is invalid, not merely untidy: the filler
        would be adding content the grader may legitimately credit.
        """
        loaded, errors = load_bank()
        if errors:
            pytest.skip("the question bank does not load in this checkout")
        keys = sorted({key for item in loaded for key in item.item.concept_keys})
        anchors = sorted({term for item in loaded for term in item.item.anchor_terms})
        assert keys and anchors, "the bank yielded no concepts to check against"
        problems = filler_problems(FILLER_200_WORDS, concept_keys=keys, anchor_terms=anchors)
        assert problems == [], problems

    def test_the_checker_catches_a_concept_mention(self):
        """The guard has to actually fire, or it proves nothing above."""
        bad = "You should think carefully about cache invalidation before answering."
        assert filler_problems(bad, concept_keys=["cache_invalidation"])

    def test_the_checker_reads_underscores_as_spaces(self):
        assert filler_problems("a note on lost update patterns", concept_keys=["lost_update"])

    def test_the_checker_does_not_fire_on_a_substring(self):
        """'stack' must not match inside 'stacked'."""
        assert filler_problems("the work is stacked up", concept_keys=["stack"]) == []

    def test_the_checker_catches_an_anchor_term(self):
        assert filler_problems("think about the queue", concept_keys=[], anchor_terms=["queue"])


class TestLengthProbeVariant:
    def test_it_keeps_the_original_answer_intact(self):
        padded = pad_with_filler(ANSWER)
        assert ANSWER.rstrip() in padded

    def test_it_adds_the_filler(self):
        padded = pad_with_filler(ANSWER)
        assert FILLER_200_WORDS in padded
        assert word_count(padded) == pytest.approx(
            word_count(ANSWER) + word_count(FILLER_200_WORDS), abs=1
        )

    def test_the_answer_comes_first(self):
        """Appended, not prepended: this isolates length from first impression.

        The confidence probe tests the prepend direction, and conflating the two
        would make a failure unattributable.
        """
        padded = pad_with_filler(ANSWER)
        assert padded.index(ANSWER[:40]) < padded.index(FILLER_200_WORDS[:40])


class TestConfidenceProbeVariant:
    def test_it_prepends_an_assertion_of_certainty(self):
        out = prepend_confidence(ANSWER)
        assert out.startswith(CONFIDENCE_PREFIX)
        assert ANSWER.lstrip() in out

    def test_it_adds_no_technical_content(self):
        """The prefix must assert, not inform."""
        loaded, errors = load_bank()
        if errors:
            pytest.skip("the question bank does not load in this checkout")
        keys = sorted({key for item in loaded for key in item.item.concept_keys})
        assert filler_problems(CONFIDENCE_PREFIX, concept_keys=keys) == []


class TestJargonProbeVariant:
    def test_it_replaces_a_precise_term_and_counts_the_substitution(self):
        out, count = jargonise("We use a write-ahead log for durability.")
        assert count == 2  # write-ahead log, durability
        assert "write-ahead log" not in out
        assert "best-practice durability layer" in out

    def test_it_reports_zero_when_nothing_matched(self):
        """A variant with no substitutions is not a jargon variant.

        The runner excludes those rows rather than counting them as a pass,
        which is why the count is returned rather than just the text.
        """
        _, count = jargonise("The weather today is pleasant and mild.")
        assert count == 0

    def test_matching_is_case_insensitive(self):
        out, count = jargonise("Atomicity matters.")
        assert count == 1
        assert "Atomicity" not in out

    def test_it_matches_whole_words_only(self):
        """'stack' must not fire inside 'stacked' or 'haystack'."""
        _, count = jargonise("The haystack was stacked high.")
        assert count == 0

    def test_a_longer_phrase_wins_over_a_shorter_one(self):
        """'optimistic concurrency control' must not be eaten by a shorter key."""
        out, _ = jargonise("I would use optimistic concurrency control here.")
        assert "modern concurrency approach" in out

    def test_every_replacement_is_vaguer_rather_than_a_synonym(self):
        """Sanity check on the table itself: no entry may reuse the term it replaces."""
        for term, replacement in JARGON_SUBSTITUTIONS.items():
            assert term.lower() not in replacement.lower(), term

    def test_it_is_deterministic(self):
        assert jargonise(ANSWER) == jargonise(ANSWER)

    def test_a_replacement_is_never_itself_substituted(self):
        """The cascade bug, pinned.

        'write-ahead log' -> 'best-practice durability layer' contains the word
        'durability', which is itself a key. A term-by-term implementation then
        rewrites its own output to 'best-practice persistence guarantees layer'
        and counts two substitutions for one term in the input.
        """
        out, count = jargonise("We use a write-ahead log here.")
        assert count == 1
        assert out == "We use a best-practice durability layer here."
        assert "persistence guarantees" not in out


class TestPositionProbeVariant:
    def test_it_really_permutes(self):
        """A shuffle that returned the input would make the probe vacuous."""
        items = ["a", "b", "c", "d"]
        out = shuffled_concepts(items, seed=7)
        assert sorted(out) == sorted(items)
        assert out != items

    def test_it_is_reproducible_for_a_seed(self):
        items = ["a", "b", "c", "d"]
        assert shuffled_concepts(items, seed=7) == shuffled_concepts(items, seed=7)

    def test_a_single_concept_is_returned_unchanged(self):
        """Nothing to permute, and no infinite loop looking for one."""
        assert shuffled_concepts(["only"], seed=1) == ["only"]
        assert shuffled_concepts([], seed=1) == []

    def test_it_preserves_the_multiset(self):
        items = ["a", "a", "b"]
        assert sorted(shuffled_concepts(items, seed=3)) == ["a", "a", "b"]
