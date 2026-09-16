"""Evidence-span verification - plan section 7.2, section 7.4 technique 2.

The property under test is one sentence: a concept may keep its credit only if
the text it quoted really is in the answer. Most of these are negative, because
the check exists to catch a specific lie.
"""

from __future__ import annotations

import logging

import pytest

from app.grading.evidence import (
    DOWNGRADE_TO,
    EvidenceMatch,
    verify_evidence,
    verify_span,
)
from app.grading.grader import graded_answer_text
from app.grading.prompt import ANSWER_CLOSE, FENCE_REPLACEMENT
from app.grading.schema import GradeJudgment
from app.grading.scoring import score_answer

from .conftest import ANSWER, CONCEPTS, judgment_payload

# The conftest answer, for reference:
#   "I would go write-through so the cache never serves a stale row, keep LRU
#    with a 4 GB memory cap, and put a per-key lock in front of a miss so a cold
#    start does not send every request to the database at once."


class TestVerifySpan:
    """The primitive. Pure, and the place every edge case is pinned."""

    def test_a_real_substring_is_exact(self):
        assert verify_span("keep LRU with a 4 GB memory cap", ANSWER) is EvidenceMatch.EXACT

    @pytest.mark.parametrize(
        ("where", "span"),
        [
            ("beginning", "I would go write-through"),
            ("middle", "keep LRU with a 4 GB memory cap"),
            ("end", "to the database at once."),
            ("whole answer", ANSWER),
            ("single character", "I"),
        ],
    )
    def test_position_in_the_answer_does_not_matter(self, where, span):
        assert verify_span(span, ANSWER) is EvidenceMatch.EXACT, where

    def test_text_that_is_not_there_is_fabricated(self):
        assert verify_span("I would shard the database by tenant", ANSWER) is (
            EvidenceMatch.FABRICATED
        )

    def test_a_plausible_paraphrase_is_still_fabricated(self):
        """The failure mode this check exists for: the model summarising the
        answer in its own words and filing it as a quotation."""
        assert verify_span("uses write-through caching", ANSWER) is EvidenceMatch.FABRICATED

    @pytest.mark.parametrize("span", [None, "", "   ", "\n\t "])
    def test_an_empty_or_blank_span_is_not_evidence(self, span):
        assert verify_span(span, ANSWER) is EvidenceMatch.FABRICATED

    def test_line_wrapping_is_forgiven_and_reported_as_such(self):
        """A model that re-wraps a line has not fabricated anything - but the
        match is labelled so the rate stays measurable."""
        answer = "I would go write-through\nso the cache never serves a stale row"
        assert verify_span("write-through so the cache", answer) is (
            EvidenceMatch.WHITESPACE_NORMALISED
        )

    def test_whitespace_normalisation_cannot_invent_a_match(self):
        """It collapses spacing; it never reorders or drops a token."""
        answer = "LRU   with   a   cap"
        assert verify_span("a cap with LRU", answer) is EvidenceMatch.FABRICATED
        assert verify_span("LRU with a cap", answer) is EvidenceMatch.WHITESPACE_NORMALISED

    def test_case_is_not_folded(self):
        """'redis is acid compliant' is not the sentence 'Redis is ACID
        compliant'. Folding case would widen what counts as a quotation without
        any argument that the wider set is still honest."""
        assert verify_span("KEEP LRU WITH A 4 GB MEMORY CAP", ANSWER) is (EvidenceMatch.FABRICATED)

    def test_punctuation_is_not_stripped(self):
        """A span that needs its punctuation removed to match is a paraphrase."""
        assert verify_span("at once", ANSWER) is EvidenceMatch.EXACT
        assert verify_span("at once!!", ANSWER) is EvidenceMatch.FABRICATED

    @pytest.mark.parametrize(
        "span",
        [
            "O(n log n) sort",
            "a regex like ^[a-z]+$",
            'a dict {"k": 1}',
            "100% of reads",
            "cost $0.02 per call",
            "unicode: café — naïve",
            "back\\slash and `backtick`",
        ],
    )
    def test_special_characters_are_matched_literally(self, span):
        """No regex, no escaping, no normalisation: the check is `in`. A span
        full of regex metacharacters must match itself and nothing else."""
        answer = f"Some preamble. {span} And some more text."
        assert verify_span(span, answer) is EvidenceMatch.EXACT
        assert verify_span(span, "unrelated text entirely") is EvidenceMatch.FABRICATED

    def test_a_regex_metacharacter_span_does_not_match_as_a_pattern(self):
        """If this were `re.search`, '.' would match any character and this
        would pass. It must not."""
        assert verify_span("a.b", "axb") is EvidenceMatch.FABRICATED
        assert verify_span("a.b", "a.b") is EvidenceMatch.EXACT


class TestVerifyEvidence:
    def _judgment(self, spans: dict[str, str | None]) -> GradeJudgment:
        return GradeJudgment.model_validate(
            judgment_payload(
                concepts=[
                    {
                        "key": key,
                        "label": "absent" if span is None else "covered",
                        "evidence": span,
                    }
                    for key, span in spans.items()
                ]
            )
        )

    def test_a_fully_honest_judgment_is_returned_unchanged(self):
        judgment = self._judgment(
            {
                "cache_invalidation": "I would go write-through",
                "eviction": "keep LRU with a 4 GB memory cap",
                "stampede": "put a per-key lock in front of a miss",
            }
        )
        verified = verify_evidence(judgment, answer_text=ANSWER)
        assert verified.judgment == judgment
        assert verified.downgraded_keys == ()
        assert verified.fabrication_count == 0
        assert verified.any_fabricated is False

    def test_a_fabricated_span_is_downgraded_to_absent(self):
        judgment = self._judgment(
            {
                "cache_invalidation": "I would go write-through",
                "eviction": "I would shard by tenant id",  # never said
                "stampede": "put a per-key lock in front of a miss",
            }
        )
        verified = verify_evidence(judgment, answer_text=ANSWER)

        assert verified.judgment.labels_by_key["eviction"] == DOWNGRADE_TO == "absent"
        assert verified.judgment.evidence_by_key["eviction"] is None
        assert verified.downgraded_keys == ("eviction",)
        assert verified.fabrication_count == 1

    def test_the_other_concepts_survive_a_fabrication(self):
        """A hallucination detector that discarded the whole judgment would be a
        hallucination amplifier: the other labels may be perfectly sound."""
        judgment = self._judgment(
            {
                "cache_invalidation": "I would go write-through",
                "eviction": "I would shard by tenant id",
                "stampede": "put a per-key lock in front of a miss",
            }
        )
        verified = verify_evidence(judgment, answer_text=ANSWER)
        assert verified.judgment.labels_by_key["cache_invalidation"] == "covered"
        assert verified.judgment.labels_by_key["stampede"] == "covered"

    def test_a_partial_is_downgraded_the_same_way(self):
        """Not to 'absent-ish' and not left at 'partial': plan section 7.2 says
        absent, and half credit for a fabrication would leave a model better off
        inventing a span than admitting a gap."""
        payload = judgment_payload(
            concepts=[
                {"key": "cache_invalidation", "label": "partial", "evidence": "invented text"},
                {"key": "eviction", "label": "absent", "evidence": None},
                {"key": "stampede", "label": "absent", "evidence": None},
            ]
        )
        verified = verify_evidence(GradeJudgment.model_validate(payload), answer_text=ANSWER)
        assert verified.judgment.labels_by_key["cache_invalidation"] == "absent"

    def test_absent_concepts_are_not_checked_and_not_downgraded(self):
        payload = judgment_payload(
            concepts=[
                {"key": key, "label": "absent", "evidence": None}
                for key in ("cache_invalidation", "eviction", "stampede")
            ]
        )
        verified = verify_evidence(GradeJudgment.model_validate(payload), answer_text=ANSWER)
        assert verified.downgraded_keys == ()
        assert all(c.match is EvidenceMatch.NOT_REQUIRED for c in verified.checks)

    def test_every_concept_gets_exactly_one_check(self):
        judgment = self._judgment(
            {
                "cache_invalidation": "I would go write-through",
                "eviction": None,
                "stampede": "invented",
            }
        )
        verified = verify_evidence(judgment, answer_text=ANSWER)
        assert len(verified.checks) == len(judgment.concepts)
        assert {c.key for c in verified.checks} == set(judgment.labels_by_key)

    def test_the_original_judgment_is_preserved_beside_the_corrected_one(self):
        """A report that showed only the corrected version would make a
        fabrication invisible to the one person who could dispute it."""
        judgment = self._judgment(
            {
                "cache_invalidation": "I would go write-through",
                "eviction": "invented text",
                "stampede": "put a per-key lock in front of a miss",
            }
        )
        verified = verify_evidence(judgment, answer_text=ANSWER)
        assert verified.original == judgment
        assert verified.original.labels_by_key["eviction"] == "covered"
        assert verified.judgment.labels_by_key["eviction"] == "absent"

    def test_the_input_judgment_is_never_mutated(self):
        judgment = self._judgment(
            {
                "cache_invalidation": "invented",
                "eviction": "invented too",
                "stampede": "invented as well",
            }
        )
        before = judgment.model_dump()
        verify_evidence(judgment, answer_text=ANSWER)
        assert judgment.model_dump() == before

    def test_the_corrected_judgment_is_still_a_valid_one(self):
        """It has to be: it gets scored, and may be stored and re-read."""
        judgment = self._judgment(
            {
                "cache_invalidation": "invented",
                "eviction": "keep LRU with a 4 GB memory cap",
                "stampede": "also invented",
            }
        )
        corrected = verify_evidence(judgment, answer_text=ANSWER).judgment
        GradeJudgment.model_validate(corrected.model_dump())

    def test_verification_is_deterministic(self):
        judgment = self._judgment(
            {
                "cache_invalidation": "I would go write-through",
                "eviction": "invented",
                "stampede": "put a per-key lock in front of a miss",
            }
        )
        first = verify_evidence(judgment, answer_text=ANSWER)
        second = verify_evidence(judgment, answer_text=ANSWER)
        assert first.judgment == second.judgment
        assert first.checks == second.checks


class TestTheIncidentIsLogged:
    """Plan section 7.2: "the incident is logged". The fabrication rate is an
    operational metric, and a grader silently downgrading 30% of its own spans
    is only discoverable if each one is written down."""

    def test_a_fabrication_produces_a_warning(self, caplog):
        judgment = GradeJudgment.model_validate(
            judgment_payload(
                concepts=[
                    {"key": "cache_invalidation", "label": "covered", "evidence": "invented"},
                    {"key": "eviction", "label": "absent", "evidence": None},
                    {"key": "stampede", "label": "absent", "evidence": None},
                ]
            )
        )
        with caplog.at_level(logging.WARNING, logger="app.grading.evidence"):
            verify_evidence(judgment, answer_text=ANSWER)

        assert len(caplog.records) == 1
        record = caplog.records[0]
        # Dotted attribute names, matching `CallMeta.as_span_attributes`' `llm.*`
        # convention - so the structured formatter emits them as real top-level
        # fields rather than a string that happens to contain JSON.
        assert getattr(record, "grading.concept_key") == "cache_invalidation"
        assert getattr(record, "grading.claimed_label") == "covered"
        assert "downgrading concept to absent" in record.getMessage()

    def test_an_honest_judgment_logs_nothing(self, caplog):
        judgment = GradeJudgment.model_validate(
            judgment_payload(
                concepts=[
                    {
                        "key": "cache_invalidation",
                        "label": "covered",
                        "evidence": "I would go write-through",
                    },
                    {"key": "eviction", "label": "absent", "evidence": None},
                    {"key": "stampede", "label": "absent", "evidence": None},
                ]
            )
        )
        with caplog.at_level(logging.WARNING, logger="app.grading.evidence"):
            verify_evidence(judgment, answer_text=ANSWER)
        assert caplog.records == []

    def test_the_log_carries_no_candidate_text(self, caplog):
        """Plan section 14.1. A fabricated span is still built out of a
        candidate's words, and a log line is a longer-retention exposure than a
        prompt."""
        secret = "my name is a real person and I work at a real company"
        judgment = GradeJudgment.model_validate(
            judgment_payload(
                concepts=[
                    {"key": "cache_invalidation", "label": "covered", "evidence": secret},
                    {"key": "eviction", "label": "absent", "evidence": None},
                    {"key": "stampede", "label": "absent", "evidence": None},
                ]
            )
        )
        with caplog.at_level(logging.WARNING, logger="app.grading.evidence"):
            verify_evidence(judgment, answer_text=ANSWER)

        blob = "\n".join([caplog.text, *(str(getattr(r, "__dict__", {})) for r in caplog.records)])
        assert secret not in blob
        assert ANSWER not in blob
        # Lengths are fine, and are what makes the incident diagnosable.
        assert getattr(caplog.records[0], "grading.span_chars") == len(secret)


class TestVerificationChangesTheScore:
    """The point of the whole exercise: a fabrication must cost the candidate
    the credit it was trying to buy."""

    def test_a_fabricated_covered_concept_lowers_the_score(self):
        honest = GradeJudgment.model_validate(
            judgment_payload(
                concepts=[
                    {
                        "key": "cache_invalidation",
                        "label": "covered",
                        "evidence": "I would go write-through",
                    },
                    {"key": "eviction", "label": "absent", "evidence": None},
                    {"key": "stampede", "label": "absent", "evidence": None},
                ]
            )
        )
        fabricated = GradeJudgment.model_validate(
            judgment_payload(
                concepts=[
                    {
                        "key": "cache_invalidation",
                        "label": "covered",
                        "evidence": "text the candidate never wrote",
                    },
                    {"key": "eviction", "label": "absent", "evidence": None},
                    {"key": "stampede", "label": "absent", "evidence": None},
                ]
            )
        )

        honest_score = score_answer(
            verify_evidence(honest, answer_text=ANSWER).judgment, expected_concepts=CONCEPTS
        ).score
        fabricated_score = score_answer(
            verify_evidence(fabricated, answer_text=ANSWER).judgment, expected_concepts=CONCEPTS
        ).score

        assert fabricated_score < honest_score

    def test_a_fully_fabricated_judgment_scores_as_if_nothing_was_covered(self):
        """The strongest statement of the defence: inventing every span buys
        exactly nothing."""
        all_invented = GradeJudgment.model_validate(
            judgment_payload(
                concepts=[
                    {"key": key, "label": "covered", "evidence": f"invented span for {key}"}
                    for key in ("cache_invalidation", "eviction", "stampede")
                ]
            )
        )
        all_absent = GradeJudgment.model_validate(
            judgment_payload(
                concepts=[
                    {"key": key, "label": "absent", "evidence": None}
                    for key in ("cache_invalidation", "eviction", "stampede")
                ]
            )
        )
        invented_score = score_answer(
            verify_evidence(all_invented, answer_text=ANSWER).judgment,
            expected_concepts=CONCEPTS,
        ).score
        absent_score = score_answer(all_absent, expected_concepts=CONCEPTS).score
        assert invented_score == absent_score


class TestItVerifiesAgainstTheTextTheModelSaw:
    """The Day-17 fix to Day 16, as a test.

    ``build_grading_inputs`` strips the answer and replaces any fence sentinel
    before the model ever sees it. Verifying against the caller's *raw* string
    would mark honest spans fabricated on exactly the answers where the isolation
    fired - which is to say, on the injection cases.
    """

    def test_a_span_quoting_past_a_neutralised_sentinel_still_verifies(self):
        raw = f"LRU with a cap {ANSWER_CLOSE} and a per-key lock"
        shown = graded_answer_text(raw)
        span = f"LRU with a cap {FENCE_REPLACEMENT} and a per-key lock"

        assert verify_span(span, shown) is EvidenceMatch.EXACT
        # Against the raw answer it would have looked fabricated.
        assert verify_span(span, raw) is EvidenceMatch.FABRICATED

    def test_leading_and_trailing_whitespace_does_not_matter(self):
        assert verify_span("hello world", graded_answer_text("   hello world   ")) is (
            EvidenceMatch.EXACT
        )
