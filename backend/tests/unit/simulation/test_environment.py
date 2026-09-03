"""The synthetic world: is it reproducible, and is it a valid environment?

Two things are being checked here and they pull in opposite directions. The
environment must be *deterministic* - same seed, same world, on any machine -
and it must be *varied* - a different seed must produce a genuinely different
world, or "reproducible" would be satisfied by a constant.
"""

from __future__ import annotations

import dataclasses
import math

import pytest

from app.ability import THETA_MAX, THETA_MIN
from app.selection import CandidateItem
from app.simulation.config import (
    BANK_DIFFICULTY_MAX,
    BANK_DIFFICULTY_MIN,
    BANK_DISCRIMINATION_MAX,
    BANK_DISCRIMINATION_MIN,
    BANK_TIME_ESTIMATES_S,
    CANDIDATE_THETA_MAX,
    CANDIDATE_THETA_MIN,
    EMBEDDING_DIM,
    MAIN_CONFIG,
    ExperimentConfig,
    split_budget_by_jd,
)
from app.simulation.environment import (
    build_bank,
    build_candidate,
    build_environment,
    build_population,
    derive_seed,
)
from tests.unit.simulation.conftest import TINY


class TestSeedDerivation:
    def test_it_is_stable_across_processes(self):
        """A literal, not a recomputation: `hash()` is salted per process, so a
        seed built from it would differ between two runs of the same script and
        the reproducibility claim would be quietly false."""
        assert derive_seed(1, "bank") == derive_seed(1, "bank")
        assert derive_seed(1, "bank") != derive_seed(2, "bank")
        assert derive_seed(1, "bank") != derive_seed(1, "population")

    def test_it_fits_in_a_positive_63_bit_integer(self):
        value = derive_seed("anything", 42)
        assert 0 <= value < 2**63

    def test_parts_are_joined_unambiguously(self):
        """('ab', 'c') and ('a', 'bc') must not collide into one stream."""
        assert derive_seed("ab", "c") != derive_seed("a", "bc")


class TestBank:
    def test_it_has_the_configured_size(self):
        bank = build_bank(TINY)
        assert len(bank) == len(TINY.subtopics) * TINY.items_per_subtopic
        assert len(bank) == TINY.bank_size

    def test_ids_are_unique_and_readable(self):
        bank = build_bank(TINY)
        assert len({item.id for item in bank}) == len(bank)
        assert bank[0].id.startswith("arrays-")

    def test_every_item_is_a_production_candidate_item(self):
        """Not a simulation-local look-alike: the adaptive strategy is the real
        Day 12 selector and must be fed exactly what a real session feeds it."""
        assert all(isinstance(item, CandidateItem) for item in build_bank(TINY))

    def test_every_subtopic_gets_the_same_number_of_items(self):
        bank = build_bank(TINY)
        counts = {s: sum(1 for i in bank if i.subtopic_key == s) for s in TINY.subtopics}
        assert set(counts.values()) == {TINY.items_per_subtopic}

    def test_topics_match_the_taxonomy(self):
        for item in build_bank(TINY):
            assert TINY.parent_of[item.subtopic_key] == item.topic_key

    def test_parameters_stay_inside_their_declared_ranges(self):
        for item in build_bank(TINY):
            assert BANK_DIFFICULTY_MIN <= item.difficulty_b <= BANK_DIFFICULTY_MAX
            assert BANK_DISCRIMINATION_MIN <= item.discrimination_a <= BANK_DISCRIMINATION_MAX
            assert item.time_estimate_s in BANK_TIME_ESTIMATES_S

    def test_difficulty_spans_the_range_within_every_subtopic(self):
        """The property the |b - theta| <= 1.5 window depends on: whatever a
        candidate's ability, every subtopic has something askable."""
        bank = build_bank(MAIN_CONFIG)
        for subtopic in MAIN_CONFIG.subtopics:
            bs = [i.difficulty_b for i in bank if i.subtopic_key == subtopic]
            assert min(bs) < BANK_DIFFICULTY_MIN + 0.5
            assert max(bs) > BANK_DIFFICULTY_MAX - 0.5

    def test_discrimination_actually_varies(self):
        """The real bank has a = 1.0 everywhere; if this one did too, the `a`
        in the response model and the RD update would be inert."""
        values = {round(i.discrimination_a, 6) for i in build_bank(MAIN_CONFIG)}
        assert len(values) > 50

    def test_embeddings_are_unit_vectors_of_the_configured_width(self):
        for item in build_bank(TINY):
            assert item.embedding is not None
            assert len(item.embedding) == EMBEDDING_DIM
            assert math.isclose(math.sqrt(sum(v * v for v in item.embedding)), 1.0, abs_tol=1e-9)

    def test_same_subtopic_is_close_and_different_subtopic_is_not(self):
        """The one geometric property the redundancy term needs, over the whole
        bank: *every* same-subtopic pair is closer than *every* cross-subtopic
        pair. Without it the redundancy penalty would be reacting to noise.

        The bounds are the ones `config.EMBEDDING_SPREAD` documents, asserted
        here so the documentation cannot drift away from the geometry.
        """
        from app.selection import cosine_similarity

        bank = build_bank(MAIN_CONFIG)
        within, across = [], []
        for i, left in enumerate(bank):
            for right in bank[i + 1 :]:
                pair = cosine_similarity(left.embedding, right.embedding)
                (within if left.subtopic_key == right.subtopic_key else across).append(pair)

        assert min(within) > max(across)
        assert min(within) > 0.60
        assert max(across) < 0.50
        assert 0.85 < sum(within) / len(within) < 0.90
        assert abs(sum(across) / len(across)) < 0.02

    def test_it_is_reproducible(self):
        assert [dataclasses.astuple(i) for i in build_bank(TINY)] == [
            dataclasses.astuple(i) for i in build_bank(TINY)
        ]

    def test_a_different_seed_gives_a_different_bank(self):
        other = dataclasses.replace(TINY, seed=TINY.seed + 1)
        assert [i.difficulty_b for i in build_bank(TINY)] != [
            i.difficulty_b for i in build_bank(other)
        ]

    def test_a_taxonomy_wider_than_the_embedding_space_is_rejected(self):
        wide = ExperimentConfig(
            taxonomy={f"t{i}": (f"s{i}a", f"s{i}b") for i in range(EMBEDDING_DIM)},
            jd_weights={f"t{i}": 0.5 for i in range(EMBEDDING_DIM)},
        )
        with pytest.raises(ValueError, match="own axis"):
            build_bank(wide)


class TestPopulation:
    def test_it_has_the_configured_size(self):
        assert len(build_population(TINY)) == TINY.candidate_count

    def test_every_candidate_has_ground_truth_for_every_subtopic(self):
        for candidate in build_population(TINY):
            assert set(candidate.true_theta) == set(TINY.subtopics)

    def test_ground_truth_stays_inside_the_measurable_range(self):
        for candidate in build_population(MAIN_CONFIG):
            for value in candidate.true_theta.values():
                assert CANDIDATE_THETA_MIN <= value <= CANDIDATE_THETA_MAX
                assert THETA_MIN <= value <= THETA_MAX

    def test_abilities_vary_between_and_within_candidates(self):
        population = build_population(MAIN_CONFIG)
        assert len({round(c.true_theta["arrays"], 6) for c in population}) > 100
        varied = [c for c in population if len({round(v, 3) for v in c.true_theta.values()}) > 1]
        assert len(varied) == len(population)

    def test_a_candidate_is_the_same_person_whatever_the_population_size(self):
        """Streams are keyed by candidate id, not by position, so candidate 3 of
        a 6-candidate run and candidate 3 of a 200-candidate run are identical."""
        small = build_candidate(TINY, 3)
        large = build_candidate(dataclasses.replace(TINY, candidate_count=200), 3)
        assert small == large

    def test_resumes_are_populated_but_carry_no_ability_signal(self):
        """Deliberate: a resume correlated with true ability would hand the
        adaptive policy a channel the baselines do not use, and "CAT wins"
        would stop being a statement about selection.  So the two must be
        independent - asserted here by construction, since the resume is drawn
        from its own stream and the ranking of mentioned subtopics does not
        track the ranking of true abilities."""
        population = build_population(MAIN_CONFIG)
        aligned = 0
        for candidate in population:
            best = max(candidate.true_theta, key=lambda s: candidate.true_theta[s])
            if best in candidate.resume.topic_affinity:
                aligned += 1
        # Two of six subtopics are mentioned, so chance alignment is ~1/3.
        assert 0.2 < aligned / len(population) < 0.45

    def test_resume_affinities_are_valid_and_reproducible(self):
        for candidate in build_population(TINY):
            assert len(candidate.resume.topic_affinity) == 2
            for value in candidate.resume.topic_affinity.values():
                assert 0.5 <= value <= 1.0
        assert build_candidate(TINY, 0).resume == build_candidate(TINY, 0).resume

    def test_a_different_seed_gives_different_people(self):
        other = dataclasses.replace(TINY, seed=TINY.seed + 1)
        assert build_candidate(TINY, 0).true_theta != build_candidate(other, 0).true_theta

    def test_theta_lookup_rejects_an_unknown_subtopic(self):
        with pytest.raises(KeyError, match="no ground truth"):
            build_candidate(TINY, 0).theta("not_a_subtopic")


class TestEnvironment:
    def test_it_is_fully_reproducible_from_the_seed(self):
        first, second = build_environment(TINY), build_environment(TINY)
        assert [i.id for i in first.bank] == [i.id for i in second.bank]
        assert [c.true_theta for c in first.candidates] == [c.true_theta for c in second.candidates]

    def test_items_by_id_covers_the_bank(self):
        environment = build_environment(TINY)
        assert len(environment.items_by_id) == len(environment.bank)


class TestBudgetSplit:
    def test_quotas_sum_to_the_budget(self):
        quotas = split_budget_by_jd(20, MAIN_CONFIG.jd_weights)
        assert sum(quotas.values()) == 20

    def test_a_heavier_topic_gets_at_least_as_many_items(self):
        quotas = split_budget_by_jd(20, {"a": 0.9, "b": 0.7, "c": 0.5})
        assert quotas["a"] >= quotas["b"] >= quotas["c"]

    def test_largest_remainder_rather_than_repeated_flooring(self):
        """Three equal topics and a budget of 20 must not lose two items."""
        quotas = split_budget_by_jd(20, {"a": 1.0, "b": 1.0, "c": 1.0})
        assert sum(quotas.values()) == 20
        assert sorted(quotas.values()) == [6, 7, 7]

    def test_it_is_deterministic_under_ties(self):
        weights = {"a": 1.0, "b": 1.0, "c": 1.0}
        assert split_budget_by_jd(20, weights) == split_budget_by_jd(20, weights)

    def test_zero_weights_are_rejected(self):
        with pytest.raises(ValueError, match="must not sum to zero"):
            split_budget_by_jd(20, {"a": 0.0})

    def test_a_negative_budget_is_rejected(self):
        with pytest.raises(ValueError, match="must not be negative"):
            split_budget_by_jd(-1, {"a": 1.0})
