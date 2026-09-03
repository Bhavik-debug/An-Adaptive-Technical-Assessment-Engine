"""The Beta response model.

Three things must hold, and they are tested separately because they fail
separately: the *expectation* comes from Day 11's 2PL and nothing else, the
*draw* is a Beta centred on it, and the whole thing is reproducible from the
experiment seed.
"""

from __future__ import annotations

import random
import statistics

import pytest

from app.ability import probability_correct
from app.simulation.config import RESPONSE_CONCENTRATION, RESPONSE_P_EPSILON
from app.simulation.environment import build_bank, build_candidate
from app.simulation.response import (
    beta_parameters,
    draw_score,
    expected_score,
    graded_score,
    response_rng,
)
from tests.unit.simulation.conftest import TINY

BANK = build_bank(TINY)
CANDIDATE = build_candidate(TINY, 0)


def item_for(subtopic: str, index: int = 0):
    return [i for i in BANK if i.subtopic_key == subtopic][index]


class TestExpectedScore:
    def test_it_is_day_elevens_2pl_on_ground_truth(self):
        """Not a reimplementation - the production function, called with the
        hidden theta instead of the estimated one."""
        item = item_for("arrays")
        assert expected_score(CANDIDATE, item) == probability_correct(
            CANDIDATE.true_theta["arrays"], item.difficulty_b, item.discrimination_a
        )

    def test_it_is_a_probability(self):
        for item in BANK:
            assert 0.0 <= expected_score(CANDIDATE, item) <= 1.0

    def test_a_harder_item_lowers_the_expectation(self):
        items = sorted(
            (i for i in BANK if i.subtopic_key == "arrays"), key=lambda i: i.difficulty_b
        )
        scores = [expected_score(CANDIDATE, i) for i in items]
        assert scores[0] > scores[-1]

    def test_a_stronger_candidate_scores_higher_on_the_same_item(self):
        item = item_for("arrays")
        population = [build_candidate(TINY, i) for i in range(TINY.candidate_count)]
        weakest = min(population, key=lambda c: c.true_theta["arrays"])
        strongest = max(population, key=lambda c: c.true_theta["arrays"])
        assert expected_score(strongest, item) > expected_score(weakest, item)

    def test_it_uses_the_items_own_discrimination(self):
        """`a` is used here and *not* in Day 12's selection score, so the world
        distinguishes a sharp item from a flat one while the policy cannot."""
        import dataclasses

        item = dataclasses.replace(item_for("arrays"), difficulty_b=0.0, discrimination_a=1.0)
        sharp = dataclasses.replace(item, discrimination_a=2.0)
        theta = CANDIDATE.true_theta["arrays"]
        if theta > 0:
            assert expected_score(CANDIDATE, sharp) > expected_score(CANDIDATE, item)
        else:
            assert expected_score(CANDIDATE, sharp) < expected_score(CANDIDATE, item)


class TestBetaParameters:
    def test_the_mean_is_p(self):
        for p in (0.1, 0.25, 0.5, 0.75, 0.9):
            alpha, beta = beta_parameters(p)
            assert alpha / (alpha + beta) == pytest.approx(p)

    def test_the_concentration_is_the_total(self):
        alpha, beta = beta_parameters(0.4, concentration=10.0)
        assert alpha + beta == pytest.approx(10.0)

    def test_a_higher_concentration_means_a_tighter_grader(self):
        loose_a, loose_b = beta_parameters(0.5, concentration=4.0)
        tight_a, tight_b = beta_parameters(0.5, concentration=100.0)
        loose_var = (loose_a * loose_b) / ((loose_a + loose_b) ** 2 * (loose_a + loose_b + 1))
        tight_var = (tight_a * tight_b) / ((tight_a + tight_b) ** 2 * (tight_a + tight_b + 1))
        assert tight_var < loose_var

    @pytest.mark.parametrize("p", [0.0, 1.0])
    def test_the_extremes_are_clamped_rather_than_degenerate(self, p):
        """alpha = 0 is not a distribution; the clamp keeps the model defined
        exactly where the bank's difficulty range is widest."""
        alpha, beta = beta_parameters(p)
        assert alpha > 0.0
        assert beta > 0.0
        assert min(alpha, beta) == pytest.approx(RESPONSE_P_EPSILON * RESPONSE_CONCENTRATION)

    @pytest.mark.parametrize("bad", [-0.01, 1.01])
    def test_a_p_outside_zero_and_one_is_rejected(self, bad):
        with pytest.raises(ValueError, match=r"p must be in \[0, 1\]"):
            beta_parameters(bad)

    def test_a_non_positive_concentration_is_rejected(self):
        with pytest.raises(ValueError, match="concentration must be positive"):
            beta_parameters(0.5, concentration=0.0)


class TestDrawScore:
    def test_it_is_always_a_valid_day_eleven_score(self):
        """Day 11 *rejects* a score outside [0, 1] rather than clamping it, so a
        floating-point edge here would look like a scoring bug there."""
        rng = random.Random(0)
        for _ in range(2000):
            assert 0.0 <= draw_score(rng.random(), rng) <= 1.0

    def test_the_sample_mean_tracks_p(self):
        for p in (0.2, 0.5, 0.8):
            rng = random.Random(7)
            draws = [draw_score(p, rng) for _ in range(4000)]
            assert statistics.fmean(draws) == pytest.approx(p, abs=0.02)

    def test_the_spread_matches_the_concentration(self):
        """Var = p(1-p)/(k+1); at p = 0.5 and k = 10 that is sd 0.151."""
        rng = random.Random(11)
        draws = [draw_score(0.5, rng, concentration=10.0) for _ in range(4000)]
        assert statistics.stdev(draws) == pytest.approx(0.151, abs=0.01)

    def test_a_tighter_grader_produces_less_noise(self):
        rng_loose, rng_tight = random.Random(3), random.Random(3)
        loose = [draw_score(0.5, rng_loose, concentration=4.0) for _ in range(2000)]
        tight = [draw_score(0.5, rng_tight, concentration=100.0) for _ in range(2000)]
        assert statistics.stdev(tight) < statistics.stdev(loose)

    def test_it_is_deterministic_under_a_seeded_generator(self):
        assert [draw_score(0.4, random.Random(5)) for _ in range(3)][0] == draw_score(
            0.4, random.Random(5)
        )


class TestCommonRandomNumbers:
    def test_the_same_pair_always_gets_the_same_score(self):
        """The fairness control: which policy asked, and when, cannot change the
        answer a candidate gives to a question."""
        item = item_for("graphs")
        first = graded_score(TINY.seed, CANDIDATE, item)
        second = graded_score(TINY.seed, CANDIDATE, item)
        assert first == second

    def test_a_different_item_gets_a_different_draw(self):
        assert graded_score(TINY.seed, CANDIDATE, item_for("arrays", 0)) != graded_score(
            TINY.seed, CANDIDATE, item_for("arrays", 1)
        )

    def test_a_different_candidate_gets_a_different_draw(self):
        item = item_for("arrays")
        assert graded_score(TINY.seed, CANDIDATE, item) != graded_score(
            TINY.seed, build_candidate(TINY, 1), item
        )

    def test_a_different_experiment_seed_gives_a_different_world(self):
        item = item_for("arrays")
        assert graded_score(TINY.seed, CANDIDATE, item) != graded_score(
            TINY.seed + 1, CANDIDATE, item
        )

    def test_the_stream_is_keyed_only_by_candidate_and_item(self):
        item = item_for("caching")
        left = response_rng(TINY.seed, CANDIDATE, item)
        right = response_rng(TINY.seed, CANDIDATE, item)
        assert [left.random() for _ in range(5)] == [right.random() for _ in range(5)]
