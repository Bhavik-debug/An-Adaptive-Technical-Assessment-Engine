"""A small, fast experiment for the Day 13 tests.

The committed experiment is 200 candidates over a 192-item bank; running that in
the unit suite would add minutes to every test run for no extra confidence.
``TINY`` is the same machinery at a size that runs in milliseconds - and every
test that cares about the *real* configuration asserts against
``app.simulation.config.MAIN_CONFIG`` directly rather than against this.
"""

from __future__ import annotations

from app.simulation.config import ExperimentConfig

#: Six candidates, four items per subtopic, six items per session.  Small enough
#: to be instant, large enough that the hard constraints, the quotas and the
#: stopping rule all still have something to do.
TINY = ExperimentConfig(
    seed=1234,
    candidate_count=6,
    items_per_subtopic=4,
    item_budget=6,
    time_budget_s=1200.0,
    label="tiny",
)
