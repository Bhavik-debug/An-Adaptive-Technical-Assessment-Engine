"""The four hard constraints - in Python, and as the SQL they compile to.

The boundaries are the whole point of this file. `|b - theta| <= 1.5` is not the
same rule as `< 1.5`, and `time_estimate_s <= time_remaining` is not the same as
`<`; each is tested at, just inside, and just outside its limit.

The SQL half needs no database: SQLAlchemy will compile a statement against the
PostgreSQL dialect on its own, which is enough to assert that all four clauses
are present and - more importantly - that every caller-supplied value arrives as
a bound parameter rather than as interpolated text.
"""

from __future__ import annotations

import pytest
from sqlalchemy.dialects import postgresql

from app.ability import AbilityState
from app.selection.constraints import (
    DIFFICULTY_WINDOW,
    REASON_ALREADY_ASKED,
    REASON_DIFFICULTY_WINDOW,
    REASON_TIME_REMAINING,
    REASON_TOPIC_QUOTA,
    eligible_items_statement,
    filter_eligible,
    ineligibility_reason,
    is_eligible,
)
from tests.unit.selection.conftest import item, state


def compiled(stmt):
    return stmt.compile(dialect=postgresql.dialect())


# ---------------------------------------------------------------------------
# 1. no repeats
# ---------------------------------------------------------------------------


class TestNoRepeats:
    def test_an_item_already_asked_is_excluded(self):
        current = state(asked=[item("q1")])
        assert not is_eligible(item("q1"), current)
        assert ineligibility_reason(item("q1"), current) == REASON_ALREADY_ASKED

    def test_a_fresh_item_survives(self):
        assert is_eligible(item("q2"), state(asked=[item("q1")]))

    def test_identical_content_under_a_different_id_is_not_a_repeat(self):
        """The rule is about ids. Near-duplicates are the redundancy term's job,
        and conflating the two would make a soft penalty into a hard filter."""
        current = state(asked=[item("q1", b=0.0)])
        assert is_eligible(item("q2", b=0.0), current)

    def test_the_constructor_refuses_a_history_with_a_duplicate(self):
        with pytest.raises(ValueError, match="twice"):
            state(asked=[item("q1"), item("q1")])


# ---------------------------------------------------------------------------
# 2. per-topic quota
# ---------------------------------------------------------------------------


class TestTopicQuota:
    def test_a_topic_under_quota_is_eligible(self):
        current = state(targets={"systems": 3}, asked=[item("q0")])
        assert is_eligible(item("q1"), current)

    def test_a_topic_exactly_at_quota_is_excluded(self):
        current = state(targets={"systems": 2}, asked=[item("q0"), item("q1")])
        assert ineligibility_reason(item("q2"), current) == REASON_TOPIC_QUOTA

    def test_a_topic_over_quota_is_excluded(self):
        asked = [item(f"q{i}") for i in range(4)]
        current = state(targets={"systems": 2}, asked=asked)
        assert ineligibility_reason(item("q9"), current) == REASON_TOPIC_QUOTA

    def test_a_topic_the_blueprint_never_asked_for_is_excluded(self):
        current = state(targets={"algorithms": 3})
        assert ineligibility_reason(item("q1", topic="systems"), current) == REASON_TOPIC_QUOTA

    def test_another_topics_items_do_not_consume_this_topics_quota(self):
        asked = [item("q0", topic="algorithms", subtopic="sorting")]
        current = state(targets={"systems": 1, "algorithms": 1}, asked=asked)
        assert is_eligible(item("q1", topic="systems"), current)

    def test_topics_with_quota_left_is_sorted_and_excludes_the_served(self):
        current = state(
            targets={"systems": 1, "algorithms": 2, "databases": 0},
            asked=[item("q0", topic="systems")],
        )
        assert current.topics_with_quota_left == ("algorithms",)


# ---------------------------------------------------------------------------
# 3. the difficulty window
# ---------------------------------------------------------------------------


class TestDifficultyWindow:
    def test_the_window_is_one_point_five(self):
        assert DIFFICULTY_WINDOW == 1.5

    @pytest.mark.parametrize("gap", [0.0, 0.75, 1.4999])
    def test_inside_the_window_is_eligible(self, gap):
        current = state(theta=0.5)
        assert is_eligible(item(b=0.5 + gap), current)
        assert is_eligible(item(b=0.5 - gap), current)

    def test_exactly_on_the_boundary_is_eligible(self):
        """`<= 1.5`, not `< 1.5`."""
        current = state(theta=0.5)
        assert is_eligible(item(b=2.0), current)
        assert is_eligible(item(b=-1.0), current)

    @pytest.mark.parametrize("gap", [1.5001, 2.0, 4.0])
    def test_outside_the_window_is_excluded(self, gap):
        current = state(theta=0.5)
        assert ineligibility_reason(item(b=0.5 + gap), current) == REASON_DIFFICULTY_WINDOW
        assert ineligibility_reason(item(b=0.5 - gap), current) == REASON_DIFFICULTY_WINDOW

    def test_the_window_is_measured_against_the_items_own_subtopic(self):
        ability = {
            "caching": AbilityState(theta=2.0, rd=0.5),
            "sorting": AbilityState(theta=-2.0, rd=0.5),
        }
        current = state(ability=ability, targets={"systems": 5})
        assert is_eligible(item("q1", subtopic="caching", b=2.0), current)
        assert not is_eligible(item("q2", subtopic="sorting", b=2.0), current)

    def test_an_unmeasured_subtopic_falls_back_to_the_cold_start_prior(self):
        """PRIOR_ABILITY has theta 0, so the window is |b| <= 1.5."""
        current = state(ability={})
        assert is_eligible(item(b=1.5, subtopic="never_seen"), current)
        assert not is_eligible(item(b=1.6, subtopic="never_seen"), current)

    def test_the_window_is_overridable_without_being_redefined(self):
        current = state(theta=0.0)
        assert not is_eligible(item(b=1.8), current)
        assert is_eligible(item(b=1.8), current, difficulty_window=2.0)


# ---------------------------------------------------------------------------
# 4. time remaining
# ---------------------------------------------------------------------------


class TestTimeRemaining:
    def test_an_item_that_fits_is_eligible(self):
        assert is_eligible(item(seconds=120), state(time_left=600.0))

    def test_an_item_that_exactly_consumes_the_time_is_eligible(self):
        """`<=`, not `<`: a question that uses the last 120 seconds is askable."""
        assert is_eligible(item(seconds=120), state(time_left=120.0))

    def test_an_item_that_does_not_fit_is_excluded(self):
        current = state(time_left=119.0)
        assert ineligibility_reason(item(seconds=120), current) == REASON_TIME_REMAINING

    def test_no_time_left_excludes_everything_with_a_positive_estimate(self):
        current = state(time_left=0.0)
        assert ineligibility_reason(item(seconds=1), current) == REASON_TIME_REMAINING

    def test_near_zero_time_is_handled_without_arithmetic_trouble(self):
        current = state(time_left=0.5)
        assert ineligibility_reason(item(seconds=1), current) == REASON_TIME_REMAINING

    def test_negative_time_is_a_caller_bug_and_is_rejected(self):
        with pytest.raises(ValueError, match="time_left_s must not be negative"):
            state(time_left=-1.0)


# ---------------------------------------------------------------------------
# 5. the four together
# ---------------------------------------------------------------------------


class TestFilteringTogether:
    def test_the_first_failing_constraint_is_the_reported_reason(self):
        """Plan order: repeats, then quota, then difficulty, then time."""
        current = state(targets={"systems": 0}, asked=[item("q1")], time_left=10.0)
        assert ineligibility_reason(item("q1", b=9.0, seconds=99), current) == REASON_ALREADY_ASKED

    def test_filter_eligible_keeps_only_the_survivors_in_order(self):
        current = state(theta=0.0, targets={"systems": 5}, asked=[item("q1")], time_left=300.0)
        candidates = [
            item("q1"),  # already asked
            item("q2", b=0.5),  # fine
            item("q3", b=2.4),  # outside the window
            item("q4", seconds=400),  # too long
            item("q5", topic="algorithms", subtopic="sorting"),  # no quota
            item("q6", b=-1.5),  # exactly on the boundary
        ]
        assert [i.id for i in filter_eligible(candidates, current)] == ["q2", "q6"]

    def test_an_empty_pool_is_an_empty_list_not_an_error(self):
        current = state(targets={})
        assert filter_eligible([item("q1")], current) == []


# ---------------------------------------------------------------------------
# 6. the same rules as SQL
# ---------------------------------------------------------------------------


class TestEligibilitySql:
    def test_all_four_constraints_appear_in_the_where_clause(self):
        current = state(theta=0.4, asked=[item("q0")], time_left=300.0)
        sql = str(compiled(eligible_items_statement(current)))
        assert "questions.id NOT IN" in sql
        assert "questions.topic_key IN" in sql
        assert "abs(questions.difficulty_b" in sql
        assert "questions.time_estimate_s <=" in sql

    def test_every_value_is_bound_rather_than_interpolated(self):
        """The security property: no caller value is ever formatted into SQL."""
        current = state(
            ability={"caching'; DROP TABLE questions; --": AbilityState(theta=1.0, rd=0.5)},
            asked=[item("q0'; DROP TABLE questions; --")],
            time_left=300.0,
        )
        statement = compiled(eligible_items_statement(current))
        sql = str(statement)
        assert "DROP TABLE" not in sql
        assert "DROP TABLE questions; --" in str(statement.params.values())

    def test_the_difficulty_window_is_a_bound_parameter(self):
        statement = compiled(eligible_items_statement(state(), difficulty_window=2.0))
        assert 2.0 in statement.params.values()

    def test_the_theta_case_carries_one_branch_per_measured_subtopic(self):
        ability = {
            "caching": AbilityState(theta=1.0, rd=0.5),
            "sorting": AbilityState(theta=-1.0, rd=0.5),
        }
        statement = compiled(eligible_items_statement(state(ability=ability)))
        assert str(statement).count("WHEN") == 2
        assert 1.0 in statement.params.values()
        assert -1.0 in statement.params.values()

    def test_with_no_measurements_the_case_collapses_to_the_prior(self):
        statement = compiled(eligible_items_statement(state(ability={})))
        assert "CASE" not in str(statement)

    def test_no_not_in_clause_is_emitted_when_nothing_has_been_asked(self):
        sql = str(compiled(eligible_items_statement(state())))
        assert "NOT IN" not in sql

    def test_the_statement_selects_only_the_columns_selection_reads(self):
        sql = str(compiled(eligible_items_statement(state())))
        assert "questions.text" not in sql
        assert "expected_concepts" not in sql
        assert "reference_answer" not in sql

    def test_it_orders_by_id_and_limits(self):
        sql = str(compiled(eligible_items_statement(state(), limit=25)))
        assert "ORDER BY questions.id" in sql
        assert "LIMIT" in sql


class TestSqlShortCircuits:
    """The two states whose answer is known without a round trip."""

    async def test_no_time_left_never_touches_the_session(self):
        from app.selection.constraints import eligible_items

        assert await eligible_items(None, state(time_left=0.0)) == []  # type: ignore[arg-type]

    async def test_no_topic_with_quota_left_never_touches_the_session(self):
        from app.selection.constraints import eligible_items

        assert await eligible_items(None, state(targets={})) == []  # type: ignore[arg-type]
