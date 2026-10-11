"""Make the interview event log append-only in the database, not just in code.

Day 21. ``interview_events`` already had everything the event log needs to be
*read* correctly - the ``(session_id, seq)`` unique index is both the ordering
guarantee and the range-scan index plan section 11.2 asks for. What it did not
have was anything stopping the log from being *rewritten*, which is the one
property every benefit of event sourcing depends on: a replay of an edited log
reproduces a history that never happened.

Two additions:

1. **``CHECK (seq >= 0)``.** ``SESSION_CREATED`` is seq 0 and every later event
   is ``previous + 1`` (enforced by ``app.interview.state.fold``). A negative seq
   cannot be produced by the application, so one in the table is corruption.

2. **A trigger that rejects ``UPDATE`` and direct ``DELETE``.** Rows are facts;
   a correction is a new event. ``DELETE`` is still allowed when it arrives by
   ``ON DELETE CASCADE`` from ``interview_sessions`` (and so from ``users``) -
   deleting an account must delete its history. The two are told apart by
   ``pg_trigger_depth()``: a direct statement fires this trigger at depth 1, a
   cascade fires it from inside the foreign key's own trigger at depth 2. That
   was verified on pg16 before relying on it, and an integration test pins it.

   ``TRUNCATE`` does not fire row triggers and is not blocked. It needs the
   table-level ``TRUNCATE`` privilege, which the application never uses; the
   test suite does, to reset state between tests.

**Safe on a populated table.** Adding the CHECK validates existing rows, which
all have seq >= 0 because nothing has ever written anything else (Day 2's
schema test writes seq 0). The trigger affects only future statements.

Revision ID: 0003
Revises: 0002
"""

from __future__ import annotations

from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None

_FUNCTION = "interview_events_append_only"
_TRIGGER = "trg_interview_events_append_only"


def upgrade() -> None:
    op.create_check_constraint(
        op.f("ck_interview_events_seq_non_negative"),
        "interview_events",
        "seq >= 0",
    )
    op.execute(
        f"""
        CREATE FUNCTION {_FUNCTION}() RETURNS trigger
        LANGUAGE plpgsql AS $$
        BEGIN
            IF TG_OP = 'DELETE' AND pg_trigger_depth() > 1 THEN
                -- Cascaded from interview_sessions: the session itself is going.
                RETURN OLD;
            END IF;
            RAISE EXCEPTION 'interview_events is append-only: % is not allowed', TG_OP
                USING ERRCODE = 'restrict_violation';
        END
        $$;
        """
    )
    op.execute(
        f"""
        CREATE TRIGGER {_TRIGGER}
        BEFORE UPDATE OR DELETE ON interview_events
        FOR EACH ROW EXECUTE FUNCTION {_FUNCTION}();
        """
    )


def downgrade() -> None:
    op.execute(f"DROP TRIGGER IF EXISTS {_TRIGGER} ON interview_events")
    op.execute(f"DROP FUNCTION IF EXISTS {_FUNCTION}()")
    op.drop_constraint(
        op.f("ck_interview_events_seq_non_negative"), "interview_events", type_="check"
    )
